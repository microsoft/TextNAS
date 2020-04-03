# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as Func
from torch import optim

from common.utils import train_ops
from common.utils import update_lr

def lstm(x, prev_c, prev_h, w):
  ifog = torch.matmul(torch.cat((x, prev_h), dim=1), w)
  i, f, o, g = torch.split(ifog, ifog.size()[1] // 4, dim=1)
  i = torch.sigmoid(i)
  f = torch.sigmoid(f)
  o = torch.sigmoid(o)
  g = torch.tanh(g)
  next_c = i * g + f * prev_c
  next_h = o * torch.tanh(next_c)
  return next_c, next_h

def stack_lstm(x, prev_c, prev_h, w):
  next_c, next_h = [], []
  for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
    inputs = x if layer_id == 0 else next_h[-1]
    curr_c, curr_h = lstm(inputs, _c, _h, _w)
    next_c.append(curr_c)
    next_h.append(curr_h)
  return next_c, next_h

class RLController(nn.Module):
  def __init__(self,
               search_for="macro",
               num_layers=4,
               num_branches=6,
               lstm_size=32,
               lstm_num_layers=2,
               tanh_constant=None,
               temperature=None,
               lr_init=1e-3,
               l2_reg=0,
               entropy_weight=None,
               clip_mode=None,
               grad_bound=5.0,
               bl_dec=0.999,
               optim_algo="adam",
               skip_target=0.8,
               skip_weight=0.5,
               pre_idxs=[],
               multi_path=True,
               is_cuda=False,
               *args,
               **kwargs):

    print("-" * 80)
    print("Building Controller")

    super(RLController, self).__init__()
    self.multi_path = multi_path
    self.search_for = search_for
    self.num_layers = num_layers
    self.num_branches = num_branches

    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers
    self.tanh_constant = tanh_constant
    self.temperature = temperature
    self.lr_init = lr_init
    self.l2_reg = l2_reg
    self.entropy_weight = entropy_weight
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.bl_dec = bl_dec

    self.skip_target = skip_target
    self.skip_weight = skip_weight

    self.optim_algo = optim_algo
    self.pre_idxs = pre_idxs
    self.is_cuda = is_cuda

    self.baseline = 0.0

    self._create_params()

    # use constant lr
    if self.optim_algo == "adam":
      self.optimizer = optim.Adam(self.parameters(), lr=self.lr_init, eps=1e-3, weight_decay=self.l2_reg)
    elif self.optim_algo == "momentum":
      self.optimizer = optim.SGD(self.parameters(), lr=self.lr_init, momentum=0.9, weight_decay=self.l2_reg, nesterov=True)
    else:
      raise ValueError("Unknown optim_algo {}".format(self.optim_algo))

  def _create_params(self):
    self.w_lstm = []
    for layer_id in range(self.lstm_num_layers):
      w = nn.Parameter(torch.Tensor(2 * self.lstm_size, 4 * self.lstm_size))
      self.w_lstm.append(w)
    self.w_lstm = nn.ParameterList(self.w_lstm)

    self.g_emb = nn.Parameter(torch.Tensor(1, self.lstm_size))
    self.w_emb = nn.Parameter(torch.Tensor(self.num_branches, self.lstm_size))

    self.layer_emb = nn.Parameter(torch.Tensor(self.num_layers, self.lstm_size))

    self.w_soft = nn.Parameter(torch.Tensor(self.lstm_size, self.num_branches))
    self.w_layer = nn.Parameter(torch.Tensor(self.lstm_size, 5))

    self.w_attn_1 = nn.Parameter(torch.Tensor(self.lstm_size, self.lstm_size))
    self.w_attn_2 = nn.Parameter(torch.Tensor(self.lstm_size, self.lstm_size))
    self.v_attn = nn.Parameter(torch.Tensor(self.lstm_size, 1))

    print("create_parameters finish")
    for name, var in self.named_parameters(): #init
      nn.init.uniform_(var, -0.1, 0.1)

  def _build_sampler(self, pre_idxs=[]):
    """Build the sampler ops and the log_prob ops."""

    anchors = []
    anchors_w_1 = []

    arc_seq = []
    entropys = []
    log_probs = []
    skip_count = []
    skip_penaltys = []

    prev_c = [torch.zeros(1, self.lstm_size, dtype=torch.float32) for _ in
              range(self.lstm_num_layers)]
    prev_h = [torch.zeros(1, self.lstm_size, dtype=torch.float32) for _ in
              range(self.lstm_num_layers)]
    inputs = self.g_emb
    skip_targets = torch.tensor([1.0 - self.skip_target, self.skip_target],
                               dtype=torch.float32, requires_grad=False)

    if self.is_cuda:
      for i in range(len(prev_c)):
        prev_c[i] = prev_c[i].cuda()
      for i in range(len(prev_h)):
        prev_h[i] = prev_h[i].cuda()
      skip_targets = skip_targets.cuda()

    idx = 0
    for layer_id in range(self.num_layers):
      # choose a previous layer as input
      if self.multi_path == True:
        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
        prev_c, prev_h = next_c, next_h

        pre_num_layers = layer_id
        if layer_id == 0:
          pre_num_layers += 1

        if pre_num_layers > 5:
          pre_num_layers = 5
        left_num_layers = 5 - pre_num_layers

        mask1 = torch.full((self.lstm_size, pre_num_layers), 1, dtype=torch.bool)
        mask2 = torch.full((self.lstm_size, left_num_layers), 0, dtype=torch.bool)
        mask = torch.cat((mask1, mask2), dim=1)
        if self.is_cuda:
          mask = mask.cuda()
        w_layer = torch.masked_select(self.w_layer, mask.bool())
        w_layer = torch.reshape(w_layer, [self.lstm_size, pre_num_layers])
        logit = torch.matmul(next_h[-1], w_layer)
        if self.temperature is not None:
          logit = logit / self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * torch.tanh(logit)
        if self.search_for == "macro" or self.search_for == "branch":
          if idx < len(pre_idxs):
            input_layer_id = pre_idxs[idx]
            idx += 1
          else:
            input_layer_id = torch.multinomial(Func.softmax(logit, dim=-1), 1)
            input_layer_id = input_layer_id.long()
            input_layer_id = torch.reshape(input_layer_id, [1])
        else:
          raise ValueError("Unknown search_for {}".format(self.search_for))
        arc_seq.append(input_layer_id)
        log_prob = Func.cross_entropy(logit, input_layer_id)
        log_probs.append(log_prob)
        entropy = torch.detach(log_prob * torch.exp(-log_prob))
        entropys.append(entropy)
        inputs = Func.embedding(input_layer_id, self.layer_emb)

      # choose branch id
      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h
      logit = torch.matmul(next_h[-1], self.w_soft)
      if self.temperature is not None:
        logit = logit / self.temperature
      if self.tanh_constant is not None:
        logit = self.tanh_constant * torch.tanh(logit)
      if self.search_for == "macro" or self.search_for == "branch":
        if idx < len(pre_idxs):
          branch_id = pre_idxs[idx]
          idx += 1
        else:
          branch_id = torch.multinomial(Func.softmax(logit, dim=-1), 1)
          branch_id = branch_id.long()
          branch_id = torch.reshape(branch_id, [1])
      elif self.search_for == "connection":
        branch_id = torch.tensor([0], dtype=torch.int32, requires_grad=False)
        if self.is_cuda:
          branch_id = branch_id.cuda()
      else:
        raise ValueError("Unknown search_for {}".format(self.search_for))
      arc_seq.append(branch_id)
      log_prob = Func.cross_entropy(logit, branch_id)
      log_probs.append(log_prob)
      entropy = torch.detach(log_prob * torch.exp(-log_prob))
      entropys.append(entropy)
      inputs = Func.embedding(branch_id, self.w_emb)

      # set skip connections
      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h

      if layer_id > 0:
        query = torch.cat(anchors_w_1, dim=0)
        query = torch.tanh(query + torch.matmul(next_h[-1], self.w_attn_2))
        query = torch.matmul(query, self.v_attn)
        logit = torch.cat([-query, query], dim=1)
        if self.temperature is not None:
          logit = logit / self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * torch.tanh(logit)
        skip = torch.multinomial(Func.softmax(logit, dim=1), 1)
        skip = skip.long()
        skip = torch.reshape(skip, [layer_id])
        if (idx < len(pre_idxs)):
          idx += layer_id
        arc_seq.append(torch.reshape(skip, [-1]))

        skip_prob = torch.sigmoid(logit)
        kl = skip_prob * torch.log(skip_prob / skip_targets)
        kl = torch.sum(kl)
        skip_penaltys.append(kl)

        log_prob = Func.cross_entropy(logit, skip)
        log_probs.append(log_prob)

        entropy = torch.detach(log_prob * torch.exp(-log_prob))
        entropys.append(entropy)

        skip = skip.float()
        skip = torch.reshape(skip, [1, layer_id])
        skip_count.append(torch.sum(skip))
        inputs = torch.matmul(skip, torch.cat(anchors, dim=0))
        inputs = inputs / (1.0 + torch.sum(skip))
      else:
        inputs = self.g_emb

      anchors.append(next_h[-1])
      anchors_w_1.append(torch.matmul(next_h[-1], self.w_attn_1))

    arc_seq = torch.cat(arc_seq, dim=0)
    self.sample_arc = torch.reshape(arc_seq, [-1]).cpu().numpy()
    entropys = torch.stack(entropys)
    self.sample_entropy = torch.sum(entropys)

    log_probs = torch.stack(log_probs)
    self.sample_log_probs = log_probs
    self.sample_log_prob = torch.sum(log_probs)

    skip_count = torch.stack(skip_count)
    self.skip_count = torch.sum(skip_count)

    skip_penaltys = torch.stack(skip_penaltys)
    self.skip_penaltys = torch.mean(skip_penaltys)

  def trainer(self, eval_acc, step):
    epoch = step // 500 #

    self.reward = torch.Tensor([eval_acc])

    normalize = self.num_layers * (self.num_layers - 1) / 2
    self.skip_rate = self.skip_count / normalize

    if self.is_cuda:
      self.reward = self.reward.cuda()

    if self.entropy_weight is not None:
      self.reward = self.reward + self.entropy_weight * self.sample_entropy

    self.sample_log_prob = torch.sum(self.sample_log_prob)
    x = (1 - self.bl_dec) * (self.baseline - self.reward)
    self.baseline = self.baseline - x

    self.loss = self.sample_log_prob * (self.reward - self.baseline)
    if self.skip_weight is not None:
      self.loss = self.loss + self.skip_weight * self.skip_penaltys

    self.loss = self.loss.mean()

    self.grad_norm = train_ops(
            self.loss,
            self.parameters(),
            self.optimizer,
            clip_mode="norm",
            grad_bound=self.grad_bound)

    return self.loss, self.sample_entropy, self.lr_init, self.grad_norm, self.reward.mean(), self.sample_log_prob, self.baseline.mean()
