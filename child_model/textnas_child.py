# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import math
import numpy as np
import torch.nn.functional as Func
from torch import nn
import torch
from torch.autograd import Variable as V

from child_model.ops import *
from child_model.utils import get_length, global_avg_pool, global_max_pool
from common.utils import count_model_params

class TextNASChild(nn.Module):
  def __init__(self,
               embedding,
               fixed_arc=None,
               num_layers=2,
               out_filters_scale=1,
               out_filters=24,
               cnn_keep_prob=1.0,
               final_output_keep_prob=1.0,
               lstm_out_keep_prob=1.0,
               embed_keep_prob=1.0,
               attention_keep_prob=1.0,
               multi_path=False,
               embedding_model="none",
               all_layer_output=False,
               output_linear_combine=False,
               num_last_layer_output=0,
               is_mask=False,
               output_type="avg_pool",
               max_input_length=32,
               class_num=5,
               is_cuda=True,
               *args,
               **kwargs):
    super(TextNASChild, self).__init__()

    self.fixed_arc = fixed_arc
    self.max_input_length = max_input_length
    self.all_layer_output = all_layer_output
    self.output_linear_combine = output_linear_combine
    self.num_last_layer_output = max(num_last_layer_output, 0)
    self.is_mask = is_mask
    self.output_type = output_type
    self.multi_path = multi_path
    self.embedding_model = embedding_model
    self.out_filters = out_filters * out_filters_scale
    self.num_layers = num_layers
    self.class_num = class_num
    self.cnn_keep_prob = cnn_keep_prob
    self.final_output_keep_prob = final_output_keep_prob
    self.lstm_out_keep_prob = lstm_out_keep_prob
    self.embed_keep_prob = embed_keep_prob
    self.attention_keep_prob = attention_keep_prob

    self.is_cuda = is_cuda

    self.fixed_flag = False
    if self.fixed_arc is not None:
      fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
      self.sample_arc = fixed_arc
      self.fixed_flag = True

    layers = []

    out_filters = self.out_filters
    if self.embedding_model == "none":
      self.embedding = nn.Parameter(torch.from_numpy(embedding["none"]))
    elif self.embedding_model == "glove":
      self.embedding = nn.Parameter(torch.from_numpy(embedding["glove"]))
    else:
      raise NotImplementedError("Unknown embedding_model '{}'".format(embedding_model))

    # _conv_opt + batch_norm
    self.init_conv = ConvOpt(1, self.embedding.size()[1], out_filters, cnn_keep_prob, False, True, is_cuda=self.is_cuda)

    for layer_id in range(self.num_layers):
      layers.append(self.make_fixed_layer(layer_id, out_filters))  # construct all branches

    self.layers = nn.ModuleList(layers)
    if self.all_layer_output and self.output_linear_combine: #use linear_combine
      self._linear_combine = LinearCombine(self.num_layers, is_cuda=self.is_cuda)
    self.linear_out = nn.Linear(out_filters, self.class_num)

    self.embed_dropout= nn.Dropout(p=(1 - embed_keep_prob))
    self.output_dropout= nn.Dropout(p=(1 - final_output_keep_prob))

  def get_sample_arc(self, sample_arc):
    self.sample_arc = sample_arc

  def forward(self, sent_ids, mask):
    seq = Func.embedding(sent_ids.long(), self.embedding)
    seq = self.embed_dropout(seq)

    seq = torch.transpose(seq, 1, 2)  # from (N, L, C) -> (N, C, L)
    inp_c, inp_l = seq.shape[1:]

    x = self.init_conv(seq, mask)

    start_idx = 0
    prev_layers = []
    final_flags = []

    for layer_id in range(self.num_layers):  # run layers
      layer = self.layers[layer_id]
      x = self.run_fixed_layer(x, mask, prev_layers, layer, layer_id, start_idx,
                               final_flags=final_flags)  # run needed branches
      prev_layers.append(x)
      final_flags.append(1)

      start_idx += 1 + layer_id
      if self.multi_path:
        start_idx += 1

    final_layers = []
    final_layers_idx = []
    for i in range(0, len(prev_layers)):
      if self.all_layer_output:
        if self.num_last_layer_output == 0:
          final_layers.append(prev_layers[i])
          final_layers_idx.append(i)
        elif i >= max((len(prev_layers) - self.num_last_layer_output), 0):
          final_layers.append(prev_layers[i])
          final_layers_idx.append(i)
      else:
        final_layers.append(final_flags[i] * prev_layers[i])

    if self.all_layer_output and self.output_linear_combine: #all layer ooutput and use linear_combine
      x = self._linear_combine(torch.stack(final_layers))
    else:
      x = sum(final_layers)
      if not self.all_layer_output:
        x /= sum(final_flags)
      else:
        x /= len(final_layers)

    class_num = self.class_num
    batch_size = x.size()[0]
    inp_c = x.size()[1]
    inp_l = x.size()[2]
    if self.output_type == "cls":
      x = x[:,:,:,0] #NCHW, H=1, W=1)
      x = torch.reshape(x, [-1, inp_c])
    elif self.output_type == "avg_pool":
      x = global_avg_pool(x, mask)  # NC
    elif self.output_type == "max_pool":
      x = global_max_pool(x, mask)  # NC
    else:
      raise ValueError("Unsupported output type.")

    x = self.output_dropout(x)
    x = self.linear_out(x)
    return x

  def make_fixed_layer(self, layer_id, out_filters):
    size = [1, 3, 5, 7]
    separables = [False, False, False, False]

    branches = []

    if self.fixed_flag:
      if self.multi_path:
        branch_id = (layer_id + 1) * (layer_id + 2) // 2
      else:
        branch_id = (layer_id) * (layer_id + 1) // 2
      bn_flag = False
      for i in range(layer_id):
        if self.sample_arc[branch_id + 1 + i] == 1:
          bn_flag = True
      branch_id = self.sample_arc[branch_id]

      for operation_id in [0, 1, 2, 3]:  # conv_opt
        if branch_id == operation_id:
          filter_size = size[operation_id]
          separable = separables[operation_id]
          op = ConvOpt(filter_size, out_filters, out_filters, self.cnn_keep_prob, False, True, is_cuda=self.is_cuda)
          branches.append(op)
      if branch_id == 4:
        branches.append(AvgPoolOpt(3, False, True, is_cuda=self.is_cuda))
      elif branch_id == 5:
        branches.append(MaxPoolOpt(3, False, True, is_cuda=self.is_cuda))
      elif branch_id == 6:
        branches.append(RnnOpt(out_filters, self.lstm_out_keep_prob, is_cuda=self.is_cuda))
      elif branch_id == 7:
        branches.append(AttentionOpt(out_filters, 4, self.attention_keep_prob,
                                    self.is_mask, is_cuda=self.is_cuda))
      branches = nn.ModuleList(branches)
      bn = None
      if bn_flag:
        bn = BatchNorm(self.out_filters, False, True, is_cuda=self.is_cuda)
    else:
      for operation_id in [0, 1, 2, 3]:  # conv_opt
        filter_size = size[operation_id]
        separable = separables[operation_id]
        op = ConvOpt(filter_size, out_filters, out_filters, self.cnn_keep_prob, False, True, is_cuda=self.is_cuda)
        branches.append(op)
      branches.append(AvgPoolOpt(3, False, True, is_cuda=self.is_cuda))
      branches.append(MaxPoolOpt(3, False, True, is_cuda=self.is_cuda))
      branches.append(RnnOpt(out_filters, self.lstm_out_keep_prob, is_cuda=self.is_cuda))
      branches.append(AttentionOpt(out_filters, 4, self.attention_keep_prob,
                                  self.is_mask, is_cuda=self.is_cuda))
      branches = nn.ModuleList(branches)
      bn = BatchNorm(self.out_filters, False, True, is_cuda=self.is_cuda)

    return nn.ModuleList([branches, bn])

  def run_fixed_layer(self, x, mask, prev_layers, layers, layer_id, start_idx, final_flags):
    layer = layers[0]
    bn = layers[1]

    if len(prev_layers) > 0:
      if self.multi_path:
        pre_layer_id = self.sample_arc[start_idx]
        num_pre_layers = len(prev_layers)
        if num_pre_layers > 5:
          num_pre_layers = 5
        if pre_layer_id >= num_pre_layers:
          final_flags[-1] = 0
          inputs = prev_layers[-1]
        else:
          layer_idx = len(prev_layers) - 1 - pre_layer_id
          final_flags[layer_idx] = 0
          inputs = prev_layers[layer_idx]
      else:
        inputs = prev_layers[-1]
        final_flags[-1] = 0
    else:
      inputs = x

    if self.multi_path:
      start_idx += 1

    branches = []
    # run branch op
    if self.fixed_flag:
      branch_id = 0
    else:
      branch_id = self.sample_arc[start_idx]
    branches.append(layer[branch_id](inputs, mask))

    if layer_id == 0:
      out = sum(branches)
    else:
      skip_start = start_idx + 1
      skip = self.sample_arc[skip_start:skip_start + layer_id]

      res_layers = []
      for i in range(layer_id):
        if skip[i] == 1:
          res_layers.append(prev_layers[i])
          final_flags[i] = 0
      prev = branches + res_layers
      out = sum(prev)  # tensor sum
      if len(prev) > 1:
        out = bn(out, mask)

    return out
