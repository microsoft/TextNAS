# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil
import sys
import random
import json

import time
import datetime
import argparse
from itertools import chain

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as Func

from child_model.sst_dataset import read_data_sst, SSTDataset
from common.utils import train_ops, count_model_params, update_lr
from common.utils import Logger, print_user_flags
from common.utils import DEFINE_boolean, DEFINE_float, DEFINE_integer, DEFINE_string

from controller.rl_controller import RLController
from child_model.textnas_child import TextNASChild

parse = argparse.ArgumentParser()

DEFINE_boolean(parse, "reset_output_dir", False, "Delete output_dir if exists.")

DEFINE_string(parse, "embedding_model", "glove", "")
DEFINE_string(parse, "child_fixed_arc", None, "")
DEFINE_string(parse, "data_path", "", "")
DEFINE_string(parse, "embedding_path", "", "")
DEFINE_string(parse, "output_dir", "", "")
DEFINE_string(parse, "search_for", "macro", "Must be [macro]")
DEFINE_string(parse, "controller_type", "rl", "random")
DEFINE_string(parse, "child_lr_decay_scheme", "cosine", "Strategy to decay learning "
              "rate, must be ['cosine', 'noam', 'exponential', 'auto']")

DEFINE_integer(parse, "batch_size", 128, "")
DEFINE_integer(parse, "eval_batch_size", 128, "")
DEFINE_integer(parse, "class_num", 5, "")
DEFINE_integer(parse, "max_input_length", 64, "")
DEFINE_integer(parse, "global_seed", 1234, "")

DEFINE_integer(parse, "num_epochs", 10, "")
DEFINE_integer(parse, "child_num_layers", 24, "")
DEFINE_integer(parse, "child_filter_size", 5, "")
DEFINE_integer(parse, "child_out_filters", 256, "")
DEFINE_integer(parse, "child_out_filters_scale", 1, "")
DEFINE_integer(parse, "child_num_branches", 8, "")
DEFINE_integer(parse, "child_progressive_branches", 4, "")
DEFINE_integer(parse, "child_lr_T_0", 10, "for lr schedule")
DEFINE_integer(parse, "child_lr_T_mul", 2, "for lr schedule")
DEFINE_integer(parse, "min_count", 1, "")
DEFINE_integer(parse, "num_last_layer_output", 0, "last n layers as output, 0 for all")
DEFINE_float(parse, "train_ratio", 1.0, "")
DEFINE_float(parse, "valid_ratio", 1.0, "")
DEFINE_float(parse, "child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float(parse, "child_lr", 0.02, "")
DEFINE_float(parse, "cnn_keep_prob", 0.8, "")
DEFINE_float(parse, "final_output_keep_prob", 1.0, "")
DEFINE_float(parse, "lstm_out_keep_prob", 0.8, "")
DEFINE_float(parse, "embed_keep_prob", 0.8, "")
DEFINE_float(parse, "attention_keep_prob", 0.8, "")
DEFINE_float(parse, "child_l2_reg", 3e-6, "")
DEFINE_float(parse, "child_lr_max", 0.002, "for lr schedule")
DEFINE_float(parse, "child_lr_min", 0.001, "for lr schedule")
DEFINE_string(parse, "child_optim_algo", "adam", "")
DEFINE_string(parse, "output_type", "avg_pool", "")
DEFINE_boolean(parse, "multi_path", False, "Search for multiple path in the architecture")
DEFINE_boolean(parse, "is_binary", False, "binary label for sst dataset")
DEFINE_boolean(parse, "all_layer_output", True, "use all layer as output")
DEFINE_boolean(parse, "output_linear_combine", True, "")
DEFINE_boolean(parse, "is_mask", True, "whether apply mask")
DEFINE_boolean(parse, "fixed_seed", True, "")
DEFINE_boolean(parse, "load_checkpoint", False, "whether load checkpoint")

DEFINE_float(parse, "controller_lr", 1e-3, "")
DEFINE_float(parse, "controller_l2_reg", 1e-3, "")
DEFINE_float(parse, "controller_bl_dec", 0.99, "")
DEFINE_float(parse, "controller_tanh_constant", None, "")
DEFINE_float(parse, "controller_temperature", None, "")
DEFINE_float(parse, "controller_entropy_weight", None, "")
DEFINE_float(parse, "controller_skip_target", 0.8, "")
DEFINE_float(parse, "controller_skip_weight", 0.0, "")
DEFINE_integer(parse, "controller_train_steps", 500, "")
DEFINE_integer(parse, "controller_train_every", 2,
               "train the controller after this number of epochs")
DEFINE_boolean(parse, "controller_training", False, "")
DEFINE_boolean(parse, "controller_use_critic", False, "")
DEFINE_boolean(parse, "is_cuda", True, "")

DEFINE_integer(parse, "log_every", 50, "How many steps to log")
DEFINE_integer(parse, "eval_every_epochs", 1, "How many epochs to eval")

FLAGS = parse.parse_args()

def set_random_seed(seed):
  print("-" * 80)
  print("set random seed for data reading: {}".format(seed))
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def get_model(embedding, num_layers, pre_idxs=[]):
  print("num layers: {0}".format(num_layers))
  assert FLAGS.search_for is not None, "Please specify --search_for"

  if FLAGS.search_for == "micro":
    raise ValueError("Currently only 'macro' search supported.")
  else:
    if FLAGS.controller_type == "rl":
      ControllerClass = RLController
    else:
      ControllerClass = None

  child_model = TextNASChild(
    embedding=embedding,
    fixed_arc=FLAGS.child_fixed_arc,
    num_layers=num_layers,
    out_filters_scale=FLAGS.child_out_filters_scale,
    out_filters=FLAGS.child_out_filters,
    cnn_keep_prob=FLAGS.cnn_keep_prob,
    final_output_keep_prob=FLAGS.final_output_keep_prob,
    embed_keep_prob=FLAGS.embed_keep_prob,
    lstm_out_keep_prob=FLAGS.lstm_out_keep_prob,
    attention_keep_prob=FLAGS.attention_keep_prob,
    multi_path=FLAGS.multi_path,
    embedding_model=FLAGS.embedding_model,
    all_layer_output=FLAGS.all_layer_output,
    output_linear_combine=FLAGS.output_linear_combine,
    num_last_layer_output=FLAGS.num_last_layer_output,
    is_mask=FLAGS.is_mask,
    output_type=FLAGS.output_type,
    max_input_length=FLAGS.max_input_length,
    class_num=FLAGS.class_num,
    is_cuda=FLAGS.is_cuda)

  if FLAGS.child_fixed_arc is None:
    controller_model = ControllerClass(
      search_for=FLAGS.search_for,
      skip_target=FLAGS.controller_skip_target,
      skip_weight=FLAGS.controller_skip_weight,
      num_layers=num_layers,
      num_branches=FLAGS.child_num_branches,
      out_filters=FLAGS.child_out_filters,
      lstm_size=64,
      lstm_num_layers=1,
      lstm_keep_prob=1.0,
      tanh_constant=FLAGS.controller_tanh_constant,
      temperature=FLAGS.controller_temperature,
      lr_init=FLAGS.controller_lr,
      l2_reg=FLAGS.controller_l2_reg,
      entropy_weight=FLAGS.controller_entropy_weight,
      bl_dec=FLAGS.controller_bl_dec,
      use_critic=FLAGS.controller_use_critic,
      optim_algo="adam",
      pre_idxs=pre_idxs,
      multi_path=FLAGS.multi_path,
      is_cuda=FLAGS.is_cuda)
  else:
    controller_model = None

  return child_model, controller_model

def print_arc(arc, num_layers):
  start = 0
  for i in range(0, num_layers):
    end = start + i + 1
    if FLAGS.multi_path:
      end += 1
    out_str = "fixed_arc=\"$fixed_arc {0}\"".format(np.reshape(arc[start: end], [-1]))
    out_str = out_str.replace("[", "").replace("]", "")
    print(out_str)

    start = end

def eval_once(child_model, eval_set, criterion, valid_dataloader=None, test_dataloader=None):
  if eval_set == "test":
    assert test_dataloader is not None
    dataloader = test_dataloader
  elif eval_set == "valid":
    assert valid_dataloader is not None
    dataloader = valid_dataloader
  else:
    raise NotImplementedError("Unknown eval_set '{}'".format(eval_set))

  tot_acc = 0
  tot = 0
  losses = []

  with torch.no_grad():  # save memory
    for batch in dataloader:
      sent_ids, mask, labels = batch

      if FLAGS.is_cuda:
        sent_ids = sent_ids.cuda()
        mask = mask.cuda()
        labels = labels.cuda()

      logits = child_model(sent_ids, mask)  # run

      loss = criterion(logits, labels.long())
      loss = loss.mean()
      preds = logits.argmax(dim=1).long()
      acc = torch.eq(preds, labels.long()).long().sum().item()

      losses.append(loss)
      tot_acc += acc
      tot += len(labels)

  losses = torch.tensor(losses)
  loss = losses.mean()
  if tot > 0:
    final_acc = float(tot_acc) / tot
  else:
    final_acc = 0
    print("Error in calculating final_acc")
  return final_acc, loss

def train(sents, mask, labels, output_dir, num_layers, embedding, pre_idxs=[]):
  print("Build dataloader")
  train_dataset = SSTDataset(sents["train"], mask["train"], labels["train"])
  test_dataset = SSTDataset(sents["test"], mask["test"], labels["test"])

  train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=True)
  test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.eval_batch_size, pin_memory=True)

  print("Build model")
  print("-" * 80)
  child_model, controller_model = get_model(embedding, num_layers, pre_idxs)
  print("Finish build model")

  for name, var in child_model.named_parameters():
    print(name, var.size(), var.requires_grad)  # output all params

  num_vars = count_model_params(child_model.parameters())
  print("Model has {} params".format(num_vars))

  for m in child_model.modules():  # initializer
    if isinstance(m, (nn.Conv1d, nn.Linear)):
      nn.init.xavier_uniform_(m.weight)

  criterion = nn.CrossEntropyLoss()

  # get optimizer
  if FLAGS.child_optim_algo == "adam":
    optimizer = optim.Adam(child_model.parameters(), eps=1e-3, weight_decay=FLAGS.child_l2_reg)  # with L2
  elif FLAGS.child_optim_algo == "momentum":
    optimizer = optim.SGD(self.parameters(), momentum=0.9, weight_decay=self.l2_reg, nesterov=True)
  else:
    raise ValueError("Unknown optim_algo {}".format(optim_algo))
  if FLAGS.is_cuda:
    child_model.cuda()
    criterion.cuda()
  if controller_model is not None:
    controller_model.cuda()
  if FLAGS.child_fixed_arc is not None:
    fixed_arc = np.array([int(x) for x in FLAGS.child_fixed_arc.split(" ") if x])
    print_arc(fixed_arc, num_layers)

  print("Start training")
  print("-" * 80)
  start_time = time.time()
  step = 0

  # save path
  model_save_path = os.path.join(FLAGS.output_dir, "model.pth")
  best_model_save_path = os.path.join(FLAGS.output_dir, "best_model.pth")
  best_acc = 0
  start_epoch = 0
  if FLAGS.load_checkpoint:
    if os.path.isfile(model_save_path):
      checkpoint = torch.load(model_save_path, map_location = torch.device('cpu'))
      step = checkpoint['step']
      start_epoch = checkpoint['epoch']
      child_model.load_state_dict(checkpoint['child_model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      if controller_model is not None:
        controller_model.load_state_dict(checkpoint['controller_model_state_dict'])
  for epoch in range(start_epoch, FLAGS.num_epochs):
    lr = update_lr(optimizer,
                   epoch,
                   lr_decay_scheme=FLAGS.child_lr_decay_scheme,
                   lr_max=FLAGS.child_lr_max,
                   lr_min=FLAGS.child_lr_min,
                   lr_T_0=FLAGS.child_lr_T_0,
                   lr_T_mul=FLAGS.child_lr_T_mul)
    child_model.train()
    if controller_model is not None:
      controller_model.eval()

    for batch in train_dataloader:
      sent_ids, mask, labels = batch

      if FLAGS.is_cuda:
        sent_ids = sent_ids.cuda()
        mask = mask.cuda()
        labels = labels.cuda()

      step += 1
      if controller_model is not None:
        controller_model._build_sampler(pre_idxs)
        child_model.get_sample_arc(controller_model.sample_arc)

      logits = child_model(sent_ids, mask)  # run

      loss = criterion(logits, labels.long())
      loss = loss.mean()
      preds = logits.argmax(dim=1).long()
      acc = torch.eq(preds, labels.long()).long().sum().item()

      gn = train_ops(
            loss,
            child_model.parameters(),
            optimizer,
            clip_mode="norm",
            grad_bound=FLAGS.child_grad_bound)

      if step % FLAGS.log_every == 0:
        curr_time = time.time()
        log_string = ""
        log_string += "epoch={:<6d}".format(epoch)
        log_string += "ch_step={:<6d}".format(step)
        log_string += " loss={:<8.6f}".format(loss)
        log_string += " lr={:<8.4f}".format(lr)
        log_string += " |g|={:<8.4f}".format(gn)
        log_string += " tr_acc={:<3d}/{:>3d}".format(acc, logits.size()[0])
        log_string += " mins={:<10.2f}".format(float(curr_time - start_time) / 60)
        print(log_string)

    epoch += 1
    save_state = {
        'step' : step,
        'epoch' : epoch,
        'child_model_state_dict' : child_model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()}
    torch.save(save_state, model_save_path)
    child_model.eval()
    if controller_model is None:
      print("Epoch {}: Eval".format(epoch))
      eval_acc, eval_loss = eval_once(child_model, "test", criterion, test_dataloader=test_dataloader)
      print("ch_step={} {}_accuracy={:<6.4f} {}_loss={:<6.4f}".format(
          step, "test", eval_acc, "test", eval_loss))
      if eval_acc > best_acc:
        best_acc = eval_acc
        print("Save best model")
        save_state = {
            'step' : step,
            'epoch' : epoch,
            'child_model_state_dict' : child_model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict()}
        torch.save(save_state, best_model_save_path)

    else:
      if (FLAGS.controller_training and epoch % FLAGS.controller_train_every == 0):
        print("Epoch {}: Training controller".format(epoch))
        controller_model.train()
        for ct_step in range(FLAGS.controller_train_steps):
          controller_model._build_sampler(pre_idxs)
          child_model.get_sample_arc(controller_model.sample_arc)
          eval_acc, eval_loss = eval_once(child_model, "test", criterion, test_dataloader=test_dataloader)
          loss, entropy, lr, gn, reward, sample_log_prob, bl = controller_model.trainer(eval_acc, ct_step)

          if ct_step % FLAGS.log_every == 0:
            curr_time = time.time()
            log_string = ""
            log_string += "ctrl_step={:<6d}".format(ct_step)
            log_string += " loss={:<7.3f}".format(loss)
            log_string += " sample_log_prob={:<8.4f}".format(sample_log_prob)
            log_string += " reward={:<8.4f}".format(reward)
            log_string += " ent={:<5.2f}".format(entropy)
            log_string += " lr={:<6.4f}".format(lr)
            log_string += " |g|={:<8.4f}".format(gn)
            log_string += " acc={:<6.4f}".format(eval_acc)
            log_string += " bl={:<5.2f}".format(bl)
            log_string += " mins={:<.2f}".format(float(curr_time - start_time) / 60)
            print(log_string)

        avg_acc = 0
        max_acc = 0
        print("Here are 10 architectures")
        for _ in range(10):
          controller_model._build_sampler(pre_idxs)
          child_model.get_sample_arc(controller_model.sample_arc)
          arc = child_model.sample_arc
          eval_acc, eval_loss = eval_once(child_model, "test", criterion, test_dataloader=test_dataloader)
          print("ch_step={} {}_accuracy={:<6.4f} {}_loss={:<6.4f}".format(
              step, "test", eval_acc, "test", eval_loss))
          if FLAGS.search_for == "micro":
            normal_arc, reduce_arc = arc
            print(np.reshape(normal_arc, [-1]))
            print(np.reshape(reduce_arc, [-1]))
          else:
            start = 0
            for layer_id in range(num_layers):
              end = start + 1 + layer_id
              if FLAGS.multi_path:
                end += 1
              out_str = "fixed_arc=\"$fixed_arc {0}\"".format(np.reshape(arc[start: end], [-1]))
              out_str = out_str.replace("[", "").replace("]", "")
              print(out_str)
              start = end
          if eval_acc > max_acc:
            max_acc = eval_acc
          avg_acc += eval_acc
          print("-" * 80)

        avg_acc /= 10.0
        threshold = max_acc
        print("sample_arc_acc, {0}, {1}".format(avg_acc, max_acc))
        if max_acc > best_acc:
          best_acc = max_acc
          print("Found better model at: {}".format(step))
          save_state = {
              'step' : step,
              'epoch' : epoch,
              'child_model_state_dict' : child_model.state_dict(),
              'controller_model_state_dict' : controller_model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict()}
          torch.save(save_state, best_model_save_path)
  return eval_acc

def main():
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir, ignore_errors=True)
    os.makedirs(FLAGS.output_dir)
  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  print_user_flags(FLAGS)

  if FLAGS.fixed_seed:
    set_random_seed(FLAGS.global_seed)
  print("load data")

  sents, mask, labels, embedding = read_data_sst(FLAGS.data_path,
                                                 FLAGS.max_input_length,
                                                 FLAGS.embedding_model,
                                                 FLAGS.min_count,
                                                 FLAGS.train_ratio,
                                                 FLAGS.valid_ratio,
                                                 FLAGS.embedding_path,
                                                 FLAGS.is_binary)
  print("load data finish")
  train(sents, mask, labels, FLAGS.output_dir, FLAGS.child_num_layers, embedding)


if __name__ == "__main__":
  main()
