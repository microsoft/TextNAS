# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import torch
from torch import nn
import math

user_flags = []

def DEFINE_string(parse, name, default_value, doc_string):
  parse.add_argument('--{0}'.format(name), default = default_value, help = doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_integer(parse, name, default_value, doc_string):
  parse.add_argument('--{0}'.format(name), type = int, default = default_value, help = doc_string)
  global user_flags
  user_flags.append(name)


def DEFINE_float(parse, name, default_value, doc_string):
  parse.add_argument('--{0}'.format(name), type = float, default = default_value, help = doc_string)
  global user_flags
  user_flags.append(name)

def DEFINE_boolean(parse, name, default_value, doc_string):
  parse.add_argument('--{0}'.format(name), action='store_true', help = doc_string)
  global user_flags
  user_flags.append(name)

def print_user_flags(FLAGS, line_limit=80):
  print("-" * 80)

  global user_flags
  log_strings = ""
  for flag_name in sorted(user_flags):
    value = "{}".format(getattr(FLAGS, flag_name))
    log_string = flag_name
    log_string += "." * (line_limit - len(flag_name) - len(value))
    log_string += value
    log_strings = log_strings + log_string
    log_strings = log_strings + "\n"
  print(log_strings)

class TextColors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "w")

  def write(self, message):
    self.terminal.write(message)
    self.terminal.flush()
    self.log.write(message)
    self.log.flush()

  def flush(self):
    self.terminal.flush()
    self.log.flush()


def count_model_params(trainable_params):
  """
  Args:
    trainable_params: list of all model trainable params
  """
  num_vars = 0
  for var in trainable_params:
    num_vars += np.prod([dim for dim in var.size()])
  return num_vars

def update_lr(
    optimizer,
    epoch,
    lr_decay_scheme="cosine",
    lr_max=0.002,
    lr_min=0.000000001,
    lr_T_0=4,
    lr_T_mul=1):
  if lr_decay_scheme == "cosine":
    assert lr_max is not None, "Need lr_max to use lr_cosine"
    assert lr_min is not None, "Need lr_min to use lr_cosine"
    assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
    assert lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"

    T_i = lr_T_0
    t_epoch = epoch
    last_reset = 0
    while True:
      t_epoch -= T_i
      if t_epoch < 0:
        break
      last_reset += T_i
      T_i *= lr_T_mul

    T_curr = epoch - last_reset

    def _update():
      rate = T_curr / T_i * 3.1415926
      lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(rate))
      return lr

    learning_rate = _update()
  else:
    raise ValueError("Unknown learning rate decay scheme {}".format(lr_decay_scheme))

  #update lr in optimizer
  for params_group in optimizer.param_groups:
    params_group['lr'] = learning_rate
  return learning_rate

def train_ops(
    loss,
    trainable_params=None,
    optimizer=None,
    clip_mode=None,
    grad_bound=None):
  """
  Args:
    clip_mode: "global", "norm", or None.
    moving_average: store the moving average of parameters
  """

  optimizer.zero_grad()
  loss.backward()
  grad_norm = 0

  #clip grads
  if clip_mode is not None:
    assert grad_bound is not None, "Need grad_bound to clip gradients."
    if clip_mode == "norm":
      grad_norm = nn.utils.clip_grad_norm_(trainable_params, 99999999) #just compute grad_norm
      for param in trainable_params:
        nn.utils.clip_grad_norm_(param, grad_bound) #clip grad
    else:
      raise NotImplementedError("Unknown clip_mode {}".format(clip_mode))

  optimizer.step()

  return grad_norm
