# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import numpy as np
import torch

def global_avg_pool(x, mask):
  x = torch.sum(x, 2)
  length = torch.sum(mask, 1, keepdim=True).float()
  length += torch.eq(length, 0.0).float() * 1e-12
  length = length.repeat(1, x.size()[1])
  x /= length
  return x

def global_max_pool(x, mask):
  mask = torch.eq(mask.float(), 0.0).long()
  mask = torch.unsqueeze(mask, dim=1).repeat(1, x.size()[1], 1)
  mask *= -(2 ** 32) + 1
  x += mask
  x = torch.max(x, 2)[0]
  return x

def get_length(mask):
  length = torch.sum(mask, 1)
  length = length.long()
  return length
