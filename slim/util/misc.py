# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:18
@file: misc.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


# additional sub-gradient descent on the sparsity-induced penalty term
def updateBN(model, sparse_rate=1e-4):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # L1
            m.weight.grad.data.add_(sparse_rate * torch.sign(m.weight.data))
