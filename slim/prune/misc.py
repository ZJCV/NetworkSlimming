# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:21
@file: misc.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


def computer_bn_threshold(model, percent):
    """
    computer prune threshold
    :param model:
    :param percent:
    :return:
    """
    total = 0

    # Count the weights of all BN
    # After collecting data, sort and count the longest
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * percent)
    thre = y[thre_index]

    return total, thre


# refert to: [Pytorch替换model对象任意层的方法](https://zhuanlan.zhihu.com/p/356273702)
# The core function refers to the implementation of torch.quantification.fuse_modules()
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def set_module_list(model, name_list, module_list, new_module_list):
    for name, module, new_module in zip(name_list, module_list, new_module_list):
        # print(name, module, new_module)
        _set_module(model, name, new_module)


def round_to_multiple_of(val, divisor):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, i.e. (83, 8) -> 88, but (84, 8) -> 88. """
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= val else new_val + divisor
