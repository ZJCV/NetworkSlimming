# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:17
@file: __init__.py.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN
from zcls.config import get_cfg_defaults


def add_custom_config(_C):
    # Add your own customized config.
    _C.SLIMMING = CN()
    _C.SLIMMING.SPARSITY_REGULARIZATION = False
    _C.SLIMMING.SPARSE_RATE = 1e-5

    return _C


cfg = add_custom_config(get_cfg_defaults())
