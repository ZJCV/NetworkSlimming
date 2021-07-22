# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:34
@file: build.py
@author: zj
@description: 
"""

from . import prune_vggnet
from . import prune_resnet
from . import prune_mobilenet_v2


def build_prune(model,
                model_name='vgg',
                ratio=0.2,
                minimum_channels=8,
                divisor=8
                ):
    if 'vgg' in model_name:
        return prune_vggnet.prune(model, ratio, minimum_channels=minimum_channels, divisor=divisor)
    elif 'resnet' in model_name:
        return prune_resnet.prune(model, ratio, minimum_channels=minimum_channels, divisor=divisor)
    elif 'mobilenet' in model_name:
        return prune_mobilenet_v2.prune(model, ratio, minimum_channels=minimum_channels, divisor=divisor)
    else:
        raise ValueError(f'{model_name} does not supports')
