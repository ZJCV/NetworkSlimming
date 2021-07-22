# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:20
@file: build.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from zcls.util.checkpoint import CheckPointer
from zcls.util import logging

logger = logging.get_logger(__name__)

from .vggnet import get_vggnet
from .resnet import get_resnet
from .mobilenet_v2 import get_mobilenet_v2


def build_model(cfg):
    num_classes = cfg.MODEL.HEAD.NUM_CLASSES
    arch_name = cfg.MODEL.RECOGNIZER.NAME

    if 'pruned' in arch_name:
        path = cfg.MODEL.RECOGNIZER.PRELOADED
        logger.info(f'load pruned model: {path}')
        model = torch.load(path)

        if 'resnet' in arch_name:
            old_fc = model.model.fc
            assert isinstance(old_fc, nn.Linear)
            predefined_num_classes = old_fc.out_features
            num_class = cfg.MODEL.HEAD.NUM_CLASSES
            if num_class != predefined_num_classes:
                in_features = old_fc.in_features
                new_fc = nn.Linear(in_features, num_classes, bias=True)
                nn.init.normal_(new_fc.weight, 0, 0.01)
                nn.init.zeros_(new_fc.bias)

                model.model.fc = new_fc
    else:
        if 'vgg' in arch_name:
            model = get_vggnet(num_classes=num_classes, arch=arch_name)
        elif 'resnet' in arch_name:
            model = get_resnet(num_classes=num_classes, arch=arch_name)
        elif 'mobilenet' in arch_name:
            model = get_mobilenet_v2(num_classes=num_classes, arch=arch_name)
        else:
            raise ValueError(f"{arch_name} doesn't exists")

        preloaded = cfg.MODEL.RECOGNIZER.PRELOADED
        if preloaded != "":
            logger.info(f'load preloaded: {preloaded}')
            cpu_device = torch.device('cpu')
            check_pointer = CheckPointer(model)
            check_pointer.load(preloaded, map_location=cpu_device)
            logger.info("finish loading model weights")

    return model
