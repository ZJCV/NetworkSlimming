# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:22
@file: prune_resnet.py
@author: zj
@description: 
"""

import copy
import numpy as np
import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from slim.prune.misc import set_module_list, round_to_multiple_of
from slim.prune.layers import create_conv2d, create_batchnorm2d, create_linear
from .misc import computer_bn_threshold


def prune_conv_bn(old_conv2d, old_batchnorm2d, bn_threshold,
                  in_channels=3, in_idx=None, minimum_channels=8, divisor=8):
    assert isinstance(old_conv2d, nn.Conv2d)
    assert isinstance(old_batchnorm2d, nn.BatchNorm2d)

    weight_copy = old_batchnorm2d.weight.data.abs().clone()
    # If the number of BN channels is less than or equal to minimum_channels, pruning is not performed
    if len(weight_copy) <= minimum_channels:
        out_idx = np.arange(minimum_channels)
    else:
        mask = weight_copy.gt(bn_threshold).float()
        # get pruning mask
        out_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if out_idx.size == 1:
            out_idx = np.resize(out_idx, (1,))

        # If the feature length after pruning is less than minimum_channels or is not a multiple of divisor,
        # round up to a multiple of divisor
        old_prune_len = len(out_idx)
        new_prune_len = round_to_multiple_of(old_prune_len, divisor)
        if new_prune_len > old_prune_len:
            mask = weight_copy.le(bn_threshold).float()
            tmp_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if tmp_idx.size == 1:
                tmp_idx = np.resize(tmp_idx, (1,))
            res_idx = np.random.choice(tmp_idx, new_prune_len - old_prune_len, replace=False)

            out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))

    # get output channels
    out_channels = len(out_idx)

    # new Conv2d/BatchNorm2d
    new_conv2d = create_conv2d(old_conv2d, in_channels, out_channels)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_channels)

    new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
    if new_conv2d.bias is not None:
        new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()
    if in_idx is not None:
        new_conv2d.weight.data = new_conv2d.weight.data[:, in_idx.tolist(), :, :].clone()

    new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
    new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
    new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
    new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

    return new_conv2d, new_batchnorm2d, out_channels, out_idx


def prune_stem(module_list, bn_threshold, minimum_channels=8, divisor=8):
    """
    The stem block is composed of conv+BN+relu+maxpool
    """
    assert len(module_list) == 4

    new_module_list = list()
    idx = 0

    in_channels = 3
    in_idx = None
    while idx < len(module_list):
        if isinstance(module_list[idx], nn.Conv2d):
            conv2d, batchnorm2d, in_channels, in_idx = prune_conv_bn(module_list[idx],
                                                                     module_list[idx + 1],
                                                                     bn_threshold,
                                                                     in_channels=in_channels,
                                                                     in_idx=in_idx,
                                                                     minimum_channels=minimum_channels,
                                                                     divisor=divisor)
            new_module_list.append(conv2d)
            new_module_list.append(batchnorm2d)
            new_module_list.append(nn.ReLU(inplace=True))
            idx += 3
        elif isinstance(module_list[idx], nn.MaxPool2d):
            new_module_list.append(module_list[idx])
            idx += 1
    return new_module_list, in_channels, in_idx


def computer_bn_length(old_batchnorm2d, bn_threshold):
    """
    calculate BN length after pruning
    """
    assert isinstance(old_batchnorm2d, nn.BatchNorm2d)

    weight_copy = old_batchnorm2d.weight.data.abs().clone()
    mask = weight_copy.gt(bn_threshold).float()

    return len(mask)


def computer_layer_out_channels(in_channels, layer_module, bn_threshold=0.5, minimum_channels=8, divisor=8):
    assert isinstance(layer_module, nn.Sequential)
    # bottleneck nums
    num = len(layer_module)
    # BN length after pruning
    bn_list = list()

    bn_list.append(in_channels)
    for i in range(num):
        # get bottleneck
        bottle = layer_module[i]
        # calculate the length of BN after pruning of each bottleneck
        out_channels = computer_bn_length(bottle.bn3, bn_threshold)
        bn_list.append(out_channels)
        # If it is the first bottleneck, we also need to consider the BN of the downsample layer
        if i == 0:
            out_channels = computer_bn_length(bottle.downsample[1], bn_threshold)
            bn_list.append(out_channels)

    # Extract the maximum value as the BN length after each bottleneck pruning
    out_channels = max(bn_list)
    # concerns the parallel principle
    out_channels = minimum_channels if out_channels < minimum_channels else out_channels
    out_channels = round_to_multiple_of(out_channels, divisor)

    return out_channels


def computer_mask(old_batchnorm2d, bn_threshold, out_channels):
    """
    Calculate pruning mask
    """
    assert isinstance(old_batchnorm2d, nn.BatchNorm2d)

    weight_copy = old_batchnorm2d.weight.data.abs().clone()
    mask = weight_copy.gt(bn_threshold).float()
    # get pruning mask
    out_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
    if out_idx.size == 1:
        out_idx = np.resize(out_idx, (1,))

    old_pruned_len = len(out_idx)
    if out_channels > old_pruned_len:
        mask = weight_copy.le(bn_threshold).float()
        tmp_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if tmp_idx.size == 1:
            tmp_idx = np.resize(tmp_idx, (1,))
        res_idx = np.random.choice(tmp_idx, out_channels - old_pruned_len, replace=False)

        out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))

    return out_idx


def prune_bottleneck(module_list, in_channels, in_idx, out_channels, bn_threshold=0.5, minimum_channels=8, divisor=8):
    """
    Each bottleneck consists of three conv-BN and a downsample layer
    """
    src_in_channels = in_channels
    src_in_idx = copy.deepcopy(in_idx)

    new_module_list = list()
    idx = 0

    while idx < 3:
        # first and second conv-bn
        if isinstance(module_list[idx], nn.Conv2d):
            conv2d, batchnorm2d, in_channels, in_idx = prune_conv_bn(module_list[idx],
                                                                     module_list[idx + 1],
                                                                     bn_threshold,
                                                                     in_channels=in_channels,
                                                                     in_idx=in_idx,
                                                                     minimum_channels=minimum_channels,
                                                                     divisor=divisor)
            new_module_list.append(conv2d)
            new_module_list.append(batchnorm2d)
            idx += 2

    # Calculate the number of output channels of the third conv-BN and calculate the number of output channels
    old_conv2d = module_list[idx]
    old_batchnorm2d = module_list[idx + 1]
    out_idx = computer_mask(old_batchnorm2d,
                            bn_threshold,
                            out_channels
                            )

    # new Conv2d/BatchNorm2d
    new_conv2d = create_conv2d(old_conv2d, in_channels, out_channels)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_channels)

    new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
    if new_conv2d.bias is not None:
        new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()
    if in_idx is not None:
        new_conv2d.weight.data = new_conv2d.weight.data[:, in_idx.tolist(), :, :].clone()

    new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
    new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
    new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
    new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

    new_module_list.append(new_conv2d)
    new_module_list.append(new_batchnorm2d)
    # There is a relu in the middle
    new_module_list.append(nn.ReLU(inplace=True))
    idx += 3

    # If there is a downsample layer, the downsample layer is also pruned
    if len(module_list) > 7:
        # yes, it has
        old_conv2d = module_list[idx]
        old_batchnorm2d = module_list[idx + 1]
        out_idx = computer_mask(old_batchnorm2d,
                                bn_threshold,
                                out_channels
                                )
        # new Conv2d/BatchNorm2d
        new_conv2d = create_conv2d(old_conv2d, src_in_channels, out_channels)
        new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_channels)

        new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
        if new_conv2d.bias is not None:
            new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()
        if src_in_idx is not None:
            new_conv2d.weight.data = new_conv2d.weight.data[:, src_in_idx.tolist(), :, :].clone()

        new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
        new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
        new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
        new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

        new_module_list.append(new_conv2d)
        new_module_list.append(new_batchnorm2d)

    return new_module_list, out_channels, out_idx


def prune_classifier(module_list, in_channels, in_idx):
    """
    For RESNET, the classifier does not contain BN layer, so it only needs to adjust the input channel
    :param module_list:
    :param in_channels:
    :return:
    """
    new_module_list = list()
    new_module_list.append(module_list[0])

    old_linear = module_list[1]
    assert isinstance(old_linear, nn.Linear)
    new_linear, in_channels = create_linear(old_linear, in_channels)

    new_linear.weight.data = old_linear.weight.data[:, in_idx].clone()
    if new_linear.bias is not None:
        new_linear.bias.data = old_linear.bias.data.clone()

    new_module_list.append(new_linear)
    return new_module_list


def prune(model, percent, minimum_channels=8, divisor=8):
    total, threshold = computer_bn_threshold(model, percent)

    model = list(model.children())[0]
    # print(model)

    # stem layer first
    stem_name_list = list()
    stem_module_list = list()
    for name, children in list(model.named_children())[:4]:
        stem_name_list.append(name)
        stem_module_list.append(children)
    new_module_list, in_channels, in_idx = prune_stem(stem_module_list,
                                                      bn_threshold=threshold,
                                                      minimum_channels=minimum_channels,
                                                      divisor=divisor)
    assert len(new_module_list) == len(stem_module_list) == len(stem_name_list)
    set_module_list(model, stem_name_list, stem_module_list, new_module_list)

    # process layer one by one
    for layer_name, layer in list(model.named_children())[4:8]:
        assert isinstance(layer, nn.Sequential)
        # First, the number of output channels after pruning of each bottleneck in the layer is calculated
        out_channels = computer_layer_out_channels(in_channels, layer)
        # traverse every bottleneck
        for submodule_name, submodule in layer.named_children():
            # Count the layers of each bottleneck and prune them
            assert isinstance(submodule, Bottleneck)

            bottleneck_name_list = list()
            bottleneck_module_list = list()
            for sub_name, sub in submodule.named_children():
                if isinstance(sub, nn.Conv2d) or isinstance(sub, nn.BatchNorm2d) or isinstance(sub, nn.ReLU):
                    bottleneck_name_list.append(f'{layer_name}.{submodule_name}.{sub_name}')
                    bottleneck_module_list.append(sub)
                elif isinstance(sub, nn.Sequential):
                    # The downsample layer needs further splitting
                    for downsample_name, downsample_module in sub.named_children():
                        bottleneck_name_list.append(f'{layer_name}.{submodule_name}.{sub_name}.{downsample_name}')
                        bottleneck_module_list.append(downsample_module)

            new_module_list, in_channels, in_idx = prune_bottleneck(bottleneck_module_list,
                                                                    in_channels,
                                                                    in_idx,
                                                                    out_channels,
                                                                    bn_threshold=threshold,
                                                                    minimum_channels=minimum_channels,
                                                                    divisor=divisor
                                                                    )
            assert len(new_module_list) == len(bottleneck_module_list) == len(bottleneck_name_list)
            set_module_list(model, bottleneck_name_list, bottleneck_module_list, new_module_list)

    # last to prune classifier
    classifier_name_list = list()
    classifier_module_list = list()
    for name, children in list(model.named_children())[8:]:
        classifier_name_list.append(name)
        classifier_module_list.append(children)
    new_module_list = prune_classifier(classifier_module_list, in_channels, in_idx)
    assert len(new_module_list) == len(classifier_module_list) == len(classifier_name_list)
    set_module_list(model, classifier_name_list, classifier_module_list, new_module_list)

    new_total, _ = computer_bn_threshold(model, percent)
    return 1 - (1.0 * new_total / total), threshold
