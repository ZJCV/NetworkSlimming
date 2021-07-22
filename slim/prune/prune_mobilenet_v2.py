# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:23
@file: prune_mobilenet_v2.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import ConvBNActivation, ConvBNReLU, InvertedResidual

from slim.prune.layers import create_conv2d, create_batchnorm2d, create_linear
from slim.prune.misc import set_module_list, round_to_multiple_of
from .misc import computer_bn_threshold


def prune_conv_bn(old_conv2d, old_batchnorm2d, bn_threshold,
                  in_channels=3, in_idx=None, minimum_channels=8, divisor=8, is_dw=False):
    """
    For deep-wise convolution, the number of groups is the same as the number of input channels,
    and the number of output channels can be divided by the number of groups
    """
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

        old_prune_len = len(out_idx)
        # For the deep-wise convolution of mobilenet-v2, the number of input and output channels is consistent
        if is_dw:
            new_prune_len = in_channels
        else:
            # If the feature length after pruning is less than minimum_channels or not multiples of divisor,
            # then round up to multiples of divisor
            new_prune_len = round_to_multiple_of(old_prune_len, divisor)
        if new_prune_len > old_prune_len:
            mask = weight_copy.le(bn_threshold).float()
            tmp_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
            if tmp_idx.size == 1:
                tmp_idx = np.resize(tmp_idx, (1,))
            res_idx = np.random.choice(tmp_idx, new_prune_len - old_prune_len, replace=False)

            out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))
        elif new_prune_len < old_prune_len:
            out_idx = np.random.choice(out_idx, new_prune_len, replace=False)

    # get output channels
    out_channels = len(out_idx)

    # new Conv2d/BatchNorm2d
    new_conv2d = create_conv2d(old_conv2d, in_channels, out_channels, old_groups=in_channels if is_dw else None)
    new_batchnorm2d = create_batchnorm2d(old_batchnorm2d, out_channels)

    new_conv2d.weight.data = old_conv2d.weight.data[out_idx.tolist(), :, :, :].clone()
    if new_conv2d.bias is not None:
        new_conv2d.bias.data = old_conv2d.bias.data[out_idx.tolist()].clone()
    if in_idx is not None and not is_dw:
        new_conv2d.weight.data = new_conv2d.weight.data[:, in_idx.tolist(), :, :].clone()

    new_batchnorm2d.weight.data = old_batchnorm2d.weight.data[out_idx.tolist()].clone()
    new_batchnorm2d.bias.data = old_batchnorm2d.bias.data[out_idx.tolist()].clone()
    new_batchnorm2d.running_mean = old_batchnorm2d.running_mean[out_idx.tolist()].clone()
    new_batchnorm2d.running_var = old_batchnorm2d.running_var[out_idx.tolist()].clone()

    return new_conv2d, new_batchnorm2d, out_channels, out_idx


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
        # If the feature length after pruning is less than the number of output channels, it will be supplemented
        mask = weight_copy.le(bn_threshold).float()
        tmp_idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
        if tmp_idx.size == 1:
            tmp_idx = np.resize(tmp_idx, (1,))
        res_idx = np.random.choice(tmp_idx, out_channels - old_pruned_len, replace=False)

        out_idx = np.array(sorted(np.concatenate((out_idx, res_idx))))
    elif out_channels < old_pruned_len:
        # If the length after pruning is greater than the number of output channels, prune again
        out_idx = np.random.choice(out_idx, out_channels, replace=False)

    return out_idx


def prune_conv_bn_activation(module_list, in_channels, in_idx, bn_threshold, minimum_channels=8, divisor=8):
    assert len(module_list) == 3
    new_module_list = list()

    old_conv = module_list[0]
    old_batchnorm2d = module_list[1]
    new_conv2d, new_batchnorm2d, out_channels, out_idx = prune_conv_bn(old_conv,
                                                                       old_batchnorm2d,
                                                                       bn_threshold,
                                                                       in_channels=in_channels,
                                                                       in_idx=in_idx,
                                                                       minimum_channels=minimum_channels,
                                                                       divisor=divisor)
    assert new_conv2d.weight.shape[:2] == torch.Size(
        [new_conv2d.out_channels, new_conv2d.in_channels]), '0, {} {}'.format(new_conv2d, new_conv2d.weight.shape)
    assert new_batchnorm2d.weight.shape == torch.Size((new_batchnorm2d.num_features,))
    new_module_list.append(new_conv2d)
    new_module_list.append(new_batchnorm2d)
    new_module_list.append(nn.ReLU6(inplace=True))

    return new_module_list, out_channels, out_idx


def prune_inverted_residual(module_list, in_channels, in_idx, bn_threshold, minimum_channels=8, divisor=8,
                            use_res_connect=False):
    assert len(module_list) in [5, 8]
    src_in_channels = in_channels
    new_module_list = list()

    idx = 0
    # first ConvBNActivation
    new_conv2d, new_batchnorm2d, out_channels, out_idx = prune_conv_bn(module_list[idx],
                                                                       module_list[idx + 1],
                                                                       bn_threshold,
                                                                       in_channels=in_channels,
                                                                       in_idx=in_idx,
                                                                       minimum_channels=minimum_channels,
                                                                       divisor=divisor,
                                                                       is_dw=len(module_list) == 5
                                                                       )
    if len(module_list) == 5:
        assert new_conv2d.weight.shape[:2] == torch.Size(
            [new_conv2d.out_channels, 1]), '0, {} {}'.format(new_conv2d, new_conv2d.weight.shape)
    else:
        assert new_conv2d.weight.shape[:2] == torch.Size(
            [new_conv2d.out_channels, new_conv2d.in_channels]), '0, {} {}'.format(new_conv2d, new_conv2d.weight.shape)
    assert new_batchnorm2d.weight.shape == torch.Size((new_batchnorm2d.num_features,))
    new_module_list.append(new_conv2d)
    new_module_list.append(new_batchnorm2d)
    new_module_list.append(nn.ReLU6(inplace=True))
    idx += 3

    in_channels = out_channels
    in_idx = out_idx
    if len(module_list) == 8:
        # second ConvBNActivation
        new_conv2d, new_batchnorm2d, out_channels, out_idx = prune_conv_bn(module_list[idx],
                                                                           module_list[idx + 1],
                                                                           bn_threshold,
                                                                           in_channels=in_channels,
                                                                           in_idx=in_idx,
                                                                           minimum_channels=minimum_channels,
                                                                           divisor=divisor,
                                                                           is_dw=True
                                                                           )
        assert new_conv2d.weight.shape[:2] == torch.Size(
            [new_conv2d.out_channels, 1]), '1, {} {}'.format(new_conv2d, new_conv2d.weight.shape)
        assert new_batchnorm2d.weight.shape == torch.Size((new_batchnorm2d.num_features,))
        new_module_list.append(new_conv2d)
        new_module_list.append(new_batchnorm2d)
        new_module_list.append(nn.ReLU6(inplace=True))
        idx += 3

    in_channels = out_channels
    in_idx = out_idx
    if use_res_connect:
        out_channels = src_in_channels
        # identity map
        old_conv2d = module_list[idx]
        old_batchnorm2d = module_list[idx + 1]
        out_idx = computer_mask(old_batchnorm2d, bn_threshold, out_channels)

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

        assert new_conv2d.weight.shape[:2] == torch.Size(
            [new_conv2d.out_channels, new_conv2d.in_channels]), '2, {} {}'.format(new_conv2d, new_conv2d.weight.shape)
        assert new_batchnorm2d.weight.shape == torch.Size((new_batchnorm2d.num_features,))
        new_module_list.append(new_conv2d)
        new_module_list.append(new_batchnorm2d)
    else:
        new_conv2d, new_batchnorm2d, out_channels, out_idx = prune_conv_bn(module_list[idx],
                                                                           module_list[idx + 1],
                                                                           bn_threshold,
                                                                           in_channels=in_channels,
                                                                           in_idx=in_idx,
                                                                           minimum_channels=minimum_channels,
                                                                           divisor=divisor,
                                                                           is_dw=False
                                                                           )

        assert new_conv2d.weight.shape[:2] == torch.Size(
            [new_conv2d.out_channels, new_conv2d.in_channels]), '3, {} {}'.format(new_conv2d, new_conv2d.weight.shape)
        assert new_batchnorm2d.weight.shape == torch.Size((new_batchnorm2d.num_features,))
        new_module_list.append(new_conv2d)
        new_module_list.append(new_batchnorm2d)

    return new_module_list, out_channels, out_idx


def prune_classifier(module_list, in_channels, in_idx):
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

    in_channels = 3
    in_idx = None

    # Process one by one
    features_name = 'features'
    features_stage = model.features
    assert isinstance(features_stage, nn.Sequential)

    for block_name, block in features_stage.named_children():
        if isinstance(block, ConvBNActivation) or isinstance(block, ConvBNReLU):
            layer_name_list = list()
            layer_list = list()
            for layer_name, layer in block.named_children():
                layer_name_list.append(f'{features_name}.{block_name}.{layer_name}')
                layer_list.append(layer)
            new_module_list, in_channels, in_idx = prune_conv_bn_activation(layer_list,
                                                                            in_channels,
                                                                            in_idx,
                                                                            threshold,
                                                                            minimum_channels,
                                                                            divisor)
            assert len(new_module_list) == len(layer_list) == len(layer_name_list)
            set_module_list(model, layer_name_list, layer_list, new_module_list)
        elif isinstance(block, InvertedResidual):
            # print(block_name)
            sub_block = block.conv

            layer_name_list = list()
            layer_list = list()
            for layer_name, layer in sub_block.named_children():
                if isinstance(layer, ConvBNActivation):
                    for sub_layer_name, sub_layer in layer.named_children():
                        layer_name_list.append(f'{features_name}.{block_name}.conv.{layer_name}.{sub_layer_name}')
                        layer_list.append(sub_layer)
                else:
                    layer_name_list.append(f'{features_name}.{block_name}.conv.{layer_name}')
                    layer_list.append(layer)
            new_module_list, in_channels, in_idx = prune_inverted_residual(layer_list,
                                                                           in_channels,
                                                                           in_idx,
                                                                           threshold,
                                                                           minimum_channels,
                                                                           divisor,
                                                                           block.use_res_connect
                                                                           )
            assert len(new_module_list) == len(layer_list) == len(layer_name_list)
            set_module_list(model, layer_name_list, layer_list, new_module_list)

    # Finally, process classifier
    classifier_name_list = list()
    classifier_module_list = list()
    for name, children in list(model.classifier.named_children()):
        classifier_name_list.append(f'classifier.{name}')
        classifier_module_list.append(children)
    new_module_list = prune_classifier(classifier_module_list, in_channels, in_idx)
    assert len(new_module_list) == len(classifier_module_list) == len(classifier_name_list)
    set_module_list(model, classifier_name_list, classifier_module_list, new_module_list)

    new_total, _ = computer_bn_threshold(model, percent)
    return 1 - (1.0 * new_total / total), threshold
