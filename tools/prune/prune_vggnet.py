# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:27
@file: prune_vggnet.py
@author: zj
@description: 
"""

import warnings

from operation import load_model, prune_model, save_model

warnings.filterwarnings('ignore')


def prune_vggnet(cfg_file, pruned_rate, minimum_channels=8, divisor=8):
    model, arch_name = load_model(cfg_file)

    pruned_model, true_pruned_ratio, threshold = prune_model(arch_name,
                                                             model,
                                                             ratio=pruned_rate,
                                                             minimum_channels=minimum_channels,
                                                             divisor=divisor
                                                             )
    print(pruned_model)
    print('predict pruned ratio:', pruned_rate)
    print('true pruned ratio:', true_pruned_ratio)
    print('threshold:', threshold)

    model_name = f'outputs/vggnet_pruned/{arch_name}_slimming_1e_4_pruned_{pruned_rate}.pkl'
    save_model(pruned_model, model_name)


if __name__ == '__main__':
    cfg_file = 'configs/vggnet/vgg16_bn_cifar100_224_e100_sgd_mslr_slim_1e_4.yaml'
    prune_vggnet(cfg_file, pruned_rate=0.2, minimum_channels=8, divisor=8)
    # prune_vggnet(cfg_file, pruned_rate=0.4, minimum_channels=8, divisor=8)
    # prune_vggnet(cfg_file, pruned_rate=0.6, minimum_channels=8, divisor=8)
