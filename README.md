<div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/NetworkSlimming"><img align="center" src="./imgs/NetworkSlimming.png"></a></div>

<p align="center">
  «NetworkSlimming» re-implements the paper <a title="" href="https://arxiv.org/abs/1708.06519">Learning Efficient Convolutional Networks through Network Slimming</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

More training statistics can see:

* [Details](./docs/details.md)

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Background](#background)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## Background

`Network Slimming` uses `L1` regularization to sparsely train the BN layer's `scaling factor`; After the training, it performs channel-level pruning operation; Finally, by fine-tuning to recovery performance. it achieves good results in practical application.

## Usage

First, you need set env for `PYTHONPATH` and `CUDA_VISIBLE_DEVICES`

```angular2html
$ export PYTHONPATH=<project root path>
$ export CUDA_VISIBLE_DEVICES=0
```

Then, begin `train-prune-finetuning`

* For train

```
$ python tools/train.py -cfg=configs/vggnet/vgg16_bn_cifar100_224_e100_sgd_mslr_slim_1e_4.yaml
```

* For prune

```angular2html
$ python tools/prune/prune_vggnet.py
```

* For fine-tuning

```angular2html
$ python tools/train.py -cfg=configs/vggnet/refine_pruned_0_2_vgg16_bn_cifar100_224_e100_sgd_mslr_slim_1e_4.yaml
```

Finally, set the fine-tuning model path in the `PRELOADED` option of the configuration file

```angular2html
$ python tools/test.py -cfg=configs/vggnet/refine_pruned_0_2_vgg16_bn_cifar100_224_e100_sgd_mslr_slim_1e_4.yaml
```

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [ Eric-mingjie/network-slimming ](https://github.com/Eric-mingjie/network-slimming)
* [ wlguan/MobileNet-v2-pruning ](https://github.com/wlguan/MobileNet-v2-pruning)
* [ 666DZY666/micronet](https://github.com/666DZY666/micronet)
* [ foolwood/pytorch-slimming ](https://github.com/foolwood/pytorch-slimming)

```
@misc{liu2017learning,
      title={Learning Efficient Convolutional Networks through Network Slimming}, 
      author={Zhuang Liu and Jianguo Li and Zhiqiang Shen and Gao Huang and Shoumeng Yan and Changshui Zhang},
      year={2017},
      eprint={1708.06519},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/ZJCV/NetworkSlimming/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2021 zjykzj