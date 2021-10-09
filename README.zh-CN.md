<div align="right">
  语言:
    🇨🇳
  <a title="英语" href="./README.md">🇺🇸</a>
</div>

 <div align="center"><a title="" href="https://github.com/ZJCV/NetworkSlimming"><img align="center" src="./imgs/NetworkSlimming.png"></a></div>

<p align="center">
  «NetworkSlimming»复现了论文<a title="" href="https://arxiv.org/abs/1708.06519">Learning Efficient Convolutional Networks through Network Slimming</a>
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square"></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg"></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg"></a>
</p>

* 解析：[ Learning Efficient Convolutional Networks through Network Slimming](https://blog.zhujian.life/posts/ec02d2a5.html)

更详细的训练数据可以查看：

* [Details](./docs/details.md)

## 内容列表

- [内容列表](#内容列表)
- [背景](#背景)
- [安装](#安装)
- [用法](#用法)
- [主要维护人员](#主要维护人员)
- [致谢](#致谢)
- [参与贡献方式](#参与贡献方式)
- [许可证](#许可证)

## 背景

`Network Slimming`利用`L1`正则化对`BN`层缩放因子进行稀疏训练，完成训练后再进行通道级别剪枝操作，最后通过微调恢复性能。在实际应用过程中实现了很好的效果

## 安装

```angular2html
$ pip install -r requirements.txt
```

## 用法

首先，设置环境变量

```angular2html
$ export PYTHONPATH=<project root path>
$ export CUDA_VISIBLE_DEVICES=0
```

然后进行`训练-剪枝-微调`

* 训练

```
$ python tools/train.py -cfg=configs/vggnet/vgg16_bn_cifar100_224_e100_sgd_mslr_slim_1e_4.yaml
```

* 剪枝

```angular2html
$ python tools/prune/prune_vggnet.py
```

* 微调

```angular2html
$ python tools/train.py -cfg=configs/vggnet/refine_pruned_0_2_vgg16_bn_cifar100_224_e100_sgd_mslr_slim_1e_4.yaml
```

最后，在配置文件的`PRELOADED`选项中设置微调后的模型路径

```angular2html
$ python tools/test.py -cfg=configs/vggnet/refine_pruned_0_2_vgg16_bn_cifar100_224_e100_sgd_mslr_slim_1e_4.yaml
```

## 主要维护人员

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## 致谢

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

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/ZJCV/NetworkSlimming/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2021 zjykzj