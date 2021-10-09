
# Training Results

## VGGNet

```
# vggnet16_bn default training
total -  top1 acc: 81.060  top5 acc: 95.770
# vggnet16_bn slim 1e-4 pruning training
total -  top1 acc: 81.120  top5 acc: 95.750
```

## VGGNet Pruning

|     arch    | flops/G | model size/MB | slim | predict pruning ratio | true pruning ratio | Flops after pruning | Model size after pruning |  top1  |  top5  |
|:-----------:|:-------:|:-------------:|:----:|:---------------------:|:------------------:|:-------------------:|:------------------------:|:------:|:------:|
| vggnet16_bn |  15.51  |     134.68    | 1e-4 |          20%          |       18.75%       |         8.36        |          130.31          | 80.670 | 95.160 |
| vggnet16_bn |  15.51  |     134.68    | 1e-4 |          40%          |       39.39%       |         4.52        |          125.72          | 79.960 | 95.030 |
| vggnet16_bn |  15.51  |     134.68    | 1e-4 |          60%          |       58.71%       |         2.45        |          120.73          | 77.620 | 93.970 |

## ResNet

```
# resnet50  default training
total -  top1 acc: 83.850  top5 acc: 96.400
# resnet50 slim 1e-5 pruning training
total -  top1 acc: 83.940  top5 acc: 96.340
```

## ResNet Pruning

|   arch   | flops/G | model size/MB | slim | predict pruning ratio | true pruning ratio | Flops after pruning | Model size after pruning |  top1  |  top5  |
|:--------:|:-------:|:-------------:|:----:|:---------------------:|:------------------:|:-------------------:|:------------------------:|:------:|:------:|
| resnet50 |   4.11  |     23.72     | 1e-5 |          20%          |       00.09%       |         4.08        |           23.67          | 83.680 | 96.260 |
| resnet50 |   4.11  |     23.72     | 1e-5 |          40%          |       05.99%       |         2.84        |           19.44          | 83.010 | 95.780 |
| resnet50 |   4.11  |     23.72     | 1e-5 |          60%          |       20.09%       |         1.12        |           7.42           | 74.580 | 92.720 |

## MobileNet_v2

```
# mobilenet_v2  default training
total -  top1 acc: 80.030  top5 acc: 95.380
# mobilenet_v2 slim 1e-5  pruning training
total -  top1 acc: 80.320  top5 acc: 95.050
```

## MobileNet_v2 Pruning

|     arch     | flops/G | model size/MB | slim | predict pruning ratio | true pruning ratio | Flops after pruning | Model size after pruning |  top1  |  top5  |
|:------------:|:-------:|:-------------:|:----:|:---------------------:|:------------------:|:-------------------:|:------------------------:|:------:|:------:|
| mobilenet_v2 |  0.313  |     2.352     | 1e-5 |           5%          |          /         |          /          |             /            | 75.260 | 92.830 |
| mobilenet_v2 |  0.313  |     2.352     | 1e-5 |          20%          |       29.08%       |        0.224        |           1.780          | 83.680 | 96.260 |
| mobilenet_v2 |  0.313  |     2.352     | 1e-5 |          40%          |       54.60%       |        0.153        |           1.206          | 83.010 | 95.780 |
| mobilenet_v2 |  0.313  |     2.352     | 1e-5 |          60%          |       73.03%       |        0.096        |           0.728          | 74.580 | 92.720 |