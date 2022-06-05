# BDMI2022S课程大作业

选题：基于人体姿态估计算法的引体向上计数算法

作业中使用的人体姿态估计算法

+ 论文链接：[Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose](https://arxiv.org/pdf/1811.12004.pdf)

+ GitHub链接：[Daniil-Osokin/lightweight-human-pose-estimation.pytorch](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)

## Requirements

+ Windows10

+ Python 3.9

+ PyTorch 1.11.0
  + 建议参考[PyTorch官网](https://pytorch.org/)的安装方式
+ `pip install -r requirements.txt`

## Pre-trained model

测试时所使用的预训练COCO模型来自[https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth)，下载好的模型保存在`/checkpoint`目录下（该目录需要用户自己创建）

## Data

离线测试时使用的数据为`REALSENSE D435`所拍摄的`*.bag`文件，数据保存在`/data`目录下（该目录需要用户自己创建）

## Demo

### 离线测试

```shell
python demo.py --checkpoint-path checkpoint/checkpoint_iter_370000.pth --inputbag data/12-1-1280.bag
```

### 在线测试

```shell
# 尚未进行场景测试
# To do...
python demo.py --checkpoint-path checkpoint/checkpoint_iter_370000.pth --realsense
```
