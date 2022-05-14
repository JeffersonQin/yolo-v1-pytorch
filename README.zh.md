# YOLO v1 PyTorch 实现

**这个项目用于学习目的，是 YOLO v1 的 PyTorch 实现。** 原文中的网络非常难以与训练，所以我将 Backbone 替换为了预训练过的 ResNet18 和 ResNet 50。原网络已经在 `yolo.py` 中实现了，如有需要可以自行训练。原网络的与训练代码我还没写（因为我没有这个需求（雾））。

此外，根据 [YOLO v2 的 Paper](https://arxiv.org/pdf/1612.08242.pdf)，我在每一个卷积层后都添加了 BatchNorm，并去掉了 Dropout。

超参数与损失函数的实现和[原文](https://arxiv.org/pdf/1506.02640.pdf) 保持了一致。此外，网络在 VOC2007-trainval+test 和 VOC2012-train 上训练，在 VOC2012-val 上测试。我的设备为 RTX2070s。

下面是项目结构：

```
webcam.py                     # 摄像头 demo
utils
├── data.py                   # 数据
├── init.py                   # 权重初始化
├── metrics.py                # mAP 计算
├── utils.py                  # helper, e.g. Accumulator, Timer
└── visualize.py              # 可视化
yolo
├── tests.py                  # 测试样例
└── yolo.py                   # YOLO module, loss, nms
```

## 性能

|         Model          | Backbone | mAP@VOC2012-val | COCOmAP@VOC2012-val |    FPS     |
| :--------------------: | :------: | :-------------: | :-----------------: | :--------: |
| YOLOv1-ResNet18 (Ours) | ResNet18 |     48.10%      |       23.18%        | **235.47** |
| YOLOv1-ResNet50 (Ours) | ResNet50 |   **49.87%**    |     **23.95**%      |   95.94    |

|         Model          | Backbone | mAP@VOC2012-test |    FPS    |
| :--------------------: | :------: | :--------------: | :-------: |
| YOLOv1-ResNet18 (Ours) | ResNet18 |      44.54%      | **97.88** |
| YOLOv1-ResNet50 (Ours) | ResNet50 |    **47.28%**    |   58.40   |
|         YOLOv1         | Darknet? |    **57.9%**     |    45     |

Leaderboard 链接:

* Our [YOLOv1-ResNet18](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb_main.php?challengeid=11&compid=3#KEY_YOLOv1-resnet-18-50)
* Our [YOLOv1-ResNet50](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb_main.php?challengeid=11&compid=4#KEY_YOLOv1-resnet-18-50)

细分比较:

|         Model          | mean  | aero plane | bicycle | bird  | boat  | bottle |  bus  |  car  |  cat  | chair |  cow  |
| :--------------------: | :---: | :--------: | :-----: | :---: | :---: | :----: | :---: | :---: | :---: | :---: | :---: |
|          YOLO          | 57.9  |    77.0    |  67.2   | 57.7  | 38.3  |  22.7  | 68.3  | 55.9  | 81.4  | 36.2  | 60.8  |
| YOLOv1-ResNet18 (Ours) | 44.5  |    64.3    |  54.2   | 47.4  | 26.8  |  16.6  | 55.4  | 44.3  | 66.5  | 23.1  | 38.1  |
| YOLOv1-ResNet50 (Ours) | 47.3  |    66.7    |  56.1   | 49.5  | 25.9  |  17.8  | 60.2  | 45.9  | 70.6  | 26.1  | 43.0  |

|         Model          | dining<br>table |  dog  | horse | motor<br>bike | person | potted<br>plant | sheep | sofa  | train | tv<br>monitor |
| :--------------------: | :-------------: | :---: | :---: | :-----------: | :----: | :-------------: | :---: | :---: | :---: | :-----------: |
|          YOLO          |      48.5       | 77.2  | 72.3  |     71.3      |  63.5  |      28.9       | 52.2  | 54.8  | 73.9  |     50.8      |
| YOLOv1-ResNet18 (Ours) |      38.5       | 62.9  | 57.6  |     60.8      |  45.0  |      15.2       | 33.3  | 43.9  | 60.0  |     37.2      |
| YOLOv1-ResNet50 (Ours) |      41.1       | 67.5  | 59.2  |     62.4      |  47.6  |      17.6       | 35.6  | 45.7  | 64.6  |     42.4      |

结果不是非常理想，现在我的主要精力放在更新的架构和论文实现上。

![](./assets/test1.png)

![](./assets/test2.png)

![](./assets/test3.png)

![](./assets/test4.png)

## 数据集

在初次运行的时候（若没有数据集），需要给 `load_data_voc` 添加 `download=True` 的参数。之后可以去除，因为每次都会解压，非常耗时。

## 训练

自行训练可以使用 `resnet18-yolo-train.ipynb` 和 `resnet50-yolo-train.ipynb`。

我用的是 RTX2070s-8GB，会碰到 OOM 问题，所以我实现了梯度累计。真正的批量样本数等于 `DataLoader` 的 `batch_size` 乘上 `train` 的 `accum_batch_num`。

以 `resnet18-yolo-train.ipynb` 为例：`batch_size = 16 (dataloader/batch_size) * 4 (accum_batch_num) = 64`。此外，还还可以通过指定 `train()` 的 `num_gpu` 实现多路 GPU 并行。

下面是一些训练的 metrics:

ResNet18 (Backbone):

<div align="center">
	<img src="./assets/resnet18-train.svg">
</div>

ResNet50 (Backbone):

<div align="center">
	<img src="./assets/resnet50-train.svg">
</div>

## 测试

模型权重已经发布在 Release 里了。将其移动到 `./model/` 即可。`resnet18-yolo-test.ipynb` 和 `resnet50-yolo-test.ipynb` 为测试 `notebook`。

Update：还实现了一个实时摄像头的 demo。

Update [2022/05/10]: 根据 VOC 官方的说法，在测试阶段需要排除包含 difficult 标签的物体，虽然检测出来也不会惩罚。之前忽略了这句话，经过测试，排除 difficult 标签的物体之后，mAP 在两个模型上都提高了 4% 左右。

Update [2022/05/13]: 添加了 VOC2012 刷榜脚本。说是说刷榜脚本其实就是生成 VOC2012 测试集的结果，最终输出到 `results.tar.gz`。测试结果已经公布在前面的章节了。如果要自己测试，将数据集测试数据集按照下面的格式放置即可（需要这么做的原因是 PyTorch 没给接口）。

```
.
README.md                   # Other files
VOC2012test                 # create dataset folder
└── VOCdevkit
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        └── JPEGImages
```

VOC2012 测试数据集下载链接

* [pjreddie 镜像](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

## 致谢

* https://github.com/abeardear/pytorch-YOLO-v1
* https://arxiv.org/pdf/1506.02640.pdf
* https://arxiv.org/pdf/1612.08242.pdf
* https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
* https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc14.pdf
* https://github.com/rafaelpadilla/Object-Detection-Metrics
* https://github.com/rafaelpadilla/review_object_detection_metrics
* 大感谢 [@dmMaze](https://github.com/dmMaze) 提供的帮助（

## YOLO v2

我的 YOLO v2 实现：

* [JeffersonQin/yolo-v2-pytorch](https://github.com/JeffersonQin/yolo-v2-pytorch)
