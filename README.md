# YOLO v1 PyTorch Implementation

**I wrote this repo for the purpose of learning, aimed to reproduce YOLO v1 using PyTorch.** It is very hard to pretrain the original network on ImageNet, so I replaced the backbone with ResNet18 and ResNet50 with PyTorch pretrained version for convenience. However, the original network backbone is also defined in `yolo.py`, and is available for training. Pretraining method is not yet finished (and maybe would never be finished since I've achieved reasonable results using other backbones), and is marked TODO in the file.

Besides, I removed the Dropout layer and added Batch Normalization after every convolution layer according to [yolo v2](https://arxiv.org/pdf/1612.08242.pdf).

The implementation of loss function is exact as the [original paper](https://arxiv.org/pdf/1506.02640.pdf). Also, I adapted all the hyper parameters from the paper, and the network is trained on VOC2007-trainval+test and VOC2012-train, tested on VOC2012-val using RTX2070s.

Here is the structure of the project.

```
webcam.py                     # webcam demo
utils
├── data.py                   # data pipeline
├── init.py                   # weight initialization
├── metrics.py                # mAP calculation
├── utils.py                  # helper, e.g. Accumulator, Timer
└── visualize.py              # visualization
yolo
├── tests.py                  # test wrapping
└── yolo.py                   # YOLO module, loss, nms implementation
```

## Performance

|         Model          | Backbone  | mAP@VOC2012-val | COCOmAP@VOC2012-val |    FPS     |
| :--------------------: | :-------: | :-------------: | :-----------------: | :--------: |
| YOLOv1-ResNet18 (Ours) | ResNet18  |     44.38%      |       21.41%        | **223.61** |
| YOLOv1-ResNet50 (Ours) | ResNet50  |     45.86%      |       22.17%        |   96.16    |
|         YOLOv1         | Darknet19 |    **57.9%**    |          /          |     45     |

## Note

When running the notebook for the first time, you should add `, download=True` param to `load_data_voc` to download dataset. It is suggested to remove the param after everything's set, since it is time-consuming to unarchive the data every time.

## Training

If you want to train the model totally by yourself, use `resnet18-yolo-train.ipynb` and `resnet50-yolo-train.ipynb`.

I trained the network using RTX2070s-8GB, so I also implemented gradient accumulation due to OOM problem. The true `batch_size` is determined by both `batch_size` of `DataLoader` and `accum_batch_num` param from `train` method. In the case of `resnet18-yolo-train.ipynb`, `batch_size = 16 (dataloader/batch_size) * 4 (accum_batch_num)`. You can adjust the param according to specific cases. Besides, DataParallel is also supported by specifying `num_gpu` param of `train()`.

Here are some training loss plot:

ResNet18 (Backbone):

<div align="center">
	<img src="./assets/resnet18-train.svg">
</div>

ResNet50 (Backbone):

<div align="center">
	<img src="./assets/resnet50-train.svg">
</div>

## Testing

Model weight are available in repo release. Place the weights in `./model/` folder, and run `resnet18-yolo-test.ipynb` and `resnet50-yolo-test.ipynb`.

Here is also a demo using using webcam (`webcam.py`).

## Thanks

* https://github.com/abeardear/pytorch-YOLO-v1
* https://arxiv.org/pdf/1506.02640.pdf
* https://arxiv.org/pdf/1612.08242.pdf
* Also big thanks to [@dmMaze](https://github.com/dmMaze)
