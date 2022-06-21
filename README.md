# CenterNet

## Introduction

![f1.png](https://s2.loli.net/2022/04/27/Bv5V7DkbJOxnhGc.png)

Network for CenterNet. The pytorch implementation for "[Objects as Points](https://arxiv.org/abs/1904.07850) ". 

## Quick start

1. Clone this repository

```shell
git clone https://github.com/Runist/torch_CenterNet
```
2. Install torch_CenterNet from source.

```shell
cd torch_CenterNet
pip installl -r requirements.txt
```
3. Download the Pascal dataset or COCO dataset. Create new folder name call "data" and symbolic link for your dataset.
```shell
mkdir data
cd data
ln -s xxx/VOCdevkit VOCdevkit
cd ..
```
4. Prepare the classes information file and place it in "data" directory, the txt file format is:
```shell
aeroplane
bicycle
...
tvmonitor
```
5. Configure the parameters in [tools/args.py](https://github.com/Runist/Siam-NestedUNet/blob/master/utils/parser.py).
6. Start train your model.

```shell
python tools/train.py
```
or use Linux shell to start.
```shell
sh scripts/train_yolo.sh
```
7. Open tensorboard to watch loss, learning rate etc. You can also see training process and training process and validation prediction.

```shell
tensorboard --logdir ./weights/yolo_voc/log/summary
```

8. After train, you can run *evaluate.py* to watch model performance.

```shell
python tools/evaluate.py
```
As well as use Linux shell to start.
```shell
sh scripts/eval_yolo.sh
```
9. Get prediction of model.

```shell
python tools/predict.py
```

Or use script to run

```shell
sh scripts/predict.sh
```

![dog.jpg](https://s2.loli.net/2022/06/21/RM9fQGgKwumy8is.jpg)

## Train your dataset

We provide three dataset format for this repository "yolo", "coco", "voc",You need create new annotation file for "yolo", the format of "yolo" is:

```shell
image_path|1,95,240,336,19
image_path|305,131,318,151,14|304,142,354,160,3
```

"coco", "voc" is  follow the format of their dataset. And prepare the classes information file and place it in "data" directory.

## Performance

| Train Dataset | Val Dataset | weight                                                       | mAP 0.5 | mAP 0.5：0.95 |
| ------------- | ----------- | ------------------------------------------------------------ | ------- | ------------- |
| VOC07+12      | VOC-Test07  | [resnet50-CenterNet.pt](https://github.com/Runist/torch_CenterNet/releases/download/v1/resnet50-CenterNet.pt) | 0.622   | 0.436         |

## Reference

Appreciate the work from the following repositories:

- [bubbliiiing](https://github.com/bubbliiiing)/[centernet-pytorch](https://github.com/bubbliiiing/centernet-pytorch)

- [katsura-jp](https://github.com/katsura-jp)/[pytorch-cosine-annealing-with-warmup](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup)

- [YunYang1994](https://github.com/YunYang1994)/[tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)

## License

Code and datasets are released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.