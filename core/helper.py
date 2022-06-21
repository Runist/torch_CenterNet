# -*- coding: utf-8 -*-
# @File : helper.py
# @Author: Runist
# @Time : 2022/3/30 12:00
# @Software: PyCharm
# @Brief: Some function

from net import CenterNet
from core.dataset import CocoDataset, PascalDataset, YoloDataset, ILSVRCDataset

import torch
from torch import nn
import numpy as np
import os
import shutil
import random
import cv2 as cv


def seed_torch(seed):
    """
    Set all random seed
    Args:
        seed: random seed

    Returns: None

    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def remove_dir_and_create_dir(dir_name, is_remove=True):
    """
    Make new folder, if this folder exist, we will remove it and create a new folder.
    Args:
        dir_name: path of folder
        is_remove: if true, it will remove old folder and create new folder

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "create.")
    else:
        if is_remove:
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            print(dir_name, "create.")
        else:
            print(dir_name, "is exist.")


def get_color_map():
    """
    Create color map.

    Returns: numpy array.

    """
    color_map = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            color_map[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return color_map


def get_model(args, dev):
    """
    Get CenterNet model.
    Args:
        args: ArgumentParser
        dev: torch dev

    Returns: CenterNet model

    """
    device_ids = [i for i in range(len(args.gpu.split(',')))]

    model = CenterNet(args.backbone, num_classes=args.num_classes)

    if args.pretrain_weight_path is not None:
        print("Loading {}.".format(args.pretrain_weight_path))
        model_state_dict = model.state_dict()
        pretrain_state_dict = torch.load(args.pretrain_weight_path)

        for k, v in pretrain_state_dict.items():
            centernet_k = "backbone." + k
            if centernet_k in model_state_dict.keys():
                model_state_dict[centernet_k] = v

        model.load_state_dict(model_state_dict)

    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(dev)

    return model


def get_class_names(classes_info_file):
    """
    Get dataset class name
    Args:
        classes_info_file: class name file path

    Returns:

    """
    class_names = []
    with open(classes_info_file, 'r') as data:
        for name in data:
            name = name.strip('\n')
            if ":" in name:
                name = name.split(": ")[0]
            class_names.append(name)

    return class_names


def get_dataset(args, class_names):
    """
    Get CenterNet dataset.
    Args:
        args: ArgumentParser
        class_names: the name of class

    Returns: dataset

    """
    if args.dataset_format == "voc":
        train_dataset = PascalDataset(args.image_train_dir,
                                      args.annotation_train_dir,
                                      class_names,
                                      args.dataset_train_path, (args.input_height, args.input_width),
                                      num_classes=args.num_classes, is_train=True)
        val_dataset = PascalDataset(args.image_val_dir,
                                    args.annotation_val_dir,
                                    class_names,
                                    args.dataset_val_path, (args.input_height, args.input_width),
                                    num_classes=args.num_classes, is_train=False)
    elif args.dataset_format == "coco":
        train_dataset = CocoDataset(args.image_train_dir,
                                    args.dataset_train_path, (args.input_height, args.input_width),
                                    num_classes=args.num_classes, is_train=True)
        val_dataset = CocoDataset(args.image_val_dir,
                                  args.dataset_val_path, (args.input_height, args.input_width),
                                  num_classes=args.num_classes, is_train=False)
    elif args.dataset_format == "yolo":
        train_dataset = YoloDataset(args.dataset_train_path, (args.input_height, args.input_width),
                                    num_classes=args.num_classes, is_train=True)
        val_dataset = YoloDataset(args.dataset_val_path, (args.input_height, args.input_width),
                                  num_classes=args.num_classes, is_train=False)
    elif args.dataset_format == "ilsvrc":
        train_dataset = ILSVRCDataset(args.image_train_dir,
                                      args.annotation_train_dir,
                                      class_names,
                                      args.dataset_train_path, (args.input_height, args.input_width),
                                      num_classes=args.num_classes, is_train=True)
        val_dataset = ILSVRCDataset(args.image_val_dir,
                                    args.annotation_val_dir,
                                    class_names,
                                    args.dataset_val_path, (args.input_height, args.input_width),
                                    num_classes=args.num_classes, is_train=False)
    else:
        raise Exception("There is no {} format for data parsing, you should choose one from 'yolo', 'coco', 'voc', 'ilsvrc'".
                        format(args.dataset_format))

    return train_dataset, val_dataset


def draw_bbox(image, bboxes, labels, class_names, scores=None, show_name=False):
    """
    Draw bounding box in image.
    Args:
        image: image
        bboxes: coordinate of bounding box
        labels: the index of labels
        class_names: the names of class
        scores: bounding box confidence
        show_name: show class name if set true, otherwise show index of class

    Returns: draw result

    """
    color_map = get_color_map()
    image_height, image_width = image.shape[:2]
    draw_image = image.copy()

    for i, c in list(enumerate(labels)):
        bbox = bboxes[i]
        c = int(c)
        color = [int(j) for j in color_map[c]]
        if show_name:
            predicted_class = class_names[c]
        else:
            predicted_class = c

        if scores is None:
            text = '{}'.format(predicted_class)
        else:
            score = scores[i]
            text = '{} {:.2f}'.format(predicted_class, score)

        x1, y1, x2, y2 = bbox

        x1 = max(0, np.floor(x1).astype(np.int32))
        y1 = max(0, np.floor(y1).astype(np.int32))
        x2 = min(image_width, np.floor(x2).astype(np.int32))
        y2 = min(image_height, np.floor(y2).astype(np.int32))

        thickness = int((image_height + image_width) / (np.sqrt(image_height**2 + image_width**2)))
        fontScale = 0.35

        t_size = cv.getTextSize(text, 0, fontScale, thickness=thickness * 2)[0]
        cv.rectangle(draw_image, (x1, y1), (x2, y2), color=color, thickness=thickness)
        cv.rectangle(draw_image, (x1, y1), (x1 + t_size[0], y1 - t_size[1]), color, -1)  # filled
        cv.putText(draw_image, text, (x1, y1), cv.FONT_HERSHEY_SIMPLEX,
                   fontScale, (255, 255, 255), thickness//2, lineType=cv.LINE_AA)

    return draw_image
