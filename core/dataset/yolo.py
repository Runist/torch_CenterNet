# -*- coding: utf-8 -*-
# @File : yolo.py
# @Author: Runist
# @Time : 2022/4/13 9:42
# @Software: PyCharm
# @Brief:
from core.dataset import CenterNetDataset

import os
import numpy as np
from PIL import Image


class YoloDataset(CenterNetDataset):
    def load_annotations(self, annotations_path):
        """
        Load image and label info from yolo format annotation file.
        The format is "image_path|x1,y1,x2,y2,l1|x1,y1,x2,y2,l2"

        Args:
            annotations_path: Annotation file path

        Returns: image and label id list

        """
        with open(annotations_path, 'r', encoding='utf-8') as f:
            txt = f.readlines()
            annotations = [line.strip().strip("|") for line in txt]

        return annotations

    def parse_annotation(self, index):
        """
        Parse self.annotation element and read image and bounding boxes.

        Args:
            index: index for self.annotation

        Returns: image, bboxes

        """
        annotation = self.annotations[index]

        line = annotation.split("|")
        image_path = line[0]

        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)

        image = Image.open(image_path)
        image = np.array(image)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = image.repeat(3, axis=-1)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]

        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])

        return image, bboxes
