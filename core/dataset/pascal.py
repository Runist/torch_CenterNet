# -*- coding: utf-8 -*-
# @File : pascal.py
# @Author: Runist
# @Time : 2022/3/28 17:32
# @Software: PyCharm
# @Brief: pascal VOC Dataset
from core.dataset import CenterNetDataset

import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


class PascalDataset(CenterNetDataset):
    def __init__(self, image_dir, annotation_dir, class_names, annotation_path, input_shape, num_classes, is_train):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.class_names = class_names

        super().__init__(annotation_path, input_shape, num_classes, is_train)

    def load_annotations(self, annotations_path):
        """
        Load image and label info from voc*.txt file.

        Args:
            annotations_path: Annotation file path

        Returns: image and label id list

        """
        with open(annotations_path, 'r', encoding='utf-8') as f:
            txt = f.readlines()
            annotations = [line.strip().split()[0] for line in txt]
        return annotations

    def parse_annotation(self, index):
        """
        Parse self.annotation element and read image and bounding boxes.

        Args:
            index: index for self.annotation

        Returns: image, bboxes

        """
        img_id = self.annotations[index]

        image_path = os.path.join(self.image_dir, img_id + ".jpg")
        xml = ET.parse(os.path.join(self.annotation_dir, img_id + ".xml")).getroot()

        bboxes = []
        for obj in xml.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False

            if difficult:
                continue

            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
            bboxes.append([xmin, ymin, xmax, ymax, self.class_names.index(name)])

        bboxes = np.array(bboxes)

        image = Image.open(image_path)
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = image.repeat(3, axis=-1)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]

        return image, bboxes

