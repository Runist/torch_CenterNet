# -*- coding: utf-8 -*-
# @File : dataset.py
# @Author: Runist
# @Time : 2022/4/12 11:06
# @Software: PyCharm
# @Brief:
from core.dataset.utils import image_resize, preprocess_input, gaussian_radius, draw_gaussian

import os
import cv2 as cv
import random
import math
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data.dataset import Dataset


class CenterNetDataset(Dataset):
    def __init__(self, annotation_path, input_shape, num_classes, is_train):
        super(CenterNetDataset, self).__init__()
        self.stride = 4

        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // self.stride, input_shape[1] // self.stride)
        self.num_classes = num_classes
        self.is_train = is_train

        self.annotations = self.load_annotations(annotation_path)

    def load_annotations(self, annotations_path):
        """
        Load image and label info from annotation file.
        Must implement in subclass. You can do this by referring to 'yolo.py'.

        Args:
            annotations_path: Annotation file path

        Returns: list or dict type. Depends on how do you read it in 'parse_annotation'

        """
        raise NotImplementedError('load_annotations method not implemented!')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_offset = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_offset_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        # Read image and bounding boxes
        image, bboxes = self.parse_annotation(index)

        if self.is_train:
            image, bboxes = self.data_augmentation(image, bboxes)

        # Image preprocess
        image, bboxes = image_resize(image, self.input_shape, bboxes)
        image = preprocess_input(image)

        # Clip bounding boxes
        clip_bboxes = []
        labels = []
        for bbox in bboxes:
            x1, y1, x2, y2, label = bbox

            if x2 <= x1 or y2 <= y1:
                # Don't use such boxes as this may cause nan loss.
                continue

            x1 = int(np.clip(x1, 0, self.input_shape[1]))
            y1 = int(np.clip(y1, 0, self.input_shape[0]))
            x2 = int(np.clip(x2, 0, self.input_shape[1]))
            y2 = int(np.clip(y2, 0, self.input_shape[0]))
            # Clipping coordinates between 0 to image dimensions as negative values
            # or values greater than image dimensions may cause nan loss.
            clip_bboxes.append([x1, y1, x2, y2])
            labels.append(label)

        bboxes = np.array(clip_bboxes)
        labels = np.array(labels)

        if len(bboxes) != 0:
            labels = np.array(labels, dtype=np.float32)
            bboxes = np.array(bboxes[:, :4], dtype=np.float32)
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]] / self.stride, a_min=0, a_max=self.output_shape[1])
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]] / self.stride, a_min=0, a_max=self.output_shape[0])

        for i in range(len(labels)):
            x1, y1, x2, y2 = bboxes[i]
            cls_id = int(labels[i])

            h, w = y2 - y1, x2 - x1
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                # Calculates the feature points of the real box
                ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # Get gaussian heat map
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)

                # Assign ground truth height and width
                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h

                # Assign center point offset
                batch_offset[ct_int[1], ct_int[0]] = ct - ct_int

                # Set the corresponding mask to 1
                batch_offset_mask[ct_int[1], ct_int[0]] = 1

        return image, batch_hm, batch_wh, batch_offset, batch_offset_mask

    def parse_annotation(self, index):
        """
        Parse self.annotation element and read image and bounding boxes.

        Args:
            index: index for self.annotation

        Returns: image, bboxes

        """

        raise NotImplementedError('parse_annotation method not implemented!')

    def data_augmentation(self, image, bboxes):
        if random.random() < 0.5:
            image, bboxes = self.random_horizontal_flip(image, bboxes)
        # if random.random() < 0.5:
        #     image, bboxes = self.random_vertical_flip(image, bboxes)
        if random.random() < 0.5:
            image, bboxes = self.random_crop(image, bboxes)
        if random.random() < 0.5:
            image, bboxes = self.random_translate(image, bboxes)

        if random.random() < 0.5:
            image = Image.fromarray(image)
            enh_bri = ImageEnhance.Brightness(image)
            # brightness = [1, 0.5, 1.4]
            image = enh_bri.enhance(random.uniform(0.6, 1.4))
            image = np.array(image)

        if random.random() < 0.5:
            image = Image.fromarray(image)
            enh_col = ImageEnhance.Color(image)
            # color = [0.7, 1.3, 1]
            image = enh_col.enhance(random.uniform(0.7, 1.3))
            image = np.array(image)

        if random.random() < 0.5:
            image = Image.fromarray(image)
            enh_con = ImageEnhance.Contrast(image)
            # contrast = [0.7, 1, 1.3]
            image = enh_con.enhance(random.uniform(0.7, 1.3))
            image = np.array(image)

        if random.random() < 0.5:
            image = Image.fromarray(image)
            enh_sha = ImageEnhance.Sharpness(image)
            # sharpness = [-0.5, 0, 1.0]
            image = enh_sha.enhance(random.uniform(0, 2.0))
            image = np.array(image)

        return image, bboxes

    def random_horizontal_flip(self, image, bboxes):
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        image = np.array(image)

        if bboxes.size != 0:
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_vertical_flip(self, image, bboxes):
        h, _, _ = image.shape
        image = image[::-1, :, :]
        image = np.array(image)

        if bboxes.size != 0:
            bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if bboxes.size == 0:
            return image, bboxes

        h, w, _ = image.shape

        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = min(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = min(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if bboxes.size == 0:
            return image, bboxes

        h, w, _ = image.shape

        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv.warpAffine(image, M, (w, h), borderValue=(128, 128, 128))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes
