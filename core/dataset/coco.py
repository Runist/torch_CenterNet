# -*- coding: utf-8 -*-
# @File : coco.py
# @Author: Runist
# @Time : 2022/4/12 11:06
# @Software: PyCharm
# @Brief:
from core.dataset import CenterNetDataset

import os
import numpy as np
from PIL import Image

from pycocotools.coco import COCO


class CocoDataset(CenterNetDataset):
    def __init__(self, data_dir, annotation_path, input_shape, num_classes, is_train):
        self.data_dir = data_dir
        super().__init__(annotation_path, input_shape, num_classes, is_train)

    def load_annotations(self, annotations_path):
        """
        Load image and label info from coco*.json.

        Args:
            annotations_path: Annotation file path

        Returns: image id list
        """
        self.coco = COCO(annotations_path)
        self.remove_useless_info()

        img_ids = self.coco.getImgIds()

        self.cat_ids = sorted(self.coco.getCatIds())

        return img_ids

    def parse_annotation(self, index):
        """
        Parse self.annotation element and read image and bounding boxes.

        Args:
            index: index for self.annotation

        Returns: image, bboxes

        """
        img_id = self.annotations[index]
        img_ann = self.coco.loadImgs(img_id)[0]

        width = img_ann["width"]
        height = img_ann["height"]
        filename = img_ann["file_name"]
        image_path = os.path.join(self.data_dir, filename)

        ann_id = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=False)
        annotation = self.coco.loadAnns(ann_id)

        bboxes = []
        for obj in annotation:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))

            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                bboxes.append([x1, y1, x2, y2, self.cat_ids.index(obj["category_id"])])

        bboxes = np.array(bboxes, np.int32)

        image = Image.open(image_path)
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = image.repeat(3, axis=-1)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]

        return image, bboxes

    def remove_useless_info(self):
        """
        Remove useless info in coco dataset. COCO object is modified inplace.
        This function is mainly used for saving memory (save about 30% mem).

        Returns: None

        """
        if isinstance(self.coco, COCO):
            dataset = self.coco.dataset
            dataset.pop("info", None)
            dataset.pop("licenses", None)
            for img in dataset["images"]:
                img.pop("license", None)
                img.pop("coco_url", None)
                img.pop("date_captured", None)
                img.pop("flickr_url", None)
            if "annotations" in self.coco.dataset:
                for anno in self.coco.dataset["annotations"]:
                    anno.pop("segmentation", None)
