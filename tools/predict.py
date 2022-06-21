# -*- coding: utf-8 -*-
# @File : predict.py
# @Author: Runist
# @Time : 2022/3/29 11:01
# @Software: PyCharm
# @Brief: Predict script

from args import args, dev, class_names
from core.detect import predict
from core.helper import draw_bbox, remove_dir_and_create_dir

from PIL import Image
import cv2 as cv
import torch
import numpy as np
import os
from tqdm import tqdm
import random


if __name__ == '__main__':
    test_folder = "./assets"
    outputs_dir = "{}/images".format(args.outputs_dir)
    remove_dir_and_create_dir(outputs_dir)

    model = torch.load(args.test_weight)

    model.eval()
    with torch.no_grad():
        for file in tqdm(os.listdir(test_folder)):
            image_path = os.path.join(test_folder, file)
            image = Image.open(image_path)
            image = np.array(image)

            outputs = predict(image, model, dev, args)

            if len(outputs) == 0:
                image = Image.fromarray(image)
                image.save("{}/{}".format(outputs_dir, file))
                continue

            outputs = outputs.data.cpu().numpy()
            labels = outputs[:, 5]
            scores = outputs[:, 4]
            bboxes = outputs[:, :4]

            image = draw_bbox(image, bboxes, labels, class_names, scores=scores, show_name=True)

            image = Image.fromarray(image)
            image.save("{}/images/{}".format(args.outputs_dir, file))
