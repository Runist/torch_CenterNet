# -*- coding: utf-8 -*-
# @File : utils.py
# @Author: Runist
# @Time : 2022/3/28 17:36
# @Software: PyCharm
# @Brief:


import numpy as np
import cv2 as cv
import math


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Get a heatmap of one class
    Args:
        heatmap: The heatmap of one class(storage in single channel)
        center: The location of object center
        radius: 2D Gaussian circle radius
        k: The magnification of the Gaussian

    Returns: heatmap

    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def gaussian2D(shape, sigma=1):
    """
    2D Gaussian function
    Args:
        shape: (diameter, diameter)
        sigma: variance

    Returns: h

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Get gaussian circle radius.
    Args:
        det_size: (height, width)
        min_overlap: overlap minimum

    Returns: radius

    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def preprocess_input(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    image = (image / 255. - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return image


def recover_input(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    image = np.transpose(image, (1, 2, 0))
    image = (image * std + mean) * 255

    return image


def image_resize(image, target_size, gt_boxes=None):
    ih, iw = target_size

    h, w = image.shape[:2]

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv.resize(image, (nw, nh))

    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

    if gt_boxes is None:
        return image_paded
    elif gt_boxes.size == 0:
        # Use no label image to train
        return image_paded, gt_boxes
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
