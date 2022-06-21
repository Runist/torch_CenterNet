# -*- coding: utf-8 -*-
# @File : loss.py
# @Author: Runist
# @Time : 2022/3/28 21:10
# @Software: PyCharm
# @Brief: Loss function


import torch
import torch.nn.functional as F


def focal_loss(pred, target):
    """
    classifier loss of focal loss
    Args:
        pred: heatmap of prediction
        target: heatmap of ground truth

    Returns: cls loss

    """
    # Find every image positive points and negative points,
    # one bounding box corresponds to one positive point,
    # except positive points, other feature points are negative sample.
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    # The negative samples near the positive sample feature point have smaller weights
    neg_weights = torch.pow(1 - target, 4)
    loss = 0
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    # Calculate Focal Loss.
    # The hard to classify sample weight is large, easy to classify sample weight is small.
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds * neg_weights

    # Loss normalization is carried out
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def l1_loss(pred, target, mask):
    """
    Calculate l1 loss
    Args:
        pred: offset detection result
        target: offset ground truth
        mask: offset mask, only center point is 1, other place is 0

    Returns: l1 loss

    """
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    # Don't calculate loss in the position without ground truth.
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')

    loss = loss / (mask.sum() + 1e-7)

    return loss
