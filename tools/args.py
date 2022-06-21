# -*- coding: utf-8 -*-
# @File : args.py
# @Author: Runist
# @Time : 2022/3/29 9:46
# @Software: PyCharm
# @Brief: Code argument parser

import argparse
import warnings
import os
import torch
import sys
sys.path.append(os.getcwd())

from core.helper import seed_torch, get_class_names


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='1', help='Select gpu device.')

parser.add_argument('--input_height', type=int, default=416, help='The height of model input.')
parser.add_argument('--input_width', type=int, default=416, help='The width of model input.')
parser.add_argument('--num_classes', type=int, default=20, help='The number of class.')

parser.add_argument('--warmup_epochs', type=int, default=5, help='The number of freeze training epochs.')
parser.add_argument('--freeze_epochs', type=int, default=50, help='The number of freeze training epochs.')
parser.add_argument('--unfreeze_epochs', type=int, default=100, help='The number of unfreeze training epochs.')

parser.add_argument('--freeze_batch_size', type=int, default=16, help='The number of examples per batch.')
parser.add_argument('--unfreeze_batch_size', type=int, default=16, help='The number of examples per batch.')

parser.add_argument('--learn_rate_init', type=float, default=2e-4,
                    help='Initial value of cosine annealing learning rate.')
parser.add_argument('--learn_rate_end', type=float, default=1e-6,
                    help='End value of cosine annealing learning rate.')
parser.add_argument('--num_workers', type=int, default=12, help='The number of torch dataloader thread.')

parser.add_argument('--backbone', type=str,
                    default="resnet50",
                    choices=["resnet50", "resnet101"],
                    help='The path of the pretrain weight.')
parser.add_argument('--pretrain_weight_path', type=str,
                    default=None,
                    help='The path of the pretrain weight.')

parser.add_argument('--dataset_train_path', type=str,
                    default="",
                    help='The file path of the train data.')
parser.add_argument('--dataset_val_path', type=str,
                    default="",
                    help='The file path of the val data.')
parser.add_argument('--image_train_dir', type=str,
                    default="",
                    help='The images directory of the train data.')
parser.add_argument('--image_val_dir', type=str,
                    default="",
                    help='The images directory of the val data.')
parser.add_argument('--annotation_train_dir', type=str,
                    default="",
                    help='The labels directory of the train data.')
parser.add_argument('--annotation_val_dir', type=str,
                    default="",
                    help='The labels directory of the val data.')
parser.add_argument('--dataset_format', type=str,
                    default="voc", choices=["coco", "voc", "yolo", "ilsvrc"],
                    help='The format of dataset, it will influence dataloader method.')

parser.add_argument('--logs_dir', type=str, default="./logs/temp",
                    help='The directory of saving weights and training log.')
parser.add_argument('--outputs_dir', type=str, default='./outputs',
                    help='The directory of the predict image.')

parser.add_argument('--confidence', type=float, default=0.3, help='The number of class.')
parser.add_argument('--classes_info_file', type=str, default="./data/voc.txt",
                    help='The text that stores classification information.')
parser.add_argument('--test_weight', type=str, help='The name of the model weight.')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
seed_torch(777)

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class_names = get_class_names(args.classes_info_file)
