# -*- coding: utf-8 -*-
# @File : evaluate.py
# @Author: Runist
# @Time : 2022/3/29 10:57
# @Software: PyCharm
# @Brief: Evaluate map

from args import args, dev, class_names
from core.map import get_map
from core.helper import remove_dir_and_create_dir
from net.centernet import CenterNet

import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np


if __name__ == '__main__':
    model = torch.load(args.test_weight)

    output_files_path = os.path.join(args.outputs_dir, "map")
    remove_dir_and_create_dir(output_files_path)

    get_map(args, output_files_path, class_names, model, dev)
