# -*- coding: utf-8 -*-
# @File : map.py
# @Author: Runist
# @Time : 2022/4/10 10:11
# @Software: PyCharm
# @Brief: Map function

import json
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from core.detect import predict
import xml.etree.ElementTree as ET


def get_ground_truth(args, class_names):
    ground_truth = {}
    categories_info = []

    for i, cls in enumerate(class_names):
        category = {}
        category['supercategory'] = cls
        category['name'] = cls
        category['id'] = i + 1
        categories_info.append(category)

    print("Creating ground-truth files ...")
    if args.dataset_format == "yolo":
        images_info, annotations_info = get_yolo_ground_truth(args.dataset_val_path)
    elif args.dataset_format == "coco":
        images_info, annotations_info = get_coco_ground_truth(args.dataset_val_path)
    elif args.dataset_format == "voc":
        images_info, annotations_info = get_voc_ground_truth(args, class_names)
    elif args.dataset_format == "ilsvrc":
        images_info, annotations_info = get_ilsvrc_ground_truth(args, class_names)
    else:
        raise Exception("There is no {} format for data parsing, you should choose one from 'yolo', 'coco', 'voc', 'ilsvrc'".
                        format(args.dataset_format))

    ground_truth['images'] = images_info
    ground_truth['categories'] = categories_info
    ground_truth['annotations'] = annotations_info

    return ground_truth


def get_map(args, output_files_path, class_names, model, dev):
    ground_truth = get_ground_truth(args, class_names)

    print("Creating detection-result files ...")
    if args.dataset_format == "yolo":
        detection_result = get_yolo_detection_result(args, model, dev)
    elif args.dataset_format == "coco":
        detection_result = get_coco_detection_result(args, model, dev)
    elif args.dataset_format == "voc":
        detection_result = get_voc_detection_result(args, model, dev)
    elif args.dataset_format == "ilsvrc":
        detection_result = get_ilsvrc_detection_result(args, model, dev)
    else:
        raise Exception("There is no {} format for data parsing, you should choose one from 'yolo', 'coco', 'voc', 'ilsvrc'".
                        format(args.dataset_format))

    print("Calculating map ...")
    gt_json_path = os.path.join(output_files_path, 'instances_gt.json')
    dr_json_path = os.path.join(output_files_path, 'instances_dr.json')

    with open(gt_json_path, "w") as f:
        json.dump(ground_truth, f, indent=4)

    with open(dr_json_path, "w") as f:
        json.dump(detection_result, f, indent=4)

    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dr_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # get all classes ap
    precisions = coco_eval.eval['precision']

    # precision: (iou, recall, cls, area range, max dets)
    assert len(coco_gt.getCatIds()) == precisions.shape[2]

    for idx, catId in enumerate(coco_gt.getCatIds()):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        name = coco_gt.loadCats(catId)[0]
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]

        if precision.size:
            ap = np.mean(precision)
        else:
            ap = float('nan')

        print("{}: {:.1f}%".format(name["name"], float(ap) * 100))


def get_coco_ground_truth(annotations_path):
    coco = COCO(annotations_path)
    image_ids = coco.getImgIds()
    cat_ids = sorted(coco.getCatIds())

    i = 0
    images_info = []
    annotations_info = []

    for image_id in tqdm(image_ids):
        image_info = {}

        image_info['file_name'] = str(image_id) + '.jpg'
        image_info['width'] = 1
        image_info['height'] = 1
        image_info['id'] = str(image_id)
        images_info.append(image_info)

        ann_id = coco.getAnnIds(imgIds=[int(image_id)], iscrowd=False)
        annotation = coco.loadAnns(ann_id)

        for obj in annotation:
            class_id = cat_ids.index(obj["category_id"])
            xmin = np.max((0, obj["bbox"][0]))
            ymin = np.max((0, obj["bbox"][1]))
            width = np.max((0, obj["bbox"][2]))
            height = np.max((0, obj["bbox"][3]))

            annotation = {}
            if obj["area"] > 0 and width > 0 and height > 0:
                iscrowd = 0

                annotation['area'] = width * height
                annotation['category_id'] = int(class_id + 1)
                annotation['image_id'] = str(image_id)
                annotation['iscrowd'] = iscrowd
                annotation['bbox'] = [int(xmin), int(ymin), int(width), int(height)]
                annotation['id'] = i
                annotations_info.append(annotation)
                i += 1

    return images_info, annotations_info


def get_coco_detection_result(args, model, dev):
    coco = COCO(args.dataset_val_path)
    image_ids = coco.getImgIds()
    detection_result = []

    for image_id in tqdm(image_ids):
        img_ann = coco.loadImgs(image_id)[0]

        filename = img_ann["file_name"]
        image_path = os.path.join(args.image_val_dir, filename)
        image = Image.open(image_path)
        image = np.array(image)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = image.repeat(3, axis=-1)

        model.eval()
        with torch.no_grad():
            outputs = predict(image, model, dev, args)

        if len(outputs) == 0:
            continue

        outputs = outputs.data.cpu().numpy()
        labels = outputs[:, 5]
        scores = outputs[:, 4]
        bboxes = outputs[:, :4]

        for bbox, class_id, score in zip(bboxes, labels, scores):
            xmin, ymin, xmax, ymax = bbox

            result = {}
            result["image_id"] = str(image_id)
            result["category_id"] = class_id + 1
            result["bbox"] = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
            result["score"] = float(score)
            detection_result.append(result)

    return detection_result


def get_yolo_ground_truth(annotations_path):
    with open(annotations_path, 'r', encoding='utf-8') as f:
        txt = f.readlines()
    annotations = [line.strip('|') for line in txt if len(line.strip("|").split("|")[1:]) != 0]

    i = 0
    images_info = []
    annotations_info = []

    for annotation in tqdm(annotations):
        annotation = annotation.strip()
        line = annotation.split("|")

        image_path = line[0]
        image_id = os.path.split(image_path)[-1].split(".")[0]

        image_info = {}

        image_info['file_name'] = image_id + '.jpg'
        image_info['width'] = 1
        image_info['height'] = 1
        image_info['id'] = str(image_id)
        images_info.append(image_info)

        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, class_id = bbox
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            class_id = int(class_id)

            w = xmax - xmin
            h = ymax - ymin

            annotation = {}
            annotation['area'] = w * h
            annotation['category_id'] = class_id + 1
            annotation['image_id'] = str(image_id)
            annotation['iscrowd'] = 0
            annotation['bbox'] = [xmin, ymin, w, h]
            annotation['id'] = i
            annotations_info.append(annotation)
            i += 1

    return images_info, annotations_info


def get_yolo_detection_result(args, model, dev):
    with open(args.dataset_val_path, 'r', encoding='utf-8') as f:
        txt = f.readlines()
    annotations = [line.strip('|') for line in txt if len(line.strip("|").split("|")[1:]) != 0]

    detection_result = []

    for annotation in tqdm(annotations):
        annotation = annotation.strip()
        line = annotation.split("|")

        image_path = line[0]
        image_id = os.path.split(image_path)[-1].split(".")[0]

        image = Image.open(image_path)
        image = np.array(image)

        model.eval()
        with torch.no_grad():
            outputs = predict(image, model, dev, args)

        if len(outputs) == 0:
            continue

        outputs = outputs.data.cpu().numpy()
        labels = outputs[:, 5]
        scores = outputs[:, 4]
        bboxes = outputs[:, :4]

        for bbox, class_id, score in zip(bboxes, labels, scores):
            xmin, ymin, xmax, ymax = bbox

            result = {}
            result["image_id"] = str(image_id)
            result["category_id"] = class_id + 1
            result["bbox"] = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
            result["score"] = float(score)
            detection_result.append(result)

    return detection_result


def get_voc_ground_truth(args, class_names):
    with open(args.dataset_val_path, 'r', encoding='utf-8') as f:
        txt = f.readlines()
    image_ids = [line.strip().split()[0] for line in txt]

    i = 0
    images_info = []
    annotations_info = []

    for image_id in tqdm(image_ids):
        xml = ET.parse(os.path.join(args.annotation_val_dir, image_id + ".xml")).getroot()

        image_info = {}

        image_info['file_name'] = image_id + '.jpg'
        image_info['width'] = 1
        image_info['height'] = 1
        image_info['id'] = str(image_id)
        images_info.append(image_info)

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

            x1 = int(float(bbox.find("xmin").text))
            y1 = int(float(bbox.find("ymin").text))
            x2 = int(float(bbox.find("xmax").text))
            y2 = int(float(bbox.find("ymax").text))
            w = x2 - x1
            h = y2 - y1

            cls_id = class_names.index(name) + 1

            annotation = {}
            annotation['area'] = w * h - 10.0
            annotation['category_id'] = cls_id
            annotation['image_id'] = str(image_id)
            annotation['iscrowd'] = 0
            annotation['bbox'] = [x1, y1, w, h]
            annotation['id'] = i
            annotations_info.append(annotation)
            i += 1

    return images_info, annotations_info


def get_voc_detection_result(args, model, dev):
    with open(args.dataset_val_path, 'r', encoding='utf-8') as f:
        txt = f.readlines()
    image_ids = [line.strip().split()[0] for line in txt]

    detection_result = []

    for image_id in tqdm(image_ids):
        image_path = os.path.join(args.image_val_dir, image_id + ".jpg")
        image = Image.open(image_path)
        image = np.array(image)
        model.eval()
        with torch.no_grad():
            outputs = predict(image, model, dev, args)

        if len(outputs) == 0:
            continue

        outputs = outputs.data.cpu().numpy()
        labels = outputs[:, 5]
        scores = outputs[:, 4]
        bboxes = outputs[:, :4]

        for bbox, class_id, score in zip(bboxes, labels, scores):
            xmin, ymin, xmax, ymax = bbox

            result = {}
            result["image_id"] = str(image_id)
            result["category_id"] = class_id + 1
            result["bbox"] = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
            result["score"] = float(score)
            detection_result.append(result)

    return detection_result


def get_ilsvrc_ground_truth(args, class_names):
    image_ids = []
    with open(args.dataset_val_path, 'r', encoding='utf-8') as f:
        txt = f.readlines()

        for line in txt:
            line = line.strip()
            if "extra" in line:
                continue
            image_ids.append(line.split()[0])

    i = 0
    images_info = []
    annotations_info = []

    for image_id in tqdm(image_ids):
        xml = ET.parse(os.path.join(args.annotation_val_dir, image_id + ".xml")).getroot()

        image_info = {}

        image_info['file_name'] = image_id + '.JPEG'
        image_info['width'] = 1
        image_info['height'] = 1
        image_info['id'] = str(image_id)
        images_info.append(image_info)

        for obj in xml.iter("object"):
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            x1 = int(float(bbox.find("xmin").text))
            y1 = int(float(bbox.find("ymin").text))
            x2 = int(float(bbox.find("xmax").text))
            y2 = int(float(bbox.find("ymax").text))
            w = x2 - x1
            h = y2 - y1

            cls_id = class_names.index(name) + 1

            annotation = {}
            annotation['area'] = w * h - 10.0
            annotation['category_id'] = cls_id
            annotation['image_id'] = str(image_id)
            annotation['iscrowd'] = 0
            annotation['bbox'] = [x1, y1, w, h]
            annotation['id'] = i
            annotations_info.append(annotation)
            i += 1

    return images_info, annotations_info


def get_ilsvrc_detection_result(args, model, dev):
    image_ids = []
    with open(args.dataset_val_path, 'r', encoding='utf-8') as f:
        txt = f.readlines()

        for line in txt:
            line = line.strip()
            if "extra" in line:
                continue
            image_ids.append(line.split()[0])

    detection_result = []

    for image_id in tqdm(image_ids):
        image_path = os.path.join(args.image_val_dir, image_id + ".JPEG")

        image = Image.open(image_path)
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = image.repeat(3, axis=-1)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]

        model.eval()
        with torch.no_grad():
            outputs = predict(image, model, dev, args)

        if len(outputs) == 0:
            continue

        outputs = outputs.data.cpu().numpy()
        labels = outputs[:, 5]
        scores = outputs[:, 4]
        bboxes = outputs[:, :4]

        for bbox, class_id, score in zip(bboxes, labels, scores):
            xmin, ymin, xmax, ymax = bbox

            result = {}
            result["image_id"] = str(image_id)
            result["category_id"] = class_id + 1
            result["bbox"] = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
            result["score"] = float(score)
            detection_result.append(result)

    return detection_result
