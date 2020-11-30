#!/usr/bin/env python3


import json
import os
import cv2
import numpy as np

from utils import makedirs

DATASET_PATH="../../dataset"

with open(f"{DATASET_PATH}/real_data/annotation/final_anns.json") as anns:
    annotations = json.load(anns)

npz_index = 0
while os.path.exists(f"{DATASET_PATH}/{npz_index}.npz"): # disgusting, but it works
    npz_index += 1

def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(f"{DATASET_PATH}"):
        np.savez(f"{DATASET_PATH}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1

for filename in os.listdir(f"{DATASET_PATH}/real_data/frames"):
    img = cv2.imread(f"{DATASET_PATH}/real_data/frames/{filename}")

    orig_y, orig_x = img.shape[0], img.shape[1]
    scale_y, scale_x = 224/orig_y, 224/orig_x

    img = cv2.resize(img, (224,224))

    boxes = []
    classes = []

    if filename not in annotations:
        continue

    for detection in annotations[filename]:
        box = detection["bbox"]
        label = detection["cat_name"]

        if label not in ["duckie", "cone"]:
            continue

        orig_x_min, orig_y_min, orig_w, orig_h = box

        x_min = int(np.round(orig_x_min * scale_x))
        y_min = int(np.round(orig_y_min * scale_y))
        x_max = x_min + int(np.round(orig_w * scale_x))
        y_max = y_min + int(np.round(orig_h * scale_y))

        boxes.append([x_min, y_min, x_max, y_max])
        classes.append(1 if label == "duckie" else 2)

    if len(boxes) == 0:
        continue

    save_npz(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        np.array(boxes),
        np.array(classes)
    )