#!/usr/bin/env python3

import cv2
import numpy as np

from agent import PurePursuitPolicy
from utils import launch_env, makedirs, seed

DATASET_DIR = "../../dataset_sim"

npz_index = 0


def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1


def clean_segmented_image(seg_img):
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes

    truck_color = [116, 114, 117]
    duckie_color = [100, 117, 226]
    bus_color = [216, 171, 15]
    cone_color = [226, 111, 101]

    boxes = []
    classes = []

    AREA_TRESHOLD = 7
    for i, color in enumerate([duckie_color, cone_color, truck_color, bus_color]):
        # We create a 224x224 mask of zeros
        class_mask = np.zeros(seg_img.shape[:-1]).astype("uint8")
        # We look only at the segmented pixels of one class at a time
        class_pixels = (seg_img == color).all(axis=2)
        class_mask[class_pixels] = 255
        resized = cv2.resize(class_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        # We find the contours of all object
        contours = cv2.findContours(resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        # We filter out objects under a certain size
        for c in contours:
            area = cv2.contourArea(c)
            if area <= AREA_TRESHOLD:
                cv2.drawContours(resized, [c], -1, (0, 0, 0), -1)
            else:
                # We save the bounding box of the larger objects and their class
                x, y, w, h = cv2.boundingRect(c)
                boxes.append(np.array([x, y, x + w, y + h]))
                classes.append(i + 1)
    return np.array(boxes), np.array(classes)


seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500
MAX_SAMPLES = 10000

while True:
    if npz_index > MAX_SAMPLES:
        break
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action)  # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)   # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        # save 1 out of 5 steps to have more variety with limited memory usage
        if nb_of_steps % 5 == 0:
            boxes, classes = clean_segmented_image(segmented_obs)
            # save only when we have at least one object detected
            if len(boxes) == 0:
                continue
            save_npz(cv2.resize(obs, (224, 224)), boxes, classes)
        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
