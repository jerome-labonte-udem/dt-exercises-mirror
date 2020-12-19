#!/usr/bin/env python3

import os

import numpy as np
import torch
import torchvision
import transforms as T
from engine import evaluate, train_one_epoch
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils

MODEL_PATH = "../exercise_ws/src/object_detection/include/object_detection/weights"
DATASET_PATH = "../dataset"


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class DuckieTownDataset:
    """
    Create a duckietown dataset
    """
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.data = [path for path in list(sorted(os.listdir(root))) if path.endswith("npz")]

    def __getitem__(self, idx):
        # load images, bounding boxes and labels
        data_path = os.path.join(self.root, self.data[idx])
        data = np.load(data_path)
        img = Image.fromarray(data["arr_0"], "RGB")
        boxes = data["arr_1"]
        labels = data["arr_2"]
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 5
    # use our dataset and defined transformations
    dataset = DuckieTownDataset(DATASET_PATH,
                                get_transform(train=True))
    dataset_test = DuckieTownDataset(DATASET_PATH,
                                     get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 2 epochs
    num_epochs = 2

    for epoch in range(num_epochs):
        # This will create 100 jpeg image to visually check the accuracy of our model at the beginning of each epoch
        evaluate(model, data_loader_test, device, epoch)

        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, f"model_real_e{epoch}.pt"))
    # evaluate last epoch
    evaluate(model, data_loader_test, device, epoch + 1)


if __name__ == "__main__":
    main()
