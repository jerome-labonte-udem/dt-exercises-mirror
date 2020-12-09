import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import glob
import torchvision.models as models
import numpy as np
class NoGPUAvailable(Exception):
    pass

class Wrapper():
    def __init__(self, model_file):
        # TODO Instantiate your model and other class instances here!
        # TODO Don't forget to set your model in evaluation/testing/production mode, and sending it to the GPU
        # TODO If no GPU is available, raise the NoGPUAvailable exception
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            # device = torch.device('cpu')
            raise NoGPUAvailable()
        self.model = Model(device, model_file)

    def predict(self, batch_or_image):
        # TODO Make your model predict here!

        # TODO The given batch_or_image parameter will be a numpy array (ie either a 224 x 224 x 3 image, or a
        # TODO batch_size x 224 x 224 x 3 batch of images)
        # TODO These images will be 224 x 224, but feel free to have any model, so long as it can handle these
        # TODO dimensions. You could resize the images before predicting, make your model dimension-agnostic somehow,
        # TODO etc.

        # TODO This method should return a tuple of three lists of numpy arrays. The first is the bounding boxes, the
        # TODO second is the corresponding labels, the third is the scores (the probabilities)

        # See this pseudocode for inspiration
        boxes = []
        labels = []
        scores = []
        if len(batch_or_image.shape) == 3:
            batch_or_image = np.array([batch_or_image])
        for img in batch_or_image:  # or simply pipe the whole batch to the model instead of using a loop!
            box, label, score = self.model.predict(img)  # TODO you probably need to send the image to a tensor, etc.
            boxes.append(box)
            labels.append(label)
            scores.append(score)

        return boxes, labels, scores

class Model():    # TODO probably extend a TF or Pytorch class!
    def __init__(self, device, model_file):
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
        # move model to the right device
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def predict(self, img):
        img = img[:, :, ::-1]
        img = np.transpose(img, (2, 0, 1))/255
        print("img shape", img.shape)
        img = torch.from_numpy(img).to(self.device).type(torch.FloatTensor)
        print("tensor shape", img.shape)
        print("dtype", img.dtype)
        predictions = self.model([img])
        print(predictions)
        boxes = predictions[0]["boxes"]
        labels = predictions[0]["labels"]
        scores = predictions[0]["scores"]
        return boxes, labels, scores
    # TODO add your own functions if need be!
