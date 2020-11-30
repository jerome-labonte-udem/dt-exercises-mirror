# dt-object-detection-exercise
Repo for the object detection homework

## Setup

IMPORTANT! Setup a virtual environment. We recommend Pycharm, a python IDE that includes support for `venv` from the get-go.

If you don't do this, your python might get confused between our different version of the simulator and the normal one that you might have installed globally.

Then, clone into the simulator coded for this homework (you might have to `chmod +x` the script).

    ./clone.sh

Finally, use your IDE of choice to install the `requirements.txt` that just got copied to the root of this directory into your `venv` (on pycharm, simply alt-enter the requirements and it should install it).

## Step one: investigation (0%)

What does an object detection dataset look like? What information do we need?

Try downloading [the PennFudanPed dataset](https://www.cis.upenn.edu/~jshi/ped_html/), a sample pedestrian detection dataset.

The first step of this exercise is simply understanding what's in that dataset. You'll notice that if you try opening 
the masks in that dataset, your computer will display a black image. That's because each 
segmented pedestrian's mask is a single digit and the image only has one channel, 
even though the mask was saved as a `.jpg`. 

Try scaling the masks from 0 to 255, using something like `np.floor(mask / np.max(mask) * 255).astype(np.uint8)`. 
This will make the masks into something akin to a `.bmp`. Then, use opencv's `applyColorMap` 
feature on that to visualize the results. Try looking at the two `display` functions found in `utils.py` for inspiration.

This is all optional, of course. But we highly recommend trying it out, so that you can
have an intuition for the type of images you should collect in step two.

You'll also notice that the dataset doesn't include any bounding boxes. 
That's okay. For training with PennFudanPed, we have to compute them through numpy and opencv, just like we will on your own 
dataset.

Actually, for our own training, we won't need the masks! All we want are the
bounding boxes. But PennFudanPed is a useful example, as it shows how
we can extract bounding boxes from masks, something we will also do for our own dataset. To see how to do
this, you may skip ahead to the tutorial linked in the Training section.

## Step two: data collection (x%)

Now that we know what data we have to collect, we can start collecting it.

Do note that in this exercise, we don't want to differentiate the objects from one another: 
they will all have the same class. Our images will include duckies, busses, trucks, and cones. 

We thus have five classes:
- 0: background
- 1: duckie
- 2: cone
- 3: truck
- 4: bus

To collect our data, we'll use the `segmented` flag in the simulator. 
Try running the `data_collection.py` file, which cycles between the segmented simulator and the normal one. 
Notice that, unfortunately, our duckie citizens are still novice in the field of computer vision,
and they couldn't figure out how to remove the noise generated from their segmentation algorithm
in the segmented images. That's why there's all this odd coloured "snow".

Notice that when we're in the segmented simulator, all the objects we're interested in 
have the 
exact same color, and the lighting and domain randomization are turned off. Just like the 
`data_collection.py` file does, we 
can also turn the segmentation back off for the 
same position of the agent. In other words, we can essentially produce two 100% 
identical images, save for the fact that one is segmented and the other is not.


Then, collect the dataset:
- We want as many images as reasonable. The more data you have, the better your model, but also, 
the longer your training time.
- We want to remove all non-`classes` pixels in the segmented images. You'll have to 
identify the white lines, the yellow lines, the stop lines, etc, and remove them from 
the masks. Do the same for the coloured "snow" that appears in the segmented images.
- We want to identify each class by the numbers mentioned above
- We also want the bounding boxes, and corresponding classes.

Your dataset must respect a certain format. The images must be 224x224x3 images. The boxes must be in `[xmin, ymin, xmax, ymax]` format.
The labels must be an `np.array` indexed the same way as the boxes (so `labels[i]` is the label of `boxes[i]`).

We want to be able to read your `.npz`, so you *must* respect this format:

    img = data[f"arr_{0}"]
    boxes = data[f"arr_{1}"]
    classes = data[f"arr_{2}"]
    
Additionally, each `.npz` file must be identified by a number. So, if your dataset contains 1000 items, you'll have
npzs ranging from `0.npz` to `999.npz`.

Do note that even though your dataset images have to be of size 224x224, you are allowed to feed smaller
or bigger images to your model. If you wish to do so, simply resize the images at train/test/validation time.

We're also going to use the Duckietown Object Detection dataset! We already wrote the transformation funtion
for you in `data_collection/data_transfer.py`. Simply run that after you've ran your own data collection
script. You might also want to look at how it saves files as inspiration.

### Evaluation

We will manually look at part of your dataset and make sure that your bounding boxes match with the images.

## Model training (x%)

Now that we have our dataset, we will train on it. You may use Pytorch or TensorFlow; 
it doesn't really matter because we'll dockerize your implementation.

The creators of this exercise do have a soft spot for Pytorch, so we'll use it as an 
example.

This being ML, and ML being a booming field dominated by blogposts and online 
tutorials, it would be folly for us not to expect you to google "how 2 obj
detection pytorch". Let us save you some time. 
Here's the first result: 
[pytorch's object detection tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
. We'll loosely follow that template.

First, define your `Dataset` class. Like in the link, for any given image index, it should provide:
- The bounding boxes for each class in each image (contrarily to the tutorial, you calculated this earlier in the
Data collection part of this exercise)
- The class labels for each bounding box
- The normal, non-segmented image
- An ID for the image (you should just use the index of the `.npz`)

Needless to say, each of the items must be index-dependent (the nth item of `boxes` must correspond to the nth item of `labels`).

We don't need the areas or the masks here: we'll change the model so that we only predict boxes and labels.
Here's the model we will use instead of the tutorial's suggestion: 

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

Then, you can use your `Dataset` class to train your model.

### A note on the tutorial

Make sure to carefully read the tutorial. Blindly copying it won't directly work. The training data it
expects is very specific, and you should make sure that you follow its structure exactly.

Additionally, weirdly enough, the tutorial expects you to have some files that it does not link to.

Perhaps having a look (and a download) at these links might save you some time:

- https://github.com/pytorch/vision/blob/master/references/detection/engine.py
- https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
- https://github.com/pytorch/vision/blob/master/references/detection/transforms.py

You can also safely remove the `evaluate` call that the tutorial uses, and it will save you the headache
of installing most of the `coco_utils` and `coco_evaluate` dependencies.

### Making sure your model does the right thing

You should probably write a way to visually evaluate the performance of your model.

Something like displaying the input image and overlaying the bounding boxes (colored by class) would
be simple but very effective.

You should also carefully read `model.py`, as there are comments in it that describe the API your
wrapper should respect.

### Changing the model

You can essentially use any model you like here. Pay attention to the grading guidelines at the end of this
section: the faster and smaller your model, the better.

The tutorial offers very good performance, but there are better options out there.

### Training hardware

But how should you actually train your model? If you have a recent-ish nvidia GPU, you can
directly train on your computer. For reference, using a dataset with 2000 items, training on
a GTX960 was very doable.

If you don't have a GPU, or if your GPU is too slow, you can use Google Colab. We included
a `.ipynb` in the `model` directory. You can open it with Google Colab, upload the root of
this exercise to your Google Drive, and the provided notebook will mount the folder from your
drive into the Colab runtime, and then call your training script. To access the saved weights,
simply download them from your Google Drive.

### Evaluation

We will evaluate this section in two ways: 

1. What is the accuracy of your model? Specifically, we will use "mean average precision" or mAP 
to evaluate your model, so you might want to optimize it for that metric.
2. What is the size of your final model? We'll look at your Docker image, and smaller models
will receive more points than bigger ones. You can't just submit an empty file to game this
criterion, though, as you must score at least 85% mAP to even score points in this criterion.

## ROS

We're using object detection for a robotics application, so your model should offer a good framerate! 
A heavy, pretrained, slow model might be the wrong choice here.
The size criterion already does a good job of evaluating this, but we also want to integrate
your model into the Duckietown ROS stack. Write a ROS node that instantiates the `Wrapper`
of your model, and make it publish object detection predictions as fast as it can. The faster
the frequency, the more points you'll score here.

You will find an almost-complete solution for this new node in 
`dt-exercises/obj_det/exercise_ws/src/obj_det/src`. You will have to add your `model.py` file along
with its weights in the `include` folder for that `obj_det` node. Make sure that the weights' path is
the right one: it'll be a bit different once inside ROS, but you can easily just run your node, let
it die, and look at the logs to infer what path you should use.

We will use your node for the `LFP` challenge. 

Write a new control node which calls your object detection one. One there's a detection in front of the
robot, on the road, switch to obstacle avoidance mode. You can use the TODOs we left in `lane_control`
as guidance.