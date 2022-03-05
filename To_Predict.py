import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import functional as FT
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import copy
import math
from PIL import Image
import cv2
import albumentations as A  # our data augmentation library
import matplotlib.pyplot as plt
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2

from torchvision.utils import save_image
import shutil
import pathlib
from torch.autograd import Variable
from torchvision.transforms import transforms as T


# User parameters
SAVE_NAME_OD = "./Models-OD/Vehicle_Detector.model"
DATA_DIR = "./Images/Training_Images/"
USE_CHECKPOINT = True
IMAGE_SIZE = 1200 # Row and column number 2180
DATASET_PATH = "./Vehicle_Detector/"
TO_PREDICT_PATH = "./Images/Prediction_Images/To_Predict/"
PREDICTED_PATH = "./Images/Prediction_Images/Predicted_Images/"
SAVE_FULL_IMAGES = True
SAVE_CROPPED_IMAGES = False
MIN_SCORE = 0.6



def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


# Creates class folder
def makeDir(dir, classes_2):
    for classIndex, className in enumerate(classes_2):
        os.makedirs(dir + className, exist_ok=True)



# Starting stopwatch to see how long process takes
start_time = time.time()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

dataset_path = DATASET_PATH



#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes_1 = len(categories.keys())
categories

classes_1 = [i[1]['name'] for i in categories.items()]
classes_1



# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)


# TESTING TO LOAD MODEL
if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD)
if USE_CHECKPOINT and os.path.isfile(SAVE_NAME_OD):
    model_1.load_state_dict(checkpoint)


device = torch.device("cuda") # use GPU to train
model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    ToTensorV2()
])





pred_dict = {}
for image_name in os.listdir(TO_PREDICT_PATH):
    image_path = os.path.join(TO_PREDICT_PATH, image_name)

    
    image_b4_color = cv2.imread(image_path)
    image_b4_color_and_rotated = cv2.rotate(image_b4_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=image)
    transformed_image = transformed_image["image"]
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    # ALLdieCoordinates x y values are SWITCHED BUT IT WRKS
    # dieCoordinates[:, 0] = pred_1['boxes'][pred_1['scores'] > 0.8][:, 1]
    # dieCoordinates[:, 1] = pred_1['boxes'][pred_1['scores'] > 0.8][:, 0]
    # dieCoordinates[:, 2] = pred_1['boxes'][pred_1['scores'] > 0.8][:, 3]
    # dieCoordinates[:, 3] = pred_1['boxes'][pred_1['scores'] > 0.8][:, 2]
    
    box_width = int(dieCoordinates[0][2]-dieCoordinates[0][0]) 
    box_height = int(dieCoordinates[0][3]-dieCoordinates[0][1])
    line_width = round(box_width * 0.05)
    
    if SAVE_FULL_IMAGES:
        test_image = draw_bounding_boxes(transformed_image,
            dieCoordinates,
            [classes_1[i] for i in pred_1['labels'][pred_1['scores'] > MIN_SCORE].tolist()], 
            width=line_width
            )
        
        # Saves full image with bounding boxes
        save_image((test_image/255), PREDICTED_PATH + image_name)



print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)