# follow this https://pytorch.org/get-started/locally/
#!pip3 install torch torchvision torchaudio

# this only works when doing in from terminal for some reason ??? 
# clone works
# then manually cd /Users/Mike/Desktop/Goose/Autonomous-Goose-Chaser/yolov5
# then do pip3 install -r requirements.txt
# IDK why cell does not work
#!git clone https://github.com/ultralytics/yolov5  # clone
#!cd /Users/Mike/Desktop/Goose/Autonomous-Goose-Chaser/yolov5
#!pip3 install -r requirements.txt  # install

import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid
import os
import time

# pretrained model SMALL VERSION hence "s"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# TO RUN LABEL IMAGER
# cd /Users/Mike/Desktop/Yolo/labelImg
# python3 labelImg.py
# DO ALL OF THIS IN TERNMINAL

# cd /Users/Mike/Desktop/Yolo/yolov5
# python3 train.py --img 320 --batch 16 --epochs 5 --data dataset.yaml --weights yolov5s.pt --workers 2
