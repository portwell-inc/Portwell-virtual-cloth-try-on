from openpose_python.openpose_keypoints import openpose_keypoint
from human_parse_LIP.human_parse import LIP_model, human_parse_predict
import json
import cv2
import sys
import os

image = cv2.imread('image.jpg')

keypoint = openpose_keypoint(image)
with open("keypoint.json","w") as f:
	json.dump(keypoint,f)

model = LIP_model()
human_parse = human_parse_predict(model, image)
cv2.imwrite("parse.jpg",human_parse)
