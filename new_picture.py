from openpose_python.openpose_keypoints import openpose_keypoint
from human_parse_LIP.human_parse import LIP_model, human_parse_predict
import json
import cv2
import sys
import os

#read input and resize
image = cv2.imread('shot.jpg')
image = image[50:450, 150:450]
image = cv2.resize(image, (192, 256), interpolation=cv2.INTER_CUBIC)
cv2.imwrite("image.jpg",image)
os.remove('shot.jpg')

keypoint = openpose_keypoint(image)
with open("keypoint.json","w") as f:
	json.dump(keypoint,f)

model = LIP_model()
human_parse = human_parse_predict(model, image)
cv2.imwrite("parse.jpg",human_parse)
