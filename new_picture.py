from openpose_python.openpose_keypoints import openpose_keypoint
from human_parse_LIP.human_parse import LIP_model, human_parse_predict
import json
import cv2
import sys
import os

#read input
input_dir = sys.stdin.read()

image = cv2.imread(input_dir + '/image.jpg')

keypoint = openpose_keypoint(image)
new_dic = {"version": 1.0, 
	"people": [{
	"face_keypoints": [], 
	"pose_keypoints": keypoint, 
	"hand_right_keypoints": [], 
	"hand_left_keypoints": []}]}
with open(input_dir + "/keypoint.json","w") as f:
	json.dump(new_dic,f)

model = LIP_model()
human_parse = human_parse_predict(model, image)
cv2.imwrite(input_dir + "/parse.jpg",human_parse)
