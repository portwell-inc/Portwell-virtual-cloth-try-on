from openpose_keypoints import openpose_keypoint
import cv2

imagePath = 'image/000005_0.jpg'
image = cv2.imread(imagePath)
dic = openpose_keypoint(image)
print(dic)