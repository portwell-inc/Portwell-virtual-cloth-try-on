from openpose_keypoints import openpose_keypoint
import cv2

imagePath = 'openpose_python/image/000001_0.jpg'
image = cv2.imread(imagePath)
dic = openpose_keypoint(image)
print(dic)