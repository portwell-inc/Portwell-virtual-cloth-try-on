# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import json
from sys import platform
import argparse
import time
import subprocess

#input:images in img file  output:several JSON file
def openpose_keypoint(img_mat):

    MODEL_POSE = 'COCO'
    MODEL_DIR = 'models/'
    OUTPUT_DIR = 'result/'

    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/Release;' +  dir_path + '/bin;'
        import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # 設定參數
    api_params = {'model_folder': MODEL_DIR, 'model_pose': MODEL_POSE}

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(api_params)
        opWrapper.start()
        datum = op.Datum()
        datum.cvInputData = img_mat  # imageToProcess就是讀入的圖片marix
        opWrapper.emplaceAndPop([datum])  #第一張圖花了5秒

        new_list = datum.poseKeypoints.reshape((54)).tolist()
        new_dic = {"version": 1.0, 
                    "people": [{
                    "face_keypoints": [], 
                    "pose_keypoints": new_list, 
                    "hand_right_keypoints": [], 
                    "hand_left_keypoints": []}]}

    except Exception as e:
        # print(e)
        sys.exit(-1)

    return new_dic