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
def openpose_keypoint():

    MODEL_POSE = 'COCO'
    MODEL_DIR = 'models/'
    INPUT_DIR = 'image/'
    OUTPUT_DIR = 'result/'

    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/Release;' +  dir_path + '/bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Set argument
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_dir", default="img/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    # parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    # args = parser.parse_known_args()
    api_params = {'model_folder': MODEL_DIR, 'model_pose': MODEL_POSE}

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(api_params)
        opWrapper.start()

        # Read frames on directory
        imagePaths = op.get_images_on_directory(INPUT_DIR);
        start = time.time()

        # Process and display images
        for imagePath in imagePaths:
            datum = op.Datum()
            imageToProcess = cv2.imread(imagePath)

            print(imageToProcess)
            print(imageToProcess.shape)

            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])

            new_filename = 'result\\' + imagePath.split("\\")[1].split(".")[0] + '.json'
            new_list = datum.poseKeypoints.reshape((54)).tolist()
            new_dic = {"version": 1.0, 
                        "people": [{
                        "face_keypoints": [], 
                        "pose_keypoints": new_list, 
                        "hand_right_keypoints": [], 
                        "hand_left_keypoints": []}]}
            with open(new_filename,"w") as f:
                json.dump(new_dic,f)

            print(new_list)

            # print("Body keypoints: \n" + str(datum.poseKeypoints))

            # if not args[0].no_display:
            #     cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
            #     key = cv2.waitKey(15)
            #     if key == 27: break

        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    except Exception as e:
        # print(e)
        sys.exit(-1)