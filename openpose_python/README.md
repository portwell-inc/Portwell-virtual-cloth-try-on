# OpenPose Module

## Requirement
1. **Window10**
2. **請務必使用python 3.6.x**
```
pip install numpy
pip install opencv-python
```

## Demo
1. 從[google dirve](https://drive.google.com/drive/u/1/folders/1D5kQKnYaSF9uIBYlf9PSMpi5BytFTYSG)下載bin跟model兩個資料夾，並放在"openpose_python"目錄下
2. 在"openpose_python"目錄下創一個"image"資料夾，將欲處理的圖片放在image裡
```
python demo.py
```

## Usage
Function **openpose_keypoint()** 輸入numpy圖片，回傳人體keypoints
```
from openpose_keypoints import openpose_keypoint

dic = openpose_keypoint(image)
```
