# Portwell-virtual-cloth-try-on
## Introduction
這是一個測試VITON的demo網頁<br/>
本專案共分為三個模組VITON,human_parse,openpose
## Requirement
**Step1：安裝三個模組VITON,human_parse,openposee共用套件**
```
pip install numpy
pip install opencv-python
pip install Pillow
```
**Step2：web sever 相關套件**
```
pip install flask
pip install flask_session
```
**Step3：安裝[VITON](./VITON/README.md), [human_parse](./human_parse_LIP/README.md) 與[openpose](./openpose_python/README.md)**


## Demo
```
python server.py
```
在瀏覽器中開啟 http://localhost:5000/demo 開始demo