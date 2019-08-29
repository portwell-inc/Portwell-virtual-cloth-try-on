# Human Parse Module

## Requirement
1. CUDA + cudnn (CUDA9,10都測試過)
2. tensorflow + keras
```
pip install tensorflow_gpu
pip install keras
```

## Demo
1. 從[google dirve]('https://drive.google.com/drive/u/1/folders/1j-l9sRqmH1pIKwnM6_9A_zTt-P2KFXlc')下載model資料夾，並放在"human_parse_LIP"目錄下
2. 在"human_parse_LIP"目錄下創一個"data"資料夾，將欲處理的圖片放在data裡

```
python human_parse_demo.py
```

## Usage
首先宣告model，Function **human_parse_predict()** 輸入numpy圖片，回傳parse完成的圖片
結果會儲存在human_parse_LIP/images
```
import keras.backend as K
from human_parse_LIP.human_parse import LIP_model, human_parse_predict

model = LIP_model()
human_parse = human_parse_predict(model, image)
```
