# VITON Module

## Requirement
1. PyTorch 安裝請參考[官網](https://pytorch.org)
```
pip install tensorboardX
```

## Test
1. 從[Google Drive](https://drive.google.com/drive/u/1/folders/1tW67WtAGv_yVjwyEc_wGGIxxbTJ7oJOJ)下載'checkpoints'資料夾(trained model)，放在目錄'VITON'

```
python VITON_demo.py
```

## Usage
```
from VITON.VITON import VITON, viton_model_init

stage1_model, stage2_model = viton_model_init()
result = VITON(cloth, cloth_mask, image, parse, pose, stage1_model, stage2_model)
```
### Input格式
 + cloth：PIL物件
 + cloth_mask：numpy array
 + image：PIL物件
 + parse：numpy array
 + pose：json
 + stage1_model,stage_model：call function **viton_model_init()**

## Train

### GMM Train ( stage 1 )
sample command
python train.py --name gmm_train_new --stage GMM --workers 4 --datamode train --data_list train_pairs.txt --save_count 5000 --shuffle
--name 訓練名稱 ( 存放checkpoint的資料夾名稱 )  
--stage GMM  
--datamode train / test (選擇資料集)  
--data_list train_pairs.txt / test_pairs.txt (資料集配對檔)  
--save_count 每訓練N次儲存一次checkpoint  
--checkpoint 可以選擇由某一次的checkpoint繼續訓練(default為從頭訓練)

### GMM Test ( stage 1 )
sample command
python test.py --name gmm_traintest_new --stage GMM --workers 4 --datamode train --data_list train_pairs.txt --checkpoint checkpoints/gmm_train_new/gmm_final.pth
--name 測試名稱 ( 存放tensorboard結果的資料夾名稱 )  
--stage GMM  
--datamode train / test (選擇資料集)  
--data_list train_pairs.txt / test_pairs.txt (資料集配對檔)  
--checkpoint 選擇某一次checkpoint作為模型使用  

----結果圖會存放在result底下以checkpoint命名的資料夾內----

### TOM Train ( stage 2 )
訓練前須有stage 1 test產生的圖片  
將warp-cloth及warp-mask移動到所屬資料集裡  

sample command
python train.py --name tom_train_new --stage TOM --workers 4 --datamode train --data_list tom_train_pairs.txt --save_count 5000 --shuffle
--name 訓練名稱 ( 存放checkpoint的資料夾名稱 )  
--stage TOM  
--datamode train / test (選擇資料集)  
--data_list tom_train_pairs.txt / tom_test_pairs.txt (資料集配對檔, 由stage 1 test產生)  
--save_count 每訓練N次儲存一次checkpoint  
--checkpoint 可以選擇由某一次的checkpoint繼續訓練(default為從頭訓練)

### TOM Test ( stage 2 )
測試前一樣須有stage 1 test產生的圖片  
將warp-cloth及warp-mask移動到所屬資料集裡

sample command
python test.py --name tom_traintest_new --stage TOM --workers 4 --datamode train --data_list tom_train_pairs.txt --checkpoint checkpoints/tom_train_new/tom_final.pth
--name 測試名稱 ( 存放tensorboard結果的資料夾名稱 )  
--stage TOM  
--datamode train / test (選擇資料集)  
--data_list tom_train_pairs.txt / tom_test_pairs.txt (資料集配對檔, 由stage 1 test產生)  
--checkpoint 選擇某一次checkpoint作為模型使用  

----結果圖會存放在result底下以checkpoint命名的資料夾內----
