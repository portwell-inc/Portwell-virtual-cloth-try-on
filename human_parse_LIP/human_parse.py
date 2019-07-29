# import the necessary packages
import os
import random

from cv2 import cv2 as cv
import keras.backend as K
import numpy as np

from config import num_classes
from data_generator import random_choice, safe_crop
from model import build_model
from utils import get_best_model

def LIP_model():
    model = build_model()
    model.load_weights(get_best_model())
    # print(model.summary())    
    return model

def human_parse_predict(model, human_img):
    img_rows, img_cols = 320, 320
    
    image_size = human_img.shape[:2]

    x, y = random_choice(image_size)
    human_img = safe_crop(human_img, x, y)

    x_test = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
    x_test[0, :, :, 0:3] = human_img / 255.

    out = model.predict(x_test)
    out = np.reshape(out, (img_rows, img_cols, num_classes))
    out = np.argmax(out, axis=2)
    for i in range(256):
        for j in range(192):
            if(out[i][j]==10):
                out[i][j] = 0
    label_out = out[:256, :192]
    return label_out

if __name__ == '__main__':
    image = cv.imread('./data/instance-level_human_parsing/Testing/Images/000001_0.jpg')
    model = LIP_model()
    human_parse = human_parse_predict(model, image)
    cv.imwrite('images/test.png', human_parse)

    K.clear_session()
