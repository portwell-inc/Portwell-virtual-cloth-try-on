import os

from cv2 import cv2 as cv
import keras.backend as K

from human_parse_LIP.human_parse import LIP_model, human_parse_predict

if __name__ == '__main__':
    image = cv.imread('./human_parse_LIP/data/000001_0.jpg')
    model = LIP_model()
    human_parse = human_parse_predict(model, image)
    cv.imwrite('./human_parse_LIP/images/test.png', human_parse)

    K.clear_session()