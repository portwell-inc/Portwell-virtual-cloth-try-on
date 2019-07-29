import os
import random

from cv2 import cv2 as cv
import numpy as np
from keras.utils import Sequence
from keras.utils import to_categorical
from config import batch_size, img_rows, img_cols, num_classes, color_map


def random_choice(image_size):
    height, width = image_size
    crop_height, crop_width = 320, 320
    x = random.randint(0, max(0, width - crop_width))
    y = random.randint(0, max(0, height - crop_height))
    return x, y


def safe_crop(mat, x, y):
    crop_height, crop_width = 320, 320
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    return ret
