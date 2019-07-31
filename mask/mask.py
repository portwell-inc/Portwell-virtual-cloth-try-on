import numpy as np
import cv2

def mask(img_mat):
	gray = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
	mask = (gray <= 230).astype(np.float32)
	return mask