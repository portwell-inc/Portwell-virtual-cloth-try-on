from mask import mask
import cv2

img = cv2.imread('0004.jpg')
out = mask(img)
print(out)