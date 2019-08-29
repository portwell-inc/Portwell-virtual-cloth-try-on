import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import time

from VITON.VITON import VITON, viton_model_init

if __name__ == "__main__":
    
    transform = transforms.Compose([  \
        transforms.ToTensor(),   \
        transforms.Normalize([0.5], [0.5])])
    
    # input for model
    # cloth
    c = Image.open('VITON/data/cloth/000192_1.jpg')
    # cloth mask
    cm = Image.open('VITON/data/cloth-mask/000192_1.jpg')

    c = transform(c)  # [-1,1]
    cm_array = np.array(cm)

    # person image 
    im = Image.open('VITON/data/image/000057_0.jpg')
    im = transform(im) # [-1,1]

    # human parse
    im_parse = Image.open('VITON/data/image-parse/000057_0.png')
    parse_array = np.array(im_parse)

    # pose points
    with open('VITON/data/pose/000057_0_keypoints.json') as f:
        pose_label = json.load(f)

    # call VITON module
    start = time.clock()
    stage1_model, stage2_model = viton_model_init()
    end = time.clock()
    print(end-start)

    start = time.clock()
    result = VITON(c, cm_array, im, parse_array, pose_label, stage1_model, stage2_model)
    end = time.clock()
    print(end-start)