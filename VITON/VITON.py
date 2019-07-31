import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json

from VITON.virtual_try_on import get_gmm_opt, get_tom_opt, test_gmm, test_tom, save_gmm_img, save_tom_img
from VITON.networks import GMM, UnetGenerator, load_checkpoint

def stage1_predict(c, cm_array, im, parse_array, pose_label):
    opt = get_gmm_opt()
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
    
    model = GMM(opt)
    load_checkpoint(model, opt.checkpoint)
    with torch.no_grad():
        ca, ma, agn = test_gmm(opt, model, c, cm_array, im, parse_array, pose_label)
    
    save_gmm_img(opt, ca, ma)
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))
    return ca, agn


def stage2_predict(c, agnostic):
    opt = get_tom_opt()
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
    
    model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    load_checkpoint(model, opt.checkpoint)
    with torch.no_grad():
        result = test_tom(opt, model, c, agnostic)

    save_tom_img(opt, result)
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))
    return result


def VITON(c, cm_array, im, parse_array, pose_label):
    ca, agn = stage1_predict(c, cm_array, im, parse_array, pose_label)

    transform = transforms.Compose([  \
        transforms.ToTensor(),   \
        transforms.Normalize([0.5], [0.5])])

    c = transform(ca)  # [-1,1]
    
    result = stage2_predict(c, agn)

    return result


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

    result = VITON(c, cm_array, im, parse_array, pose_label)