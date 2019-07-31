#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import numpy as np
import json

import argparse
import os
import time
from VITON.networks import GMM, UnetGenerator, load_checkpoint


def get_gmm_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--result_dir', type=str, default='VITON/data', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='VITON/checkpoints/gmm_final.pth', help='model checkpoint for test')
    
    opt = parser.parse_args()
    return opt

def get_tom_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TOM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    
    parser.add_argument("--stage", default = "TOM")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--result_dir', type=str, default='VITON/data', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='VITON/checkpoints/tom_final.pth', help='model checkpoint for test')

    opt = parser.parse_args()
    return opt

def save_gmm_img(opt, cloth_array, mask_array):
    save_dir = opt.result_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    cloth_array.save(os.path.join(warp_cloth_dir, 'test.jpg'))
    mask_array.save(os.path.join(warp_mask_dir, 'test.jpg'))

def save_tom_img(opt, p_tryon_array):
    save_dir = os.path.join(opt.result_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)

    p_tryon_array.save(os.path.join(try_on_dir, 'test.jpg'))
    

def test_gmm(opt, model, c, cm_array, im, parse_array, pose_label):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)

    transform = transforms.Compose([  \
        transforms.ToTensor(),   \
        transforms.Normalize([0.5], [0.5])])

    # cloth mask
    cm_array = (cm_array >= 128).astype(np.float32)
    cm = torch.from_numpy(cm_array) # [0,1]
    cm.unsqueeze_(0)

    # human parse
    for i in range(256):
        for j in range(192):
            if(parse_array[i][j]==10):
                parse_array[i][j] = 0
    parse_shape = (parse_array > 0).astype(np.float32)
    parse_head = (parse_array == 1).astype(np.float32) + \
            (parse_array == 2).astype(np.float32) + \
            (parse_array == 4).astype(np.float32) + \
            (parse_array == 13).astype(np.float32)
    parse_cloth = (parse_array == 5).astype(np.float32) + \
            (parse_array == 6).astype(np.float32) + \
            (parse_array == 7).astype(np.float32)
    # shape downsample
    parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
    parse_shape = parse_shape.resize((opt.fine_width//16, opt.fine_height//16), Image.BILINEAR)
    parse_shape = parse_shape.resize((opt.fine_width, opt.fine_height), Image.BILINEAR)
    shape = transform(parse_shape) # [-1,1]
    phead = torch.from_numpy(parse_head) # [0,1]
    pcm = torch.from_numpy(parse_cloth) # [0,1]

    # upper cloth
    im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
    im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts
    
    # pose points
    pose_data = pose_label['people'][0]['pose_keypoints']
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1,3))

    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, opt.fine_height, opt.fine_width)
    r = opt.radius
    im_pose = Image.new('L', (opt.fine_width, opt.fine_height))
    pose_draw = ImageDraw.Draw(im_pose)
    for i in range(point_num):
        one_map = Image.new('L', (opt.fine_width, opt.fine_height))
        draw = ImageDraw.Draw(one_map)
        pointx = pose_data[i,0]
        pointy = pose_data[i,1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
        one_map = transform(one_map)
        pose_map[i] = one_map[0]

    # cloth-agnostic representation
    agnostic = torch.cat([shape, im_h, pose_map], 0)
    
    # gpu
    agnostic_list = agnostic.expand(4, 22, 256, 192)
    agnostic_list = agnostic_list.cuda()

    c_list = c.expand(4, 3, 256, 192)
    c_list = c_list.cuda()

    cm_list = cm.expand(4, 1, 256, 192)
    cm_list = cm_list.cuda()

    # predict 
    grid, theta = model(agnostic_list, c_list)
    warped_cloth = F.grid_sample(c_list, grid, padding_mode='border')
    warped_mask = F.grid_sample(cm_list, grid, padding_mode='zeros')
    
    # warp cloth
    cloth_tensor = (warped_cloth[0].clone()+1)*0.5 * 255
    cloth_tensor = cloth_tensor.cpu().clamp(0,255)
    
    cloth_array = cloth_tensor.numpy().astype('uint8')
    cloth_array = cloth_array.swapaxes(0, 1).swapaxes(1, 2)

    ca = Image.fromarray(cloth_array)

    # warp mask
    warped_mask_save = warped_mask[0]*2-1
    mask_tensor = (warped_mask_save[0].clone()+1)*0.5 * 255
    mask_tensor = mask_tensor.cpu().clamp(0,255)

    mask_array = mask_tensor.numpy().astype('uint8')

    ma = Image.fromarray(mask_array)

    return ca, ma, agnostic


def test_tom(opt, model, c, agnostic):
    model.cuda()
    model.eval()
    
    base_name = os.path.basename(opt.checkpoint)
    
    y = torch.cat([agnostic, c])

    # gpu
    c = c.cuda()
    y_list = y.expand(4, 25, 256, 192)
    y_list = y_list.cuda()

    outputs = model(y_list)
    p_rendered, m_composite = torch.split(outputs, 3,1)
    p_rendered = F.tanh(p_rendered)
    m_composite = F.sigmoid(m_composite)
    
    p_tryon = c * m_composite + p_rendered * (1 - m_composite)
    
    # try on cloth
    p_tryon_tensor = (p_tryon[0].clone()+1)*0.5 * 255
    p_tryon_tensor = p_tryon_tensor.cpu().clamp(0,255)

    p_tryon_array = p_tryon_tensor.numpy().astype('uint8')
    p_tryon_array = p_tryon_array.swapaxes(0, 1).swapaxes(1, 2)

    pto = Image.fromarray(p_tryon_array)

    return pto