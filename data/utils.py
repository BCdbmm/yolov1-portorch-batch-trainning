# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/5/4 20:41
# @Author  : 孙玉龙
# @File    : utils.py

import cv2
import numpy as np
import torch


def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    x_scale = out_size[1] / in_size[1]
    y_scale = out_size[0] / in_size[0]
    bbox[:, 0] = bbox[:, 0] * y_scale
    bbox[:, 1] = bbox[:, 1] * x_scale
    bbox[:, 2] = bbox[:, 2] * y_scale
    bbox[:, 3] = bbox[:, 3] * x_scale
    return bbox


def flip_bbox(bbox, img_size, y_flip=False, x_flip=False):
    H, W = img_size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:,:, 0]
        y_min = H - bbox[:,:, 2]
        y_max[(bbox==0).all(axis=2)]=0
        y_min[(bbox == 0).all(axis=2)] = 0
        bbox[:,:, 0] = y_min
        bbox[:,:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:,:, 1]
        x_min = W - bbox[:,:, 3]
        x_max[(bbox == 0).all(axis=2)] = 0
        x_min[(bbox == 0).all(axis=2)] = 0
        bbox[:,:, 1] = x_min
        bbox[:,:, 3] = x_max
    return bbox


def flip_img(img, x_flip=False, y_flip=False):
    img = img.copy()
    if y_flip:
        img = img[::-1, :, :]
    if x_flip:
        img = img[:, ::-1, :]
    return img



class Resize():
    def __call__(self, sample, size=(448, 448)):
        img = sample['image']
        bbox = sample['bbox']
        label = sample['label']
        difficult = sample['difficult']
        fg=sample['fg']
        in_size = img.shape[:2]
        img = cv2.resize(img, size)
        bbox = resize_bbox(bbox, in_size, size)
        return {'image': img, 'bbox': bbox,'fg':fg ,'label': label, 'difficult': difficult}


class Totensor():
    def __call__(self, sample):
        img = sample['image']
        bbox = sample['bbox']
        label = sample['label']
        difficult = sample['difficult']
        fg=sample['fg']
        img = (img.transpose(2, 0, 1) - 128.) / 128.
        if torch.cuda.is_available():
            img = torch.Tensor(img).cuda()
            bbox = torch.Tensor(bbox).cuda()
            label = torch.LongTensor(label).cuda()
            difficult = torch.Tensor(difficult).cuda()
            fg=torch.LongTensor(fg).cuda()
        else:
            img = torch.Tensor(img)
            bbox = torch.Tensor(bbox)
            label = torch.LongTensor(label)
            difficult = torch.Tensor(difficult)
            fg = torch.LongTensor(fg)
        return {'image': img, 'bbox': bbox,'fg':fg ,'label': label, 'difficult': difficult}


class Flip():
    def __call__(self, sample):
        img = sample['image']
        bbox = sample['bbox']
        label = sample['label']
        difficult = sample['difficult']
        fg=sample['fg']
        x_flip = np.random.choice([True, False])
        y_flip = np.random.choice([True, False])
        bbox = flip_bbox(bbox, img.shape[:2], x_flip=x_flip, y_flip=y_flip)
        img = flip_img(img, x_flip=x_flip, y_flip=y_flip)
        return {'image': img, 'bbox': bbox,'fg':fg ,'label': label, 'difficult': difficult}
