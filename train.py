# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/5/8 17:29
# @Author  : 孙玉龙
# @File    : train.py

from data.datasets import *
from models.yolo import *
from torch.optim import Adam
from models.darknet import *

bb = DarkNet()
yolov1 = YOLO(bb)
yolov1.to(device=torch.device('cuda:0'))
opt=Adam(yolov1.parameters(),lr=1e-2)

for epoch in range(10):
    for sample in dataloader:
        img = sample['image']
        bbox = sample['bbox']
        label = sample['label']
        difficult = sample['difficult']
        fg=sample['fg']
        bbox[:,:,:,2:]=torch.sqrt(bbox[:,:,:,2:]/448.)
        # bbox[:, :, :, 2] = torch.sqrt(bbox[:, :, :, 2] / 448.)
        fg_score, loc, cls_score=yolov1(img)
        loss=yolov1.get_loss(fg_score,loc,cls_score,fg,bbox,label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
