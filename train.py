# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/5/8 17:29
# @Author  : 孙玉龙
# @File    : train.py

from data.datasets import *
from models.yolo import *
from torch.optim import Adam
from models.darknet import *
from models.backbones import *

# bb = DarkNet()
bb=ResNetBackBone()
yolov1 = YOLO(bb)
yolov1.to(device=torch.device('cuda:0'))
lr=1e-6
opt=Adam(yolov1.parameters(),lr=1e-6)

i=0
epoch_loss=0
for epoch in range(10):
    for sample in dataloader:
        img = sample['image']
        bbox = sample['bbox']
        label = sample['label']
        difficult = sample['difficult']
        fg=sample['fg']
        bbox=bbox/448.
        bbox[:,:,:,2:]=torch.sqrt(bbox[:,:,:,2:])

        fg_score, loc, cls_score=yolov1(img)
        loss=yolov1.get_loss(fg_score,loc,cls_score,fg,bbox,label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i%100==0:
            print(loss.item())
        i+=1
        epoch_loss+=loss.item()
    print('epoch %s loss:'%epoch,epoch_loss/i)
    epoch_loss=0
    i=0
    torch.save(yolov1,'yolov1-%s.pth'%epoch)

