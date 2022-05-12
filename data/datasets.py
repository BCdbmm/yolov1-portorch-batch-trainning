# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/5/4 22:09
# @Author  : 孙玉龙
# @File    : datasets.py

from __future__ import absolute_import
from __future__ import division
from xml.etree import ElementTree as ET
from torchvision import transforms as tvtfs
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import cv2
from data.utils import *

class voc_data(Dataset):
    def __init__(self, transform=None,data_dir='D:/datasets/VOCdevkit2007/VOCdevkit2007/VOC2007',return_difficult=True,is_train=True):
        self.data_dir = data_dir
        ids_dir = os.path.join(data_dir, 'ImageSets/Main/', 'trainval.txt')
        ids=[id_.strip() for id_ in open(ids_dir, encoding='UTF-8')]
        length=len(ids)
        np.random.shuffle(ids)
        if is_train:
            self.ids = ids[:int(length * 0.7)]
        else:
            self.ids = ids[int(length * 0.7):]
        self.return_difficult=return_difficult
        self.VOC_BBOX_LABEL_NAMES=(
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor')
        self.transform=transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id_ = self.ids[item]
        ann_pth = os.path.join(self.data_dir, "Annotations", id_ + '.xml')
        img_pth=os.path.join(self.data_dir,'JPEGImages',id_+'.jpg')
        ann = ET.parse(ann_pth)
        objs=ann.findall('object')
        labels=[]
        bboxes=[]
        difficults=[]
        # img=plt.imread(img_pth)
        # plt.imshow(img)
        # ax=plt.gca()
        fg=np.zeros((7,7))
        cls_=np.zeros((7,7,20))
        bb=np.zeros((7,7,4))
        df=np.zeros((7,7))
        size=ann.find('size')
        W=int(size.find('width').text)
        H=int(size.find('height').text)
        for obj in objs:
            difficult = int(obj.find('difficult').text)
            if not self.return_difficult and difficult==1:
                continue
            name=obj.find('name').text
            bbox_obj=obj.find('bndbox')
            bbox=[int(bbox_obj.find(e).text) for e in ['ymin','xmin','ymax','xmax']]
            cty=int(np.floor(((bbox[2]+bbox[0])/2)/H)*7)
            ctx=int(np.floor(((bbox[3]+bbox[1])/2)/W)*7)
            fg[cty,ctx]=1
            label=self.VOC_BBOX_LABEL_NAMES.index(name)+1
            cls_[cty,ctx,label-1]=1
            bb[cty,ctx,0]=bbox[0]
            bb[cty, ctx, 1] = bbox[1]
            bb[cty, ctx, 2] = bbox[2]
            bb[cty, ctx, 3] = bbox[3]
            # labels.append(label)
            # bboxes.append(bbox)
            df[cty, ctx]=difficult
            difficults.append(difficult)
            # ax.add_patch(plt.Rectangle((bbox[1],bbox[0]),bbox[3]-bbox[1],bbox[2]-bbox[0],color='r',fill=False,linewidth=1))
            # ax.text(bbox[1],bbox[0],name)
        # label=np.stack(labels)
        # bbox=np.stack(bboxes)
        # difficult=np.stack(difficults)
        img=read_image(img_pth)

        sample={'image':img,'bbox':bb,'fg':fg,'label':cls_,'difficult':df}
        if self.transform is None:
            return sample
        return self.transform(sample)

        # plt.show()


trans=tvtfs.Compose([Flip(),Resize(),Totensor()])
dd=voc_data(transform=trans)

dataloader=DataLoader(dd,batch_size=6,shuffle=True,num_workers=0)
