# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/5/12 22:52
# @Author  : 孙玉龙
# @File    : backbones.py

from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F


class ResNetBackBone(nn.Module):
    def __init__(self):
        super(ResNetBackBone, self).__init__()
        base_model=resnet50(pretrained=True)
        layers=[]
        for layer_name,layer in base_model._modules.items():
            for param in layer.parameters():
                param.requires_grad=False
            layers.append(layer)
        self.layers=nn.Sequential(*layers[:-2])
        self.head_conv=nn.Conv2d(2048,1024,1,1)
        self.out_conv = nn.Conv2d(1024, 1024, 3, 2, padding=1)
        self.out_conv_ = nn.Conv2d(1024, 30, 1, 1)
    
    def forward(self,x):
        x=F.leaky_relu(self.layers(x))
        x=F.leaky_relu(self.head_conv(x))
        x=F.leaky_relu(self.out_conv(x))
        x=self.out_conv_(x)
        return x.permute(0, 2, 3, 1)
        
