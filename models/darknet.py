# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/5/7 21:15
# @Author  : 孙玉龙
# @File    : darknet.py

import torch
import torch.nn  as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        mid_channel = int(self.out_channel / 2)
        self.conv1 = nn.Conv2d(self.in_channel, mid_channel, 1, stride=1)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, self.out_channel, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x))
        return x


class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.input_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(64, 192, 3, 1, padding=1)
        self.block1 = self._conv_block(192, 256, 1)
        self.block1_ = self._conv_block(256, 512, 1)
        self.block2 = self._conv_block(512, 512, 4)
        self.block2_ = self._conv_block(512, 1024, 1)
        self.block3 = self._conv_block(1024, 1024, 2)
        self.conv4 = nn.Conv2d(1024, 1024, 3, 1, padding=1)
        self.conv4_ = nn.Conv2d(1024, 1024, 3, 2, padding=1)
        self.conv5 = nn.Conv2d(1024, 1024, 3, 1, padding=1)
        self.conv5_ = nn.Conv2d(1024, 1024, 3, 1, padding=1)
        self.out_conv = nn.Conv2d(1024, 1024, 3, 1, padding=1)
        self.out_conv_ = nn.Conv2d(1024, 30, 1, 1)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.input_conv(x)), 2, 2)
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), 2, 2)
        x = self.block1(x)
        x = F.max_pool2d(self.block1_(x), 2, 2)
        x = self.block2(x)
        x = F.max_pool2d(self.block2_(x), 2, 2)
        x = self.block3(x)
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv4_(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv5_(x))
        x = F.leaky_relu(self.out_conv(x))
        x = self.out_conv_(x)
        return x.permute(0, 2, 3, 1)

    def _conv_block(self, in_channl, out_channel, num_block):
        Convs = []
        for i in range(num_block):
            if i == 0:
                c = ConvLayer(in_channl, out_channel)
                Convs.append(c)
            else:
                c = ConvLayer(out_channel, out_channel)
                Convs.append(c)
        return nn.Sequential(*Convs)

# x=torch.Tensor(64,3,448,448)
#
# m=DarkNet()
# y=m(x)
# print(y.shape)
