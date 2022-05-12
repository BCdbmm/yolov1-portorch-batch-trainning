# !/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2022/5/8 12:19
# @Author  : 孙玉龙
# @File    : yolo.py

import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import nms


class YOLO(nn.Module):
    def __init__(self, backbone):
        super(YOLO, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        feature = self.backbone(x)
        cls_score = torch.softmax(feature[:, :, :, 10:], dim=-1)
        fg_score = torch.softmax(feature[:, :, :, 8:10], dim=-1)[:, :, :, 0]
        loc = torch.sigmoid(feature[:, :, :, :8]).view(-1, 7, 7, 2, 4)
        return fg_score, loc, cls_score

    def get_loss(self, fg_score, loc, cls_score, gt_fg, gt_bbox, gt_label):
        x = np.arange(7)
        y = np.arange(7)
        offset_x, offset_y = np.meshgrid(x, y)
        bbox = torch.empty(loc.shape).cuda()
        bbox[:, :, :, 0, 0] = loc[:, :, :, 0, 0] + torch.from_numpy(offset_y).cuda()
        bbox[:, :, :, 1, 0] = loc[:, :, :, 1, 0] + torch.from_numpy(offset_y).cuda()
        bbox[:, :, :, 0, 1] = loc[:, :, :, 0, 1] + torch.from_numpy(offset_x).cuda()
        bbox[:, :, :, 1, 1] = loc[:, :, :, 1, 1] + torch.from_numpy(offset_x).cuda()
        fg_loss = torch.square(gt_fg - fg_score).sum()

        cls_loss = torch.square(cls_score-gt_label).sum()
        bb=bbox[(gt_fg>0),:,:]
        gt_bb=gt_bbox[(gt_fg>0),:]
        loc_loss=torch.square(bb[:,0,:]-gt_bb).sum()+torch.square(bb[:,1,:]-gt_bb).sum()
        loss = 5 * loc_loss + fg_loss + cls_loss

        return loss / loc.shape[0]

    def detector(self, img, img_size):
        H, W = img_size
        fg_score, loc, cls_score = self(img)
        cell_H = loc.shape[1]
        cell_W = loc.shape[2]
        x = np.arange(7)
        y = np.arange(7)
        offset_x, offset_y = np.meshgrid(x, y)
        bbox = torch.empty(loc.shape)
        bbox[:, :, :, 0, 0] = loc[:, :, :, 0, 0] + torch.from_numpy(offset_y)
        bbox[:, :, :, 1, 0] = loc[:, :, :, 1, 0] + torch.from_numpy(offset_y)
        bbox[:, :, :, 0, 1] = loc[:, :, :, 0, 1] + torch.from_numpy(offset_x)
        bbox[:, :, :, 1, 1] = loc[:, :, :, 1, 1] + torch.from_numpy(offset_x)
        bbox[:, :, :, :, 2:] = torch.square(loc[:, :, :, :, 2:])
        bbox[:, :, :, :, 0] = bbox[:, :, :, :, 0] / cell_H
        bbox[:, :, :, :, 1] = bbox[:, :, :, :, 1] / cell_W
        bbox[:, :, :, :, 0::2] = bbox[:, :, :, :, 0::2] * H
        bbox[:, :, :, :, 1::2] = bbox[:, :, :, :, 1::2] * W
        idx_li = []
        bboxes = []
        probs = []
        for idx in range(img.shape[0]):
            fg_b1 = fg_score[idx].view(-1)
            fg_b2 = fg_score[idx].view(-1)
            sc1 = cls_score[idx].view(-1, 20)
            sc2 = cls_score[idx].view(-1, 20)
            bb = bbox[idx].view(-1, 4)
            sc = torch.cat([sc1, sc2], dim=0)
            fg = torch.cat([fg_b1, fg_b2], dim=0)
            keep = torch.where(fg > 0.5)[0]
            bb = bb[keep]
            fg = fg[keep]
            sc = sc[keep]
            res_bbox = []
            res_prob = []
            for cls in range(20):
                sc_cls = sc[:, cls]
                kp = nms(bb, sc_cls, 0.7)
                sc_bb = bb[kp]
                prob = sc_cls[kp] * fg[kp]
                res_bbox.append(sc_bb)
                res_prob.append(prob)
            bb = torch.cat(res_bbox, dim=0)
            prob = torch.cat(res_prob, dim=0)
            idx_li.append(torch.ones(prob.shape) * idx)
            bboxes.append(bb)
            probs.append(prob)
        bboxes = torch.cat(bboxes, dim=0)
        probs = torch.cat(probs, dim=0)
        indexes = torch.cat(idx_li, dim=0)
        return bboxes, probs, indexes

#
#
#
#
#
# x = torch.Tensor(32, 3, 448, 448)
# yolov1=YOLO()
# print(yolov1.detector(x, x.shape[2:])[0].shape)
