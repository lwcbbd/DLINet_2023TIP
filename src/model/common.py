import math

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size=3,bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, bias=bias, kernel_size=kernel_size, padding='same'))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MaskSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(MaskSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        x = x*mask
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ActivationBinarizerFunction01(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.sign().clamp(0, 1)
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.input.clamp(-1, 1)
        return (2 - 2 * x * x.sign()) * grad_output

class ActBinarizer_01(nn.Module):
    def forward(self, input):
        return ActivationBinarizerFunction01.apply(input)