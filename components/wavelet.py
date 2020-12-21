# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\components\wavelet.py
###   @Author: Ziang Liu
###   @Date: 2020-12-16 14:31:03
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-16 14:36:58
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import torch
import torch.nn as nn
import numpy as np

def getWavelet(in_channels, stride=2, pool=True):
    """wavelet decomposition using conv2d"""

    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=stride, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=stride, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=stride, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=stride, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePoolCover(nn.Module):
    def __init__(self, in_channels, stride=2):
        super(WavePoolCover, self).__init__()
        self.LL, self.LH, self.HL, self.HH = getWavelet(in_channels, stride)

    def forward(self, x):
        out = [self.LL(x), self.LH(x), self.HL(x), self.HH(x)]
        out1 = torch.cat([out[0], out[1], out[2]], 1)
        out2 = torch.cat([out[0], out[2], out[3]], 1)
        out3 = torch.cat([out[0], out[1], out[3]], 1)
        out4 = torch.cat([out[1], out[2], out[3]], 1)
        return out1, out2, out3, out4

class WavePoolDense(nn.Module):
    def __init__(self, in_channels, stride=2):
        super(WavePoolDense, self).__init__()
        self.LL, self.LH, self.HL, self.HH = getWavelet(in_channels, stride)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x)

class WavePool(nn.Module):
    def __init__(self, in_channels, stride=2):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = getWavelet(in_channels, stride)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, stride=2, option_unpool='sum'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = getWavelet(self.in_channels, stride, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError
            

