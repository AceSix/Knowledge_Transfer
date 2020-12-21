# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\components\utils.py
###   @Author: Ziang Liu
###   @Date: 2020-12-16 14:31:03
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-16 15:02:53
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################
import os
import math
import numpy as np

import torch
from torchvision import transforms

trans = transforms.Compose([transforms.ToTensor()])

def selfnorm(x):
    norm = torch.norm(x.reshape(x.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)
    noramalized = x / norm
    return noramalized

from .wavelet import WavePool
def reduce_dim(patches, wave_level):
    wavepool = WavePool(256, 2).to(patches.device)
    waves = patches.unsqueeze(0)
    epoch = 0
    while epoch<wave_level:
        waves = torch.cat([torch.stack(wavepool(wave)) for wave in waves], 0)
        epoch += 1
    return waves

def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))