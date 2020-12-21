# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\components\Decoder.py
###   @Author: Ziang Liu
###   @Date: 2020-12-16 14:31:02
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-16 14:36:35
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg19
import math


class DeCov(nn.Module):
    def __init__(self,in_channels,out_channels,factor=2,kernel_size=3,stride=1,bias=False,activation=nn.SELU()):
        super(DeCov, self).__init__()
        padding_size=(kernel_size-1)//2
        self.dconv=nn.Sequential(
            nn.Upsample(scale_factor=factor), 
            nn.ReflectionPad2d((padding_size)),
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,bias=bias),
            # nn.InstanceNorm2d(out_channels),
            activation
        )
        for layer in self.dconv:
            if isinstance(layer,nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if bias:
                    nn.init.zeros_(layer.bias)
            
    def forward(self,x):
        y=self.dconv(x)
        return y

class RC(nn.Module):
    #A wrapper of ReflectionPad2d and Conv2d
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1,stride=1, bias=True, activated=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,stride,bias=bias)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.tanh(h)
        else:
            return h.clamp(0,1)


class GenResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, activation=F.selu):
        super(GenResBlock, self).__init__()

        self.activation = activation
        if h_ch is None:
            h_ch = out_ch
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        # self.b1 = nn.InstanceNorm2d(in_ch,affine=True)
        # self.b2 = nn.InstanceNorm2d(h_ch,affine=True)

    def forward(self, x):
        return x + self.residual(x)

    def residual(self, x):
        h = self.c1(x)
        # h = self.b1(h)
        h = self.activation(h)
        h = self.c2(h)
        # h = self.b2(h)
        h = self.activation(h)
        return h
    

act_dict = {
    'relu':[F.relu, nn.ReLU()],
    'selu':[F.selu, nn.SELU()]
}
class Decoder(nn.Module):
    def __init__(self, bias=True, activated=False, activation='selu'):
        super().__init__()
        
        self.rb1 = GenResBlock(256, 256, activation=act_dict[activation][0])
        self.rb2 = GenResBlock(256, 256, activation=act_dict[activation][0])
        self.rb3 = GenResBlock(256, 256, activation=act_dict[activation][0])
        self.rb4 = GenResBlock(256, 256, activation=act_dict[activation][0])
        self.rb5 = GenResBlock(256, 256, activation=act_dict[activation][0])
        self.dc1 = DeCov(256, 128, 2, activation=act_dict[activation][1])
        self.dc2 = DeCov(128, 64, 2, activation=act_dict[activation][1])
        self.dc3 = DeCov(64, 3, 1, bias=False, activation=act_dict[activation][1])
        self.rc = RC(3,3,bias=bias, activated=activated)

    def forward(self, features):
        h = self.rb1(features)
        h = self.rb2(h)
        h = self.rb3(h)
        h = self.rb4(h)
        h = self.rb5(h)
        h = self.dc1(h)
        h = self.dc2(h)
        h = self.dc3(h)
        h = self.rc(h)
        return h
