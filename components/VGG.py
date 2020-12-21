# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\components\VGG.py
###   @Author: Ziang Liu
###   @Date: 2020-12-16 14:31:03
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-16 14:49:22
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import torch
import torch.nn as nn

def GetVGGModel(pretrained_ckpt, output_layer=31):
    """
        pretrained_ckpt:    the file path to pretrained vgg model
        outputlayer:        selected out put layer number
    """
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),            # layer 1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 2
        nn.Conv2d(3, 64, (3, 3)),           # layer 3
        nn.ReLU(),                          # layer 4 relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 5
        nn.Conv2d(64, 64, (3, 3)),          # layer 6
        nn.ReLU(),                          # layer 7 relu1-2
        nn.MaxPool2d((2, 2), (2, 2), 
                (0, 0), ceil_mode=True),    # layer 8
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 9
        nn.Conv2d(64, 128, (3, 3)),         # layer 10
        nn.ReLU(),                          # layer 11 relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 12
        nn.Conv2d(128, 128, (3, 3)),        # layer 13
        nn.ReLU(),                          # layer 14 relu2-2
        nn.MaxPool2d((2, 2), (2, 2),
                (0, 0), ceil_mode=True),    # layer 15
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 16
        nn.Conv2d(128, 256, (3, 3)),        # layer 17
        nn.ReLU(),                          # layer 18 relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 19
        nn.Conv2d(256, 256, (3, 3)),        # layer 20
        nn.ReLU(),                          # layer 21 relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 22
        nn.Conv2d(256, 256, (3, 3)),        # layer 23
        nn.ReLU(),                          # layer 24 relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 25
        nn.Conv2d(256, 256, (3, 3)),        # layer 26
        nn.ReLU(),                          # layer 27 relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0),# layer 28
                ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 29
        nn.Conv2d(256, 512, (3, 3)),        # layer 30
        nn.ReLU(),                          # layer 31 relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),   # layer 32
        nn.Conv2d(512, 512, (3, 3)),        # layer 33
        nn.ReLU(),                          # layer 34 relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    vgg.load_state_dict(torch.load(pretrained_ckpt))
    vgg = nn.Sequential(*list(vgg.children())[:output_layer])
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg