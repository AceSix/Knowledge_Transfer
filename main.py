# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\main.py
###   @Author: Ziang Liu
###   @Date: 2020-12-16 14:31:05
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-21 16:47:50
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import torch
import os

from parameter import getParameters
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True
    device = torch.device('cuda')

    if config.content == 'bank':
        if not os.path.exists(config.exp_bank_dir):
            os.mkdir(config.exp_bank_dir)

        from components.bank import Bank

        bank = Bank(config.S, config.style_dir, config.exp_bank_dir, config.vgg_checkpoint,
                    config.style_size, config.kernel_size, config.stride_size, config.layers_max, 
                    config.clusters, device)
        
        bank.create_bank()
        print("Style bank creation finished.")

    else:
        package  = __import__('scripts.trainer_'+config.content, fromlist=True)
        Trainer  = getattr(package, 'Trainer')
        
        for i,item in enumerate(config.__dict__.items()):
            print("[%d]-[parameters] %s--%s"%(i,item[0],str(item[1])))
        if not os.path.exists(config.save_dir):
            os.mkdir(config.save_dir)

        trainer = Trainer(config)
        trainer.train()

if __name__ == '__main__':
    config = getParameters()
    main(config)
