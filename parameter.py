# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\parameter.py
###   @Author: Ziang Liu
###   @Date: 2020-12-16 14:38:14
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-16 15:29:59
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import  argparse


def str2bool(v):
    return v.lower() in ('true')

def getParameters():
    parser = argparse.ArgumentParser()

    # Basic information
    parser.add_argument('--content', type=str, default='AE', choices=['AE', 'bank'])
    parser.add_argument('--cuda_id', type=int, default=0)

    # Bank setting
    parser.add_argument('--S', type=list, default=['patchsize13'], nargs='+')
    parser.add_argument('--kernel_size', type=int, default=2)

    # AE training setting
    parser.add_argument('--iter_size', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    
    # Path 
    parser.add_argument('--save_dir', type=str, default='./logs')
    parser.add_argument('--style_dir', type=str, default='./style')
    parser.add_argument('--exp_bank_dir', type=str, default='./bank')
    parser.add_argument('--train_data_dir', type=str, default='../train/Place365')
    parser.add_argument('--style_data_dir', type=str, default='../train/style')
    parser.add_argument('--vgg_checkpoint', type=str, default='./logs/vgg_normalised.pth')

    # Test setting
    parser.add_argument('--statistic', type=str2bool, default=True)
    parser.add_argument('--exp_content_dir', type=str, default='../experiments/test_content')
    parser.add_argument('--decoder_checkpoint', type=str, default='./logs/340000_iter.pth')


    return parser.parse_args()

