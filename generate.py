# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\generate.py
###   @Author: Ziang Liu
###   @Date: 2020-12-21 15:19:25
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-21 16:39:50
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import os
import argparse
from PIL import Image
from tqdm import tqdm
import glob

import torch
from torchvision import transforms
from components import Decoder, GetVGGModel
from components import Swap

def load_PIL(image_path, size, ratio = 1.2):
    trans = transforms.Compose([
                transforms.Resize(int(size*ratio)),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
            ])
    image = Image.open(image_path)
    image = trans(image).unsqueeze(0).to(device)
    image = image[:,:,:image.shape[2]//16*16, :image.shape[3]//16*16]
    return image

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    if image.shape[0] == 1:
        image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image

def str2bool(v):
    return v.lower() in ('true')
    
def getParameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda_id', type=int, default=0)

    parser.add_argument('--S', type=str, default='1')
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--stride_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--statistic', type=str2bool, default=True)

    parser.add_argument('--decoder_checkpoint', type=str, default='./logs/decoder.pth')
    parser.add_argument('--vgg_checkpoint', type=str, default='./logs/vgg_normalised.pth')

    parser.add_argument('--input_dir', type=str, default='./content')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--bank_dir', type=str, default='./bank')
    
    return parser.parse_args()

if __name__ == "__main__":
    
    config = getParameters()
    contents = glob.glob(os.path.join(config.input_dir, '*.jpg'))
    style = config.S
    
    device = torch.device(f'cuda:{config.cuda_id}')

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    encoder = GetVGGModel(config.vgg_checkpoint, 21).to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(config.decoder_checkpoint, map_location=device))
    
    def generate(feature, method):
        with torch.no_grad():
            sf, _, _ = method.wavelet_swap_LSH(cf=feature, stat=config.statistic)
            out = decoder(sf)
        return out
    
    swap = Swap(config.bank_dir, device)
    swap.choose_bank(kernel_size=2, clusters_num=10, S=style)
    swap.stride = config.stride_size
    
    for content in tqdm(contents):
        feature = encoder(load_PIL(content, config.image_size, 1.0))
        stylized = generate(feature, swap)

        tensor_to_PIL(stylized).save(os.path.join(config.output_dir, 
                                    f"{style}-{os.path.basename(content).replace('.jpg', '')}-{config.statistic}.png"))
        
    