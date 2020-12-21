# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\components\bank.py
###   @Author: Ziang Liu
###   @Date: 2020-12-16 14:31:02
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-21 16:25:13
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import os
from PIL import Image
import numpy as np
import joblib
import glob

import torch
from torchvision import transforms

from kmeans_pytorch import kmeans
from .VGG import GetVGGModel
from .utils import reduce_dim

# %matplotlib inline


class Bank(object):
    def __init__(self, S, style_path, exp_bank_dir, vgg_dir, kernel_size, device):
        self.S_list = S
        self.kernel = kernel_size
        self.layers_max = 2
        self.clusters_list = [10]
        self.thredhold = 0.9
        self.image_size = 256
        self.stride = kernel_size//2

        self.style_path = style_path
        self.exp_bank_dir = exp_bank_dir
        self.wave_level = 1

        self.encoder = GetVGGModel(vgg_dir, 21)
        self.encoder = self.encoder.to(device)
        self.device = device
    
    def create_bank(self):
        for S in self.S_list:
            print('Style: ', S)
            bankpath = os.path.join(self.exp_bank_dir, 'S' + str(S))
            os.makedirs(bankpath, exist_ok=True)

            sf_patch, index_list, image_list, shape_list = self.__calcu_sf_patch__(S)

            bank_size = sf_patch.shape[0]
            bank_info = {'image':image_list, 'shape':shape_list, 'index':index_list, 'size':bank_size}

            for clusters_num in self.clusters_list:
                layers_num = self.layers_max

                version_path =  os.path.join(bankpath, f"C{clusters_num}-K{self.kernel}")
                os.makedirs(version_path,exist_ok=True)

                joblib.dump(bank_info, os.path.join(version_path, f'info.pkl'))
                print('clusters_num: ', clusters_num)
                print('layers_num: ', layers_num)

                init_index = torch.arange(sf_patch.shape[0])
                self.__cluster_sub__(sf_patch, clusters_num, layers_num, version_path, 
                                 1, 1, init_index)
                torch.cuda.empty_cache()
            del sf_patch
    
    def __cluster_sub__(self, patches, clusters_num, layers_num, version_path, cluster, layer, indexes, name=None):
        name = f'{cluster}' if name is None else f'{name}-{cluster}'

        subset_name = f'{name}_subset.pkl'
        sf_wave = reduce_dim(patches, self.wave_level)
        joblib.dump([sf_wave.cpu(), indexes.cpu(), patches.cpu()], os.path.join(version_path, subset_name))

        if layer<layers_num+1:
            ori_shape = patches.shape
            to_fit = patches.reshape(ori_shape[0], -1)
            label, centers = kmeans(to_fit, clusters_num, 'cosine', device=self.device)
            centers = centers.reshape(centers.shape[0], *ori_shape[1:])

            center_name = f'{name}_centers.pkl'
            joblib.dump(centers.cpu(), os.path.join(version_path, center_name))
            for c in range(clusters_num):
                index = np.argwhere(label.cpu().numpy() == c).reshape(-1)
                sub_patches = patches[index]
                self.__cluster_sub__(sub_patches, clusters_num, layers_num, version_path, 
                                 c+1, layer+1, indexes[index], name)
            del centers, label 

    def __calcu_sf_patch__(self, S):
        self.bankpath = os.path.join(self.exp_bank_dir, 'S' + str(S))
        if not os.path.exists(self.bankpath):
            os.mkdir(self.bankpath)

        ## create an folder inside style folder and place resized image in it
        self.style_dir = os.path.join(self.style_path, str(S)) 
        
        trans = transforms.Compose([transforms.Resize(self.image_size),
                                    transforms.ToTensor()])

        patch_list = torch.empty([0, 256, self.kernel * 2, self.kernel* 2]).to(self.device)
        index_list = torch.empty([0, 2]).to(self.device)
        image_list = []
        shape_list = []
        ### split the image's VGG output features into small size patches
        image_pathes = []
        for t in ['*.jpg', '*.png', '*.jpeg']:
            image_pathes += glob.glob(os.path.join(self.style_dir, t))
        for picture_id, path in enumerate(image_pathes):
            image = Image.open(path)

            img_tensor = trans(image).unsqueeze(0).to(self.device)
            img_tensor = img_tensor[:,:,:img_tensor.shape[2]//4*4, :img_tensor.shape[3]//4*4]
            image_list.append(img_tensor)

            img_feature = self.encoder(img_tensor)

            patches =  img_feature.unfold(2, self.kernel*2, self.stride).unfold(3, self.kernel*2, self.stride)
            patches = patches.permute(0, 2, 3, 1, 4, 5)
            _,h,w,_,_,_ = patches.shape
            shape_list.append((h,w))

            patches = patches.reshape( -1, *patches.shape[-3:]) # (subset, patch_numbers, C, kernel, kernel)
            sf_neat, init_index = self.__removeDuplicate__(patches.reshape(patches.shape[0], -1))
            patches = sf_neat.reshape(sf_neat.shape[0], *patches.shape[1:])
            patch_list = torch.cat([patch_list, patches], 0)

            init_index = init_index.unsqueeze(1)
            seq = torch.ones(init_index.shape).to(self.device)*picture_id
            init_index = torch.cat([init_index, seq], -1)
            index_list = torch.cat([index_list, init_index], 0)

        del patches, img_feature, img_tensor, image, init_index
        print('patch_list:', patch_list.shape)
        return patch_list, index_list, image_list, shape_list

    def __removeDuplicate__(self, patches):
        cosine = self.__cosine_matrix__(patches)
        cosine = torch.lt(torch.ones(cosine.shape).to(self.device)*self.thredhold, cosine).float()
        selected = (cosine.tril().sum(-1)<1.5).float()
        choose = torch.diag(selected)[:, selected.nonzero()].squeeze(-1)

        original, choosed = choose.shape
        print(f"Selected {choosed} patchs from {original}.")
        
        out = torch.mm(choose.T, patches)
        index = torch.mm(choose.T, torch.arange(patches.shape[0]).unsqueeze(-1).float().to(self.device)).squeeze()
        return out, index

    def __cosine_matrix__(self, patches):
        # transfer to device
        data = patches.to(self.device)
        A_normalized = data / data.norm(dim=-1, keepdim=True)
        B_normalized = A_normalized.permute(1,0)
        cosine = A_normalized.mm(B_normalized)
        return cosine 


