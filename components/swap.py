# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\components\swap.py
###   @Author: Ziang Liu
###   @Date: 2020-12-16 14:31:03
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-21 16:40:39
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import os
import numpy as np
import joblib
import glob
import time

import torch
import torch.nn.functional as F
import torch.nn as nn

from .utils import reduce_dim
from components.utils import reduce_dim


def calc_similarity(x, y, norm=True):
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    y = y.reshape(y.shape[0], -1, y.shape[-1])
    if norm:
        return torch.cosine_similarity(x, y).unsqueeze(0)
    else:
        return F.conv1d(x, y, stride=1)
    
def batch_similarity(x, y, norm=True):
    x = x.reshape(*x.shape[:2], -1, x.shape[-1])
    y = y.reshape(*y.shape[:2], -1, y.shape[-1])
    if norm:
        return torch.cosine_similarity(x, y, dim=-2).sum(dim=0, keepdim=True)
    else:
        return F.conv1d(x, y, stride=1)

class Swap(object):
    def __init__(self, exp_bank_dir, device):

        self.Base = 'base'
        self.bank_dir = os.path.join(exp_bank_dir, 'S' + str(self.Base))
        self.exp_bank_dir = exp_bank_dir

        self.wave_level = 1
        self.weight = [1,1,1,0]

        self.device = device
        
    def choose_bank(self, kernel_size = 2, stride_size = 1, clusters_num = 50, S = '1'):
        print(f"[kernel_size]-[{kernel_size}]-[cluster]-[{clusters_num}]-[S]-[{S}]")
        self.stylepath = os.path.join(self.exp_bank_dir, 'S' + str(S))
        self.bankpath = os.path.join(self.stylepath, f"C{clusters_num}-K{kernel_size}")

        self.kernel = kernel_size
        self.stride = stride_size
        self.S = S
        self.clusters_num = clusters_num

        self.dict = {}
        files = glob.glob(self.bankpath+"/*-*-*subset.pkl") + glob.glob(self.bankpath+"/*centers.pkl")
        for file in files:
            self.dict[os.path.basename(file)] = joblib.load(file)
            if 'subset' in file:
                self.dict[os.path.basename(file)][0] = reduce_dim(self.dict[os.path.basename(file)][2], self.wave_level).cpu()
#                 print(self.dict[os.path.basename(file)][2].shape)
        self.StyleInfoPath = os.path.join(self.bankpath, f'info.pkl')


    def wavelet_swap_LSH(self, cf, layer_num=2, stat=True, c_norm=True):
        clusters_num = self.clusters_num
        self.c_norm = c_norm

        t0 = time.time()
        cf_query, cf_wave, original_shape = self.__query_prepare__(cf)

        filename = '1_centers.pkl'
        init_centers = self.dict[filename].to(self.device)
        cs_total, choice_total = self.__query__(cf_query, cf_wave, 1, layer_num, init_centers, '1')

        if stat:
            mu_cf = cf_query.mean(2, keepdim=True)
            std_cf = cf_query.std(2, keepdim=True)+2e-5
            mu_out = cs_total.mean(2, keepdim=True)
            std_out = cs_total.std(2, keepdim=True)
            mid = (cf_query - mu_cf)/std_cf
            cs_total = mid*std_out+mu_out
        
        fn_fold = nn.Fold(original_shape[-2:], kernel_size=self.kernel*2, stride=self.stride*2)
        fn_unfold = nn.Unfold(kernel_size=self.kernel*2, stride=self.stride*2)

        cs_total = cs_total.reshape(cs_total.shape[0], -1, cs_total.shape[-1])
        cs_total = fn_fold(cs_total)
        input_ones = torch.ones(original_shape).to(self.device)
        divisor = fn_fold(fn_unfold(input_ones))
        cs_total = cs_total/divisor
        
        time_cost = time.time()-t0
        return cs_total, time_cost, choice_total

    
    def __query__(self, cf, cf_wave, layer, layer_num, sf_center, path):
        sf_center = sf_center.reshape(sf_center.shape[:2]+(-1,1))
        conv_center = calc_similarity(cf, sf_center, self.c_norm)
                
        bucket_index = torch.zeros_like(conv_center).to(self.device)
        bucket_index.scatter_(1, conv_center.argmax(dim=1, keepdim=True), 1)    
        del conv_center
        index_mask = bucket_index.view(bucket_index.shape[:2]+(-1,))

        cs_total = torch.zeros_like(cf).to(self.device)
        choice_total = torch.zeros([1,1,cs_total.shape[-1]]).to(self.device)
        for c in range(self.clusters_num):
            diag_mid =  torch.diag(index_mask[0][c]).to(self.device)
            mask_matrix = diag_mid[:, index_mask[0][c].nonzero().squeeze(-1)]
            if mask_matrix.shape[-1]==0:
                continue
            cf_wave_tmp = torch.matmul(cf_wave, mask_matrix)
            cf_tmp = torch.matmul(cf, mask_matrix)
            if layer<layer_num:
                filename = f'{path}-{c+1}_centers.pkl'
                tmp_centers = self.dict[filename]
                cs, choice = self.__query__(cf_tmp, cf_wave_tmp, layer+1, layer_num, 
                                            tmp_centers.to(self.device), f'{path}-{c+1}')
            else:
                filename = f'{path}-{c+1}_subset.pkl'
                [sf_patches, index, patches] = self.dict[filename]
                cs, one_hots = self.__feature_swap__(cf_wave_tmp, sf_patches.to(self.device), 
                                                                patches.to(self.device))

                one_hots = one_hots.squeeze(2).squeeze(0)
                index_ = torch.tensor(index.clone().detach()).unsqueeze(0).float().to(self.device)
                choice = torch.matmul(index_, one_hots)
            
            try:
                cs_total += torch.matmul(cs, mask_matrix.T)
                choice_total += torch.matmul(choice, mask_matrix.T)
            except:
                cs_total += torch.matmul(cs, mask_matrix.t())
                choice_total += torch.matmul(choice, mask_matrix.t())
        return cs_total, choice_total

    def __feature_swap__(self, cf_wave, sf_wave, sf_patches):
        sf_wave = sf_wave.reshape(*sf_wave.shape[:3],-1,1)
        
        wave_mask = np.argwhere(np.array(self.weight)!=0).reshape(-1)
        conv_out = calc_similarity(cf_wave.unsqueeze(1)[wave_mask].transpose(1,0), sf_wave[wave_mask].transpose(1,0), self.c_norm).unsqueeze(2)

        one_hots = torch.zeros_like(conv_out).to(self.device)
        one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)
        del conv_out

        sf_patches = sf_patches.reshape(sf_patches.shape[:2]+(-1,1))
        clus_out = F.conv_transpose1d(one_hots, sf_patches, stride=1)
        return clus_out, one_hots
    
    
    def __query_prepare__(self, cf):
        fn_unfold = nn.Unfold(kernel_size=self.kernel//self.wave_level, stride=self.stride//self.wave_level)
        wave = reduce_dim(cf, self.wave_level)
        wave = fn_unfold(wave.squeeze())
        wave = wave.reshape(wave.shape[0], -1, (self.kernel//self.wave_level)**2, wave.shape[-1])

        original_shape = cf.shape
        fn_unfold = nn.Unfold(kernel_size=self.kernel*2, stride=self.stride*2)
        query = fn_unfold(cf)
        query_ = query.reshape(query.shape[0], -1, self.kernel*2, self.kernel*2, query.shape[-1])
        query = query.reshape(query.shape[0], -1, self.kernel**2*4, query.shape[-1])
        return query, wave, original_shape