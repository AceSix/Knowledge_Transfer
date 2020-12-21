# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \Knowledge_Transfer\scripts\trainer_AE.py
###   @Author: Ziang Liu
###   @Date: 2020-12-16 14:31:05
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2020-12-16 15:08:05
###   @Copyright (C) 2020 SJTU. All rights reserved.
###################################################################

import os
import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image

from components import GetVGGModel, Decoder
from components.utils import PSNR

class Trainer(object):
    def __init__(self, config):

        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(config.train_data_dir, transforms.Compose([
                transforms.RandomSizedCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

        self.style_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(config.style_data_dir, transforms.Compose([
                transforms.RandomSizedCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

        self.version_dir = f'{config.save_dir}/{config.version}'
        self.model_state_dir = f'{self.version_dir}/model_state'
        self.image_dir = f'{self.version_dir}/image'

        if not os.path.exists(self.version_dir):
            os.mkdir(self.version_dir)
            os.mkdir(self.model_state_dir)
            os.mkdir(self.image_dir)

        self.encoder = GetVGGModel(config.vgg_checkpoint, 21).cuda()
        self.decoder = Decoder().cuda()

        self.config = config

    def train(self):
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.SmoothL1Loss()

        styles = iter(self.style_loader)
        contents = iter(self.train_loader)
        for i in range(0, self.config.iter_size+1):
            try:
                style, _ = next(styles)
            except:
                styles = iter(self.style_loader)
                style, _ = next(styles)
            try:
                content, _ = next(contents)
            except:
                contents = iter(self.train_loader)
                content, _ = next(contents)

            content, style = content.cuda(), style.cuda()
            sample = torch.cat([content, style], 0)

            feature = self.encoder(sample)
            output = self.decoder(feature)

            loss = criterion(sample, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%self.config.log_interval == 0:
                with torch.no_grad():
                    psnr = PSNR(sample.cpu().numpy(),output.cpu().numpy())
                if i%(self.config.log_interval*10) == 0:
                    torch.save(self.decoder.state_dict(), f'{self.model_state_dir}/{i}_iter.pth')
                    print(f"[iter]-[{i}]-[psnr]-[{round(psnr,2)}]-[checkpoint]")
                    save_image(torch.cat([sample, output], -1), os.path.join(self.image_dir, f"iter-{i}.jpg"))
                else:
                    print(f"[iter]-[{i}]-[psnr]-[{round(psnr,2)}]")
