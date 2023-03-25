import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
import time
import argparse
from pathlib import Path
from util import *
from abc import ABC
from tqdm import tqdm

"""
model
    content_layer -> config.layer_nums | None
    style_layer -> confg.layer_nums | None
"""
class VGG(nn.Module):
    def __init__(self, device):
        super(VGG, self).__init__()
        self.req_features = [0,5,10,19,28]
        self.model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:30]
        self.device = device
        
    def forward(self, x, config):
        content_features = []
        style_features = []
        val = 0 if config.use_type == "conv" else 1 
        x = x.to(self.device)
        if config.content_layers is None and config.style_layers is None:
            raise ValueError("content_layers and style_layers both cannot be None")
            return 
        if config.reconstruct and config.visualize != "none":
            raise ValueError("Either you can reconstruct or visualize, not both")
            return
        if config.content_layers is None and config.style_layers is not None and config.visualize == "content":
            raise ValueError("content_layers cannot be none when visualize==content")
            return
        if config.style_layers is None and config.content_layers is not None and config.visualize == "style":
            raise ValueError("style_layers cannot be none when visualize==style")
            return
        if (config.content_layers is None or config.style_layers is None) and config.visualize == "both":
            raise ValueError("either content layer or style layer cannot be None when visulize==both")
            return
        if (config.content_layers is None or len(config.content_layers)>1) and config.reconstruct:
            raise ValueError("only one layer shall be provided in content_layers arg for reconstruction")
            return
        if (config.style_layers is None or len(config.style_layers)<5) and config.reconstruct:
            raise ValueError("need all layers values in style_layers arg for reconstruction")
            return 
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            n = layer_num-val
            if n in self.req_features and config.content_layers is not None and n in [self.req_features[i] for i in config.content_layers]:
                # print (f"content layer appended layer_num: {n}")
                content_features.append(x)
            if n in self.req_features and config.style_layers is not None and n in [self.req_features[i] for i in config.style_layers]:
                # print (f"style layer appended layer_num: {n}")
                style_features.append(x)
        return content_features, style_features

class GeneratedImage:
    def __init__(self, config, content_image, style_image, device):
        self.config = config
        self.generated_image = 0
        self.device = device
        self.content_image = content_image
        self.style_image = style_image
        
    def Get(self):
        if self.config.init_image == "noise":
            self.generated_image = torch.randn(self.content_image.size())
            self.generated_image.to(self.device, torch.float)
            self.generated_image.requires_grad = True
        elif self.config.init_image == "content":
            self.generated_image = self.content_image.clone().requires_grad_(True)
        elif self.config.init_image == "style":
            self.generated_image = self.style_image.clone().requires_grad_(True)
        return self.generated_image
            
class Command(ABC):
    def ContentLoss(self, generated_feature, content_feature):
        content_loss = torch.mean((generated_feature-content_feature.detach())**2)
        return content_loss
    
    def StyleLoss(self, generated_feature, style_feature):
        _, channel, height, width = generated_feature.shape
        G = torch.mm(generated_feature.view(channel, height*width), 
                     generated_feature.view(channel, height*width).t())
        A = torch.mm(style_feature.view(channel, height*width),
                     style_feature.view(channel, height*width).t())
        norm = torch.tensor(4*channel*(height*width)**2)
        style_l = torch.div(torch.mean((G-A.detach())**2), norm)
        return style_l
    
    def TotalVariationLoss(self, Y_hat):
        return torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() + \
            torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean()
    
    def Optimizer(self, config, generated_image):
        if config.optimizer == "Adam":
            return optim.Adam([generated_image], lr=config.lr)
        elif config.optimizer == "LBFGS":
            return optim.LBFGS([generated_image])
    
    def SaveFeature(self, iter, config, total_loss, gen_img):
        # print(f"{iter}th epoch: loss = ", round(total_loss.item(),2))
        if not iter%config.sav_freq:
            save_image(gen_img[0], os.path.join(os.path.dirname(__file__), f"data/transfer/{int(iter/config.sav_freq)}.jpg"))

class NeuralStyleTransfer(Command):
    def __init__(self, config, device):
        self.config = config
        self.model = None
        self.content_image = image_loader(self.config.content_image, self.config, device)
        self.style_image = image_loader(self.config.style_image, self.config, device)
        self.gen_image = GeneratedImage(self.config, self.content_image, self.style_image, device).Get()
        self.optimizer = self.Optimizer(self.config, self.gen_image)
        self.device = device
        
    def Loss(self, style_gen_features, style_style_features, content_content_features, 
                content_gen_features, gen_image):
        i = style_loss = content_loss = tv_loss = 0
        
        for cont, gen in zip(content_content_features, content_gen_features):
            content_loss += self.ContentLoss(gen, cont)
        for gen, style in zip(style_gen_features, style_style_features):
            style_loss += self.StyleLoss(gen, style)*self.config.style_weights[i]
            i += 1
        tv_loss = self.TotalVariationLoss(gen_image)
        i = 0
        total_loss = self.config.alpha*content_loss + self.config.beta*style_loss + self.config.gamma*tv_loss
        return total_loss
    
    def RunAdam(self):
        self.model = VGG(self.device).eval()
        self.model.to(self.device)
        clearDir()
        start = perf_counter()
        with tqdm(total=self.config.iterations) as pbar:
            for iter in range(self.config.iterations):
                self.optimizer.zero_grad()
                content_gen_features, style_gen_features = self.model(self.gen_image, self.config)
                content_content_features, _ = self.model(self.content_image, self.config)
                _, style_style_features = self.model(self.style_image, self.config)
                total_loss = self.Loss(
                    style_gen_features, style_style_features, content_content_features, 
                    content_gen_features,self.gen_image
                )
                total_loss.backward()
                self.optimizer.step()
                self.SaveFeature(iter, self.config, total_loss, self.gen_image)
                pbar.update(1)
        end = perf_counter()
        print(f"took {round(end-start)} s")
        time.sleep(2)
        create_video_from_intermediate_results(self.config, os.path.join(os.path.dirname(__file__), "data/transfer"))
    
    def RunLBFGS(self):
        self.model = VGG(self.device).eval()
        self.model.to(self.device)
        clearDir()
        start = perf_counter()
        run = [0]
        pbar = tqdm(total=int(self.config.iterations/20))
        while run[0]<=self.config.iterations:
            def closure():
                self.optimizer.zero_grad()
                content_gen_features, style_gen_features = self.model(self.gen_image, self.config)
                content_content_features, _ = self.model(self.content_image, self.config)
                _, style_style_features = self.model(self.style_image, self.config)
                total_loss = self.Loss(
                    style_gen_features, style_style_features, content_content_features, 
                    content_gen_features,self.gen_image
                )
                total_loss.backward()
                self.SaveFeature(run[0], self.config, total_loss, self.gen_image)
                run[0] += 1
                return total_loss

            self.optimizer.step(closure)
            pbar.update(1)
        pbar.close()
        end = perf_counter()
        print(f"took {round(end-start)} s")
        time.sleep(2)
        create_video_from_intermediate_results(self.config, os.path.join(os.path.dirname(__file__), "data/transfer"))
            
    def Execute(self):
        if self.config.optimizer == "Adam":
            self.RunAdam()
        elif self.config.optimizer == "LBFGS":
            self.RunLBFGS()
