import pickle
import tensorflow
import io
import os, time
import shutil
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import clip
from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from IPython.display import display
from einops import rearrange
import CLIP
import necessary_functions as nf

class StyleGAN(object):
    def __init__(self, model_path, device='cuda'):
        # Load StyleGAN model
        with open(model_path, 'rb') as fp:
            self.G = pickle.load(fp)['G_ema'].to(device)

        self.device = device
        self.w_stds = None  # Define w_stds as an attribute

    def fix_camera(self):
        # Fix the coordinate grid to w_avg
        shift = self.G.synthesis.input.affine(self.G.mapping.w_avg.unsqueeze(0))
        self.G.synthesis.input.affine.bias.data.add_(shift.squeeze(0))
        self.G.synthesis.input.affine.weight.data.zero_()

    def generate_images(self, num_images):
        # Generate images
        zs = torch.randn([num_images, self.G.mapping.z_dim], device=self.device)
        ws = self.G.mapping(zs, None)  # Mapping network
        return self.G.synthesis(ws)  # Synthesis network


