import torch
import clip
import torchvision.transforms as transforms
import tensorflow
import io
import os, time
import pickle
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from IPython.display import display
from einops import rearrange
import necessary_functions as nf

class CLIP(object):
    def __init__(self, clip_model_path, device='cuda'):
        clip_model_name = "ViT-B/32"
        self.device = device
        # Load the CLIP model architecture
        self.model, _ = clip.load(clip_model_name, device=device)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        # Load the trained model weights
        self.load_trained_model(clip_model_path)

    def load_trained_model(self, clip_model_path):
        try:
            state_dict = torch.load(clip_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Trained CLIP model loaded successfully.")
        except Exception as e:
            print(f"Error loading trained CLIP model: {e}")
            raise e

    @torch.no_grad()
    def embed_text(self, prompt):
        "Normalized clip text embedding."
        return nf.norm1(self.model.encode_text(clip.tokenize(prompt).to(self.device)).float())

    def embed_cutout(self, image):
        "Normalized clip image embedding."
        return nf.norm1(self.model.encode_image(self.normalize(image.to(self.device))))

