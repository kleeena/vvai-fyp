import tensorflow
import os, time
import sys
import numpy as np
from PIL import Image
from IPython.display import display
import torch
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from IPython.display import display
from einops import rearrange
from  . import clip, stylegan
import necessary_functions as nf

text_prompt_weight = 1 #@param {type: "number"}

# Fix camera so it's not moving around
fix_camera = True 
# Speed at which to try approximating the text. Too fast seems to give strange results. Maximum is 100.
speed = 20  

steps = 20 
# Change the seed to generate variations of the same prompt
seed = 21 
# We haven't completely understood which parameters influence the generation of this model. Changing the learning rate could help (between 0 and 100)
learning_rate = 10 
social = False
smoothing = (100.0-speed)/100.0

device = torch.device('cuda')
print('Using device:', device, file=sys.stderr)
torch.manual_seed(seed)
clip_model_path = './Trained_Models/TrainedCLIP.pth'
clip_model = clip.CLIP(clip_model_path)



def embed_image(image):
  n = image.shape[0]
  cutouts = nf.make_cutouts(image)
  embeds = clip_model.embed_cutout(cutouts)
  embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
  return embeds

# Example usage:
model_path = './Trained_Models/TrainedStyleGAN3.pkl'
stylegan_model = stylegan.StyleGAN(model_path)

# Optionally fix camera (if needed)
stylegan_model.fix_camera()

# Global variable definition
global G, w_stds
G = stylegan_model.G
device = stylegan_model.device

# Calculate w_stds globally
zs = torch.randn([10000, G.mapping.z_dim], device=device)
w_stds = G.mapping(zs, None).std(0)



promptoo="A green v neck shirt"
target = clip_model.embed_text(promptoo)
prompts = [(text_prompt_weight, target)]

# Actually do the run
output_path = '/output'
tf = Compose([
  Resize(224),
  lambda x: torch.clamp((x+1)/2,min=0,max=1),
  ])

def run():
  torch.manual_seed(seed)
  timestring = time.strftime('%Y%m%d%H%M%S')

  with torch.no_grad():
    qs = []
    losses = []
    for _ in range(8):
      q = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
      images = G.synthesis(q * w_stds + G.mapping.w_avg)
      embeds = embed_image(images.add(1).div(2))
      loss = 0
      for (w, t) in prompts:
        loss += w * nf.spherical_dist_loss(embeds, t).mean(0)
      i = torch.argmin(loss)
      qs.append(q[i])
      losses.append(loss[i])
    qs = torch.stack(qs)
    losses = torch.stack(losses)
    print(losses)
    print(losses.shape, qs.shape)
    i = torch.argmin(losses)
    q = qs[i].unsqueeze(0).requires_grad_()

  #Sampling loop
  q_ema = q
  opt = torch.optim.AdamW([q], lr=learning_rate/250.0, betas=(0.0,0.999))
  loop = tqdm(range(steps))
  for i in loop:
    opt.zero_grad()
    w = q * w_stds
    image = G.synthesis(w + G.mapping.w_avg, noise_mode='const')
    embed = embed_image(image.add(1).div(2))
    loss = 0
    for (w, t) in prompts:
      loss += w * nf.spherical_dist_loss(embed, t).mean(0)
    print(loss)
    loss.backward()
    opt.step()
    loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())

    q_ema = q_ema * smoothing + q * (1-smoothing)
    latent = q_ema * w_stds + G.mapping.w_avg
    image = G.synthesis(latent, noise_mode='const')

    #if i % 10 == 0:
    pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1))
    os.makedirs(output_path, exist_ok=True)
    os.makedirs("/tmp/ffmpeg", exist_ok=True)
    if i % 5 == 0:
      pil_image.save(f'{output_path}/output_{i:04}.jpg')
      display(pil_image)
    pil_image.save(f'/tmp/ffmpeg/output_{i:04}.jpg')
  return latent

try:
  latent = run()
  torch.save(latent, f"{output_path}/latent.pt")
except KeyboardInterrupt:
  pass