from ddpm import Diffusion
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm
from torch import optim
import logging
import argparse

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

from ddpm import Diffusion
from model import UNet
import yaml
from util import set_seed, prepare_dataloaders, CLASS_LABELS
set_seed()





def create_result_folders(experiment_name):
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("weights", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)


def test(img_size, T=500, cfg=True, input_channels=3, channels=32, 
          time_dim=256, batch_size=1, lr=1e-3, num_epochs=30, 
          experiment_name="DDPM", show=False, device='cpu'):

    create_result_folders(experiment_name)

    num_classes = 2

    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, 
                 num_classes=num_classes, time_dim=time_dim,channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, diff_type=diff_type, device=device)

    # sample
    sample = torch.randn((batch_size, input_channels, img_size, img_size)).to(device)
