import os
from tqdm import tqdm

import numpy as np
import random

# torch imports
import torch
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter

# custom imports
from ddpm import Diffusion
from model import Classifier 
from utils import *


EPOCHS = 20

def create_result_folders(experiment_name):
    os.makedirs(os.path.join("weights", experiment_name), exist_ok=True)


def train(device='cpu', T=500, img_size=16, input_channels=3, channels=32, time_dim=256, epochs=1):

    exp_name = 'classifier'
    create_result_folders(exp_name)
    train_loader, val_loader, _  = prepare_dataloader(keep_label='cg', label='shortcut')

    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    model = Classifier(img_size=img_size, c_in=input_channels, labels=2, 
        time_dim=time_dim,channels=channels, device=device
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params/1_000_000:.2f}M')
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=-1)
    pbar = tqdm(range(1, epochs + 1), desc='Training')



    for epoch in pbar:
        model.train()
        for images, labels in train_loader:
            images = images.to(device).float() 
            labels = labels.to(device).long()

            # Reset gradients
            optimizer.zero_grad()
            
            # Sample noisy image
            t = torch.randint(0, T, (images.size(0),)).to(device)
            x_t, noise = diffusion.q_sample(images, t)
            
            # Calculate loss
            logits = model(x_t, t)                      
            loss = loss_fn(logits, labels)
            
            # Update gradients
            loss.backward()
            optimizer.step()
    
    # save your checkpoint in weights/classifier/model.pth
    torch.save(model.state_dict(), os.path.join("weights", exp_name, 'model.pth'))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    config = yaml.safe_load(open("config.yaml"))
    size_w, size_h = config["size"][0], config["size"][1]
    print(f"Model will run on {device}")
    set_seed()
    train(device=device, 
          img_size=np.array([size_h, size_w]),
          epochs=30)

if __name__ == '__main__':
    main()
    

        