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

config = yaml.safe_load(open("config.yaml"))

def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()


def create_result_folders(experiment_name):
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("weights", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)


def train(img_size, T=500, cfg=True, input_channels=3, channels=32, 
          time_dim=256, batch_size=1, lr=1e-3, num_epochs=30, 
          experiment_name="DDPM", show=False, device='cpu'):

    create_result_folders(experiment_name)
    train_loader,_,_ = prepare_dataloaders(batch_size)
    
    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, time_dim=time_dim,channels=channels, device=device).to(device)
    
    diff_type = 'DDPM-cFg' if cfg else 'DDPM'
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, diff_type=diff_type, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    
    logger = SummaryWriter(os.path.join("runs", experiment_name))
    l = len(train_loader)

    min_train_loss = 1e10
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)
        epoch_loss = 0
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)

            if diff_type == 'DDPM-cFg':
                # one-hot encode labels for classifier-free guidance
                labels = labels.to(device)
                labels = F.one_hot(labels, num_classes=num_classes).float()
            else :
                labels = None

            # Train a diffusion model with classifier-free guidance
            # Do not forget randomly discard labels
            p_uncod = 0.1

            if np.random.rand() < p_uncod:
                labels = None

            t = torch.randint(0, time_dim, (images.shape[0],), device=device)

            x_t, noise = diffusion.q_sample(images, t)

            predicted_noise = model(x_t, t)
            
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        epoch_loss /= l
        if epoch_loss <= min_train_loss:
            torch.save(model.state_dict(), os.path.join("weights", experiment_name, f"model.pth"))
            min_train_loss = epoch_loss

            
        if diffusion.diff_type == 'DDPM-cFg':
            y = torch.tensor([np.random.randint(0,5)], device=device)
            title = f'Epoch {epoch} with label:{CLASS_LABELS[y.item()]}'
            y = F.one_hot(y, num_classes=num_classes).float()
        else:
            y = None
            title = f'Epoch {epoch}'

        sampled_images = diffusion.p_sample_loop(model, batch_size=images.shape[0], y=y)
        save_images(images=sampled_images, path=os.path.join("results", experiment_name, f"{epoch}.jpg"),
                    show=show, title=title)
        


def main():    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}\n")
    
    set_seed()
    size_w, size_h = config["size"][0], config["size"][1]
    batch_size = config['batch_size']
    train(batch_size=batch_size , 
          device=device, 
          num_epochs=2,
          img_size=np.array([size_w, size_h]))

if __name__ == '__main__':
    main()
    

        