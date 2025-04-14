import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from torch import optim
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

from ddpm import Diffusion
from model import UNet

##############
from torchvision import transforms
from itertools import compress
import numpy as np
from utils import *

def train(img_size, device='cpu', T=500, input_channels=3, channels=32, time_dim=256,
          batch_size=100, lr=1e-3, num_epochs=30, experiment_name="ddpm", show=False):
    """Implements algrorithm 1 (Training) from the ddpm paper at page 4"""
    create_result_folders(experiment_name)
    train_loader, val_loader, test_loader = prepare_dataloader(batch_size, label='shortcut', keep_label=0)

    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, 
                 time_dim=time_dim,channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss() # use MSE loss 
    
    logger = SummaryWriter(os.path.join("runs", experiment_name))
    l = len(train_loader)

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)
        
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)

            # TASK 4: implement the training loop
            t = diffusion.sample_timesteps(images.shape[0]).to(device) # line 3 from the Training algorithm
            x_t, noise = diffusion.q_sample(images, t) # inject noise to the images (forward process), HINT: use q_sample
            predicted_noise = model(x_t, t) # predict noise of x_t using the UNet
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 5 == 0:
            sampled_images = diffusion.p_sample_loop(model, batch_size=batch_size)
            save_images(images=sampled_images, path=os.path.join("results", experiment_name, f"{epoch}.jpg"),
                        show=show, title=f'Epoch {epoch}')
        torch.save(model.state_dict(), os.path.join("models", experiment_name, f"weights-{epoch}.pt"))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=SEED)
    
    size_w, size_h = config["size"][0], config["size"][1]
    batch_size = config['batch_size']
    train(batch_size=batch_size , 
          device=device, 
          num_epochs=30,
          img_size=np.array([size_w, size_h]))

if __name__ == '__main__':
    main()
    

        