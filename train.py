import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import time
datestr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

from PIL import Image
import random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from torch import optim
import logging
import wandb

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

from ddpm import Diffusion
from ddim import DiffusionImplicit
from model import UNet

##############
from torchvision import transforms
from itertools import compress
import numpy as np
from utils import *



def train(img_size, device='cpu', T=500, input_channels=3, channels=32, time_dim=256,
          batch_size=100, lr=1e-3, num_epochs=30, experiment_name="DDPM-cg", show=False, wandb_run=None):
    """Implements algrorithm 1 (Training) from the ddpm paper at page 4"""
    create_result_folders(experiment_name)

    transform = transforms.Compose([
                                transforms.Resize((img_size[0], img_size[1])), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                # transforms.RandomRotation(degrees=180)
                                # transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                # transforms.RandomAffine(degrees=180)#, scale=(0.8,1.2))
                 ])
    train_loader, val_loader, test_loader = prepare_dataloader(batch_size, label='shortcut', keep_label=0, transform=transform)

    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, 
                 time_dim=time_dim,channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss() # use MSE loss 


    accumulation_steps = 16  # for example, simulate a batch 4x larger
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)
        optimizer.zero_grad()  # move this outside the loop
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.q_sample(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            # Normalize loss to account for accumulation
            loss = loss / accumulation_steps
            loss.backward()
            # Update weights only every accumulation_steps iterations
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(pbar):
                optimizer.step()
                optimizer.zero_grad()

    # for epoch in range(1, num_epochs + 1):
    #     # logging.info(f"Starting epoch {epoch}:")
    #     # pbar = tqdm(train_loader)
        
    #     for i, (images, labels) in enumerate(train_loader): #enumerate(pbar):
    #         images = images.to(device)

    #         # TASK 4: implement the training loop
    #         t = diffusion.sample_timesteps(images.shape[0]).to(device) # line 3 from the Training algorithm
    #         x_t, noise = diffusion.q_sample(images, t) # inject noise to the images (forward process), HINT: use q_sample
    #         predicted_noise = model(x_t, t) # predict noise of x_t using the UNet
    #         loss = mse(noise, predicted_noise)
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # pbar.set_postfix(MSE=loss.item())
    #         # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
    #         if wandb_run is not None:
    #             wandb_run.log({"MSE": loss.item()})

        if epoch % 5 == 0:
            # sampled_images = diffusion.p_sample_loop(model, batch_size=batch_size)
            # # create folder results/<datestr>
            # os.makedirs("results/"+datestr, exist_ok=True)
            os.makedirs("models/"+datestr, exist_ok=True)
            # save_images(images=sampled_images, path=os.path.join("results", datestr, f"{epoch}.jpg"),
            #             show=show, title=f'Epoch {epoch}')
            torch.save(model.state_dict(), os.path.join("models", datestr, f"weights.pt"))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=SEED)
    

    # Start a new wandb run to track this script.
    num_epochs = 100000
    size_w, size_h = config["size"][0], config["size"][1]
    batch_size = config['batch_size']
    run = wandb.init(
        entity="crulotest",
        project="ADLCV_exam",
        config={
            "batch_size": batch_size,
            "size_w": size_w,
            "size_h": size_h,
            "epochs": num_epochs,
        },
    )
    train(batch_size=batch_size , 
          device=device, 
          num_epochs=num_epochs,
          time_dim=256,
          img_size=np.array([size_h, size_w]),
          wandb_run=run)

if __name__ == '__main__':
    main()
    

        