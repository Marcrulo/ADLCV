import os
import numpy as np
import matplotlib.pyplot as plt

# torch imports
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# custom imports
from ddpm import Diffusion
from ddim import DiffusionImplicit
from model import UNet

from utils import *


# init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('assets/', exist_ok=True)

# dataset and dataloaders
transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])

ddpm = Diffusion(img_size=img_size, device=device)
ddim = DiffusionImplicit(img_size=img_size, device=device)

###########################################
###########################################
###########################################

# 1. Get an image
train_loader, val_loader, test_loader = prepare_dataloader(batch_size, label='shortcut', keep_label=1)
images, labels  = next(iter(test_loader))
L = torch.Tensor([100]).long().to(device)
x0 = images[0].unsqueeze(0).to(device) # add batch dimenstion

# 2. DDPM forward process that image up to a timestep t: 1<L<T
xt, noise = ddpm.q_sample(x0, L)

# 3. DDIM denoiseing process (std=0)
model = UNet(img_size=img_size, device=device)
model.eval()
model.to(device)
model.load_state_dict(torch.load('models/ddpm/weights.pt', map_location=device, weights_only=False)) # load the given model

x_new = ddim.p_sample_loop(model=model, batch_size=xt.shape[0], partly_noised=xt, L=L.item())


# 4. Calculate difference
diff_img = (x_new - x0)**2



fig, axs = plt.subplots(1,4, figsize=(20,5))

# print(x0.shape, type(x0))
# print(xt.shape, type(xt))
# print(x_new.shape, type(x_new))
# print(diff_img.shape, type(diff_img))


im1 = axs[0].imshow(im_normalize(tens2image(x0.cpu())))
fig.colorbar(im1, ax=axs[0])
im2 = axs[1].imshow(im_normalize(tens2image(xt.cpu())))
fig.colorbar(im2, ax=axs[1])
im3 = axs[2].imshow(im_normalize(tens2image(x_new.cpu())))
fig.colorbar(im3, ax=axs[2])
im4 = axs[3].imshow(im_normalize(tens2image(diff_img.cpu())))
fig.colorbar(im4, ax=axs[3])

titles = ["Original", f"Noised to t={L.item()}", "Denoised", "Diff image"]
for i, title in enumerate(titles):
    axs[i].set_title(title)

plt.tight_layout()
plt.savefig("assets/inference.png")
plt.close()


###########################################
###########################################
###########################################
