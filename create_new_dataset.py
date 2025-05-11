import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import PIL.Image as Image
from PIL import ImageDraw

# torch imports
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# custom imports
from ddpm import Diffusion
from ddim import DiffusionImplicit
from model import UNet, Classifier

from utils import *

# clear cache 
torch.cuda.empty_cache()



class AddColorBlobs:
    def __init__(self, num_blobs=5, size_range=(3, 5)):
        self.num_blobs = num_blobs
        self.size_range = size_range

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = transforms.ToPILImage()(img)

        draw = ImageDraw.Draw(img)
        width, height = img.size

        for _ in range(self.num_blobs):
            # Random blob position and size
            x = 60#random.randint(0, width)
            y = 15#random.randint(0, height)
            r = 4#random.randint(*self.size_range)

            # Random color
            color = tuple(np.array([76, 230, 71]))

            # Draw a circle
            bbox = (x - r, y - r, x + r, y + r)
            draw.ellipse(bbox, fill=color, outline=None)

        return img



# init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('assets/', exist_ok=True)

L = torch.tensor([int(sys.argv[1])]).long().to(device)
gradient_scale = float(sys.argv[2])

# dataset and dataloaders
transform = transforms.Compose([
    transforms.Resize((img_size[0], img_size[1])),
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    # AddColorBlobs(num_blobs=1),
    # transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])

ddpm = Diffusion(        img_size=img_size, device=device, diff_type='DDPM-cg')
ddim = DiffusionImplicit(img_size=img_size, device=device, diff_type='DDPM-cg')

###########################################
###########################################
###########################################

# DDIM denoiseing process (std=0)
model = UNet(img_size=img_size, device=device)
model.eval()
model.to(device)
folder = "2025_05_10_06_52_13"
model.load_state_dict(torch.load(f'models/{folder}/weights.pt', map_location=device, weights_only=False)) # load the given model

# Classifier
classifier = Classifier(
    img_size=img_size, c_in=3, labels=2,
    time_dim=256,channels=32, device=device
)
classifier.to(device)
classifier.eval()
classifier.load_state_dict(torch.load('weights/classifier/model.pth', map_location=device))
ddim.classifier = classifier


# Get an image
train_loader, val_loader, test_loader = prepare_dataloader(batch_size, label='shortcut', keep_label=None, transform=transform)

for ex_index in range(len(train_loader.dataset.dataset)):
    print(ex_index)
    image, label = train_loader.dataset.dataset[ex_index]
    x0 = image.unsqueeze(0).to(device) # add batch dimenstion

    # 2. DDPM forward process that image up to a timestep t: 1<L<T
    xt, noise = ddpm.q_sample(x0, L)

    x_new = ddim.p_sample_loop(model=model, batch_size=1, partly_noised=xt, L=L.item(), y=torch.tensor([0]), gradient_scale=gradient_scale)
    x_new = x_new.squeeze(0).permute(1, 2, 0).cpu().numpy()

    name = train_loader.dataset.dataset.image_paths[ex_index]
    name = name.split("/")[-1]
    name = name.split(".")[0]
    name = name.split("_")[1]
    image = Image.fromarray(x_new)
    image.save(f"../../data/trainwoDots/{name}.png")


###########################################
###########################################
###########################################
