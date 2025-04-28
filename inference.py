import os
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
from model import UNet

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
            x = random.randint(0, width)
            y = random.randint(0, height)
            r = random.randint(*self.size_range)

            # Random color
            color = tuple(np.random.randint(0, 256, size=3))

            # Draw a circle
            bbox = (x - r, y - r, x + r, y + r)
            draw.ellipse(bbox, fill=color, outline=None)

        return img



# init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('assets/', exist_ok=True)

# dataset and dataloaders
transform = transforms.Compose([
    transforms.Resize((img_size[0], img_size[1])),
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    AddColorBlobs(num_blobs=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
])

ddpm = Diffusion(img_size=img_size, device=device)
ddim = DiffusionImplicit(img_size=img_size, device=device)

###########################################
###########################################
###########################################

# 1. Get an image
train_loader, val_loader, test_loader = prepare_dataloader(batch_size, label='shortcut', keep_label=1, transform=transform)
images, labels = next(itertools.islice(val_loader, 1, None))
L = torch.Tensor([5,25,45,65,85]).long().to(device)  
x0 = images[0].unsqueeze(0).to(device) # add batch dimenstion

# 2. DDPM forward process that image up to a timestep t: 1<L<T
xt, noise = ddpm.q_sample(x0, L)

# 3. DDIM denoiseing process (std=0)
model = UNet(img_size=img_size, device=device)
model.eval()
model.to(device)
model.load_state_dict(torch.load('models/ddpm/weights.pt', map_location=device, weights_only=False)) # load the given model

Lsize = L.shape[0]
fig, axs = plt.subplots(Lsize, 4, figsize=(10,2*Lsize))
x_new = torch.zeros_like(x0).repeat(Lsize,1,1,1)
for i, l in enumerate(L):
    x_new[i] = ddim.p_sample_loop(model=model, batch_size=1, partly_noised=xt[i].unsqueeze(0), L=l.item())
    img1 = im_normalize(tens2image(x0.cpu()))
    img2 = im_normalize(tens2image(xt[i].cpu()))
    img3 = im_normalize(tens2image(x_new[i].cpu()))
    diff_img = np.linalg.norm( img1 - img3, axis=2)
    img4 = im_normalize( diff_img )

    im1 = axs[i,0].imshow(img1)
    im2 = axs[i,1].imshow(img2)
    im3 = axs[i,2].imshow(img3)
    im4 = axs[i,3].imshow(img4, vmin=0, vmax=1, cmap='jet')
    fig.colorbar(im4, ax=axs[i,3])


    titles = ["Original", f"Noised to t={l}", "Denoised", "Diff image"]
    for j, title in enumerate(titles):
        axs[i,j].set_title(title)

plt.tight_layout()
plt.savefig("assets/inference.png")
plt.close()


###########################################
###########################################
###########################################
