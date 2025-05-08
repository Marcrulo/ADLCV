import os
import numpy as np
import matplotlib.pyplot as plt

# torch imports
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# custom imports
from ddpm import Diffusion
from model import UNet

from utils import *

import PIL.Image as Image
from PIL import ImageDraw

class AddColorBlobs:
    def __init__(self, num_blobs=5, size_range=(5, 50)):
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



def show(imgs, title=None, fig_titles=None, save_path=None): 

    if fig_titles is not None:
        assert len(imgs) == len(fig_titles)

    fig, axs = plt.subplots(1, ncols=len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        axs[i].imshow(img)
        axs[i].axis('off')
        if fig_titles is not None:
            axs[i].set_title(fig_titles[i])

    if title is not None:
        plt.suptitle(title)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('assets/', exist_ok=True)
    
    # dataset and dataloaders
    transform = transforms.Compose([
                                transforms.Resize((img_size[0], img_size[1])), 
                                transforms.ToTensor(),
                                AddColorBlobs(num_blobs=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                 ])

    train_loader, val_loader, test_loader = prepare_dataloader(batch_size, label='shortcut', keep_label=1, transform=transform)
    images, labels  = next(iter(test_loader))
    example_images = np.stack([im_normalize(tens2image(images[idx])) for idx in range(batch_size)], axis=0)
    show(example_images, 'Example images', save_path='assets/example.png')

    ################## Diffusion class ##################
    # TASK 1: Implement beta, alpha, and alpha_hat 
    diffusion = Diffusion(img_size=img_size, device=device)
    plt.figure()
    plt.plot(range(1,diffusion.T+1), diffusion.alphas.cpu().numpy(), label='alphas', linewidth=3)
    plt.plot(range(1,diffusion.T+1), diffusion.alphas_bar.cpu().numpy(), label='alphas_bar',linewidth=3)
    plt.plot(range(1,diffusion.T+1), diffusion.betas.cpu().numpy(), label='betas', linewidth=3)
    plt.title('Diffusion parameters')
    plt.legend()
    plt.savefig('assets/diffusion_params.png', bbox_inches='tight')
    plt.show()
    #####################################################
    

    # timesteps for forward

    t = torch.Tensor([0, 50, 100, 150, 200, 300, 499]).long().to(device)
    fig_titles = [f'Step {ti.item()}' for ti in t]
    x0 = images[0].unsqueeze(0).to(device) # add batch dimenstion

    ################## Forward process ##################
    # TASK 2: Implement it in the diffusion class
    xt, noise = diffusion.q_sample(x0, t)
    #####################################################

    noised_images = np.stack([im_normalize(tens2image(xt[idx].cpu())) for idx in range(t.shape[0])], axis=0)
    show(noised_images, title='Forward process', fig_titles=fig_titles, save_path='assets/forward.png')



    ################## Inverse process ##################
    model = UNet(img_size=img_size, device=device)
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load('models/ddpm/weights.pt', map_location=device, weights_only=False)) # load the given model


    ###########################################
    ###########################################
    ###########################################

    # 1. Get an image

    # 2. DDPM forward process that image up to a timestep t: 1<L<T

    # 3. DDIM denoiseing process (std=0)

    # 4. Calculate difference


    ###########################################
    ###########################################
    ###########################################




    # TASK 3: Implement it in the diffusion class
    x_new, intermediate_images = diffusion.p_sample_loop(model, 1, timesteps_to_save=t)
    intermediate_images = [tens2image(img.cpu()) for img in intermediate_images]
    show(intermediate_images, title='Reverse process', fig_titles=fig_titles, save_path='assets/reverse.png')
    #####################################################


    # Diff between x_new and x0
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow( x0[0].cpu().permute(1,2,0), cmap='jet', vmin=-1, vmax=1)
    axs[0].set_title('x0')
    axs[1].imshow( x_new[0].cpu().permute(1,2,0), cmap='jet', vmin=-1, vmax=1)
    axs[1].set_title('x_new')
    axs[2].set_title('Diff')
    axs[2].axis('off')
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].axis('off')
    axs[2].set_xticks([])
    diff_img = x_new[0].cpu().permute(1,2,0) - x0[0].cpu().permute(1,2,0)
    diff_img = im_normalize(diff_img)
    diff_img = np.clip(diff_img, 0, 1)
    axs[2].imshow( diff_img, cmap='jet', vmin=-1, vmax=1)
    plt.savefig('assets/diff.png', bbox_inches='tight', pad_inches=0)
    plt.show()
