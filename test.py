import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT
# from dataloader import train_loader, trainset
from dataloader import LesionDataset
import yaml
from torch.utils.data import DataLoader


# Files
import os
import sys
import glob 

# Images
import PIL.Image as Image

# Data
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Data loader
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import utils
from torch.utils.data import DataLoader

val_set = False

# Config
import yaml

def prepare_dataloaders(batch_size):

    config = yaml.safe_load(open("config.yaml"))
    data_path = config['data_path']

    size_w, size_h = config['size'][0], config['size'][1]
    transform = transforms.Compose([transforms.Resize((size_h, size_w)), 
                                        transforms.ToTensor()])

    batch_size = config['batch_size']
    trainset = LesionDataset(transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    valset = LesionDataset(transform=transform, val_data=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=1)
    

    return train_loader, trainset, val_loader, valset

if __name__ == '__main__':
    train_loader, trainset, val_loader, valset = prepare_dataloaders(batch_size=3)

    for i, (X, Y) in enumerate(train_loader):
        # print(X.shape, Y)
        # plt.imshow(X[0].permute(1, 2, 0))
        # plt.show()
        print(Y)
        break

    for i, (X, Y) in enumerate(val_loader):
        # print(X.shape, Y)
        # plt.imshow(X[0].permute(1, 2, 0))
        # plt.show()
        break