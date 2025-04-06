
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



# Config
import yaml

# load from config file

config = yaml.safe_load(open("config.yaml"))
data_path = config['data_path']

class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, transform, data_path=data_path, val_data=False):
        'Initialization'
        self.transform = transform

        if val_data:
            self.image_paths = sorted(glob.glob(data_path + f"/{config['images_folder'][1]}/*.jpg"))
            self.lesions     = pd.read_csv(data_path + f"/{config['lesion_file'][1]}")
        else:
            self.image_paths = sorted(glob.glob(data_path + f"/{config['images_folder'][0]}/*.jpg"))
            self.lesions     = pd.read_csv(data_path + f"/{config['lesion_file'][0]}")

        # Unncomment if shortcuts are needed
        # self.shortcuts   = pd.read_csv(data_path + f"/{config['shortcut_file']}")#[["image","ruler"]]
        # self.shortcuts = self.shortcuts[self.shortcuts["ink"] != 1]
        # self.shortcuts = self.shortcuts[self.shortcuts["sticker"] != 1]
        # self.shortcuts = self.shortcuts[["image","ruler"]]
        

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert('RGB')
        X = self.transform(image)
        # Y = {'lesion'  : torch.tensor(self.lesions[self.lesions.columns[1:]].iloc[idx].values[1:])}
        Y = torch.tensor(self.lesions[self.lesions.columns[1:]].iloc[idx].values[1:])

        # Add in first position in Y if needed
        # 'shortcut': torch.tensor(self.shortcuts.iloc[idx]['ruler'])

        return X, Y
    
# size_w, size_h = config['size'][0], config['size'][1]
# train_transform = transforms.Compose([transforms.Resize((size_h, size_w)), 
#                                       transforms.ToTensor()])

# batch_size = config['batch_size']
# trainset = LesionDataset(transform=train_transform)
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

if __name__ == '__main__':
    print('Tried to run dataloader file')
    # print(len(train_loader))
    # for i, (X, Y) in enumerate(train_loader):
    #     print(X.shape, Y)
    #     plt.imshow(X[0].permute(1, 2, 0))
    #     plt.show()
    #     break
