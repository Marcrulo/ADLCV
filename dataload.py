# Files
import os
import sys
import glob
from itertools import compress

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


class LesionDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, data_path=None, label="lesion", keep_label=None):
        "Initialization"
        # load from config file
        self.config = yaml.safe_load(open("config.yaml"))
        if data_path is None:
            self.data_path = self.config['data_path'] 
        else:
            self.data_path = data_path
        self.size_w, self.size_h = self.config["size"][0], self.config["size"][1]

        self.transform = transform
        self.label = label
        self.image_paths = sorted(
            glob.glob(self.data_path + f"/{self.config['images_folder']}/*.jpg")
        )
        if self.label == "lesion":
            self.labels = pd.read_csv(self.data_path + f"/{self.config['lesion_file']}")
        else:
            self.labels = pd.read_csv(self.data_path + f"/{self.config['shortcut_file']}")[["image", "ruler"]]

        if keep_label == 'cg':
            self.labels = self.labels[['ruler']]
        elif keep_label is not None:
            shortcut_mask = self.labels['ruler'] == keep_label
            self.labels = self.labels[shortcut_mask]
            self.image_paths = list(np.array(self.image_paths)[shortcut_mask])

        if transform == None:
            train_transform = transforms.Compose(
                [transforms.Resize((self.size_h, self.size_w)), transforms.ToTensor()]
            )
            self.transform = train_transform

    def __len__(self):
        "Returns the total number of samples"
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        image_path = self.image_paths[idx] 
        image = Image.open(image_path).convert("RGB")
        X = self.transform(image)

        if self.label == "lesion":
            Y = torch.tensor(self.labels[self.labels.columns[1:]].iloc[idx].values[1:])
        else:
            Y = torch.tensor(self.labels.iloc[idx]["ruler"])

        return X, Y




if __name__ == "__main__":
    
    config = yaml.safe_load(open("config.yaml"))
    data_path = config["data_path"]
    size_w, size_h = config["size"][0], config["size"][1]
    train_transform = transforms.Compose(
        [transforms.Resize((size_h, size_w)), transforms.ToTensor()]
    )

    batch_size = config["batch_size"]
    trainset = LesionDataset(
        transform=train_transform, data_path=data_path, label="shortcut"
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )
    
    for i, (X, Y) in enumerate(train_loader):
        print(X.shape, Y)
        plt.imshow(X[0].permute(1, 2, 0))
        plt.show()
        break
