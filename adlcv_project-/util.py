import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import yaml
import dataload
import numpy as np
#CLASS_LABELS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF, 'VASC']
#CLASS_LABELS = ['ruler']
config = yaml.safe_load(open("config.yaml"))
data_path = config["data_path"]
SEED = config['seed']
CLASS_LABELS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
train_size = config['lesion_train']
val_size = config['lesion_val']
test_size = config['lesion_test']
DATASET_SIZE = train_size + val_size + test_size
batch_size = config['batch_size']

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True


def prepare_dataloaders(batch_size=batch_size, val_batch_size=batch_size, label="lesion"):
    dataset = dataload.LesionDataset(transform=None, data_path=data_path, label=label)
    
    #print(len(dataset), "images in dataset")
    #print("Train size: ", train_size)
    #print("Validation size: ", val_size)
    #print("Test size: ", test_size)
    #print("Total size: ", DATASET_SIZE)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def show(imgs, title=None, fig_titles=None, save_path=None):

    if fig_titles is not None:
        assert len(imgs) == len(fig_titles)

    fig, axs = plt.subplots(1, ncols=len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        axs[i].imshow(img)
        axs[i].axis("off")
        if fig_titles is not None:
            axs[i].set_title(fig_titles[i], fontweight="bold")

    if title is not None:
        plt.suptitle(title, fontweight="bold")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    plt.show()
