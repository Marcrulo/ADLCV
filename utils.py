from dataload import LesionDataset
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
import yaml
import random
import os
import torchvision
from matplotlib import pyplot as plt

config = yaml.safe_load(open("config.yaml"))
data_path = config["data_path"]
SEED = config['seed']
CLASS_LABELS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
train_size = config['lesion_train']
val_size = config['lesion_val']
test_size = config['lesion_test']
DATASET_SIZE = train_size + val_size + test_size
batch_size = config['batch_size']

size_w, size_h = config["size"][0], config["size"][1]
img_size = np.array([size_w, size_h])

def prepare_dataloader(batch_size=batch_size, val_batch_size=batch_size, label="lesion", keep_label=None):
    dataset = LesionDataset(transform=None, data_path=data_path, label=label, keep_label=keep_label)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def create_result_folders(experiment_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)
    
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()
    
def im_normalize(im):
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn

def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))