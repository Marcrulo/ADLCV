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
from torch.utils.data import DataLoader, WeightedRandomSampler

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# def prepare_dataloaders(batch_size):

#     config = yaml.safe_load(open("config.yaml"))
#     data_path = config['data_path']

#     size_w, size_h = config['size'][0], config['size'][1]
#     transform = transforms.Compose([transforms.Resize((size_h, size_w)), 
#                                         transforms.ToTensor()])

#     batch_size = config['batch_size']
#     trainset = LesionDataset(transform=transform)
#     train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

#     valset = LesionDataset(transform=transform, val_data=True)
#     val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=1)
    

#     return train_loader, val_loader, trainset, valset

def prepare_dataloaders(batch_size):
    config = yaml.safe_load(open("config.yaml"))
    data_path = config['data_path']

    size_w, size_h = config['size'][0], config['size'][1]
    transform = transforms.Compose([transforms.Resize((size_h, size_w)), 
                                    transforms.ToTensor()])

    batch_size = config['batch_size']
    trainset = LesionDataset(transform=transform)

    print()
    print('Transforming data')
    print()
    labels = [label.argmax().item() for _, label in trainset]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = [weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler) #sampler=sampler

    valset = LesionDataset(transform=transform, val_data=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, trainset, valset


def main(image_size=(180,240), patch_size=(12,12), channels=3, 
         embed_dim=128, num_heads=8, num_layers=16, num_classes=6,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=100, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1
    ):

    model_name = 'm_180_240x12x8_16x100x_1e-4_org.pth'
    stats_name = 's_180_240x12x8_16x100x_1e-4_org.pth'

    loss_function = nn.CrossEntropyLoss()

    train_loader, val_loader, trainset, valset = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    # training loop
    best_val_loss = 1e10

    train_accs = []
    train_losses = []

    val_accs = []
    val_losses = []

    for e in range(num_epochs):
        print(f'\n epoch {e}')
        model.train()
        train_loss = 0
        tot_train, cor_train = 0.0, 0.0
        for image, label in tqdm.tqdm(train_loader, desc="Training", mininterval=0.5, miniters=10):
            label = label.argmax(dim=1).long()

            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            train_loss += loss.item()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()

            sch.step()

            out = out.argmax(dim=1)
            tot_train += float(image.size(0))
            cor_train += float((label == out).sum().item())

            
        acc_train = cor_train / tot_train    
        train_loss /= len(train_loader)

        train_accs.append(acc_train)
        train_losses.append(train_loss)

        val_loss = 0
        with torch.no_grad():
            model.eval()
            tot_val, cor_val = 0.0, 0.0
            for image, label in val_loader:
                label = label.argmax(dim=1).long()

                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out = model(image)
                loss = loss_function(out, label)
                val_loss += loss.item()
                out = out.argmax(dim=1)
                tot_val += float(image.size(0))
                cor_val += float((label == out).sum().item())


            acc_val = cor_val / tot_val
            val_loss /= len(val_loader)

            val_accs.append(acc_val)
            val_losses.append(val_loss)

            print(f'-- train loss {train_loss:.3f} -- train accuracy {acc_train:.3f} -- validation accuracy {acc_val:.3f} -- validation loss: {val_loss:.3f}')
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), 'model.pth')
                best_val_loss = val_loss

        stats = {
            'train_accs': train_accs,
            'train_losses' : train_losses,
            'val_accs' : val_accs,
            'val_losses' : val_losses
        }
        torch.save(stats, 'stats.pth')


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()
