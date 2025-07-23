import os
import torch
from torch.utils.data import DataLoader, random_split
from dataset.unet_dataset import UNetDataset

def dataset_loader(data_dir, batch_size, image_size, val_ratio=0.2):
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "masks")

    dataset = UNetDataset(image_dir, label_dir, image_size=image_size)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader