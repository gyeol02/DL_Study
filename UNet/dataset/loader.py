import os
import torch
from torch.utils.data import DataLoader, random_split
from dataset.unet_dataset import UNetDataset

def dataset_loader(data_dir, batch_size, image_size, val_ratio=0.2, test_ratio=0.1, num_workers=4):
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "masks")

    dataset = UNetDataset(image_dir, label_dir, image_size=image_size)

    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size

    splits = [train_size, val_size, test_size] if test_ratio > 0 else [train_size, val_size]
    subsets = random_split(dataset, splits)

    train_loader = DataLoader(subsets[0], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(subsets[1], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(subsets[2], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader