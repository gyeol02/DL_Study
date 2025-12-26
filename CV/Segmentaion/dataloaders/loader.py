import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from .ISICDataset import ISICDataset

def get_dataloaders(data_dir="./dataset/data", batch_size=16, image_size=[256, 256], val_split=0.2, test_split=0.1, num_workers=2, seed=42):
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "masks")

    dataset = ISICDataset(image_dir, label_dir, image_size=image_size)

    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    indices = list(range(total_size))

    np.random.seed(seed)
    np.random.shuffle(indices)

    splits = [train_size, val_size, test_size] if test_split > 0 else [train_size, val_size]
    subsets = random_split(dataset, splits)

    train_loader = DataLoader(subsets[0], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(subsets[1], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(subsets[2], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader