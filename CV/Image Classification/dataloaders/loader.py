import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_dir="./dataset/data", batch_size=128, val_split=0.1, num_workers=2, seed=42):
    """
    Args:
        data_dir (str): 데이터셋 경로
        batch_size (int): 배치 크기
        val_split (float): 검증 데이터셋 비율 (0.0 ~ 1.0)
        num_workers (int): 데이터 로딩에 사용할 CPU 코어 수
    """
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset_aug = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    num_train = len(train_dataset_aug)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    np.random.seed(seed)
    np.random.shuffle(indices)

    val_size = int(len(train_dataset_aug) * val_split)
    train_size = len(train_dataset_aug) - val_size
    train_dataset, val_dataset = random_split(train_dataset_aug, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader