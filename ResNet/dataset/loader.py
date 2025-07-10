import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def dataset_loader(data_dir: str = "./dataset/data", batch_size: int = 128):
    # CIFAR-10
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = datasets.CIFAR10(root=data_dir, train=True, transform=transforms_train, download=True)
    val_data = datasets.CIFAR10(root=data_dir, train=False, transform=transforms_val, download=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



