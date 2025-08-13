import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def dataset_loader(data_dir, batch_size, image_size, val_ratio=0.1, num_workers=2):
    # CIFAR-100  Normalize
    mean = [0.5071, 0.4867, 0.4408]
    std  = [0.2675, 0.2565, 0.2761]

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)

    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader