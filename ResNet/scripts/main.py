import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import Type, TypeVar
from dataclasses import dataclass, field, fields

import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet import ResNet50, ResNet101, ResNet152
from dataset.loader import dataset_loader

T = TypeVar('T')

def parse_config(config_path: str, cls: Type[T]) -> T:
    """
    Parse a JSON config file and convert it to an instance of the given dataclass.
    
    Args:
        config_path (str): The path to the JSON config file.
        cls (Type[T]): The dataclass type to map the JSON data to.
    
    Returns:
        T: An instance of the dataclass populated with the JSON data.
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    cls_fields = {field.name for field in fields(cls)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}
    
    return cls(**filtered_data)

@dataclass
class Config:
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to use: cuda, mps, or cpu"}
    )

    model_name: str = field(
        default="resnet50",
        metadata={"help": "The name of the model to use."}
    )

    n_classes: int = field(
        default=10,
        metadata={"help": "The number of classes to classify."}
    )

    batch_size: int = field(
        default=128,
        metadata={"help": "The batch size to use."}
    )

    opt_name: str = field(
        default="adam",
        metadata={"help": "The name of the optimizer to use."}
    )

    lr: float = field(
        default=1e-3,
        metadata={"help": "The learning rate to use."}
    )

    data_dir: str = field(
        default="./dataset/data",
        metadata={"help": "Dataset root directory"}
    )

    num_epochs: int = field(
        default=100,
        metadata={"help": "The number of epochs to train for."}
    )

    early_stopping: int = field(
        default=None,
        metadata={"help": "The number of epochs to wait before early stopping."}
    )

def model_type(config: Config):
    if config.model_name == "resnet50":
        return ResNet50(n_classes = config.n_classes)
    elif config.model_name == "resnet101":
        return ResNet101(n_classes = config.n_classes)
    elif config.model_name == "resnet152":
        return ResNet152(n_classes = config.n_classes)
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_correct_cnt = 0
    total = 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        out = model(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.topk(out, k=1, dim=-1)
        pred = pred.squeeze(dim=1)

        train_correct_cnt += int(torch.sum(pred == label))
        total += label.size(0)

        batch_correct = int(torch.sum(pred == label))
        batch_total = label.size(0)
        batch_acc = batch_correct / batch_total
        print(f"Train Batch Accuarcy: {batch_acc:.2%}")

    epoch_acc = train_correct_cnt / total
    print(f"Train Epoch Accuracy: {epoch_acc:.2%}")


def validate(model, val_loader, criterion, device):
    model.eval()
    val_correct_cnt = 0
    total = 0
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            _, pred = torch.topk(out, k=1, dim=-1)
            pred = pred.squeeze(dim=1)

            val_correct_cnt += int(torch.sum(pred == label))
            total += label.size(0)

            batch_correct = int(torch.sum(pred == label))
            batch_total = label.size(0)
        batch_acc = batch_correct / batch_total
        print(f"Validation Batch Accuarcy: {batch_acc:.2%}")

    epoch_acc = val_correct_cnt / total
    print(f"Validation Epoch Accuracy: {epoch_acc:.2%}")

            

def main():
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parse_config(sys.argv[1], Config)

    device = torch.device(config.device)

    print(f"Using device: {device}")
    print(f"Model: {config.model_name}")

    model = model_type(config).to(device)
    train_loader, val_loader = dataset_loader(config.data_dir, config.batch_size)

    criterion = nn.CrossEntropyLoss()

    if config.opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config.opt_name}")

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        train(model, train_loader, criterion, optimizer, device)
        validate(model, val_loader, criterion, device)

    
if __name__ == "__main__":
    main()