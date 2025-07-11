import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import Type, TypeVar
from dataclasses import dataclass, field, fields

import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

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

def train(model, train_loader, criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False, dynamic_ncols=True)
    for img, label in loop:
        img, label = img.to(device), label.to(device)
        out = model(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.topk(out, 1, dim=1)
        pred = pred.squeeze(1)

        correct += (pred == label).sum().item()
        total += label.size(0)
        total_loss += loss.item()

        loop.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{(correct / total):.2%}"
        })

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = correct / total

    writer.add_scalar("Train/Loss", epoch_loss, epoch)
    writer.add_scalar("Train/Accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False, dynamic_ncols=True)
        for img, label in loop:
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)

            _, pred = torch.topk(out, 1, dim=1)
            pred = pred.squeeze(1)

            correct += (pred == label).sum().item()
            total += label.size(0)
            total_loss += loss.item()

            loop.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{(correct / total):.2%}"
        })

    epoch_loss = total_loss / len(val_loader)
    epoch_acc = correct / total

    writer.add_scalar("Val/Loss", epoch_loss, epoch)
    writer.add_scalar("Val/Accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc

            

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
    
    log_dir = os.path.join("runs")
    writer = SummaryWriter(log_dir=f"{log_dir}/{config.model_name}")


    for epoch in range(config.num_epochs):
        start_time = time.time()

        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, writer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, writer, epoch)

        epoch_time = time.time() - start_time
        mins, secs = divmod(int(epoch_time), 60)

        print(f"Epoch Time: {mins}m {secs}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2%}")

    writer.close()
    
if __name__ == "__main__":
    main()