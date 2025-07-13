import os
import sys

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

from models.unet import UNet
from dataset.loader import dataset_loader
from utils.metrics import compute_iou

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

    in_channels: int = field(
        default=10,
        metadata={"help": "The number of in channels to use."}
    )

    out_channels: int = field(
        default=10,
        metadata={"help": "The number of out channels to use."}
    )

    batch_size: int = field(
        default=128,
        metadata={"help": "The batch size to use."}
    )

    crop_size: int = field(
        default=32,
        metadata={"help": "The crop size to use."}
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
    if config.model_name == "unet":
        return UNet(in_ch=config.in_channels, out_ch=config.out_channels)
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

def train(model, train_loader, criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False, dynamic_ncols=True)

    for img, mask in loop:
        img, mask = img.to(device), mask.to(device)
        out = model(img)
        loss = criterion(out, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    return avg_loss


def validate(model, val_loader, criterion, device, writer, epoch, num_classes):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False, dynamic_ncols=True)

        for img, mask in loop:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            loss = criterion(out, mask)

            iou = compute_iou(out, mask, num_classes=num_classes)

            total_loss += loss.item()
            total_iou += iou
            loop.set_postfix({"Loss": f"{loss.item():.4f}", "IoU": f"{iou:.4f}"})

    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)

    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/IoU", avg_iou, epoch)

    return avg_loss, avg_iou


def main():
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        config = parse_config(sys.argv[1], Config)

    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Model: {config.model_name}")

    model = model_type(config).to(device)
    train_loader, val_loader = dataset_loader(config.data_dir, config.batch_size, config.crop_size)

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
        train_loss = train(model, train_loader, criterion, optimizer, device, writer, epoch)
        val_loss, val_iou = validate(model, val_loader, criterion, device, writer, epoch, num_classes=config.out_channels)

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print(f"Epoch Time: {mins}m {secs}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Val   IoU  : {val_iou:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
