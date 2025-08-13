import os
import random
import sys
import numpy as np

import json
from typing import Type, TypeVar
from dataclasses import dataclass, field, fields

import torch
import torch.nn as nn
import torch.optim as optim

from models.mobilenet import MobileNet
from dataset.loader import dataset_loader
from scripts.train_val_test import Trainer, Validator, create_checkpoint_folder

T = TypeVar('T')

def set_seed(seed=42):
    print(f"Using seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        # MPS doesn't need deterministic/benchmark settings
        pass
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_config(config_path: str, cls: Type[T]) -> T:
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    cls_fields = {field.name for field in fields(cls)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}
    
    return cls(**filtered_data)

@dataclass
class Config:
    device: str = field(default="mps" if torch.backends.mps.is_available() else (
                            "cuda" if torch.cuda.is_available() else "cpu"))
    seed: int = field(default=42)
    model_name: str = field(default="mobilnet")

    n_classes: int = field(default=10)
    image_size: list = field(default_factory=lambda: [256, 256])

    batch_size: int = field(default=128)
    num_epochs: int = field(default=100)

    opt_name: str = field(default="adam")
    lr: float = field(default=1e-3)

    early_stopping: int = field(default=None)

    data_dir: str = field(default="./dataset/data")



def model_type(config: Config):
    if config.model_name == "mobilenet":
        return MobileNet(num_classes=config.n_classes)
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else "config.json"
    config = parse_config(config_path, Config)

    set_seed(config.seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Model: {config.model_name}")

    model = model_type(config).to(device)
    train_loader, val_loader, _ = dataset_loader(config.data_dir, config.batch_size, config.image_size)

    if config.opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    elif config.opt_name == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=config.lr, momentum=0.9, alpha=0.9, weight_decay=4e-5)
    else:
        raise ValueError(f"Unsupported optimizer: {config.opt_name}")
    
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = create_checkpoint_folder(base_path="checkpoints")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=config.num_epochs,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    trainer.train()


if __name__ == "__main__":
    main()
