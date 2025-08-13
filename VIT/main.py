import os
import random
import sys
import numpy as np

import json
from typing import Type, TypeVar
from dataclasses import dataclass, field, fields

import torch

from model.vit import ViT
from dataset.loader import dataset_loader
from scripts.train_val_test import Trainer, Validator, create_checkpoint_folder
from utils.scheduler import vit_scheduler

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
    model_name: str = field(default="vit")

    patch_size: int = field(default=16)
    in_channels: int = field(default=3)
    n_classes: int = field(default=100)
    embed_dim: int = field(default=384)
    layers: int = field(default=8)
    num_heads: int = field(default=6)
    mlp_ratio: float = field(default=4.0)
    drop_rate: float = field(default=0.1)
    attn_drop_rate: float = field(default=0.1)
    drop_path_rate: float = field(default=0.1)

    num_epochs: int = field(default=20)
    lr: float = field(default=0.003)
    weight_decay: float = field(default=0.05)
    warmup_epochs: int = field(default=5)

    data_name: str = field(default="CIFAR100")
    data_dir: str = field(default="./dataset/data")
    image_size: int = field(default=256)
    val_ratio: float = field(default=0.2)
    batch_size: int = field(default=128)
    num_workers: int = field(default=4)

def model_type(config: Config):
    if config.model_name == "vit":
        return ViT(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_channels,
            num_classes=config.n_classes,
            embed_dim=config.embed_dim,
            depth=config.layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
        )
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")
    
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else "config.json"
    config = parse_config(config_path, Config)

    set_seed(config.seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Model: {config.model_name} | Data: {config.data_name}")

    model = model_type(config).to(device)
    train_loader, val_loader, _ = dataset_loader(config.data_dir, config.batch_size, config.image_size,
                                                 config.val_ratio, config.num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))

    total_steps = config.num_epochs * len(train_loader)
    warmup_steps = max(1, config.warmup_epochs * len(train_loader))
    scheduler = vit_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_dir = create_checkpoint_folder(base_path="checkpoints")

    trainer = Trainer(
        model = model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=config.num_epochs,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    trainer.train()

if __name__ == "__main__":
    main()