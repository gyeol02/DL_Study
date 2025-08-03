import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from Transformer.scripts.train_val_train import Tester,  create_checkpoint_folder
from dataset.loader import dataset_loader
from model.Transformer import Transformer
from dataclasses import dataclass, field, fields
from typing import Type, TypeVar

T = TypeVar("T")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
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
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")
    seed: int = field(default=42)
    model_name: str = field(default="transformer")

    d_model: int = field(default=512)
    num_heads: int = field(default=8)
    d_ff: int = field(default=2048)
    num_layers: int = field(default=6)

    batch_size: int = field(default=32)
    lr: float = field(default=1e-4)

    data_dir: str = field(default="./dataset/data")
    num_epochs: int = field(default=10)

    tokenizer_name: str = field(default="t5-small")
    max_length: int = field(default=128)


def model_type(config: Config, vocab_size: int):
    if config.model_name == "transformer":
        return Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_layers
        )
    else:
        raise ValueError(f"Unsupported model: {config.model_name}")

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else "config.json"
    config = parse_config(config_path, Config)

    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    print(f"[Device] Using {device}")

    # Load test dataset and tokenizer
    test_loader, tokenizer = dataset_loader(
        data_dir=config.data_dir,
        tokenizer_name=config.tokenizer_name,
        split="test",
        max_length=config.max_length,
        batch_size=config.batch_size
    )

    vocab_size = tokenizer.vocab_size

    model = model_type(config, vocab_size).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    checkpoint_dir = create_checkpoint_folder(base_path="checkpoints")

    testor = Tester(
        model=model,
        test_loader=test_loader,
        tokenizer=tokenizer,
        criterion=criterion,
        checkpoint_dir=checkpoint_dir
    )

    testor.train()

if __name__ == "__main__":
    main()