import json
import torch
from dataclasses import dataclass, field, fields
from typing import Dict, Any

@dataclass
class Config:
    
    device: str = field(
        default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device: mps, cuda, or cpu"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    save_dir: str = field(default="./saved", metadata={"help": "Root directory for saving experiments"})
    model_name: str = field(default="ResNet50", metadata={"help": "Model architecture name"})

    data_dir: str = field(default="./dataset/data", metadata={"help": "Dataset root path"})
    num_workers: int = field(default=2, metadata={"help": "DataLoader workers"})

    batch_size: int = field(default=128, metadata={"help": "Batch size"})
    num_epochs: int = field(default=100, metadata={"help": "Total epochs"})
    lr: float = field(default=1e-3, metadata={"help": "Learning rate"})
    early_stopping: int = field(default=10, metadata={"help": "Patience for early stopping"})

    tokenizer_name: int = field(default=10, metadata={"help": "Pre-training Tokenizer"})
    max_length: int = field(default=10, metadata={"help": "Tokenizer Max Length"})

    model_params: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Specific model kwargs"})

def parse_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Config 클래스에 정의된 필드만 필터링
    cls_fields = {f.name for f in fields(Config)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}
    
    return Config(**filtered_data)