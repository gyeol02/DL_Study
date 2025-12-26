import json
import torch
from dataclasses import dataclass, field, fields
from typing import Dict, Any, List

@dataclass
class Config:
    
    device: str = field(
        default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device: mps, cuda, or cpu"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    save_dir: str = field(default="./saved", metadata={"help": "Root directory for saving experiments"})
    model_name: str = field(default="UNet", metadata={"help": "Model architecture name"})

    data_dir: str = field(default="./dataset/data/ISIC", metadata={"help": "Dataset root path"})
    image_size: List[int] = field(default_factory=lambda: [256, 256], metadata={"help": "Input image size [H, W]"})
    num_workers: int = field(default=2, metadata={"help": "DataLoader workers"})

    batch_size: int = field(default=16, metadata={"help": "Batch size"})
    num_epochs: int = field(default=20, metadata={"help": "Total epochs"})
    lr: float = field(default=1e-3, metadata={"help": "Learning rate"})
    opt_name: str = field(default="sgd", metadata={"help": "Optimizer: adam or sgd"})
    early_stopping: int = field(default=10, metadata={"help": "Patience for early stopping"})

    model_params: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Specific model kwargs"})

def parse_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Config 클래스에 정의된 필드만 필터링하여 매핑
    cls_fields = {f.name for f in fields(Config)}
    filtered_data = {k: v for k, v in config_data.items() if k in cls_fields}
    
    return Config(**filtered_data)