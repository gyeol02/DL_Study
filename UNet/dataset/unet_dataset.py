import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class UNetDataset(Dataset):
    def __init__(self, image_dir, label_dir, crop_size, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # 흑백 영상 기준
        label = Image.open(self.label_paths[idx])

        image = np.array(image)
        label = np.array(label)

        label[label == 255] = 0    # 배경
        label[label > 1] = 1       # foreground

        h, w = image.shape
        ch = self.crop_size
        cw = self.crop_size
        
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)

        image = image[top:top+ch, left:left+cw]
        label = label[top:top+ch, left:left+cw]

        # To tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
        label = torch.tensor(label, dtype=torch.long)                          # [H, W] (class index)

        return image, label