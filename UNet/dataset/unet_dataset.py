import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Using ISIC Dataset
class UNetDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=(256, 256)):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.image_size = image_size

        self.trainsform_img = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

        self.trainsform_mask = T.Compose([
            T.Resize(image_size, interpolation=Image.NEAREST),
            T.ToTensor()
        ])

        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.label_paths[idx]).convert("L")

        image = self.trainsform_img(image)
        mask = self.trainsform_mask(mask)
        mask = (mask > 0.5).long().squeeze(0)


        return image, mask