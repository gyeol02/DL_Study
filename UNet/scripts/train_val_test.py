# scripts/train_val_test.py
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import compute_iou

def create_checkpoint_folder(base_path='checkpoints'):
    setting_number = 1
    while True:
        folder_name = f'setting_#{setting_number}'
        path = os.path.join(base_path, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(os.path.join(path, "logs"))
            os.makedirs(os.path.join(path, "ckpt"))
            return path
        setting_number += 1

class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, criterion, num_epochs,
                 checkpoint_dir, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs

        self.device = device
    
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, "logs"))

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            self.model.train()

            loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}", leave=False, dynamic_ncols=True)
            for img, mask in loop:
                img, mask = img.to(self.device), mask.to(self.device)
                out = self.model(img)
                loss = self.criterion(out, mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.train_loader)
            self.writer.add_scalar("Train/Loss", avg_loss, epoch)

            val_loss, val_iou = Validator.validate(self.model, self.val_loader, self.criterion, self.device, self.writer, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "ckpt", "best_model.pth"))
                
            print(f"EPOCH[{epoch+1}/{self.num_epochs}]: Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val_IOU: {val_iou:.4f} ")


class Validator:    
    @staticmethod
    def validate(model, val_loader, criterion, device, writer, epoch):
        model.eval()
        total_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False, dynamic_ncols=True)

            for img, mask in loop:
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                loss = criterion(out, mask)

                iou = compute_iou(out, mask)

                total_loss += loss.item()
                total_iou += iou
                loop.set_postfix({"Loss": f"{loss.item():.4f}", "IoU": f"{iou:.4f}"})

        avg_loss = total_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)

        writer.add_scalar("Val/Loss", avg_loss, epoch)
        writer.add_scalar("Val/IoU", avg_iou, epoch)

        return avg_loss, avg_iou


class Testor:
    def __init__(self, model, test_loader, criterion, device):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.num_classes = 2

    def test(self):
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for img, mask in tqdm(self.test_loader, desc="Testing", dynamic_ncols=True):
                img, mask = img.to(self.device), mask.to(self.device)
                out = self.model(img)
                loss = self.criterion(out, mask)
                iou = compute_iou(out, mask, num_classes=self.num_classes)

                total_loss += loss.item()
                total_iou += iou

        avg_loss = total_loss / len(self.test_loader)
        avg_iou = total_iou / len(self.test_loader)

        print(f"[Test] Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")
        return avg_loss, avg_iou