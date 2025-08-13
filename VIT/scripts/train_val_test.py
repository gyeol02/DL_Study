import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
                 optimizer, scheduler, criterion, num_epochs,
                 checkpoint_dir, device):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs

        self.device = device

        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, "logs"))

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            self.model.train()

            loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}", leave=False, dynamic_ncols=True)
            for img, label in loop:
                img, label = img.to(self.device), label.to(self.device)
                out = self.model(img)
                loss = self.criterion(out, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                _, pred = torch.topk(out, 1, dim=1)
                pred = pred.squeeze(1)

                correct += (pred == label).sum().item()
                total += label.size(0)
                total_loss += loss.item()

                loop.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{(correct / total):.2%}"
                })

            avg_loss = total_loss / len(self.train_loader)
            avg_acc = correct / total
            
            self.writer.add_scalar("Train/Loss", avg_loss, epoch)
            self.writer.add_scalar("Train/Acc", avg_acc, epoch)

            val_loss, val_acc = Validator.validate(self.model, self.val_loader, self.criterion, self.device, self.writer, epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "ckpt", "best_model.pth"))
                
            print(f"EPOCH[{epoch+1}/{self.num_epochs}]: Train ACC: {avg_acc:.4f} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} |") 

class Validator:    
    @staticmethod
    def validate(model, val_loader, criterion, device, writer, epoch):
        total_loss = 0.0
        correct = 0
        total = 0
        model.eval()

        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False, dynamic_ncols=True)

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

        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total

        writer.add_scalar("Val/Loss", avg_loss, epoch)
        writer.add_scalar("Val/Acc", avg_acc, epoch)

        return avg_loss, avg_acc