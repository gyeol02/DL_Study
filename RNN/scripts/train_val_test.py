import os
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm


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
                 checkpoint_dir, device='cpu'):
        self.model = model.to(device)
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
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                loop.set_postfix({"Loss": loss.item(), "Acc": f"{correct/total:.2%}"})

            avg_loss = total_loss / len(self.train_loader)
            avg_acc = correct / total
            self.writer.add_scalar("Train/Loss", avg_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", avg_acc, epoch)

            val_loss, val_acc = Validator.validate(self.model, self.val_loader, self.criterion, self.device, self.writer, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "ckpt", "best_model.pth"))

            print(f"EPOCH[{epoch+1}/{self.num_epochs}]: Train ACC: {avg_acc:.4f} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")


class Validator:
    @staticmethod
    def validate(model, val_loader, criterion, device, writer, epoch):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        loop = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                loop.set_postfix({"Loss": loss.item(), "Acc": f"{correct/total:.2%}"})

        val_loss = total_loss / len(val_loader)
        val_acc = correct / total

        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)

        return val_loss, val_acc
    
class Tester:
    def __init__(self, model, test_loader, criterion, tokenizer, device=None, checkpoint_dir="./checkpoints"):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_dir = create_checkpoint_folder(base_path=checkpoint_dir)
        self.model.to(self.device)

    def load_weights(self, weight_path=None):
        if weight_path is None:
            weight_path = os.path.join(self.checkpoint_dir, "ckpt", "best_model.pth")
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"No checkpoint found at {weight_path}")
        
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        print(f"[Tester] Loaded weights from: {weight_path}")

    def test(self):
        print("[TransformerTester] Start Testing...")
        self.model.eval()
        total_loss = 0.0
        predictions = []
        references = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total
        print(f"[Tester] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
        return avg_loss, accuracy