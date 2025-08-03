import os
import torch
from torch.utils.tensorboard import SummaryWriter
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
                 optimizer, scheduler, criterion, num_epochs, 
                 checkpoint_dir, device='cpu'):
        self.model = model.to(device)
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
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
            for batch in loop:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=labels)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).float().sum().item()
                total += labels.numel()

                loop.set_postfix({"Loss": loss.item(), "Acc": f"{correct/total:.2%}"})

            epoch_loss = total_loss / len(self.train_loader)
            epoch_acc = correct / total
            self.writer.add_scalar("Train/Loss", epoch_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", epoch_acc, epoch)

            val_loss, val_acc = Validator.validate(self.model, self.val_loader, self.criterion, self.device, self.writer, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "ckpt", "best_model.pth"))

            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "ckpt", "final_model.pth"))


class Validator:
    @staticmethod
    def validate(model, val_loader, criterion, device, writer, epoch):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        loop = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).float().sum().item()
                total += labels.numel()

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
            for batch in self.test_loader:
                src, tgt = batch['src'].to(self.device), batch['tgt'].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                pred_logits = self.model(src, tgt_input)  # (B, T, vocab_size)

                loss = self.criterion(pred_logits.view(-1, pred_logits.size(-1)), tgt_output.reshape(-1))
                total_loss += loss.item()

                pred_ids = pred_logits.argmax(dim=-1)  # (B, T)
                predictions.extend(pred_ids.cpu().tolist())
                references.extend(tgt_output.cpu().tolist())

        avg_loss = total_loss / len(self.test_loader)
        print(f"[TransformerTester] Average Test Loss: {avg_loss:.4f}")
        return avg_loss, predictions, references

    def decode_and_show(self, predictions, references, idx=0):
        pred_tokens = self.tokenizer.decode(predictions[idx], skip_special_tokens=True)
        ref_tokens = self.tokenizer.decode(references[idx], skip_special_tokens=True)

        print(f"[Example {idx}]")
        print(f"Prediction : {pred_tokens}")
        print(f"Reference  : {ref_tokens}")
