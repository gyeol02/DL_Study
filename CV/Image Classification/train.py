import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils.config import parse_config
from utils.utils import set_seed, create_experiment_folders, save_config
from utils.trainer import run_epoch
from dataloaders.loader import get_dataloaders
from models import get_model

def main():

    if len(sys.argv) < 2 or not sys.argv[1].endswith(".json"):
        print("Usage: python train.py configs/config_DenseNet.json")
        return
    
    config = parse_config(sys.argv[1])
    print(f"Config Loaded: {config.model_name}")

    set_seed(config.seed)

    log_dir, ckpt_dir, exp_dir = create_experiment_folders(config.save_dir, config.model_name)
    save_config(config, exp_dir)
    
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device(config.device)
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir, 
        batch_size=config.batch_size, 
        val_split=config.val_split, 
        num_workers=config.num_workers,
        seed=config.seed
    )

    print(f"Model: {config.model_name} (Params: {config.model_params})")
    try:
        model = get_model(config.model_name, **config.model_params).to(device)
    except Exception as e:
        print(f"Model Error: {e}")
        return

    criterion = nn.CrossEntropyLoss()
    if config.opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    best_acc = 0.0
    patience = 0
    best_model_path = os.path.join(ckpt_dir, "best.pth")

    print("\nStart Training...")
    for epoch in range(1, config.num_epochs + 1):
        start = time.time()
        
        train_loss, train_acc = run_epoch("train", model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, val_acc = run_epoch("val", model, val_loader, criterion, optimizer, device, epoch, writer)

        duration = time.time() - start
        mins, secs = divmod(int(duration), 60)

        print(f"Ep {epoch:03d}/{config.num_epochs} | ⏳ {mins}m {secs}s | "
              f"Train: Loss={train_loss:.4f} Acc={train_acc:.2%} | "
              f"Val: Loss={val_loss:.4f} Acc={val_acc:.2%}")

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✅ Best Saved! ({best_acc:.2%})")
        else:
            patience += 1
            if config.early_stopping and patience >= config.early_stopping:
                print(f"⏹️  Early Stopping at Epoch {epoch}")
                break
    
    print("="*10)
    print("\nEvaluating Best Model...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc = run_epoch("test", model, test_loader, criterion, optimizer, device, epoch, writer=None)
    print(f"Final Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")
    
    writer.close()

if __name__ == "__main__":
    main()