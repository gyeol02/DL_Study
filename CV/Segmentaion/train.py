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
        print("Usage: python train.py configs/config_unet.json")
        return
    
    config = parse_config(sys.argv[1])
    print(f"Config Loaded: {config.model_name}")

    set_seed(config.seed)

    log_dir, ckpt_dir, exp_dir = create_experiment_folders(config.save_dir, config.model_name)
    save_config(config, exp_dir)
    
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device(config.device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
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
    
    if config.opt_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    best_iou = 0.0
    patience = 0
    best_model_path = os.path.join(ckpt_dir, "best.pth")

    print("\nStart Training...")
    for epoch in range(1, config.num_epochs + 1):
        start = time.time()
        
        train_loss, train_iou = run_epoch("train", model, train_loader, criterion, optimizer, device, epoch, writer)
        
        if val_loader:
            val_loss, val_iou = run_epoch("val", model, val_loader, criterion, optimizer, device, epoch, writer)
        else:
            val_loss, val_iou = 0.0, 0.0

        duration = time.time() - start
        mins, secs = divmod(int(duration), 60)

        print(f"Ep {epoch:03d}/{config.num_epochs} | ⏳ {mins}m {secs}s | "
              f"Train: Loss={train_loss:.4f} IoU={train_iou:.4f} | "
              f"Val: Loss={val_loss:.4f} IoU={val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            patience = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✅ Best Saved! (IoU: {best_iou:.4f})")
        else:
            patience += 1
            if config.early_stopping and patience >= config.early_stopping:
                print(f"⏹️  Early Stopping at Epoch {epoch}")
                break
    
    print("="*10)
    print("\nEvaluating Best Model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_loss, test_iou = run_epoch("test", model, test_loader, criterion, optimizer, device, epoch, writer=None)
    print(f"Final Test Loss: {test_loss:.4f} | mIoU: {test_iou:.4f}")
    
    writer.close()

if __name__ == "__main__":
    main()