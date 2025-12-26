import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils.scheduler import get_transformer_scheduler
from utils.config import parse_config
from utils.utils import set_seed, create_experiment_folders, save_config
from utils.trainer import run_epoch
from dataloaders.loader import get_dataloaders
from model import get_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    if len(sys.argv) < 2 or not sys.argv[1].endswith(".json"):
        print("Usage: python train.py configs/config_DenseNet.json")
        return
    
    config = parse_config(sys.argv[1])
    print(f"Config Loaded: {config.model_name}")

    set_seed(config.seed)

    log_dir, ckpt_dir, exp_dir = create_experiment_folders(config.save_dir, config.model_name)
    
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device(config.device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        data_dir=config.data_dir,
        tokenizer_name=config.tokenizer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    vocab_size = tokenizer.vocab_size
    config.model_params['src_vocab_size'] = vocab_size
    config.model_params['tgt_vocab_size'] = vocab_size
    # Padding Token ID 확인
    # pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    save_config(config, exp_dir)

    print(f"Model: {config.model_name} (Vocab: {vocab_size})")
    try:
        model = get_model(config.model_name, **config.model_params).to(device)
    except Exception as e:
        print(f"Model Error: {e}")
        raise e

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scheduler = get_transformer_scheduler(optimizer, d_model=512, warmup_steps=4000)

    best_loss = float('inf') # NLP는 보통 Loss나 BLEU로 판단 (여기선 Loss 기준)
    patience = 0
    best_model_path = os.path.join(ckpt_dir, "best.pth")

    print("\nStart Training...")
    for epoch in range(1, config.num_epochs + 1):
        start = time.time()
        
        train_loss, train_acc = run_epoch("train", model, train_loader, criterion, optimizer, device, epoch, scheduler, writer)
        val_loss, val_acc = run_epoch("val", model, val_loader, criterion, optimizer, device, epoch, scheduler, writer)

        duration = time.time() - start
        mins, secs = divmod(int(duration), 60)

        print(f"Ep {epoch:03d}/{config.num_epochs} | ⏳ {mins}m {secs}s | "
              f"Train: Loss={train_loss:.4f} Acc={train_acc:.2%} | "
              f"Val: Loss={val_loss:.4f} Acc={val_acc:.2%}")

        # Checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✅ Best Loss Saved! ({best_loss:.4f})")
        else:
            patience += 1
            if config.early_stopping and patience >= config.early_stopping:
                print(f"⏹️  Early Stopping at Epoch {epoch}")
                break
    
    # Evaluation
    print("="*10)
    print("\nEvaluating Best Model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = run_epoch("test", model, test_loader, criterion, optimizer, device, epoch, writer=None)
    print(f"Final Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")
    
    writer.close()

if __name__ == "__main__":
    main()