import torch
from tqdm import tqdm

def run_epoch(mode, model, loader, criterion, optimizer, device, epoch, writer=None):
    
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    # Train일 때만 Gradient 계산
    with torch.set_grad_enabled(is_train):
        loop = tqdm(loader, desc=f"{mode.upper()} Ep {epoch}", leave=False, dynamic_ncols=True)
        
        for img, label in loop:
            img, label = img.to(device), label.to(device)

            out = model(img)
            loss = criterion(out, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, pred = torch.topk(out, 1, dim=1)
            total_correct += (pred.squeeze(1) == label).sum().item()
            total += label.size(0)
            total_loss += loss.item()

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(total_correct/total):.2%}")

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total

    if writer:
        writer.add_scalar(f"{mode.capitalize()}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{mode.capitalize()}/Accuracy", avg_acc, epoch)
        if is_train:
            writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)

    return avg_loss, avg_acc