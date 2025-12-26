import torch
from tqdm import tqdm

def run_epoch(mode, model, loader, criterion, optimizer, device, epoch, scheduler, writer=None):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.set_grad_enabled(is_train):
        loop = tqdm(loader, desc=f"{mode.upper()} Ep {epoch}", leave=False, dynamic_ncols=True)
        
        for batch in loop:
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, labels=labels) 
            # logits shape: (Batch, Seq_Len, Vocab_Size)

            if logits.size(1) == labels.size(1) - 1:
                labels = labels[:, 1:]

            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Transformer 학습 안정화
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            preds = torch.argmax(logits, dim=-1) # (Batch, Seq_Len)

            mask = labels != 0 
            
            correct = (preds == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            total_loss += loss.item()

            avg_loss_step = loss.item()
            acc_step = (correct.sum().item() / mask.sum().item()) if mask.sum().item() > 0 else 0
            
            loop.set_postfix(loss=f"{avg_loss_step:.4f}", acc=f"{acc_step:.2%}")

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

    if writer:
        writer.add_scalar(f"{mode.capitalize()}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{mode.capitalize()}/Accuracy", avg_acc, epoch)
        if is_train:
            writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)

    return avg_loss, avg_acc