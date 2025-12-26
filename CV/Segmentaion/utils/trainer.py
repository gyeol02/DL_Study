import torch
from tqdm import tqdm

def calculate_pixel_accuracy(pred, target):

    pred_mask = torch.argmax(pred, dim=1) # [B, H, W]
    correct = (pred_mask == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()

def compute_iou(pred, target, num_classes=2):
    
    pred_mask = torch.argmax(pred, dim=1)
    
    ious = []

    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).float().sum().item()
        union = (pred_inds | target_inds).float().sum().item()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    return sum(ious) / len(ious) if ious else 0.0

def run_epoch(mode, model, loader, criterion, optimizer, device, epoch, writer=None):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_pixel_acc = 0.0
    num_batches = len(loader)

    with torch.set_grad_enabled(is_train):
        loop = tqdm(loader, desc=f"{mode.upper()} Ep {epoch}", leave=False, dynamic_ncols=True)
        
        for batch in loop:
            images = batch[0].to(device)
            masks = batch[1].to(device) # [B, H, W]

            outputs = model(images) # [B, 2, H, W]
            loss = criterion(outputs, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # Pixel Acc & IoU
            p_acc = calculate_pixel_accuracy(outputs, masks)
            iou = compute_iou(outputs, masks, num_classes=2)
            
            total_pixel_acc += p_acc
            total_iou += iou

            loop.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    avg_pixel_acc = total_pixel_acc / num_batches

    if writer:
        writer.add_scalar(f"{mode.capitalize()}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{mode.capitalize()}/mIoU", avg_iou, epoch)
        writer.add_scalar(f"{mode.capitalize()}/Pixel_Acc", avg_pixel_acc, epoch)
        
        if is_train:
            writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], epoch)

    return avg_loss, avg_iou