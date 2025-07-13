import torch

def compute_iou(pred, target, num_classes=2):
    """
    Compute mean IoU (Intersection over Union)
    Args:
        pred: Tensor of shape [B, C, H, W] (logits or probabilities)
        target: Tensor of shape [B, H, W] (ground-truth class indices)
        num_classes: number of classes
    Returns:
        mean_iou: average IoU over classes (excluding background if needed)
    """
    with torch.no_grad():
        pred = torch.argmax(pred, dim=1)  # shape: [B, H, W]

        ious = []
        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)

            intersection = (pred_inds & target_inds).sum().item()
            union = (pred_inds | target_inds).sum().item()

            if union == 0:
                ious.append(float("nan"))  # ignore class if no presence
            else:
                ious.append(intersection / union)

        # nanmean to ignore classes not present in both pred and target
        valid_ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
        return sum(valid_ious) / len(valid_ious) if valid_ious else 0.0