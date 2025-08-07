import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset.loader import dataset_loader
from models.unet import UNet
from utils.metrics import compute_iou

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        # MPS doesn't need deterministic/benchmark settings
        pass
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# ===== 사용자 설정 =====
CHECKPOINT_PATH = "checkpoints/setting_#1/ckpt/best_model.pth"  # 저장된 best.pth 경로
DATA_DIR = "./dataset/data/ISIC"  # images/, masks/ 가 들어있는 경로
BATCH_SIZE = 1
IMAGE_SIZE = [256, 256]
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
OUT_CHANNELS = 2  # segmentation 클래스 개수
NUM_CLASSES = 2   # IoU 계산 시 클래스 개수
# =====================

# 1. 데이터 로더 (여기서는 val_loader 사용)
train_loader, val_loader, test_loader = dataset_loader(DATA_DIR, BATCH_SIZE, IMAGE_SIZE, num_workers=0)

# 2. 모델 로드
model = UNet(in_ch=3, out_ch=OUT_CHANNELS).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# 3. 손실 함수
criterion = nn.CrossEntropyLoss()

# 4. 평가
total_loss = 0.0
total_iou = 0.0

with torch.no_grad():
    for idx, (img, mask) in enumerate(tqdm(test_loader, desc="Testing", dynamic_ncols=True)):
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        # forward
        out = model(img)
        loss = criterion(out, mask)
        iou = compute_iou(out, mask, num_classes=NUM_CLASSES)

        total_loss += loss.item()
        total_iou += iou

        # 예시 3개만 시각화
        if 40< idx < 50:
            pred_mask = torch.argmax(out, dim=1).squeeze().cpu().numpy()
            true_mask = mask.squeeze().cpu().numpy()
            original_img = img.squeeze().permute(1, 2, 0).cpu().numpy()

            # 시각화
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(original_img)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(true_mask, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Predicted Mask")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

avg_loss = total_loss / len(val_loader)
avg_iou = total_iou / len(val_loader)

print(f"[Test Results] Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")