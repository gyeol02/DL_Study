import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.config import parse_config
from dataloaders.loader import get_dataloaders
from models import get_model

def test(model, test_loader, device, criterion, classes):
    
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = list(0. for _ in range(len(classes)))
    class_total = list(0. for _ in range(len(classes)))

    print(f"\nStarting Evaluation on {len(test_loader.dataset)} images...")

    with torch.no_grad(): # Gradient 계산 끔 (메모리 절약, 속도 향상)
        loop = tqdm(test_loader, desc="Testing", leave=True)
        
        for img, label in loop:
            img, label = img.to(device), label.to(device)

            # 1. Inference
            out = model(img)
            loss = criterion(out, label)

            # 2. Metric Calculation
            total_loss += loss.item()
            _, pred = torch.topk(out, 1, dim=1)
            pred = pred.squeeze()

            # 전체 정확도용
            correct += (pred == label).sum().item()
            total += label.size(0)

            # 클래스별 정확도용
            for i in range(len(label)):
                label_idx = label[i]
                pred_idx = pred[i]
                if label_idx == pred_idx:
                    class_correct[label_idx] += 1
                class_total[label_idx] += 1

    # 3. 결과 집계
    avg_loss = total_loss / len(test_loader)
    total_acc = correct / total

    print("-" * 50)
    print(f"Test Results")
    print("-" * 50)
    print(f"Loss     : {avg_loss:.4f}")
    print(f"Accuracy : {total_acc:.2%} ({correct}/{total})")
    print("-" * 50)
    
    # 4. 클래스별 정확도 출력 (분석용)
    print("📈 Class-wise Accuracy:")
    for i in range(len(classes)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"   {classes[i]:>12s} : {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"   {classes[i]:>12s} : N/A")
    print("-" * 50)

def main():
    
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument("config", type=str, help="Path to config file (e.g., configs/config_VGGNet.json)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    config = parse_config(args.config)
    
    device = torch.device(config.device)
    print(f"Device: {device}")

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    print("Loading Test Data...")
    _, _, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    print(f"Building Model: {config.model_name}")
    try:
        model = get_model(config.model_name, **config.model_params).to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file not found: {args.checkpoint}")
        return
    
    print(f"Loading Weights from: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    criterion = nn.CrossEntropyLoss()
    test(model, test_loader, device, criterion, classes)

if __name__ == "__main__":
    main()