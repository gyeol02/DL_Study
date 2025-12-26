import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

# 기존 모듈 임포트
from utils.config import parse_config
from utils.utils import set_seed
from dataloaders.loader import get_dataloaders
from model import get_model

# MPS(Mac) 가속 지원
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_model(config_path, checkpoint_path=None):
    # 1. 설정 로드
    config = parse_config(config_path)
    device = torch.device(config.device if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
    print(f"🔹 Config Loaded: {config.model_name}")
    print(f"🔹 Device: {device}")

    set_seed(config.seed)

    # 2. 데이터 로더 준비
    _, _, test_loader, tokenizer = get_dataloaders(
        data_dir=config.data_dir,
        tokenizer_name=config.tokenizer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    # 3. 모델 초기화
    vocab_size = tokenizer.vocab_size
    config.model_params['src_vocab_size'] = vocab_size
    config.model_params['tgt_vocab_size'] = vocab_size
    
    print(f"🔹 Model Structure: {config.model_name} (Vocab: {vocab_size})")
    model = get_model(config.model_name, **config.model_params).to(device)

    # 4. 가중치 로드 (Checkpoint)
    if checkpoint_path is None:
        checkpoint_path = os.path.join("experiments", config.model_name, "checkpoints", "best.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"🔹 Loading Checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return

    # 5. 평가 설정 (BLEU 제거됨)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    # 샘플 출력을 위한 리스트
    samples = []

    print("\nStart Testing...")
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing", leave=True)
        for batch_idx, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, labels=labels)
            
            if logits.size(1) == labels.size(1) - 1:
                labels = labels[:, 1:]

            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            
            mask = labels != 0
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

            if batch_idx == 0:
                clean_labels = labels.clone()
                clean_labels[clean_labels == -100] = tokenizer.pad_token_id

                src_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
                tgt_texts = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)

                for i in range(min(3, len(src_texts))):
                    samples.append({
                        "src": src_texts[i],
                        "tgt": tgt_texts[i],
                        "pred": pred_texts[i]
                    })

    avg_loss = total_loss / len(test_loader)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0

    print("\n" + "="*30)
    print(f"Final Test Results")
    print(f"   • Loss : {avg_loss:.4f}")
    print(f"   • Acc  : {avg_acc:.2%}")
    print("="*30)

    print("\nSample Predictions (Top 3):")
    for i, sample in enumerate(samples):
        print(f"\n[Sample {i+1}]")
        print(f"   Input : {sample['src']}")
        print(f"   Target: {sample['tgt']}")
        print(f"   Pred  : {sample['pred']}")
        print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py configs/config.json [path/to/checkpoint.pth]")
    else:
        cfg_path = sys.argv[1]
        ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None
        test_model(cfg_path, ckpt_path)