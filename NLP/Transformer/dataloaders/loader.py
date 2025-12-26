from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

def get_dataloaders(data_dir = "./dataset/data", tokenizer_name = "t5-small",
                   max_length = 128, batch_size = 32, num_workers=2):
    """
    Args:
        data_dir (str): 데이터셋 경로
        tokenizer_name (str): 사전 학습된 tokenizer 이름
        split (str): 분할 유형 ('train', 'validation', 'test')
        max_length (int): tokenize 최대 token 길이
        batch_size (int): 배치 크기
        num_workers (int): 데이터 로딩에 사용할 CPU 코어 수
    """

    # 데이터셋 로드
    print("[Data] Loading WMT14 dataset...")
    train_dataset = load_dataset("wmt14", "fr-en", split="train[:20000]", cache_dir=data_dir)
    val_dataset = load_dataset("wmt14", "fr-en", split="validation[:1000]", cache_dir=data_dir)
    test_dataset = load_dataset("wmt14", "fr-en", split="test", cache_dir=data_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 전처리 함수
    def preprocess(example):
        en_sentences = [item["en"] for item in example["translation"]]
        fr_sentences = [item["fr"] for item in example["translation"]]

        model_input = tokenizer(en_sentences, truncation=True, padding="max_length", max_length=max_length)
        model_output = tokenizer(fr_sentences, truncation=True, padding="max_length", max_length=max_length)

        model_input["labels"] = model_output["input_ids"]
        return model_input

    # 데이터 전처리
    print("[Data] Tokenizing train/val/test datasets...")
    train_data = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    val_data = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)
    test_data = test_dataset.map(preprocess, batched=True, remove_columns=test_dataset.column_names)

    # DataLoader
    print("[Data] Creating DataLoaders...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=default_data_collator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=default_data_collator)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=default_data_collator)
    
    return train_loader, val_loader, test_loader, tokenizer