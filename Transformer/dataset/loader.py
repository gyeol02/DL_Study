from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

def dataset_loader(data_dir: str = "./dataset/data", 
                   tokenizer_name: str = "t5-small",
                   split: str = "train",
                   max_length: int = 128,
                   batch_size: int = 32):

    # 1. WMT14 En-Fr 데이터셋 로드 (HuggingFace에서 자동 다운로드)
    dataset = load_dataset("wmt14", "fr-en", split=split, cache_dir=data_dir)

    # 2. Tokenizer 불러오기 (여기선 T5의 tokenizer 사용 예시)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 3. 전처리 함수 정의
    def preprocess(example):
        en_sentences = [item["en"] for item in example["translation"]]
        fr_sentences = [item["fr"] for item in example["translation"]]

        model_input = tokenizer(en_sentences, truncation=True, padding="max_length", max_length=max_length)
        model_output = tokenizer(fr_sentences, truncation=True, padding="max_length", max_length=max_length)

        model_input["labels"] = model_output["input_ids"]
        return model_input

    # 4. 데이터 전처리 적용
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    # 5. PyTorch DataLoader 생성
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    return dataloader, tokenizer
