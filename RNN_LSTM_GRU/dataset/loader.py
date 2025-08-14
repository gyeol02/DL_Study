import torch
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader, random_split
from utils.preprocess import TextPreprocessor

def dataset_loader(data_dir, batch_size, max_vocab_size, val_ratio=0.2):
    train_iter, test_iter = IMDB(root=data_dir, split=('train', 'test'))
    train_data = list(train_iter)
    test_data = list(test_iter)

    train_data = [(text, label) for label, text in train_iter]
    test_data = [(text, label) for label, text in test_iter]
    

    preprocessor = TextPreprocessor(max_vocab_size=max_vocab_size)
    preprocessor.build_vocab([text for text, _ in train_data])

    val_size = int(len(train_data) * val_ratio)
    train_size = len(train_data) - val_size
    train_subset, val_subset = random_split(train_data, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=preprocessor.collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=preprocessor.collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=preprocessor.collate_fn)

    return train_loader, val_loader,test_loader, preprocessor