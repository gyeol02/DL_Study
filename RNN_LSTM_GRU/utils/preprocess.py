import torch
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter


class TextPreprocessor:
    def __init__(self, tokenizer_name="basic_english", max_vocab_size=25000):
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.max_vocab_size = max_vocab_size
        self.vocab = None
        self.pad_idx = None
        self.unk_idx = None

    def build_vocab(self, texts):
        # 토큰 생성기
        def yield_tokens(texts):
            for text in texts:
                yield self.tokenizer(str(text))

        self.vocab = build_vocab_from_iterator(
            yield_tokens(texts),
            specials=['<pad>', '<unk>'],
            max_tokens=self.max_vocab_size
        )

        self.vocab.set_default_index(self.vocab['<unk>'])

        self.pad_idx = self.vocab['<pad>']
        self.unk_idx = self.vocab['<unk>']

    def encode(self, text):
        return [self.vocab[token] for token in self.tokenizer(str(text))]

    def collate_fn(self, batch):
        texts, labels = zip(*batch)
        encoded = [torch.tensor(self.encode(text), dtype=torch.long) for text in texts]
        padded = pad_sequence(encoded, batch_first=True, padding_value=self.pad_idx)
        labels = torch.tensor([1 if label == 'pos' else 0 for label in labels], dtype=torch.long)
        return padded, labels