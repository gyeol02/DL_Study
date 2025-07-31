import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from Transformer.model.Encoder_Decoder import Encoder, Decoder
from Transformer.utils.mask import create_padding_mask, create_look_ahead_mask

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # 1. Embedding + PositionalEncoding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = Positional_Encoding(d_model, max_len, dropout)

        # 2. Encoder & Decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        # 3. Final output projection (to vocab size)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        src: (batch, src_len) - token indices
        tgt: (batch, tgt_len) - token indices
        """

        B, src_len = src.shape
        B, tgt_len = tgt.shape

        if src_mask is None:
            src_mask = create_padding_mask(src)                               # (B, 1, 1, src_len)

        if tgt_mask is None:
            padding_mask = create_padding_mask(tgt)                           # (B, 1, 1, tgt_len)
            look_ahead_mask = create_look_ahead_mask(tgt_len).to(tgt.device)  # (tgt_len, tgt_len)
            look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)       # (1, 1, tgt_len, tgt_len)
            tgt_mask = padding_mask & look_ahead_mask                         # (B, 1, tgt_len, tgt_len)

        if memory_mask is None:
            memory_mask = create_padding_mask(src)                            # (B, 1, 1, src_len)

        # 1. Embedding + Positional Encoding
        src_emb = self.pos_encoding(self.src_embedding(src) * (self.d_model ** 0.5))  # (B, src_len, D)
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * (self.d_model ** 0.5))  # (B, tgt_len, D)

        # 2. Encoder
        memory = self.encoder(src_emb, src_mask)                              # (B, src_len, D)

        # 3. Decoder
        out = self.decoder(tgt_emb, memory, tgt_mask, memory_mask)            # (B, tgt_len, D)

        # 4. Output Projection to vocab
        logits = self.output_layer(out)                                       # (B, tgt_len, vocab_size)
        return logits
    
""" Positional Encoding"""
class Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding matrix 초기화 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)                         # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)   # 홀수 인덱스
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 학습 X, 저장은 O

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 위치 벡터를 앞 seq_len만큼 잘라서 더함
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)