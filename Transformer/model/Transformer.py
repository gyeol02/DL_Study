import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.Encoder_Decoder import Encoder, Decoder
#from utils.mask import create_padding_mask, create_look_ahead_mask

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

    def forward(self, input_ids, decoder_input_ids, attention_mask=None, decoder_attention_mask=None, labels=None):
        """
        src: (batch, src_len) - input_ids
        tgt: (batch, tgt_len) - decoder_input_ids
        attention_mask: (batch, src_len) - encoder mask
        decoder_attention_mask: (batch, tgt_len) - decoder mask (look-ahead mask는 외부에서 넣는다고 가정)
        """

        # 1. Embedding + PositionalEncoding
        src_emb = self.pos_encoding(self.src_embedding(input_ids) * (self.d_model ** 0.5))  # (B, src_len, D)
        tgt_emb = self.pos_encoding(self.tgt_embedding(decoder_input_ids) * (self.d_model ** 0.5))  # (B, tgt_len, D)

        # 2. Encoder
        memory = self.encoder(src_emb, attention_mask)  # (B, src_len, D)

        # 3. Decoder
        out = self.decoder(tgt_emb, memory, decoder_attention_mask, attention_mask)  # (B, tgt_len, D)

        # 4. Output Projection
        logits = self.output_layer(out)  # (B, tgt_len, vocab_size)
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