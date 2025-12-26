import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Sub-Modules (Attention, FFN, Norm, PE)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, Seq_Len, D)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # scores: (B, H, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
            # 0인 위치(padding)에 -inf 마스킹
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

    def forward(self, x, mask=None, kv=None):
        batch_size, seq_len, _ = x.size()
        
        # Self Attention이면 kv=x, Cross Attention이면 kv=EncoderOutput
        if kv is None:
            kv = x

        # Linear Projections & Split Heads
        # (B, L, H, D_k) -> (B, H, L, D_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        kv_len = kv.size(1)
        K = self.W_k(kv).view(batch_size, kv_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kv).view(batch_size, kv_len, self.num_heads, self.d_k).transpose(1, 2)
        
        out, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concat Heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(out)

class FFNBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))


# Encoder / Decoder Blocks
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = FFNBlock(d_model, d_ff, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        x = self.addnorm1(x, attn_out)
        ffn_out = self.ffn(x)                    
        x = self.addnorm2(x, ffn_out)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = FFNBlock(d_model, d_ff, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # 1. Masked Self Attention
        self_attn_out = self.self_attn(x, tgt_mask)              
        x = self.addnorm1(x, self_attn_out)

        # 2. Cross Attention (Query=Decoder, Key/Val=Encoder)
        cross_attn_out = self.cross_attn(x, memory_mask, kv=enc_output)
        x = self.addnorm2(x, cross_attn_out)

        # 3. FFN
        ffn_out = self.ffn(x)
        x = self.addnorm3(x, ffn_out)
        return x


# Encoder / Decoder Stacks
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return x

# Main Transformer Model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8, d_ff=2048,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dropout=0.1, max_len=5000, **kwargs):
        super().__init__()
        
        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stacks (Encoder & Decoder Depth Separation)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_decoder_layers, dropout)

        # Output Projection
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # Initialize Weights (Xavier)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # src: (B, L) -> Pad Token(0) Masking
        # (B, 1, 1, L) 형태로 확장하여 Attention Score에 Broadcasting 가능하게 함
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        # tgt: (B, L)
        # 1. Pad Mask
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
        
        # 2. Look Ahead Mask (Triangular)
        seq_len = tgt.size(1)
        look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len)).type_as(tgt).bool() # (L, L)
        
        return pad_mask & look_ahead_mask # (B, 1, L, L)

    def forward(self, input_ids, labels=None, **kwargs):
        """
        Args:
            input_ids: (B, Src_Len) - Encoder Input
            labels: (B, Tgt_Len) - Decoder Input + Target
        """
        # Training 시에는 labels가 필수 (Teacher Forcing)
        if labels is None:
            raise ValueError("Labels are required for training forward pass.")

        # Decoder Input: <SOS> ... <Last Token> (Right Shifted)
        # 여기서는 간단히 Labels의 맨 뒤를 자른 것을 Decoder Input으로 가정 (실제로는 SOS 토큰 처리 필요)
        # 하지만 T5 Tokenizer 등은 이미 처리가 되어 있을 수 있으므로, 
        # 가장 일반적인 방식: Decoder Input = Labels[:, :-1]
        decoder_input = labels[:, :-1]
        
        # Masks 생성
        src_mask = self.make_src_mask(input_ids)
        tgt_mask = self.make_tgt_mask(decoder_input)

        # Embedding
        src_emb = self.pos_encoding(self.src_embedding(input_ids) * (self.d_model ** 0.5))
        tgt_emb = self.pos_encoding(self.tgt_embedding(decoder_input) * (self.d_model ** 0.5))

        # Encoder
        enc_output = self.encoder(src_emb, src_mask)

        # Decoder
        # memory_mask는 보통 src_mask와 동일 (Encoder의 Pad를 보지 않기 위함)
        dec_output = self.decoder(tgt_emb, enc_output, tgt_mask, memory_mask=src_mask)

        # Projection
        logits =self.output_layer(dec_output) 
        return logits