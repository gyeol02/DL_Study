import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.Attention import Multi_Head_Attention

""" Encoder & Decoder Stack"""
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            Encoder_Block(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        x    : Embedded source input (batch_size, src_len, d_model)
        mask : Optional self-attention mask for encoder
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        x           : Embedded target input (batch_size, tgt_len, d_model)
        enc_output  : Output from encoder (batch_size, src_len, d_model)
        tgt_mask    : Mask for target self-attention
        memory_mask : Mask for encoder-decoder attention
        """
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return x

""" Encoder & Decoder Block"""
class Encoder_Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 1. Self-attention
        self.self_attn = Multi_Head_Attention(d_model, num_heads)
        self.addnorm1 = Add_Norm(d_model, dropout)

        # 2. Feed Forward Network
        self.ffn = FFN_Block(d_model, d_ff, dropout)
        self.addnorm2 = Add_Norm(d_model, dropout)

    def forward(self, x, mask=None):
        """
        x    : Embedded source input (batch_size, src_len, d_model)
        mask : Optional self-attention mask for encoder
        """
        # 1. Self-attention
        attn_output = self.self_attn(x, mask)
        x = self.addnorm1(x, attn_output)

        # 2. Feed Forward
        ffn_output = self.ffn(x)                    
        x = self.addnorm2(x, ffn_output)

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 1. Masked Self-Attention
        self.self_attn = Multi_Head_Attention(d_model, num_heads)
        self.addnorm1 = Add_Norm(d_model, dropout)

        # 2. Encoder-Decoder Cross Attention
        self.cross_attn = Multi_Head_Attention(d_model, num_heads)
        self.addnorm2 = Add_Norm(d_model, dropout)

        # 3. Feed Forward Network
        self.ffn = FFN_Block(d_model, d_ff, dropout)
        self.addnorm3 = Add_Norm(d_model, dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        x          : Decoder input (batch_size, tgt_len, d_model)
        enc_output : Encoder output (batch_size, src_len, d_model)
        tgt_mask   : Mask for decoder self-attention (to prevent looking ahead)
        memory_mask: Optional mask for encoder-decoder attention
        """
        # 1. Masked Self-Attention
        self_attn_out = self.self_attn(x, tgt_mask)              
        x = self.addnorm1(x, self_attn_out)

        # 2. Cross Attention (queries = decoder x, keys/values = encoder output)
        cross_attn_out = self.cross_attn(x, memory_mask, kv=enc_output)
        x = self.addnorm2(x, cross_attn_out)

        # 3. Feed Forward
        ffn_out = self.ffn(x)
        x = self.addnorm3(x, ffn_out)

        return x

""" FFN & ADD & NORM"""
class FFN_Block(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x               : (batch_size, seq_len, d_model)
        """
        out = self.linear1(x)          # (batch_size, seq_len, d_ff)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)        # (batch_size, seq_len, d_model)
        out = self.dropout(out)
        return out

class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        """
        x               : input (Residual)
        sublayer_output : sublayer (ex. MHA or FFN) output
        """
        return self.norm(x + self.dropout(sublayer_output))