import torch
import torch.nn as nn
import torch.nn.functional as F
import math

""" Attention """
class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, H, L, L)

        if mask is not None:
            # mask: (B, L) → (B, 1, 1, L)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))  # broadcast to (B, H, L, L)

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

    def forward(self, x, mask=None, kv=None):
        batch_size, seq_len, _ = x.size()

        if kv is None:
            kv = x

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(kv).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kv).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        out, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(out)