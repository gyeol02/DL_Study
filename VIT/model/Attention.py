import torch
import torch.nn as nn

class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, d_model, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(attn_drop)  
        self.proj_drop = nn.Dropout(proj_drop)

    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        output = torch.matmul(attn, V)
        return output, attn

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        
        # (B,N,D) → (B, N, H, d_k) → (B, H, N, d_k)
        Q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        out, _ = self.scaled_dot_product_attention(Q, K, V)

        # (B, H, N, d_k) → (B, N, H, d_k) → (B,N,D) 
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.proj_drop(self.W_o(out))
        return out