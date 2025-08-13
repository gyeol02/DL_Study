import torch
import torch.nn as nn
from model.Attention import Multi_Head_Self_Attention

class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)

        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):    # (B, C, H, W) → (B, D, N) → (B, N, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):

        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() # keep:1 / drop:0
        return x / keep_prob * random_tensor


class Encoder_Block(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        # 1) Pre-LN
        self.norm1 = nn.LayerNorm(d_model)

        # 2) MSA
        self.attn  = Multi_Head_Self_Attention(d_model, num_heads, attn_drop=attn_drop, proj_drop=drop)

        # 3) Dropout Path
        self.drop_path1 = DropPath(drop_path)

        # 4) Pre-LN
        self.norm2 = nn.LayerNorm(d_model)

        # 5) MLP
        self.mlp   = MLP(d_model, mlp_ratio, drop)

        # 6) Dropout Path
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x): # (B, N, D) → (B, N, D)
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()

        # 0) Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 1) [CLS] Token -> (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))

        # 2) Position Embedding, (N + 1 ([CLS] Token)) -> (1, N+1, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))

        # 3) Token Dropout
        self.pos_drop  = nn.Dropout(drop_rate)

        # 4) Stochastic Depth Schedule
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        # 5) Transformer Encoder
        self.blocks = nn.ModuleList([Encoder_Block(embed_dim, num_heads, mlp_ratio, drop_rate,
                                           attn_drop_rate, dpr[i]) for i in range(depth)])
        
        # 6) Layer Norm (To stablize the final output in a Pre-LN structure )
        self.norm = nn.LayerNorm(embed_dim)

        # 7) Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # -- Initialize Parameter --
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d): 
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:,0])