import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Helper function to create sinusoidal positional embeddings
def get_positional_embeddings(sequence_length, d_model):
    result = torch.ones(sequence_length, d_model)
    for i in range(sequence_length):
        for j in range(d_model // 2):
            result[i, 2*j] = torch.sin(i / (10000 ** (2 * j / d_model)))
            result[i, 2*j+1] = torch.cos(i / (10000 ** (2 * j / d_model)))
    return result

# Multi-head self-attention layer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3*d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b s (qkv h d) -> qkv b h s d', qkv=3, h=self.num_heads)
        
        attn_weights = torch.einsum('bhqd,bhkd->bhqk', q, k) / self.head_dim ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.einsum('bhqk,bhvd->bhqd', attn_weights, v)
        out = rearrange(out, 'b h s d -> b s (h d)')
        out = self.out_proj(out)
        
        return out

# MLP block
class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Transformer encoder block with residual connections
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = residual + x

        return x

# ViT model
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.patch_embedding = nn.Linear(patch_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        patches = self.patch_embedding(patches)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, patches), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.fc(cls_output)
        
        return logits