"""Module 6 - Pre-LN transformer block."""
from __future__ import annotations

import torch
import torch.nn as nn

from backend.multi_head_attention import CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act = nn.GELU()                     # GELU, not ReLU (PLAN.md)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, seq_length: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, seq_length, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))           # sub-layer 1: LN -> attention -> residual
        x = x + self.mlp(self.ln2(x))            # sub-layer 2: LN -> FFN -> residual
        return x
