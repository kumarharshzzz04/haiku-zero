"""Module 5 - causal multi-head attention, hand-rolled (the learning goal).

No nn.MultiheadAttention: explicit Q/K/V projections, explicit scaled dot product,
explicit causal mask, explicit head concat.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, seq_length: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)          # dropout on the attention output
        mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, seq_length, seq_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(~self.causal_mask[:, :, :t, :t], float("-inf"))
        att = torch.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.drop(self.proj(y))
