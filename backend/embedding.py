"""Module 4 - token embedding + learned positional embedding."""
from __future__ import annotations

import torch
import torch.nn as nn


class TokenPosEmbedding(nn.Module):
    def __init__(self, vocab_size: int, seq_length: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.seq_length = seq_length
        self.token = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(seq_length, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        if t > self.seq_length:
            raise ValueError(f"sequence of {t} > seq_length {self.seq_length}")
        pos = torch.arange(t, device=idx.device)
        return self.drop(self.token(idx) + self.pos(pos))
