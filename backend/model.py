"""Module 7 - the full HAIKU ZERO model.

embeddings -> N pre-LN blocks -> final LayerNorm -> lm_head, weights tied to the
token embedding and the head kept in float32 (the #1 mixed-precision bug).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ""):  # allow: python backend/model.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import Config  # noqa: E402
from backend.embedding import TokenPosEmbedding  # noqa: E402
from backend.meter_check import count_syllables  # noqa: E402
from backend.transformer_block import TransformerBlock  # noqa: E402

MAX_LINE_STATE = 12      # accumulated-syllable buckets; long lines clamp here, beyond any target
MAX_LINE_INDEX = 3       # seed line + three poem lines


class HaikuZeroLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = TokenPosEmbedding(cfg.vocab_size, cfg.seq_length, cfg.embed_dim, cfg.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.seq_length, cfg.ffn_dim, cfg.dropout)
            for _ in range(cfg.num_layers)
        )
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # Meter state (Config.meter_state) is additive and optional: without it the model is
        # byte-for-byte the old architecture, so every existing checkpoint loads unchanged.
        self.meter_state = bool(getattr(cfg, "meter_state", False))
        if self.meter_state:
            self.syl_emb = nn.Embedding(MAX_LINE_STATE + 1, cfg.embed_dim)
            self.line_emb = nn.Embedding(MAX_LINE_INDEX + 1, cfg.embed_dim)
            # derived from the tokenizer, never trained: a buffer that stays out of state_dict
            # so a checkpoint cannot carry a table that disagrees with its tokenizer
            self.register_buffer("syl_table", torch.zeros(cfg.vocab_size, dtype=torch.long),
                                 persistent=False)
            self.newline_id = self.eos_id = self.bos_id = -1

        self.apply(self._init_weights)
        scale = (2 * cfg.num_layers) ** -0.5      # GPT-2 residual scaling, keeps deep stacks stable
        for block in self.blocks:
            block.attn.proj.weight.data.mul_(scale)
            block.mlp.fc2.weight.data.mul_(scale)
        self.lm_head.weight = self.emb.token.weight   # weight tying: both are (vocab, dim)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_syllable_table(self, tok) -> None:
        """Fill the per-token syllable counts and the reset tokens from the tokenizer.

        A no-op when meter_state is off, so it is safe to call on any model. Multi-byte pieces
        are counted on their decoded text, which is approximate for a subword fragment - the
        accumulators that matter (5 and 7 syllables) are dominated by whole words.
        """
        if not self.meter_state:
            return
        table = []
        for i in range(self.cfg.vocab_size):
            text = tok.decode([i])
            table.append(0 if "\n" in text
                         else min(self.cfg.meter_max_syllables, count_syllables(text)))
        self.syl_table.copy_(torch.tensor(table, dtype=torch.long))
        self.newline_id = tok.encode("\n").ids[0]
        self.eos_id = tok.token_to_id("<eos>")
        self.bos_id = tok.token_to_id("<bos>")

    def _meter_state(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(syllables accumulated in the current line, line index) for every position.

        Computed from the ids alone, so training, strict decoding and free sampling all get it
        with no call-site changes. A line starts after a newline (and after <eos>/<bos>, which
        start a new poem inside a packed window); `base` is the syllable total before the
        current line, taken with a cumulative max because it is non-decreasing by construction.
        """
        resets = torch.zeros_like(idx, dtype=torch.bool)
        for tid in (self.newline_id, self.eos_id, self.bos_id):
            if tid >= 0:
                resets |= idx == tid
        table = self.syl_table.to(idx.device)
        syl = torch.where(resets, torch.zeros_like(idx), table[idx])
        total = torch.cumsum(syl, dim=1)
        starts = torch.zeros_like(resets)
        starts[:, 0] = True
        if idx.size(1) > 1:
            starts[:, 1:] = resets[:, :-1]
        line = torch.cumsum(starts.long(), dim=1) - 1
        base = torch.cummax(torch.where(starts, total - syl, torch.full_like(total, -1)),
                            dim=1).values
        acc = (total - base).clamp_(0, MAX_LINE_STATE)
        return acc, line.clamp_(max=MAX_LINE_INDEX)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if self.meter_state:
            acc, line = self._meter_state(idx)
            h = self.emb(idx) + self.syl_emb(acc) + self.line_emb(line)
        else:
            h = self.emb(idx)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.lm_head(h).float()          # float32 logits, always
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())   # tied weight counted once


if __name__ == "__main__":
    cfg = Config()
    model = HaikuZeroLM(cfg)
    n = model.num_params()
    print(f"params {n / 1e6:.2f}M | layers {cfg.num_layers} dim {cfg.embed_dim} "
          f"vocab {cfg.vocab_size} tied head")
    if n > 10e6:
        print("WARNING: >10M params - vocab or weight tying is wrong (PLAN.md)")
    logits, loss = model(torch.randint(0, cfg.vocab_size, (2, cfg.seq_length)),
                         torch.randint(0, cfg.vocab_size, (2, cfg.seq_length)))
    print(f"forward ok: logits {tuple(logits.shape)} {logits.dtype}, loss {loss.item():.3f}")
