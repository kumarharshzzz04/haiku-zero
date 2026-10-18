"""Central hyperparameters for HAIKU ZERO. Edit here, every module reads this."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- tokenizer / data (Modules 1-3) ---
    vocab_size: int = 4096            # custom BPE, matched to corpus size
    seq_length: int = 64              # never truncates a haiku, leaves prompt room
    special_tokens: tuple = ("<pad>", "<bos>", "<eos>", "<seed>")
    val_fraction: float = 0.03
    prefix_mask_prob: float = 0.5      # train on random partial poems so prompts continue cleanly
    seed_prob: float = 0.5             # train "<seed> word -> write a poem about it"
    seed: int = 1337

    # --- model (Modules 4-7) ---
    # meter state: feed "syllables so far in this line / which line this is" to the model, so
    # 5-7-5 can be learned from the data instead of enforced by the decoder. Off by default
    # because a checkpoint trained without it must load and behave identically (RUNBOOK 4c).
    meter_state: bool = False
    meter_max_syllables: int = 4       # per-token syllable cap for the state table
    embed_dim: int = 256
    num_heads: int = 8                # 32 dims per head
    num_layers: int = 6
    ffn_dim: int = 1024               # 4x embed_dim
    dropout: float = 0.1

    # --- training (Module 8) ---
    batch_size: int = 128
    epochs: int = 25
    lr: float = 3e-4                  # peak LR (warmup target)
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True                  # mixed precision (CUDA only)
    log_every: int = 25
    preview_prompt: str = "autumn moonlight"

    # --- generation (Module 9) ---
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.3     # 1.1 repeated content words after the poetry pretrain
    max_new_tokens: int = 64

    def to_dict(self) -> dict:
        return asdict(self)
