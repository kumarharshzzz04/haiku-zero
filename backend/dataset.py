"""Module 3 - encode the corpus and pack it into causal-LM windows (the tf.data part).

Split happens at haiku level BEFORE packing, so train/val never share a haiku.
Windows are packed continuously (only an <eos> boundary is ever shared), which is
the standard nanoGPT-style setup.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):  # allow: python backend/dataset.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import Config  # noqa: E402
from backend.train_tokenizer import read_records  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "haikus_clean.txt"
METER_PATH = ROOT / "data" / "haikus_meter.txt"
TOKENIZER_PATH = ROOT / "tokenizer.json"


def load_tokenizer(path: Path = TOKENIZER_PATH) -> Tokenizer:
    if not Path(path).exists():
        raise SystemExit(f"{path} missing - run backend.train_tokenizer first")
    return Tokenizer.from_file(str(path))


THEME_WORD_RE = re.compile(r"[a-z]{3,}")


def theme_word(text: str, rng: random.Random) -> str | None:
    words = THEME_WORD_RE.findall(text.lower())
    return rng.choice(words) if words else None


def encode_records(records: list[str], tok: Tokenizer, prefix_mask_prob: float = 0.0,
                   seed_prob: float = 0.0, rng: random.Random | None = None,
                   max_mask: int | None = None):
    """Encode each poem as [<bos>, *tokens, <eos>].

    Returns (chunks, masks); mask True means "given context, do not train the model to
    predict it". Two augmentations share that mechanism:

    - seed conditioning: `[<bos> <seed> word \\n poem <eos>]`, so a typed prompt means
      "write a poem about this" instead of "finish this phrase" (which a tiny model reads
      as a completed line and answers with a 2-syllable fragment);
    - prefix masking: the model also learns to continue a partial poem.

    Without <bos>, a poem start is out-of-distribution and the model wastes the first line.
    """
    bos_id, eos_id = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    seed_id = tok.token_to_id("<seed>")
    nl_ids = tok.encode("\n").ids
    rng = rng or random.Random(0)
    chunks, masks = [], []
    for record in records:
        poem_ids = tok.encode(record).ids
        ids = [bos_id] + poem_ids + [eos_id]
        mask = [False] * len(ids)
        word = theme_word(record, rng) if seed_prob and rng.random() < seed_prob else None
        if word is not None:
            seed_ids = [bos_id, seed_id] + tok.encode(" " + word).ids + nl_ids
            ids = seed_ids + poem_ids + [eos_id]
            mask = [True] * len(seed_ids) + [False] * (len(poem_ids) + 1)
        elif prefix_mask_prob and rng.random() < prefix_mask_prob:
            given = rng.randrange(1, max(2, len(ids) - 1))   # at least one given token
            # A masked run must stay shorter than one window, or the window can lose
            # every target and pack() refuses to build the dataset. Haiku records are
            # ~20 tokens so this never bound; the poetry pretraining corpus produces
            # 64+ token records, where the mask could swallow an entire window.
            if max_mask is not None:
                given = min(given, max_mask)
            for i in range(1, given + 1):
                mask[i] = True
        chunks.append(ids)
        masks.append(mask)
    return chunks, masks


def split_records(records: list[str], val_fraction: float, seed: int):
    order = torch.randperm(len(records), generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = max(1, int(len(records) * val_fraction))
    val_idx = set(order[:n_val])
    train = [records[i] for i in order[n_val:]]
    val = [records[i] for i in order[:n_val]]
    return train, val


def pack(chunks: list[list[int]], seq_length: int,
         masks: list[list[bool]] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate everything, cut into non-overlapping windows of seq_length+1.

    Masked (given-context) positions become target -100, which cross entropy ignores.
    """
    flat = torch.tensor([t for chunk in chunks for t in chunk], dtype=torch.long)
    n = (flat.numel() - 1) // seq_length
    if n < 1:
        raise SystemExit("corpus too small for even one window")
    x = flat[: n * seq_length].view(n, seq_length)
    y = flat[1 : n * seq_length + 1].view(n, seq_length)
    if masks is not None:
        flat_mask = torch.tensor([m for row in masks for m in row], dtype=torch.bool)
        y = y.masked_fill(flat_mask[1 : n * seq_length + 1].view(n, seq_length), -100)
        if not (y != -100).any(dim=1).all():
            raise SystemExit("a training window lost every target - lower prefix_mask_prob")
    return x, y


class HaikuWindows(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x, self.y = x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def build_loaders(cfg: Config | None = None, corpus: Path = CLEAN_PATH,
                  tokenizer_path: Path = TOKENIZER_PATH, batch_size: int | None = None,
                  verbose: bool = True):
    cfg = cfg or Config()
    tok = load_tokenizer(tokenizer_path)
    records = read_records(corpus)
    if not records:
        raise SystemExit(f"no records in {corpus} - run backend.prepare_data first")

    train_recs, val_recs = split_records(records, cfg.val_fraction, cfg.seed)
    rng = random.Random(cfg.seed)
    train_ids, train_masks = encode_records(train_recs, tok, cfg.prefix_mask_prob, cfg.seed_prob, rng,
                                            max_mask=max(1, cfg.seq_length - 2))
    val_ids, _ = encode_records(val_recs, tok)            # val stays unmasked: honest val loss
    x_train, y_train = pack(train_ids, cfg.seq_length, train_masks)
    x_val, y_val = pack(val_ids, cfg.seq_length)

    bs = batch_size or cfg.batch_size
    train_loader = DataLoader(
        HaikuWindows(x_train, y_train), batch_size=bs, shuffle=True, drop_last=True,
        pin_memory=torch.cuda.is_available(), num_workers=0,  # all data is already in RAM, Windows workers only add spawn cost
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    val_loader = DataLoader(
        HaikuWindows(x_val, y_val), batch_size=bs, shuffle=False, drop_last=False,
        pin_memory=torch.cuda.is_available(), num_workers=0,
    )

    stats = {
        "haiku_train": len(train_recs), "haiku_val": len(val_recs),
        "tokens_train": int(x_train.numel()), "tokens_val": int(x_val.numel()),
        "windows_train": len(train_loader.dataset), "windows_val": len(val_loader.dataset),
        "steps_per_epoch": len(train_loader),
        "masked_targets_pct": round(100 * float((y_train == -100).float().mean()), 1),
        "vocab_size": tok.get_vocab_size(),
        "expected_initial_loss": round(torch.log(torch.tensor(float(tok.get_vocab_size()))).item(), 2),
    }
    if verbose:
        print(f"train: {stats['haiku_train']} haiku, {stats['tokens_train'] / 1e6:.2f}M tokens, "
              f"{stats['windows_train']} windows, {stats['steps_per_epoch']} steps/epoch at batch {bs}")
        print(f"val:   {stats['haiku_val']} haiku, {stats['tokens_val'] / 1e6:.2f}M tokens, "
              f"{stats['windows_val']} windows")
        print(f"initial loss must start near ln({stats['vocab_size']}) = {stats['expected_initial_loss']}")
    return train_loader, val_loader, tok, stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Module 3: report the dataset split")
    ap.add_argument("--corpus", type=Path, default=CLEAN_PATH)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args(argv)
    build_loaders(corpus=args.corpus, tokenizer_path=args.tokenizer, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
