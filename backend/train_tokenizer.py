"""Module 2 - train the custom ByteLevel BPE tokenizer (vocab 4096) -> tokenizer.json.

The <eos> marker is a special token, so it is stripped before training: otherwise
BPE learns junk merges for the literal string "<eos>".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

if __package__ in (None, ""):  # allow: python backend/train_tokenizer.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import Config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "haikus_clean.txt"
OUT_PATH = ROOT / "tokenizer.json"
EOS = "<eos>"


def read_records(path: Path) -> list[str]:
    """3-line texts out of the canonical corpus (drops the <eos> marker lines)."""
    records, buf = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() == EOS:
            if len(buf) == 3:
                records.append("\n".join(buf))
            buf = []
        elif line.strip():
            buf.append(line.strip())
    return records


def train_tokenizer(clean_path: Path = CLEAN_PATH, out_path: Path = OUT_PATH,
                    vocab_size: int | None = None, show: bool = True) -> Tokenizer:
    cfg = Config()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size or cfg.vocab_size,
        special_tokens=list(cfg.special_tokens),
        # Without the full byte alphabet, bytes never seen in training are *dropped*
        # at encode time and generation silently loses characters.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show,
    )
    tok.train_from_iterator(read_records(clean_path), trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


TRICKY = ["autumn moonlight", "jazz?", "3.14/2", "\u00e9t\u00e9 caf\u00e9", "-- \u2014 ...", "Q\u00f6z"]


def sanity(tok: Tokenizer, sample: list[str] | None = None, show: bool = True) -> dict:
    """Checks PLAN.md asks for: round-trip, special ids, tokens per haiku."""
    cfg = Config()
    vocab = tok.get_vocab()
    roundtrip = tok.decode(tok.encode("autumn moonlight").ids) == "autumn moonlight"
    lossy = [s for s in TRICKY if tok.decode(tok.encode(s).ids) != s]
    special_ids = {s: vocab.get(s) for s in cfg.special_tokens}
    eos_single = len(tok.encode(EOS).ids) == 1
    sample = sample or []
    lens = [len(tok.encode(t).ids) for t in sample[:5000]]
    avg = sum(lens) / len(lens) if lens else float("nan")
    info = {
        "real_vocab_size": len(vocab),
        "roundtrip_ok": roundtrip,
        "byte_lossless": not lossy,
        "lossy_samples": lossy,
        "special_ids": special_ids,
        "eos_is_single_token": eos_single,
        "avg_tokens_per_haiku": round(avg, 1) if lens else None,
    }
    if show:
        print(f"vocab size: {len(vocab)}")
        print(f"round-trip 'autumn moonlight': {'OK' if roundtrip else 'FAILED'}")
        print(f"byte-lossless decode: {'OK' if not lossy else 'FAILED on ' + repr(lossy)}")
        print(f"special token ids: {special_ids}")
        print(f"'<eos>' is a single token: {eos_single}")
        if lens:
            in_range = 15 <= avg <= 40
            print(f"avg tokens/haiku: {avg:.1f} ({'in' if in_range else 'OUTSIDE'} the 15-40 band)")
    return info


def main(argv=None) -> Tokenizer:
    ap = argparse.ArgumentParser(description="Module 2: train the custom BPE tokenizer")
    ap.add_argument("--clean", type=Path, default=CLEAN_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--vocab-size", type=int, default=None)
    args = ap.parse_args(argv)

    records = read_records(args.clean)
    if not records:
        raise SystemExit(f"no records in {args.clean} - run backend.prepare_data first")
    print(f"training BPE on {len(records)} haiku -> {args.out}")
    tok = train_tokenizer(args.clean, args.out, args.vocab_size)
    sanity(tok, records)
    return tok


if __name__ == "__main__":
    main()
