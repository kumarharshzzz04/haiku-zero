"""Teach the weights what the decoder already does (self-distillation).

PROGRESS.md known limitation #1: every quality rule lives in the decoder — exact meter,
no repeats, no invented words, seed adherence — and the raw model would break them all.
This script moves some of that into the weights by training on the decoder's own best
output, and then measures the weights alone.

    python scripts/self_distill.py --stage pairs --pairs 900 --best-of 6
    python scripts/self_distill.py --stage train --epochs 4 --lr 5e-5
    python scripts/self_distill.py --stage probe --models checkpoints,checkpoints_selfdistill --n 4

Stage `pairs` draws poems the way the live decoder does (strict 5-7-5, reranker on, seed
as context so the word has to land on its own) and keeps ONLY those that pass every rule
the decoder enforces: exact 5-7-5, the seed word present, no repeated content word, every
word attested in the training corpora, no junk, no banned word. One poem per seed word,
deduplicated, so 900 seeds give 900 distinct targets instead of 900 variations of ten.

Stage `train` encodes each pair exactly the way `backend.dataset.encode_records` does for
seed conditioning — `[<bos> <seed> word \\n poem <eos>]` with the seed prefix masked out of
the loss — and fine-tunes from the live checkpoint with the same `train_run` engine the
real runs use (AMP, warmup + cosine, grad clip, best/last, CSV log, preview). `--mix
<corpus>:<count>` adds rows from any corpus file (the authentic corpus as a regulariser, or
`data/haikus_exact.txt` when the goal is form); `--meter-state` turns on the syllable/line
inputs of RUNBOOK §4c, whose two new tensors load from an older checkpoint because `train_run`
now loads init weights with `strict=False` and prints exactly what was freshly initialised.

Stage `probe` answers the question the whole experiment exists for: does the model behave
better *without* the decoder's help? It neutralises the candidate weights
(`REPEAT_WEIGHT`, `SEAM_WEIGHT`, `UNATTESTED_WEIGHT` to 1.0), the reranker bonus
(`word_weight=0`, `SEED_BONUS=0`) and the reranking itself (`--best-of 1`), then samples.
Any improvement that survives that is in the weights.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import generate, meter_check  # noqa: E402
from backend.config import Config  # noqa: E402
from backend.dataset import HaikuWindows, load_tokenizer, pack  # noqa: E402
from backend.train import train_run  # noqa: E402
PAIRS_PATH = ROOT / "data" / "self_distill_pairs.jsonl"
AUTHENTIC_PATH = ROOT / "data" / "haikus_classical.txt"
TOKENIZER_PATH = ROOT / "tokenizer.json"
WORD_RE = re.compile(r"[a-z']{2,}")


# ------------------------------------------------------------------ stage: pairs

def corpus_seed_words(count: int, rng: random.Random) -> list[str]:
    """Word pool the way training itself samples it: a random 3+ letter word from a poem.

    `backend.dataset.theme_word` does exactly this, so the distilled pairs are conditioned
    the same way the model was taught, not on some nicer distribution invented here.
    """
    poems = [b.strip() for b in AUTHENTIC_PATH.read_text(encoding="utf-8").split("<eos>") if b.strip()]
    rng.shuffle(poems)
    words: list[str] = []
    seen: set[str] = set()
    for block in poems:
        text = block.replace("\n", " ").lower()
        candidates = [w for w in re.findall(r"[a-z]{3,}", text) if w not in generate.STOPWORDS]
        if not candidates:
            continue
        word = rng.choice(candidates)
        if word in seen:
            continue
        seen.add(word)
        words.append(word)
        if len(words) >= count:
            break
    return words


def clean_pair(model, tok, word: str, best_of: int, seed: int, words: frozenset,
               banned: frozenset, dev) -> tuple[list[str] | None, str]:
    """One decoder-quality poem for `word`, or the reason it was rejected.

    The rejection list IS the decoder's rule set, restated once here instead of being
    trusted implicitly: a target that breaks a rule would teach the model to break it.
    """
    lines, pat, _ = generate.best_of_haiku(model, tok, word, n=best_of, strict=True,
                                           seed_first=False, device=dev, seed=seed)
    if len(lines) != 3 or any(len(l.split()) < 2 for l in lines):
        return None, "short_line"
    if not meter_check.is_haiku(lines):
        return None, "off_meter"
    if generate.seed_adherence(lines, word) != 1.0:
        return None, "no_seed"
    if generate.repeated_content_words(lines):
        return None, "repeat"
    if generate.attested_fraction(lines, words) < 1.0:
        return None, "invented"
    if any(generate.has_junk(l) for l in lines):
        return None, "junk"
    if banned & set(WORD_RE.findall(" ".join(lines).lower())):
        return None, "banned"
    return lines, "kept"


def stage_pairs(args) -> None:
    rng = random.Random(args.seed)
    model, tok, cfg, dev, payload = generate.load_model(args.checkpoint)
    words = generate.load_wordlist()
    banned = generate.banned_words()
    print(f"loaded {args.checkpoint} (epoch {payload.get('epoch')}, val {payload.get('val_loss'):.4f})")
    print(f"{args.pairs} seeds, best-of {args.best_of}, strict, seed as context (word must land on its own)\n")

    seeds = corpus_seed_words(args.pairs * 2, rng)   # headroom for rejected seeds
    kept: list[dict] = []
    stats = Counter()
    seen_poems: set[str] = set()
    placements: Counter = Counter()
    for i, word in enumerate(seeds):
        if len(kept) >= args.pairs:
            break
        lines, why = clean_pair(model, tok, word, args.best_of, seed=i * 13 + 1,
                                words=words, banned=banned, dev=dev)
        if lines is None:
            stats[why] += 1
            continue
        key = "".join(c for c in " ".join(lines).lower() if c.isalnum())
        if key in seen_poems:
            stats["duplicate"] += 1
            continue
        seen_poems.add(key)
        stats["kept"] += 1
        first_line = lines[0].lower()
        position = next((j for j, w in enumerate(WORD_RE.findall(first_line))
                         if w == word or w.startswith(word[:max(4, len(word) - 2)])), None)
        placements["line 1" if position is not None else "later"] += 1
        kept.append({"seed": word, "lines": lines})
        if len(kept) % 100 == 0:
            print(f"  {len(kept)} kept ...", flush=True)

    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PAIRS_PATH.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"\nkept {len(kept)} pairs of {sum(stats.values())} draws -> {PAIRS_PATH.relative_to(ROOT)}")
    print(f"rejected: {dict(stats.most_common())}")
    print(f"seed word appeared in line 1 in {placements['line 1']} of {len(kept)} poems, "
          f"later in {placements['later']} (opening the poem is only the first-word subset of this)")


# ------------------------------------------------------------------ stage: train

def corpus_rows(path: Path, count: int, rng: random.Random) -> list[dict]:
    """Poems from any corpus file in the pair format, seed word sampled the training way.

    Used for two jobs: real poems as a regulariser (a set of the decoder's own low-entropy
    output drifts on repeats - measured 20% -> 40% raw) and the exact-5-7-5 corpus when the
    goal is form rather than vocabulary. `backend.dataset.theme_word` picks the seed the same
    way, so these rows are conditioned exactly like a normal training batch.
    """
    blocks = [b.strip() for b in Path(path).read_text(encoding="utf-8").split("<eos>") if b.strip()]
    rng.shuffle(blocks)
    rows = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) != 3:
            continue
        words = [w for w in re.findall(r"[a-z]{3,}", " ".join(lines).lower())
                 if w not in generate.STOPWORDS]
        if not words:
            continue
        rows.append({"seed": rng.choice(words), "lines": lines})
        if len(rows) >= count:
            break
    return rows


def encode_pairs(pairs: list[dict], tok, max_mask: int) -> tuple[list[list[int]], list[list[bool]]]:
    """`[<bos> <seed> word \\n poem <eos>]`, seed prefix masked - dataset.encode_records' shape."""
    bos_id, eos_id = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    seed_id = tok.token_to_id("<seed>")
    nl_ids = tok.encode("\n").ids
    chunks, masks = [], []
    for row in pairs:
        poem_ids = tok.encode("\n".join(row["lines"])).ids
        seed_ids = [bos_id, seed_id] + tok.encode(" " + row["seed"]).ids + nl_ids
        ids = seed_ids + poem_ids + [eos_id]
        mask = [True] * len(seed_ids) + [False] * (len(poem_ids) + 1)
        if len(ids) > max_mask + len(seed_ids) + 2:      # a target must survive the window
            continue
        chunks.append(ids)
        masks.append(mask)
    return chunks, masks


def stage_train(args) -> None:
    pairs = [json.loads(l) for l in PAIRS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not pairs:
        raise SystemExit(f"no pairs in {PAIRS_PATH} - run --stage pairs first")
    rng0 = random.Random(args.seed)
    for spec in args.mix or []:
        path, _, count = spec.partition(":")
        extra = corpus_rows(Path(path), int(count or 0), rng0)
        print(f"mixing in {len(extra)} poems from {path}")
        pairs = pairs + extra
    init = Path(args.init_from)
    payload = torch.load(init, map_location="cpu", weights_only=False)
    cfg = Config(**payload["config"])            # architecture comes from the checkpoint itself
    cfg.lr, cfg.epochs, cfg.batch_size = args.lr, args.epochs, args.batch_size
    cfg.preview_prompt = args.preview
    cfg.meter_state = args.meter_state          # new embeddings load from init with strict=False
    tok = load_tokenizer(init.parent / "tokenizer.json" if (init.parent / "tokenizer.json").exists()
                         else TOKENIZER_PATH)

    rng = random.Random(args.seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_rows = [pairs[i] for i in order[:n_val]]
    train_rows = [pairs[i] for i in order[n_val:]]
    max_mask = max(1, cfg.seq_length - 2)
    tr_chunks, tr_masks = encode_pairs(train_rows, tok, max_mask)
    va_chunks, va_masks = encode_pairs(val_rows, tok, max_mask)
    x_tr, y_tr = pack(tr_chunks, cfg.seq_length, tr_masks)
    x_va, y_va = pack(va_chunks, cfg.seq_length, va_masks)
    train_loader = DataLoader(HaikuWindows(x_tr, y_tr), batch_size=cfg.batch_size, shuffle=True,
                              drop_last=False, num_workers=0,
                              generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(HaikuWindows(x_va, y_va), batch_size=cfg.batch_size, shuffle=False,
                            drop_last=False, num_workers=0)
    cfg.warmup_steps = max(10, len(train_loader) * args.epochs // 20)

    print(f"pairs {len(pairs)} ({len(train_rows)} train / {len(val_rows)} val), "
          f"{train_loader.dataset.x.numel()} train tokens, {len(train_loader)} steps/epoch")
    print(f"init {init} -> {args.ckpt_dir}, lr {cfg.lr}, {args.epochs} epochs, "
          f"warmup {cfg.warmup_steps} steps, meter_state {cfg.meter_state}, "
          f"mask {100 * float((y_tr == -100).float().mean()):.0f}% of targets")
    train_run(cfg, train_loader, val_loader, tok, ckpt_dir=Path(args.ckpt_dir),
              epochs=args.epochs, device=args.device, run_name="self-distill",
              log_path=Path(args.log), init_from=init,
              tokenizer_path=init.parent / "tokenizer.json"
              if (init.parent / "tokenizer.json").exists() else TOKENIZER_PATH)


# ------------------------------------------------------------------ stage: probe

@torch.no_grad()
def raw_probe(ckpt: Path, seeds: list[str], n: int, device, ablate_meter: bool = False) -> dict:
    """Sample with the decoder's opinion switched off, so only the weights can speak.

    `REPEAT_WEIGHT`/`SEAM_WEIGHT`/`UNATTESTED_WEIGHT` are the candidate multipliers the
    decoder uses to kill a repeat, a fused word or an invention; 1.0 means "no preference".
    Combined with best-of 1 and word_weight 0, nothing is ranked, chosen or rejected for
    quality - which is exactly PROGRESS.md's stated way to see the model's real behaviour.
    """
    generate.REPEAT_WEIGHT = 1.0
    generate.SEAM_WEIGHT = 1.0
    generate.UNATTESTED_WEIGHT = 1.0
    if hasattr(generate, "SEED_BONUS"):
        generate.SEED_BONUS = 0.0
    # free-meter shaping is a decoder opinion too: the 10-syllable ceiling and the "this line is
    # long enough, close it" boost both choose line lengths for the model. A 12-syllable ceiling
    # still bounds run-ons (nothing can close them otherwise), the boost is switched off.
    generate.FREE_MAX_SYLLABLES = 12
    generate.FREE_MIN_SYLLABLES = 4
    generate.FREE_NEWLINE_BOOST = 1.0
    model, tok, cfg, dev, payload = generate.load_model(ckpt, device=device)
    if ablate_meter and model.meter_state:
        # zero the two state embeddings: the body keeps its trained weights, the model just stops
        # being told how many syllables it is into the line. If the score does not move, the
        # feature was decoration; if it drops, the feature is what the model counts with.
        model.syl_emb.weight.data.zero_()
        model.line_emb.weight.data.zero_()
    words = generate.load_wordlist()
    rows = []
    for i, seed_word in enumerate(seeds):
        for k in range(n):
            lines, _, _ = generate.best_of_haiku(model, tok, seed_word, n=1, strict=False,
                                                 seed_first=False, word_weight=0.0,
                                                 device=dev, seed=1000 + i * 31 + k)
            pat = meter_check.pattern(lines)
            rows.append({
                "lines": lines,
                "three": len(lines) == 3,
                "exact": meter_check.is_haiku(lines),
                "near": meter_check.distance(pat) <= 1,
                "repeat": generate.repeated_content_rate(lines),
                "unattested": 1.0 - generate.attested_fraction(lines, words),
                "seed": generate.seed_adherence(lines, seed_word) == 1.0,
                "junk": any(generate.has_junk(l) for l in lines),
                "dist": meter_check.distance(pat),
            })
    total = len(rows)
    return {
        "model": ckpt.parent.name,
        "epoch": payload.get("epoch"),
        "n": total,
        "three": sum(r["three"] for r in rows) / total,
        "exact": sum(r["exact"] for r in rows) / total,
        "near": sum(r["near"] for r in rows) / total,
        "repeat": sum(r["repeat"] > 0 for r in rows) / total,
        "unattested": sum(r["unattested"] > 0 for r in rows) / total,
        "seed": sum(r["seed"] for r in rows) / total,
        "junk": sum(r["junk"] for r in rows) / total,
        "dist": sum(r["dist"] for r in rows) / total,
        "rows": rows,
    }


def stage_probe(args) -> None:
    seeds = [w.strip() for w in args.words.split(",") if w.strip()]
    print(f"raw probe: {len(seeds)} seeds x {args.n}, free meter, best-of 1, "
          f"candidate weights neutralised\n")
    header = (f"{'model':<24}{'epoch':>6}{'3 lines':>9}{'exact':>7}{'+-1':>7}{'repeat':>8}"
              f"{'invent':>8}{'seed':>7}{'junk':>7}{'dist':>7}")
    print(header)
    print("-" * len(header))
    for name in args.models.split(","):
        ckpt = ROOT / name / "best.pt"
        if not ckpt.exists():
            print(f"{name}: missing")
            continue
        r = raw_probe(ckpt, seeds, args.n, args.device, args.ablate_meter)
        r["model"] = ckpt.parent.name + (" (state zeroed)" if args.ablate_meter else "")
        print(f"{r['model']:<24}{str(r['epoch']):>6}{100 * r['three']:>8.0f}%{100 * r['exact']:>6.0f}%"
              f"{100 * r['near']:>6.0f}%{100 * r['repeat']:>7.0f}%{100 * r['unattested']:>7.0f}%"
              f"{100 * r['seed']:>6.0f}%{100 * r['junk']:>6.0f}%{r['dist']:>7.2f}")
        for row in r["rows"][:args.show]:
            print("    " + " / ".join(row["lines"]))
        print()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="self-distillation: decoder rules into the weights")
    ap.add_argument("--stage", choices=("pairs", "train", "probe"), required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    ap.add_argument("--init-from", type=Path, default=ROOT / "checkpoints" / "best.pt")
    ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints_selfdistill")
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "training" / "training_log_self_distill.csv")
    ap.add_argument("--pairs", type=int, default=900, help="clean pairs to keep (one per seed)")
    ap.add_argument("--best-of", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.08)
    ap.add_argument("--mix", action="append", metavar="PATH:COUNT",
                    help="extra corpus rows to train alongside the pairs, repeatable")
    ap.add_argument("--meter-state", action="store_true",
                    help="train with the syllable/line state inputs (RUNBOOK 4c)")
    ap.add_argument("--preview", default="sea")
    ap.add_argument("--models", default="checkpoints,checkpoints_selfdistill")
    ap.add_argument("--words", default="rain,night,moon,wind,light,snow,sea,river,stone,bird,fire,cloud")
    ap.add_argument("--n", type=int, default=4, help="poems per word in the raw probe")
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--ablate-meter", action="store_true",
                    help="zero the meter-state embeddings, to test whether the model uses them")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)
    if args.batch_size is None:
        args.batch_size = 64
    if args.stage == "pairs":
        stage_pairs(args)
    elif args.stage == "train":
        stage_train(args)
    else:
        stage_probe(args)


if __name__ == "__main__":
    main()
