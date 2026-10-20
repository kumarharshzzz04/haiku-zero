"""Repeat / seed / meter probe - measures what the decoder actually does after the poetry pretrain.

    python scripts/eval/repeat_probe.py [--checkpoint checkpoints/best.pt] [--n 4] [--free]

Repetition is measured on content words only (stopwords repeated are grammar) and on stems, so
"snow / snow's / snows" all count as the same word being spent twice. Seed adherence is measured
the same way, so "river's" counts as saying "river".
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend import generate  # noqa: E402


def content_stems(lines) -> list[str]:
    words = [w for line in lines for w in generate.WORD_RE.findall(str(line).lower())
             if w not in generate.STOPWORDS]
    return [generate._stem(w) for w in words]


def run(ckpt: Path, seeds: list[str], n: int, strict: bool) -> dict:
    model, tok, cfg, dev, payload = generate.load_model(Path(ckpt), device="cuda")
    repeats = exacts = seed_hits = total = 0
    examples = []
    for seed in seeds:
        want = generate._stem(seed.split()[0])
        for i in range(n):
            lines, pat, exact = generate.best_of_haiku(model, tok, seed, n=2, strict=strict,
                                                       seed_first=True, seed=i, amp=True)
            total += 1
            exacts += bool(exact if exact is not None else False)
            stems = content_stems(lines)
            if len(set(stems)) < len(stems):
                repeats += 1
                if len(examples) < 3:
                    examples.append((seed, " / ".join(lines)))
            seed_hits += sum(1 for s in stems if s == want)
    return {"ckpt": str(ckpt), "total": total, "repeat": repeats / total,
            "exact": exacts / total, "seed_uses": seed_hits / total, "examples": examples}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, action="append")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--free", action="store_true", help="free meter instead of strict")
    ap.add_argument("--seeds", default="blood,snow,autumn,river,harbor,iron,candle,dust")
    args = ap.parse_args(argv)

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    for ckpt in (args.checkpoint or [ROOT / "checkpoints" / "best.pt",
                                     ROOT / "checkpoints_tweets_live" / "best.pt"]):
        r = run(ckpt, seeds, args.n, not args.free)
        mode = "free" if args.free else "strict"
        print(f"{r['ckpt']:38s} {mode:6s} repeat {100*r['repeat']:3.0f}%  "
              f"exact {100*r['exact']:3.0f}%  seed-word uses {r['seed_uses']:.2f}/poem")
        for seed, text in r["examples"]:
            print(f"    [{seed}] {text}")


if __name__ == "__main__":
    main()
