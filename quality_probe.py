"""Fixed probes for output quality, so every change is measured on the same sled.

    python quality_probe.py                                  # every checkpoints*/best.pt
    python quality_probe.py --models checkpoints             # one
    python quality_probe.py --n 20 --best-of 4 --seeds autumn,breeze

Reports, per model: junk rate (characters a haiku line cannot contain - the '?????' bug),
'?' rate, strict 5-7-5 exact rate, seed adherence, repeat rate and non-word rate (share of words
the training corpus never contained). The repeat column counts CONTENT words by stem, the same
number the API and UI show; the old all-word metric charged grammar and reported a clean poem
as 29% repeats. Cross-run val loss is NOT comparable (different splits), so these probes are the
comparison that is.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import generate, meter_check  # noqa: E402

ROOT = Path(__file__).resolve().parent
SEED_STEM = re.compile(r"[a-z']+")


def probe(ckpt: Path, seeds: list[str], n: int, best_of: int, strict: bool,
          seed_first: bool = True, word_weight: float | None = None) -> dict:
    model, tok, cfg, dev, payload = generate.load_model(ckpt)
    extra = {} if word_weight is None else {"word_weight": word_weight}
    rows = []
    for prompt in seeds:
        for i in range(n):
            lines, pat, ok = generate.best_of_haiku(model, tok, prompt, n=best_of, strict=strict,
                                                    seed_first=seed_first, device=dev,
                                                    seed=i * 7 + 1, **extra)
            text = " ".join(lines)
            rows.append({
                "seed": prompt,
                "lines": lines,
                "junk": any(generate.has_junk(l) for l in lines),
                "q": "?" in text,
                "exact": meter_check.is_haiku(lines),
                "seeded": generate.seed_adherence(lines, prompt) == 1.0 if prompt else None,
                "repeat": generate.repeated_content_rate(lines),
                "unattested": 1.0 - generate.attested_fraction(lines),
                "padded": tuple(pat) + (0,) * (3 - len(pat)),
            })
    total = len(rows)
    seeded = [r for r in rows if r["seed"]]
    return {
        "model": ckpt.parent.name,
        "epoch": payload.get("epoch"),
        "val": payload.get("val_loss"),
        "n": total,
        "junk_rate": sum(r["junk"] for r in rows) / total,
        "q_rate": sum(r["q"] for r in rows) / total,
        "exact_rate": sum(r["exact"] for r in rows) / total,
        "seed_rate": (sum(r["seeded"] for r in seeded) / len(seeded)) if seeded else None,
        "repeat_rate": sum(r["repeat"] for r in rows) / total,
        "unattested_rate": sum(r["unattested"] for r in rows) / total,
        "rows": rows,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="fixed-probe quality comparison")
    ap.add_argument("--models", default=None, help="comma-separated checkpoint dirs")
    ap.add_argument("--seeds", default="autumn,breeze,old pond,")
    ap.add_argument("--n", type=int, default=10, help="samples per seed")
    ap.add_argument("--best-of", type=int, default=4)
    ap.add_argument("--free", action="store_true", help="free meter instead of strict")
    ap.add_argument("--no-seed-first", dest="seed_first", action="store_false",
                    help="do not start poems with the seed word (then the seed column is a real test)")
    ap.add_argument("--word-weight", type=float, default=None,
                    help="override the non-word rerank weight (0 = off, to measure its effect)")
    ap.add_argument("--show", type=int, default=3, help="samples to print per model")
    args = ap.parse_args(argv)

    if args.models:
        ckpts = [ROOT / m / "best.pt" for m in args.models.split(",")]
    else:
        ckpts = sorted(p / "best.pt" for p in ROOT.glob("checkpoints*") if (p / "best.pt").exists())

    seeds = [s for s in args.seeds.split(",")]
    print(f"probes: {len(ckpts)} model(s) x {len(seeds)} seeds x {args.n} samples, best-of "
          f"{args.best_of}, {'free meter' if args.free else 'strict 5-7-5'}, "
          f"seed_first {'on (seed column is by construction)' if args.seed_first else 'OFF'}, "
          f"word rerank {generate.WORD_WEIGHT if args.word_weight is None else args.word_weight}\n")
    header = (f"{'model':<24}{'epoch':>6}{'junk':>8}{'?':>8}{'exact':>8}{'seed':>8}{'repeat':>8}"
              f"{'nonword':>9}")
    print(header)
    print("-" * len(header))
    results = []
    for ckpt in ckpts:
        if not ckpt.exists():
            print(f"{ckpt} missing")
            continue
        r = probe(ckpt, seeds, args.n, args.best_of, not args.free, args.seed_first,
                  args.word_weight)
        results.append(r)
        seed_txt = "n/a" if r["seed_rate"] is None else f"{100 * r['seed_rate']:.0f}%"
        print(f"{r['model']:<24}{str(r['epoch']):>6}{100 * r['junk_rate']:>7.0f}%"
              f"{100 * r['q_rate']:>7.0f}%{100 * r['exact_rate']:>7.0f}%{seed_txt:>8}"
              f"{100 * r['repeat_rate']:>7.0f}%{100 * r['unattested_rate']:>8.0f}%")
        for row in r["rows"][: args.show]:
            label = row["seed"] or "(none)"
            tag = "exact" if row["exact"] else f"near {row['padded']}"
            print(f"    seed {label!r:<10} [{tag}] " + " / ".join(row["lines"]))
        print()


if __name__ == "__main__":
    main()
