"""Arrange the raw sources that the two corpora are built from.

    python scripts/data/prepare_corpora.py

Why this exists: the original corpus was 78% twaiku (tweets), which is what made the model
sound like a Twitter feed. Decisions, all reversible:

- the untouched twaiku file moves to `data/raw_twaiku/` (kept, never edited);
- a deterministic sample of it (`data/raw/twaiku_capped.csv`, seed 1337) goes back into the
  full corpus, so the model still learns English volume and haiku shape from it - but at a
  minority weight instead of 78%;
- the imported authentic corpus (`data/raw/statworx.csv`, no tweets) is copied into both
  `data/raw/` (full) and `data/raw_classical/` (authentic only).

Then rebuild with Module 1:
    python -m backend.prepare_data --raw-dir data/raw            --out data/haikus_clean.txt
    python -m backend.prepare_data --raw-dir data/raw_classical  --out data/haikus_classical.txt
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
RAW_CLASSICAL = ROOT / "data" / "raw_classical"
TWAIKU_STORE = ROOT / "data" / "raw_twaiku"
STATWORX = RAW / "statworx.csv"


def row_count(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="arrange raw sources for the two corpora")
    ap.add_argument("--twaiku-cap", type=int, default=35000,
                    help="how many twaiku rows stay in the full corpus")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    if not STATWORX.exists():
        raise SystemExit(f"{STATWORX} missing - run: python scripts/data/import_external.py")

    # 1. park the original tweet file, then put a capped sample back in the full corpus
    TWAIKU_STORE.mkdir(parents=True, exist_ok=True)
    original = RAW / "twaiku.csv"
    stored = TWAIKU_STORE / "twaiku.csv"
    if original.exists() and not stored.exists():
        shutil.move(str(original), str(stored))
        print(f"moved {original.relative_to(ROOT)} -> {stored.relative_to(ROOT)} (untouched)")
    if not stored.exists():
        raise SystemExit(f"{stored} missing and no {original} to move")

    lines = stored.read_text(encoding="utf-8", errors="replace").splitlines(True)
    header, body = lines[0], lines[1:]
    cap = min(args.twaiku_cap, len(body))
    rng = random.Random(args.seed)
    kept = sorted(rng.sample(range(len(body)), cap))
    capped = RAW / "twaiku_capped.csv"
    with capped.open("w", encoding="utf-8", newline="") as f:
        f.write(header)
        for i in kept:
            f.write(body[i])
    print(f"twaiku: {len(body):,} rows -> {cap:,} kept in {capped.name} (seed {args.seed})")

    # 2. the authentic corpus belongs in both source sets
    RAW_CLASSICAL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STATWORX, RAW_CLASSICAL / STATWORX.name)
    print(f"copied {STATWORX.name} into data/raw_classical/")

    for folder in (RAW, RAW_CLASSICAL):
        print(f"\n{folder.relative_to(ROOT)}/")
        total = 0
        for path in sorted(folder.glob("*.csv")):
            n = row_count(path)
            total += n
            print(f"    {path.name:<24}{n:>9,} rows")
        print(f"    {'TOTAL':<24}{total:>9,} rows (before cleaning)")


if __name__ == "__main__":
    main()
