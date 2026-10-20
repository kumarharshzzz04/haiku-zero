"""Import an external haiku corpus into data/raw/ in the schema Module 1 expects.

    python scripts/data/import_external.py   # statworx/haiku -> data/raw/statworx.csv (twaiku rows skipped)

Source: https://huggingface.co/datasets/statworx/haiku (49,024 rows, public, not gated,
column `text` holds the poem, `source` labels where it came from). Its card declares no
license, so this import is for local experimentation and the file is kept out of the repo
import path (`data/external/`) until you decide you want it in the mix.

Requires pyarrow (data prep only, not a training dependency):
    pip install pyarrow
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
URL = "https://huggingface.co/datasets/statworx/haiku/resolve/main/train.parquet"
PARQUET = ROOT / "data" / "external" / "statworx_train.parquet"
OUT = ROOT / "data" / "raw" / "statworx.csv"

# this corpus stores one poem per row with the three lines joined by " / "
LINE_SPLIT_RE = re.compile(r"\r?\n|<br\s*/?>| / ")
SKIP_SOURCES = {"twaiku"}          # tweets: already in our corpus, and they are what we are cutting back
JUNK = re.compile(r"[^\w\s'’\-–—.,;:!?]|\d")
WORD_RE = re.compile(r"[A-Za-z]{2,}")


def rel(path: Path) -> str:
    """Displayable path: relative to the repo when possible.

    A relative `--out` used to crash the final print with ValueError *after* the file was written
    (same bug found three times now: seed_placement_probe.py, fetch_poetry.py, and here), so a
    successful run reported failure. resolve() before relative_to() is the whole fix.
    """
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"cached: {rel(path)} ({path.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as out:
        out.write(response.read())
    print(f"saved: {rel(path)} ({path.stat().st_size / 1e6:.1f} MB)")


def three_lines(text: str) -> list[str] | None:  # noqa: D401
    parts = [re.sub(r"\s+", " ", p).strip() for p in LINE_SPLIT_RE.split(str(text))]
    parts = [p for p in parts if p]
    return parts[:3] if len(parts) >= 3 else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="import statworx/haiku into data/raw/")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    download(args.url, PARQUET)
    import pyarrow.parquet as pq

    table = pq.read_table(PARQUET)
    print(f"parquet: {table.num_rows:,} rows, columns {table.column_names}")
    rows = table.to_pylist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    sources = collections.Counter()
    written = 0
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "line1", "line2", "line3", "source", "hash"])
        for i, row in enumerate(rows):
            lines = three_lines(row.get("text", ""))
            if not lines:
                stats["not_3_lines"] += 1
                continue
            if any(len(l) > 60 for l in lines):
                stats["line_too_long"] += 1
                continue
            joined = " ".join(lines)
            if not WORD_RE.search(joined):
                stats["no_words"] += 1
                continue
            if JUNK.search(joined):
                stats["junk_chars"] += 1
                continue
            source = str(row.get("source") or "statworx").strip().lower() or "statworx"
            if source in SKIP_SOURCES:
                stats[f"skipped_{source}"] += 1
                continue
            sources[source] += 1
            writer.writerow([i, *lines, f"statworx-{source}", f"SX{i}"])
            written += 1

    print(f"kept {written:,} poems -> {rel(args.out)}")
    print(f"dropped: {dict(stats)}")
    print(f"by source: {dict(sources.most_common())}")
    sample = args.out.read_text(encoding="utf-8").splitlines()[1:5]
    for line in sample:
        print("   ", line)


if __name__ == "__main__":
    main()
