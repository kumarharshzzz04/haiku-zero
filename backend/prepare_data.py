"""Module 1 - merge, clean and normalize every source in data/raw/ into one corpus.

Outputs
    data/haikus_clean.txt   canonical corpus, one haiku = 3 lines + "<eos>"
    data/haikus_meter.txt   subset whose syllable pattern is within +-1 of 5-7-5
    data/stats.json         every count printed below, for the record

Usage
    python -m backend.prepare_data
    python -m backend.prepare_data --lowercase        # force all-lowercase corpus
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # allow: python backend/prepare_data.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.meter_check import line_syllables  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_PATH = ROOT / "data" / "haikus_clean.txt"
METER_PATH = ROOT / "data" / "haikus_meter.txt"
STATS_PATH = ROOT / "data" / "stats.json"

EOS = "<eos>"
MAX_LINE_CHARS = 60
MAX_NONASCII_RATIO = 0.50
URL_RE = re.compile(r"https?://|www\.|@")
HAS_LETTER_RE = re.compile(r"[A-Za-z]")
LINE_SPLIT_RE = re.compile(r"\r?\n|<br\s*/?>| / ")
# '?' and dot runs are line-end markers in this corpus, and the model copied them as zero-syllable
# FILLER when strict decoding blocked a newline (the "?????" bug). Stripping them from the targets
# removes the signal; generation also refuses them outright (backend/generate.has_junk).
QUESTION_RE = re.compile(r"\s*\?\s*")
DOT_RUN_RE = re.compile(r"(?:\s*\.\s*){2,}")
EDGE_PUNCT_RE = re.compile(r"^[\s'\"\u201c\u201d\u2018\u2019,;:\-\u2013\u2014.]+|[\s'\"\u201c\u201d\u2018\u2019,;:\-\u2013\u2014.]+$")


# ---------------------------------------------------------------- reading raw

def lines_from_cells(cells) -> list[str] | None:
    """Pull 3 poem lines out of one CSV row / JSON value list."""
    cells = ["" if c is None else str(c) for c in cells]
    # Shape A: index-like first column, then line1, line2, line3 (the given dataset).
    if len(cells) >= 4 and all(c.strip() for c in cells[1:4]):
        return cells[1:4]
    # Shape B: one blob column holding the whole poem.
    for c in cells:
        parts = [p for p in LINE_SPLIT_RE.split(c) if p.strip()]
        if len(parts) >= 3:
            return parts[:3]
    return None


def lines_from_block(block: str) -> list[str] | None:
    """3 lines out of a plain-text block, or out of a JSON value."""
    parts = [p for p in LINE_SPLIT_RE.split(block) if p.strip()]
    return parts[:3] if len(parts) >= 3 else None


def read_json(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        items = json.loads(text)
        if isinstance(items, dict):
            items = next((v for v in items.values() if isinstance(v, list)), [])
    blobs = []
    for it in items:
        if isinstance(it, dict):
            blobs.append("\n".join(str(v) for v in it.values() if v is not None))
        else:
            blobs.append(str(it))
    return blobs


def read_txt(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    return blocks


def candidates_from_file(path: Path) -> list[tuple[str, list[str]]]:
    """Yield (source, lines) candidates from one raw file of any format."""
    out: list[tuple[str, list[str]]] = []
    try:
        if path.suffix.lower() in (".csv", ".tsv"):
            delim = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open(encoding="utf-8", errors="replace", newline="") as f:
                for row in csv.reader(f, delimiter=delim):
                    lines = lines_from_cells(row)
                    if lines:
                        out.append((path.stem, lines))
        elif path.suffix.lower() in (".json", ".jsonl"):
            for blob in read_json(path):
                lines = lines_from_block(blob)
                if lines:
                    out.append((path.stem, lines))
        elif path.suffix.lower() in (".txt", ".text"):
            for block in read_txt(path):
                lines = lines_from_block(block)
                if lines:
                    out.append((path.stem, lines))
    except Exception as exc:  # a broken source must not kill the run
        print(f"  ! skipped {path.name}: {exc}")
    return out


# ------------------------------------------------------------------ cleaning

def normalize(lines: list[str], lowercase: bool) -> list[str]:
    out = []
    for line in lines:
        line = QUESTION_RE.sub(" ", line)
        line = DOT_RUN_RE.sub(" ", line)
        line = re.sub(r"\s+", " ", line).strip()
        line = EDGE_PUNCT_RE.sub("", line).strip()
        out.append(line.lower() if lowercase else line)
    return out


def reject_reason(lines: list[str]) -> str | None:
    """None = keep. Otherwise the filter that kills this candidate."""
    if len(lines) != 3 or any(not l for l in lines):
        return "not_3_lines"
    if not all(HAS_LETTER_RE.search(l) for l in lines):
        return "no_letters"          # kills header rows and pure-number junk
    if any(len(l) > MAX_LINE_CHARS for l in lines):
        return "line_too_long"
    joined = " ".join(lines)
    non_ascii = sum(1 for c in joined if ord(c) > 127)
    if non_ascii > len(joined) * MAX_NONASCII_RATIO:
        return "too_much_non_ascii"
    if URL_RE.search(joined):
        return "url_or_mention"
    return None


def dedupe_key(lines: list[str]) -> str:
    return "".join(c for c in "".join(lines).casefold() if c.isalnum())


def collect(raw_dir: Path, lowercase: bool) -> tuple[list[list[str]], Counter, Counter]:
    kept: list[list[str]] = []
    seen: set[str] = set()
    sources = Counter()
    rejected = Counter()
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"no raw files in {raw_dir} - drop CSVs/JSON/TXT there first")
    for path in files:
        raw_count = 0
        for source, lines in candidates_from_file(path):
            raw_count += 1
            lines = normalize(lines, lowercase)
            reason = reject_reason(lines)
            if reason:
                rejected[reason] += 1
                continue
            key = dedupe_key(lines)
            if key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(key)
            kept.append(lines)
            sources[source] += 1
        print(f"  {path.name}: {raw_count} candidates")
    return kept, sources, rejected


# -------------------------------------------------------------------- writing

def write_corpus(records: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for lines in records:
            f.write("\n".join(lines) + f"\n{EOS}\n")


def meter_filter(records: list[list[str]]) -> list[list[str]]:
    out = []
    for lines in records:
        p = tuple(line_syllables(l) for l in lines)
        if abs(p[0] - 5) <= 1 and abs(p[1] - 7) <= 1 and abs(p[2] - 5) <= 1:
            out.append(lines)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Module 1: clean every raw haiku source")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=CLEAN_PATH)
    ap.add_argument("--meter-out", type=Path, default=METER_PATH)
    ap.add_argument("--stats", type=Path, default=STATS_PATH)
    ap.add_argument("--lowercase", action="store_true",
                    help="PLAN.md default was lowercase; off by default so proper nouns survive")
    args = ap.parse_args(argv)

    print(f"reading {args.raw_dir}")
    kept, sources, rejected = collect(args.raw_dir, args.lowercase)
    if not kept:
        raise SystemExit("nothing survived the filters")

    write_corpus(kept, args.out)
    meter = meter_filter(kept)
    write_corpus(meter, args.meter_out)

    chars = sum(len(" ".join(r)) for r in kept)
    stats = {
        "kept": len(kept),
        "rejected": dict(rejected),
        "by_source": dict(sources),
        "meter_subset": len(meter),
        "avg_chars_per_haiku": round(chars / len(kept), 1),
        "est_bpe_tokens": int(chars / 3.5),     # empirical: 19.8 tok/haiku at vocab 4096
        "lowercased": args.lowercase,
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nkept {len(kept)} haiku  ->  {args.out}")
    print(f"rejected: {dict(rejected)}")
    print(f"by source: {dict(sources)}")
    print(f"meter-ish subset: {len(meter)}  ->  {args.meter_out}")
    print(f"avg {stats['avg_chars_per_haiku']} chars/haiku, ~{stats['est_bpe_tokens'] / 1e6:.2f}M BPE tokens")
    return stats


if __name__ == "__main__":
    main()
