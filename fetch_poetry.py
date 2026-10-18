"""Fetch a public-domain English poetry corpus from Project Gutenberg -> data/poetry_pretrain.txt

Why this exists: the haiku corpus is only ~1.4M BPE tokens, and starving a 15.8M-param
model on that is what produces invented words ("windshavel", "cicadases"). Growing the
model did not help; more *English* is the fix. Stage 1 pretrains on this file, stage 2
fine-tunes on data/haikus_classical.txt. Everything here is unambiguously public domain.

The corpus is written in the SAME canonical format as the haiku corpus (3 lines then an
`<eos>` marker line), so backend.train_tokenizer.read_records and the whole existing
dataset/training pipeline consume it with no new code. Consecutive verse lines from one
book are chopped into 3-line blocks; blocks never cross a book boundary.

    python fetch_poetry.py --catalog-only     # how many poetry books qualify?
    python fetch_poetry.py                    # download + build (resumable, cached)
    python fetch_poetry.py --max-chars 40e6   # explicit text budget

Downloads are cached in data/raw_poetry/ so re-runs cost nothing. Project Gutenberg asks
only for polite use, so requests are sequential with a small delay and a real User-Agent.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
CATALOG = ROOT / "data" / "external" / "pg_catalog.csv"
BOOKS = ROOT / "data" / "raw_poetry"
OUT = ROOT / "data" / "poetry_pretrain.txt"
UA = "Mozilla/5.0 (haiku-zero; local research corpus build)"
EOS = "<eos>"

START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", re.I)
# A verse line: letters with light punctuation, no digits, no markup, no double spaces.
VERSE_OK_RE = re.compile(r"^[A-Za-z\u00c0-\u017f][A-Za-z\u00c0-\u017f'’\- ,.;:!?\"()\u2014\u2019]*$")
PROSE_OR_STAGE_RE = re.compile(r"^\s*[\[(]|^(ACT|SCENE|CHAPTER|CONTENTS|PREFACE|NOTE)\b", re.I)
WORD_RE = re.compile(r"[A-Za-z\u00c0-\u017f]{2,}")

MIN_LINE_CHARS = 10
MAX_LINE_CHARS = 72
MAX_WORDS = 13


def download(url: str, path: Path, timeout: int = 60) -> bool:
    """Fetch url -> path, skipping if already cached. False on failure."""
    if path.exists() and path.stat().st_size > 500:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) < 500:
            return False
        path.write_bytes(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"  ! {url}: {exc}", flush=True)
        return False


def load_catalog() -> list[dict]:
    if not download(CATALOG_URL, CATALOG, timeout=120):
        raise SystemExit("could not fetch the Project Gutenberg catalog")
    with CATALOG.open(encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def is_poetry(row: dict) -> bool:
    """English, plain text, and actually classified as poetry.

    Deliberately NOT using the LoCC fallback (PR/PS = all English literature): that
    swept in 23,355 books including Moby-Dick and Alice in Wonderland, i.e. novels.
    """
    if (row.get("Type") or "").strip().lower() != "text":
        return False
    if (row.get("Language") or "").strip().lower() != "en":
        return False
    shelf = (row.get("Bookshelves") or "").lower()
    subject = (row.get("Subjects") or "").lower()
    if "poetry" in shelf or "poetry" in subject:
        return True
    locc = (row.get("LoCC") or "").strip().upper().replace(" ", "")
    return locc.startswith("PR1") or locc.startswith("PS1")   # PR5000-/PS = poetry classes


def book_text(row: dict) -> str | None:
    """Boilerplate-stripped body of one book, or None if it could not be fetched."""
    gid = (row.get("Text#") or "").strip()
    if not gid.isdigit():
        return None
    path = BOOKS / f"pg{gid}.txt"
    if not download(f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt", path):
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    start = START_RE.search(text)
    end = END_RE.search(text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start()] if end.start() > 0 else text
    return text


def verse_lines(text: str) -> list[str]:
    """Candidate verse lines: short, wordy, no markup, no prose paragraphs."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not (MIN_LINE_CHARS <= len(line) <= MAX_LINE_CHARS):
            continue
        if "  " in line or "_" in line or "=" in line or "http" in line:
            continue
        if any(ch.isdigit() for ch in line):
            continue
        if line.isupper():
            continue
        if line.endswith("-"):        # hyphenated word broken across lines = wrapped prose
            continue
        if PROSE_OR_STAGE_RE.match(line):
            continue
        if not VERSE_OK_RE.match(line):
            continue
        words = WORD_RE.findall(line)
        if not (2 <= len(words) <= MAX_WORDS):
            continue
        # a long line closing on a full stop is a prose sentence, not a verse line
        if line.endswith((".", ":", ";")) and len(words) > 9:
            continue
        out.append(re.sub(r"\s+", " ", line))
    return out


def drop_prose_paragraphs(lines: list[str], min_len: int = 58, run: int = 3) -> list[str]:
    """Remove runs of near-full-width lines: those are wrapped prose paragraphs.

    Measured before this existed: about a third of the first corpus was prose from
    prefaces, notes and introductions, because a wrapped prose line is short, wordy
    and punctuation-light exactly like a verse line. What prose does NOT do is vary
    its line width - it re-wraps at a fixed margin, so it shows up as a run of long
    lines. Verse almost never puts three 58+ character lines in a row.
    """
    out, i = [], 0
    while i < len(lines):
        if len(lines[i]) >= min_len:
            j = i
            while j < len(lines) and len(lines[j]) >= min_len:
                j += 1
            if j - i >= run:
                i = j                     # whole paragraph dropped
                continue
        out.append(lines[i])
        i += 1
    return out


def english_share(lines: list[str]) -> float:
    """Percent of words that are "the" - a language tell that needs no dependency.

    Measured over the 1,012 cached books: English verse runs a median 5.9% "the"
    (p10 4.4%, p95 8.3%), while the Portuguese/Italian/French books that PG's catalog
    labels "English" sit at or below ~2%. 2.5 sits in that gap. Catching a mislabeled
    poetry book matters here: foreign words in a vocabulary corpus are indistinguishable
    from the invented words we are trying to eliminate.
    """
    words = WORD_RE.findall(" ".join(lines).lower())
    if len(words) < 50:
        return 0.0
    return 100 * words.count("the") / len(words)


def fold(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum())


def rel(path: Path) -> str:
    """Displayable path: relative to the repo when possible.

    A bare `--out data/x.txt` (relative) crashed the final summary print with ValueError, after
    the corpus had already been written - so a successful run reported failure. Same fix as
    seed_placement_probe.py; `runner` paths in this repo are not always absolute.
    """
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="build a public-domain poetry pretraining corpus")
    ap.add_argument("--catalog-only", action="store_true", help="just report candidate counts")
    ap.add_argument("--max-chars", type=float, default=42e6,
                    help="text budget in characters (~4 chars/token for this BPE)")
    ap.add_argument("--max-books", type=int, default=1200)
    ap.add_argument("--delay", type=float, default=0.12, help="seconds between downloads")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    rows = load_catalog()
    print(f"catalog: {len(rows):,} rows, columns {list(rows[0].keys())}")
    poetry = [r for r in rows if is_poetry(r)]
    print(f"english text-classified poetry books: {len(poetry):,}")

    if args.catalog_only:
        from collections import Counter
        shelves = Counter((r.get("Bookshelves") or "?").split(";")[0].strip() for r in poetry)
        print("top shelves:", shelves.most_common(8))
        print("first 8 candidates:")
        for r in poetry[:8]:
            print(f"  {r['Text#']:>6}  {r['Title'][:60]:<60} {r.get('Authors','')[:26]}")
        return

    # biggest books first would skew the mix; shuffle deterministically so a budget stop
    # still spans many authors instead of 40 volumes of one collected works
    order = sorted(poetry, key=lambda r: fold(r.get("Authors") or "") + (r.get("Text#") or ""))
    seen_records: set[str] = set()
    kept = 0
    chars = 0
    books_used = 0
    dupes = 0
    rejected_lines = 0
    skipped_lang = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # write to a .part file and rename at the end: a killed run must never leave a
    # truncated corpus where training would happily read it
    part = args.out.with_suffix(args.out.suffix + ".part")

    with part.open("w", encoding="utf-8") as f:
        for i, row in enumerate(order[: args.max_books]):
            text = book_text(row)
            if not text:
                continue
            lines = verse_lines(text)
            lines = drop_prose_paragraphs(lines)
            if english_share(lines) < 2.5:
                skipped_lang += 1
                print(f"  - skipping non-English book {row.get('Text#')}: {row.get('Title', '')[:50]}",
                      flush=True)
                continue
            # trim the head and tail of each book: title page, table of contents and
            # preface at the front, index/licence boilerplate at the back
            if len(lines) > 80:
                lines = lines[40:-20]
            if len(lines) < 12:
                continue
            books_used += 1
            buf: list[str] = []
            for line in lines:
                if fold(line) == "":
                    continue
                buf.append(line)
                if len(buf) == 3:
                    key = fold("".join(buf))
                    if len(key) < 12 or key in seen_records:
                        dupes += 1
                    else:
                        seen_records.add(key)
                        f.write("\n".join(buf) + f"\n{EOS}\n")
                        kept += 1
                        chars += sum(len(b) for b in buf)
                    buf = []
            rejected_lines += len(buf)
            if i % 25 == 0:
                print(f"  [{i}/{min(len(order), args.max_books)}] {books_used} books, "
                      f"{kept:,} records, {chars / 1e6:.1f}M chars", flush=True)
            if chars >= args.max_chars:
                print(f"budget reached at book {i}")
                break
            if args.delay:
                time.sleep(args.delay)

    part.replace(args.out)      # atomic: corpus only appears once it is complete

    est_tokens = chars / 4.0
    print(f"\nwrote {kept:,} 3-line records ({chars / 1e6:.1f}M chars ~ {est_tokens / 1e6:.1f}M tokens) "
          f"from {books_used} books -> {rel(args.out)}")
    print(f"deduped {dupes:,} repeated blocks | {rejected_lines} trailing lines dropped | "
          f"{skipped_lang} non-English books skipped")
    sample = args.out.read_text(encoding="utf-8").splitlines()[:12]
    print("sample:\n  " + "\n  ".join(sample))


if __name__ == "__main__":
    main()
