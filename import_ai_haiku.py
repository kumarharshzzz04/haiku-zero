"""Intake and vetting for AI-written haiku dropped into a folder.

    python import_ai_haiku.py                          # scans "haikus from ai"
    python import_ai_haiku.py --dir "some folder"      # any folder of .txt/.json/.jsonl/.csv
    python import_ai_haiku.py --no-unique-lines        # keep poems that reuse a line the batch used

Reads every file in the folder whatever the format, cleans each poem exactly like the
authentic pipeline does (`backend.prepare_data`), then answers one question per poem: is this
really a 5-7-5 haiku by our own counter, and is it already in the corpus?

Outputs
    data/raw_ai/           tier A - exact 5-7-5, unique lines, not in the corpus (training-ready)
    data/raw_ai_near/      tier B - every line within +-1 of 5/7/5 (kept for inspection, not mixed)
    data/ai_intake_stats.json
    AI_HAIKU_INTAKE.md     the report: per file counts, meter spread, repeats, samples, rejects

Re-runnable: it always rescans the folder, so adding batch 3 and running it again is the
whole workflow. Nothing is trained here; `build_synth_corpus.py --ai-repeat N` mixes the result.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.meter_check import distance, pattern  # noqa: E402
from backend.prepare_data import (  # noqa: E402
    reject_reason, dedupe_key, normalize, collect,
)

DEFAULT_DIR = ROOT / "haikus from ai"
AI_OUT = ROOT / "data" / "raw_ai"
NEAR_OUT = ROOT / "data" / "raw_ai_near"
STATS_OUT = ROOT / "data" / "ai_intake_stats.json"
REPORT_OUT = ROOT / "AI_HAIKU_INTAKE.md"
CORPUS = ROOT / "data" / "haikus_clean.txt"
AUTHORED = ROOT / "data" / "raw_synthetic"
# lines that are file furniture, never poems: # comments, "Theme 3: Nature", "Batch 07", a bare
# title ending in a colon. Getting this wrong shifts every poem after it by one line, which is how
# "Theme 1: Nature & Seasons" once became the first line of a poem.
HEADER_RE = re.compile(
    r"^\s*#"                                     # markdown / comment lines
    r"|^\s*[\*\-_=]{3,}\s*$"                       # rules
    r"|^\s*\**\s*(theme|batch|part|section|set|prompt|source|format)\b.*$"
    r"|^\s*[A-Za-z][A-Za-z ]{0,40}:\s*$",
    re.IGNORECASE)
TEXT_FIELDS = ("haiku", "poem", "text", "lines_text", "content")
LIST_FIELDS = ("lines", "haiku_lines")
WORD_RE = re.compile(r"[a-z']+")
STOP = {"the", "a", "an", "and", "of", "in", "on", "its", "it", "is", "are", "was", "were",
        "no", "not", "this", "that", "with", "to", "at", "by", "for", "from", "my", "your",
        "his", "her", "their", "our", "as", "but", "or", "into", "over", "under", "still",
        "its", "i", "you", "we", "they", "he", "she", "there", "here", "all", "one", "of"}


def content_words(line: str) -> list[str]:
    return [w for w in WORD_RE.findall(line.lower()) if w not in STOP and len(w) > 2]


def repeated_content(lines: list[str], allowed_lines: int = 1) -> bool:
    """True when one content word is spent in more than `allowed_lines` lines of the poem.

    The default (1) is the strict AI setting: a poem that says "dusk" in two of its three lines
    is usually a recombination artefact, not a deliberate echo. The authentic corpus fails this
    test far more often than the AI batches do in spirit, which is why the flag is configurable
    (`--allow-echoes` turns it off).
    """
    seen: Counter = Counter()
    for line in lines[:3]:
        seen.update(set(content_words(line)))
    return any(count > allowed_lines for count in seen.values())


def self_repeat(lines: list[str]) -> bool:
    """True when two lines of the same poem are the same materials rearranged.

    Recombination batches (1,000 poems built from a pool of lines) produce poems like
    "the clouds open wide / the clouds sail over the fields / clouds drift over fields":
    every line is meter-perfect and the poem is meaningless. Two lines sharing two content
    words, or opening on the same two words, is the cheap detector for that.
    """
    heads = [tuple(WORD_RE.findall(l.lower())[:2]) for l in lines[:3]]
    if len(set(heads)) < len(heads):
        return True
    sets = [set(content_words(l)) for l in lines[:3]]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if len(sets[i] & sets[j]) >= 2:
                return True
    return False


# ------------------------------------------------------------------ parsing

def from_json_value(value) -> list[str] | None:
    """3 lines out of one JSON value, whatever shape it has."""
    if isinstance(value, str):
        parts = [p.strip() for p in value.splitlines() if p.strip()]
        return parts[:3] if len(parts) >= 3 else None
    if isinstance(value, dict):
        for field in LIST_FIELDS:
            got = value.get(field)
            if isinstance(got, list) and len(got) >= 3:
                return [str(x).strip() for x in got[:3]]
        # a string field, possibly joined by " / " the way several datasets store a whole poem
        for field in TEXT_FIELDS:
            got = value.get(field)
            if isinstance(got, str) and got.strip():
                parts = [p.strip() for p in re.split(r"\r?\n|<br\s*/?>| / ", got) if p.strip()]
                if len(parts) >= 3:
                    return parts[:3]
        # any value that is itself a 3-line string, whatever the key is called
        for got in value.values():
            if isinstance(got, str) and got.count("\n") >= 2:
                parts = [p.strip() for p in got.splitlines() if p.strip()]
                if len(parts) >= 3:
                    return parts[:3]
    return None


def parse_json_file(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[list[str]] = []
    if path.suffix.lower() == ".jsonl":
        for raw in text.splitlines():
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            got = from_json_value(value)
            if got:
                out.append(got)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return out
        items = data if isinstance(data, list) else next(
            (v for v in data.values() if isinstance(v, list)), []) if isinstance(data, dict) else []
        for value in items:
            got = from_json_value(value)
            if got:
                out.append(got)
    return out


def parse_text_file(path: Path) -> list[list[str]]:
    """Blank-line blocks when the file uses them, otherwise groups of 3 lines inside a block.

    Blocks are chunked independently on purpose: letting lines from one block run into the next is
    what turns a stray two-line header into a shifted corpus.
    """
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    blocks = [b for b in re.split(r"\n\s*\n", raw) if b.strip()]
    poems: list[list[str]] = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines()]
        lines = [l for l in lines if l and not HEADER_RE.match(l)]
        if not lines:
            continue
        if len(lines) == 3:
            poems.append(lines)
            continue
        for i in range(0, len(lines) - 2, 3):
            poems.append(lines[i:i + 3])
    return poems


def parse_parquet_file(path: Path) -> list[list[str]]:
    """Hugging Face downloads are parquet; read the text column whatever it is called."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f"  {path.name}: parquet needs pyarrow in this env - skipped", file=sys.stderr)
        return []
    out: list[list[str]] = []
    for row in pq.read_table(path).to_pylist():
        got = from_json_value(row)      # same field hunt as json: lines / haiku / poem / text
        if got:
            out.append(got)
    return out


def parse_file(path: Path) -> list[list[str]]:
    suffix = path.suffix.lower()
    if suffix in (".json", ".jsonl"):
        return parse_json_file(path)
    if suffix == ".parquet":
        return parse_parquet_file(path)
    if suffix in (".txt", ".md", ".text"):
        return parse_text_file(path)
    if suffix in (".csv", ".tsv"):
        rows = []
        delim = "\t" if suffix == ".tsv" else ","
        for block in (path.read_text(encoding="utf-8", errors="replace").splitlines()):
            cells = [c.strip() for c in block.split(delim)]
            cells = [c for c in cells if c]
            if len(cells) >= 3:
                rows.append(cells[:3])
        return rows
    return []


# ------------------------------------------------------------------ vetting

def corpus_keys() -> set[str]:
    """Every poem key we already train on: the authentic corpus plus the authored set."""
    keys: set[str] = set()
    if CORPUS.exists():
        records, _, _ = collect(ROOT / "data" / "raw_classical", lowercase=False)
        keys = {dedupe_key(r) for r in records}
    if AUTHORED.exists():
        authored, _, _ = collect(AUTHORED, lowercase=False)
        keys |= {dedupe_key(r) for r in authored}
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="vet AI-written haiku before mixing them in")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--unique-content", dest="unique_content", action="store_true", default=True,
                    help="reject a poem that spends the same content word in two of its three lines")
    ap.add_argument("--allow-echoes", dest="unique_content", action="store_false",
                    help="keep poems with a repeated content word (the authentic corpus has ~9%)")
    ap.add_argument("--max-line-uses", type=int, default=2,
                    help="how often one line may appear across the poems that are kept (default 2;"
                         " AI batches recycle stock lines, and a line trained 30 times is a template)")
    ap.add_argument("--report", type=Path, default=REPORT_OUT)
    ap.add_argument("--show", type=int, default=12, help="samples per tier in the report")
    args = ap.parse_args(argv)

    folder = args.dir
    if not folder.exists():
        raise SystemExit(f"{folder} does not exist - drop the haiku files there first")
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"no files in {folder}")

    known = corpus_keys()
    print(f"existing corpus keys: {len(known)} poems")

    per_file: dict[str, dict] = {}
    rejected: Counter = Counter()
    tier_a: list[list[str]] = []
    tier_b: list[list[str]] = []
    accepted_lines: Counter = Counter()
    line_counts: Counter = Counter()
    reject_samples: list[tuple[str, str, str]] = []

    for path in files:
        if path.suffix.lower() not in (".txt", ".md", ".text", ".json", ".jsonl", ".csv", ".tsv"):
            continue
        raw_poems = parse_file(path)
        counts = Counter()
        kept_a, kept_b = [], []
        for lines in raw_poems:
            counts["rows"] += 1
            lines = normalize(lines, lowercase=False)
            reason = reject_reason(lines)
            if reason:
                counts[reason] += 1
                rejected[reason] += 1
                if len(reject_samples) < 40:
                    reject_samples.append((path.name, reason, " / ".join(lines)))
                continue
            key = dedupe_key(lines)
            if key in known:
                counts["already_in_corpus"] += 1
                rejected["already_in_corpus"] += 1
                continue
            if self_repeat(lines):
                counts["self_repeat"] += 1
                rejected["self_repeat"] += 1
                continue
            if args.unique_content and repeated_content(lines):
                counts["echoed_word"] += 1
                rejected["echoed_word"] += 1
                continue
            pat = pattern(lines)
            if distance(pat) == 0:
                if any(accepted_lines[l.lower()] >= args.max_line_uses for l in lines):
                    counts["reused_line"] += 1
                    rejected["reused_line"] += 1
                    continue
                known.add(key)
                for l in lines:
                    accepted_lines[l.lower()] += 1
                kept_a.append(lines)
                counts["tier_a"] += 1
            elif all(abs(pat[i] - (5, 7, 5)[i]) <= 1 for i in range(3)):
                kept_b.append(lines)
                counts["tier_b"] += 1
            else:
                counts["off_meter"] += 1
                rejected["off_meter"] += 1
            for l in lines:
                line_counts[l.lower()] += 1
        tier_a.extend(kept_a)
        tier_b.extend(kept_b)
        per_file[path.name] = dict(counts)
        print(f"  {path.name}: {counts['rows']} rows -> {counts['tier_a']} exact, "
              f"{counts['tier_b']} near, " + ", ".join(
                  f"{k} {v}" for k, v in counts.items() if k not in ("rows", "tier_a", "tier_b")))

    # raw-source format (blank line between poems), NOT the <eos> corpus format: these files are
    # read back by prepare_data.collect(), which splits on blank lines. Writing them as corpus
    # lines silently collapsed 125 poems into one candidate.
    def write_raw(records: list[list[str]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n".join("\n".join(lines) for lines in records) + "\n", encoding="utf-8")

    write_raw(tier_a, AI_OUT / "ai_intake.txt")
    write_raw(tier_b, NEAR_OUT / "ai_intake_near.txt")

    repeats = {l: c for l, c in line_counts.items() if c > 1}
    stats = {
        "folder": str(folder.relative_to(ROOT)) if str(folder).startswith(str(ROOT)) else str(folder),
        "files": per_file,
        "tier_a_exact": len(tier_a),
        "tier_b_near": len(tier_b),
        "rejected": dict(rejected),
        "unique_lines_in_intake": len(line_counts),
        "repeated_lines": len(repeats),
        "top_repeated_lines": [{"line": l, "count": c} for l, c in
                               sorted(repeats.items(), key=lambda kv: -kv[1])[:10]],
        "max_line_uses": args.max_line_uses,
        "unique_content_words": args.unique_content,
    }
    STATS_OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    report = [
        "# AI haiku intake",
        "",
        f"Source folder: `{stats['folder']}` — scanned {len(per_file)} file(s).",
        "",
        "Every poem is cleaned with the same code the authentic pipeline uses "
        "(`backend.prepare_data`), then measured with `backend.meter_check`, then looked up "
        "against the corpus we already train on.",
        "",
        f"- **tier A (exact 5-7-5, unique lines): {len(tier_a)}** → `data/raw_ai/` (training-ready)",
        f"- tier B (every line within ±1): {len(tier_b)} → `data/raw_ai_near/` (inspection only)",
        f"- dropped: " + ", ".join(f"{k} {v}" for k, v in rejected.most_common()) if rejected else "- dropped: none",
        "",
        "## Per file",
        "",
        "| file | rows | exact 5-7-5 | near | off meter | already in corpus | reused line | junk/length |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in per_file.items():
        junk = sum(c.get(k, 0) for k in ("not_3_lines", "no_letters", "line_too_long",
                                         "too_much_non_ascii", "url_or_mention", "self_repeat",
                                         "echoed_word"))
        report.append(f"| `{name}` | {c.get('rows', 0)} | {c.get('tier_a', 0)} | {c.get('tier_b', 0)} | "
                      f"{c.get('off_meter', 0)} | {c.get('already_in_corpus', 0)} | "
                      f"{c.get('reused_line', 0)} | {junk} |")

    if repeats:
        report += ["", "## Lines the batch reuses", "",
                   "| line | times |", "|---|---|"]
        report += [f"| {l} | {c} |" for l, c in sorted(repeats.items(), key=lambda kv: -kv[1])[:10]]

    rng = __import__("random").Random(7)
    if tier_a:
        report += ["", f"## {min(args.show, len(tier_a))} samples from tier A", ""]
        for i, poem in enumerate(rng.sample(tier_a, min(args.show, len(tier_a))), 1):
            report.append(f"{i}. " + " / ".join(poem))
    if reject_samples:
        report += ["", "## What was dropped, and why", ""]
        for name, reason, text in reject_samples[:15]:
            report.append(f"- `{reason}` ({name}): {text}")

    report += ["", "## Next", "",
               "```bash",
               "python build_synth_corpus.py --ai-repeat 6 --repeat 6   # rebuild the mix with this batch",
               "python -m backend.train --corpus data/haikus_mix.txt --init-from checkpoints_mix/best.pt \\",
               "  --ckpt-dir checkpoints_mix --lr 6e-5 --embed-dim 384 --layers 8 --ffn-dim 1536 --epochs 12",
               "```",
               "", "Drop the next batch in the folder and re-run `python import_ai_haiku.py`; nothing "
               "is trained until you ask for the mix.", ""]
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\ntier A {len(tier_a)} exact 5-7-5  ->  {AI_OUT / 'ai_intake.txt'}")
    print(f"tier B {len(tier_b)} within +-1   ->  {NEAR_OUT / 'ai_intake_near.txt'}")
    print(f"dropped: {dict(rejected)}")
    print(f"lines reused inside the batch: {len(repeats)}")
    print(f"wrote {args.report.name} and {STATS_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
