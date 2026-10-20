"""Check every authored haiku in data/raw_synthetic against the project's own meter.

The point is not to trust my counting: each line goes through backend.meter_check.line_syllables,
the same counter the strict decoder enforces, and anything that is not exactly 5-7-5 is printed so
it can be fixed here rather than quietly entering the training corpus.

Usage
    python scripts/data/synth_verify.py              # report only
    python scripts/data/synth_verify.py --json       # also write data/synth_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.meter_check import line_syllables  # noqa: E402
from backend.prepare_data import dedupe_key, normalize, reject_reason  # noqa: E402

RAW = ROOT / "data" / "raw_synthetic"
REPORT = ROOT / "data" / "synth_report.json"


def blocks(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    out = []
    for chunk in text.split("\n\n"):
        lines = [l.rstrip() for l in chunk.strip().splitlines() if l.strip()]
        if lines:
            out.append(lines[:3])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(RAW.glob("*.txt"))
    if not files:
        raise SystemExit(f"no authored haiku in {RAW}")

    total = good = 0
    seen: set[str] = set()
    bad: list[tuple[str, list[str], tuple[int, ...], str | None]] = []
    by_file: dict[str, dict[str, int]] = {}
    dupes = 0

    for path in files:
        count = exact = 0
        for raw_lines in blocks(path):
            total += 1
            count += 1
            lines = normalize(raw_lines, lowercase=False)
            reason = reject_reason(lines)
            pattern = tuple(line_syllables(l) for l in lines) if not reason else (0, 0, 0)
            if reason is None and pattern != (5, 7, 5):
                reason = f"{pattern[0]}-{pattern[1]}-{pattern[2]}"
            key = dedupe_key(lines)
            if key in seen:
                dupes += 1
                reason = reason or "duplicate"
            seen.add(key)
            if reason is None:
                good += 1
                exact += 1
            else:
                bad.append((path.name, lines, pattern, reason))
        by_file[path.name] = {"poems": count, "exact": exact}

    print(f"{good}/{total} exact 5-7-5 across {len(files)} files   ({dupes} duplicates)")
    for name, stats in by_file.items():
        print(f"  {name}: {stats['exact']}/{stats['poems']}")
    if bad:
        print(f"\n{len(bad)} to fix:")
        for name, lines, pattern, reason in bad:
            print(f"  [{name}] {reason}: {' / '.join(lines)}")

    if args.json:
        REPORT.write_text(
            json.dumps(
                {
                    "poems": total,
                    "exact": good,
                    "duplicates": dupes,
                    "by_file": by_file,
                    "problems": [
                        {"file": n, "lines": ls, "pattern": list(p), "reason": r}
                        for n, ls, p, r in bad
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {REPORT}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
