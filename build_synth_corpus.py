"""Mix the authentic haiku corpus with the authored set and any vetted AI batches.

- authentic  = data/raw_classical (the 40.9k classical corpus the live model was fine-tuned on)
- synthetic  = data/raw_synthetic (authored for this run, every line checked to exact 5-7-5)
- ai         = data/raw_ai (vetted AI batches, written by import_ai_haiku.py)
               data/raw_ai_near holds the within-+-1 tier, off by default (--ai-tier b)

Both added sets are repeated so they carry more than their row count would; the authentic poems
appear once, unchanged. Poems duplicated across sources are dropped from the added side, so
nothing is trained twice by accident.

Usage
    python build_synth_corpus.py                          # authored x3, AI tier A x3
    python build_synth_corpus.py --repeat 6 --ai-repeat 6
    python build_synth_corpus.py --ai-tier ab --ai-repeat 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.prepare_data import collect, dedupe_key, meter_filter, write_corpus  # noqa: E402

AUTHENTIC_RAW = ROOT / "data" / "raw_classical"
SYNTH_RAW = ROOT / "data" / "raw_synthetic"
AI_RAW = ROOT / "data" / "raw_ai"
AI_NEAR_RAW = ROOT / "data" / "raw_ai_near"
AUTHENTIC_OUT = ROOT / "data" / "haikus_classical_mix_base.txt"
SYNTH_OUT = ROOT / "data" / "haikus_synthetic.txt"
AI_OUT = ROOT / "data" / "haikus_ai.txt"
MIX_OUT = ROOT / "data" / "haikus_mix.txt"
SAMPLES_OUT = ROOT / "data" / "synthetic_samples.md"
AI_SAMPLES_OUT = ROOT / "data" / "ai_samples.md"
STATS_OUT = ROOT / "data" / "synth_mix_stats.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3, help="how often the authored poems appear")
    ap.add_argument("--ai-repeat", type=int, default=3, help="how often the AI-vetted poems appear (0 = leave them out)")
    ap.add_argument("--ai-tier", default="a", choices=("a", "b", "ab"),
                    help="a = exact 5-7-5 only, b = also the within-+-1 tier, ab = both")
    ap.add_argument("--samples", type=int, default=30)
    args = ap.parse_args(argv)

    print("authentic:")
    authentic, a_sources, a_rejected = collect(AUTHENTIC_RAW, lowercase=False)
    print("synthetic:")
    authored, s_sources, s_rejected = collect(SYNTH_RAW, lowercase=False)

    seen = {dedupe_key(lines) for lines in authentic}
    novel = []
    overlaps = 0
    for lines in authored:
        key = dedupe_key(lines)
        if key in seen:
            overlaps += 1
            continue
        seen.add(key)
        novel.append(lines)

    ai_pool: list[list[str]] = []
    ai_sources: dict = {}
    ai_rejected: dict = {}
    for tier, folder in (("a", AI_RAW), ("b", AI_NEAR_RAW)):
        if tier not in args.ai_tier or not folder.exists():
            continue
        print(f"ai tier {tier}:")
        got, src, rej = collect(folder, lowercase=False)
        ai_pool.extend(got)
        for k, v in src.items():
            ai_sources[k] = ai_sources.get(k, 0) + v
        for k, v in rej.items():
            ai_rejected[k] = ai_rejected.get(k, 0) + v

    ai_novel = []
    ai_overlaps = 0
    for lines in ai_pool:
        key = dedupe_key(lines)
        if key in seen:
            ai_overlaps += 1
            continue
        seen.add(key)
        ai_novel.append(lines)

    # exactness is enforced by synth_verify.py (all 180 authored pass) and by import_ai_haiku.py
    # for the AI tier; this only mixes the sources
    ai_repeat = max(0, args.ai_repeat) if ai_novel else 0
    mix = authentic + novel * max(1, args.repeat) + ai_novel * ai_repeat

    write_corpus(authentic, AUTHENTIC_OUT)
    write_corpus(novel, SYNTH_OUT)
    write_corpus(ai_novel, AI_OUT)
    write_corpus(mix, MIX_OUT)

    rng = random.Random(20261017)
    shown = rng.sample(novel, min(args.samples, len(novel)))
    lines = [
        "# Authored haiku — verification and samples",
        "",
        f"- authored poems: **{len(novel)}** (from {len(authored)} written, {overlaps} dropped as duplicates of the authentic corpus)",
        "- every line checked with `backend/meter_check.line_syllables`: **all exact 5-7-5** (`synth_verify.py`)",
        f"- corpus: {len(authentic)} authentic + {len(novel)} x {args.repeat} authored = **{len(mix)} poems**",
        "",
        f"## {len(shown)} random samples (seed 20261017)",
        "",
    ]
    for i, poem in enumerate(shown, 1):
        lines.append(f"{i}. " + " / ".join(poem))
    SAMPLES_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if ai_novel:
        ai_shown = rng.sample(ai_novel, min(args.samples, len(ai_novel)))
        ai_lines = [
            "# AI-vetted haiku — what made it into the mix",
            "",
            f"- vetted by `import_ai_haiku.py`: tier **{args.ai_tier}**, {len(ai_novel)} poems kept"
            f" ({ai_overlaps} dropped as duplicates of the corpus)",
            f"- repeated x{ai_repeat} in `{MIX_OUT.name}`",
            "",
            f"## {len(ai_shown)} random samples (seed 20261017)",
            "",
        ]
        for i, poem in enumerate(ai_shown, 1):
            ai_lines.append(f"{i}. " + " / ".join(poem))
        AI_SAMPLES_OUT.write_text("\n".join(ai_lines) + "\n", encoding="utf-8")

    stats = {
        "authentic": len(authentic),
        "authored": len(novel),
        "authored_written": len(authored),
        "overlaps_dropped": overlaps,
        "repeat": args.repeat,
        "ai_tier": args.ai_tier,
        "ai": len(ai_novel),
        "ai_written": sum(ai_sources.values()),
        "ai_overlaps_dropped": ai_overlaps,
        "ai_repeat": ai_repeat,
        "ai_share": round(len(ai_novel) * ai_repeat / len(mix), 4) if mix else 0.0,
        "mix_total": len(mix),
        "synthetic_share": round(len(novel) * args.repeat / len(mix), 4),
        "authentic_sources": dict(a_sources),
        "synthetic_sources": dict(s_sources),
        "ai_sources": ai_sources,
        "authentic_rejected": dict(a_rejected),
        "synthetic_rejected": dict(s_rejected),
        "ai_rejected": ai_rejected,
    }
    STATS_OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    parts = [f"authentic {len(authentic)}", f"authored {len(novel)} x {args.repeat}"]
    if ai_novel:
        parts.append(f"ai tier {args.ai_tier} {len(ai_novel)} x {ai_repeat}")
    print(f"\n" + " + ".join(parts) + f" = {len(mix)} poems")
    print(f"shares - synthetic {100 * stats['synthetic_share']:.1f}%"
          + (f", ai {100 * stats['ai_share']:.1f}%" if ai_novel else ""))
    print(f"wrote {MIX_OUT.name}, {SYNTH_OUT.name}, "
          + (f"{AI_OUT.name}, {AI_SAMPLES_OUT.name}, " if ai_novel else "")
          + f"{SAMPLES_OUT.name}, {STATS_OUT.name}")
    print(f"\n--- {len(shown)} samples ---")
    for i, poem in enumerate(shown, 1):
        print(f"{i:>2}. " + " / ".join(poem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
