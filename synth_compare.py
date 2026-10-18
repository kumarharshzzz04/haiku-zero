"""Compare the production checkpoint with the synthetic-mix fine-tune.

    python synth_compare.py                                  # both models, 5 prompts
    python synth_compare.py --n 4 --models checkpoints,checkpoints_mix

Prints, per prompt, the poems each model writes with strict 5-7-5 decoding, then a
memorisation report: how many generated lines copy a line of the authored corpus
(data/raw_synthetic) or of the authentic corpus (data/haikus_clean.txt) verbatim.
A fine-tune on 180 authored poems fails if it regurgitates them.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import generate  # noqa: E402

ROOT = Path(__file__).resolve().parent
PROMPTS = ["autumn", "breeze", "old pond", "snow", ""]


def corpus_lines() -> tuple[set[str], set[str], set[str]]:
    authored: set[str] = set()
    ai: set[str] = set()
    for folder in (ROOT / "data" / "raw_synthetic",):
        for path in sorted(folder.glob("*.txt")):
            for block in path.read_text(encoding="utf-8").split("\n\n"):
                authored.update(l.strip().lower() for l in block.splitlines() if l.strip())
    for folder in (ROOT / "data" / "raw_ai", ROOT / "data" / "raw_ai_near"):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.txt")):
            for block in path.read_text(encoding="utf-8").split("\n\n"):
                ai.update(l.strip().lower() for l in block.splitlines() if l.strip())
    authentic: set[str] = set()
    clean = ROOT / "data" / "haikus_clean.txt"
    if clean.exists():
        text = clean.read_text(encoding="utf-8")
        authentic.update(l.strip().lower() for l in text.splitlines()
                         if l.strip() and l.strip() != "<eos>")
    return authored, authentic, ai


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="production vs synthetic-mix comparison")
    ap.add_argument("--models", default="checkpoints,checkpoints_mix")
    ap.add_argument("--prompts", default="|".join(PROMPTS))
    ap.add_argument("--n", type=int, default=5, help="poems per prompt per model")
    ap.add_argument("--best-of", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "synth_compare.md")
    args = ap.parse_args(argv)

    authored, authentic, ai = corpus_lines()
    print(f"authored lines {len(authored)}, authentic lines {len(authentic)}, AI-vetted lines {len(ai)}")
    prompts = args.prompts.split("|")
    out: list[str] = ["# Production vs synthetic-mix fine-tune", ""]
    copies = Counter()

    for name in args.models.split(","):
        ckpt = ROOT / name / "best.pt"
        if not ckpt.exists():
            print(f"{ckpt} missing")
            continue
        model, tok, cfg, dev, payload = generate.load_model(ckpt)
        print(f"\n=== {name} (epoch {payload.get('epoch')}, val {payload.get('val_loss'):.4f}) ===")
        out.append(f"## {name} (epoch {payload.get('epoch')}, val {payload.get('val_loss'):.4f})")
        for prompt in prompts:
            label = prompt or "(no seed)"
            print(f"\n-- seed {label!r}")
            out.append(f"\n### seed `{label}`")
            for i in range(args.n):
                lines, pat, ok = generate.best_of_haiku(model, tok, prompt, n=args.best_of,
                                                        strict=True, device=dev, seed=i * 13 + 3)
                tag = " ".join(map(str, pat)) if len(pat) == 3 else "?"
                mark = "5-7-5" if ok else f"near {tag}"
                text = " / ".join(lines)
                print(f"   [{mark}] {text}")
                out.append(f"- [{mark}] {text}")
                for line in lines:
                    low = line.strip().lower()
                    if low in authored:
                        copies["authored"] += 1
                        print(f"       ^ verbatim from the authored corpus: {line}")
                    if low in ai:
                        copies["ai"] += 1
                    if low in authentic:
                        copies["authentic"] += 1

    total = copies["authored"] + copies["authentic"] + copies["ai"]
    out.append("\n## memorisation report\n")
    out.append(f"- lines copied verbatim from the 180 authored poems: **{copies['authored']}**")
    out.append(f"- lines copied verbatim from the vetted AI batches: **{copies['ai']}**")
    out.append(f"- lines copied verbatim from the authentic corpus: **{copies['authentic']}**")
    print(f"\nverbatim copies: authored {copies['authored']}, ai {copies['ai']}, "
          f"authentic {copies['authentic']} (of {total} lines drawn from known corpora)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
