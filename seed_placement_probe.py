"""Where should the word you type go: opening the poem, or somewhere inside it?

    python seed_placement_probe.py                            # 12 corpus words, 4 poems, best-of 5
    python seed_placement_probe.py --words sea,moon --per-word 2
    python seed_placement_probe.py --checkpoint checkpoints/best.pt

Two placements, same model, same reranker, same draw seeds:

  first  (seed_first=True)   the word is written as the poem's opening word and the model
                             continues it - "sea roars loud / river runs red"
  topic  (seed_first=False)  the word is context ([<bos> <seed> word \\n]) and the model
                             writes a poem ABOUT it, so the word lands wherever it fits
                             - "the white sea shines / ..." Half of every training batch is
                             exactly this shape (dataset.encode_records puts a random corpus
                             word in front of a poem), so this is the placement the model was
                             trained for; the question is whether it is reliable.

Reliability is the thing to measure, so `topic` gets a second column that prices the
guarantee: how many best-of draws it takes before a topic-mode poem both says the word and
holds exact 5-7-5. If that number is small, integration is free; if it is large, writing the
word first is the honest way to keep a typed word in the poem.

Columns
    exact     strict 5-7-5
    seed      the poem says the seed word (or an inflection)
    first     the seed word opens the poem
    inside    the poem says it, but not as the first word (the "integrated" shape)
    missing   the poem never says it
    repeat    content word said twice, invented words, mean log-likelihood
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import generate, meter_check  # noqa: E402

ROOT = Path(__file__).resolve().parent


def rel(path: Path) -> str:
    """Displayable path: relative to the repo when possible (a bare --out crashed on this)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
# all of these occur in data/haikus_mix.txt (42,760 poems); document frequency in brackets:
# rain 1527, night 1435, moon 1321, wind 1121, light 1073, snow 842, sea 416, river 376,
# stone 265, bird 241, fire 227, cloud 194
DEFAULT_WORDS = "rain,night,moon,wind,light,snow,sea,river,stone,bird,fire,cloud"
WORD_RE = generate.WORD_RE


def placements(lines, prompt: str) -> list[tuple[int, int]]:
    """(line index, word index) of every hit of the seed word, using the adherence rule."""
    head = prompt.strip().lower().split()[0]
    stem = head[:max(4, len(head) - 2)]
    hits = []
    for i, line in enumerate(lines):
        for j, word in enumerate(WORD_RE.findall(str(line).lower())):
            if word == head or word.startswith(stem):
                hits.append((i, j))
    return hits


def draw(model, tok, prompt: str, seed_first: bool, n: int, seed: int, dev):
    lines, pat, exact = generate.best_of_haiku(model, tok, prompt, n=n, strict=True,
                                               seed_first=seed_first, device=dev, seed=seed)
    ctx = generate.seed_context(tok, prompt) if prompt else [tok.token_to_id("<bos>")]
    hits = placements(lines, prompt)
    return {
        "lines": lines,
        "exact": bool(exact) and meter_check.is_haiku(lines),
        "hits": hits,
        "first": hits == [(0, 0)],
        "repeat": generate.repeated_content_rate(lines) > 0,
        "invented": generate.attested_fraction(lines) < 1.0,
        "ll": generate.score_text(model, tok, ctx, "\n".join(lines)),
        "pat": tuple(pat) + (0,) * (3 - len(pat)),
    }


def forced_topic(model, tok, prompt: str, n: int, per_word: int, max_attempts: int, dev):
    """Attempts needed until a topic-mode poem says the word AND holds 5-7-5 (None = gave up)."""
    used = []
    for i in range(per_word):
        attempts = None
        for attempt in range(1, max_attempts + 1):
            row = draw(model, tok, prompt, False, n, seed=1000 + i * 13 + attempt, dev=dev)
            if row["hits"] and row["exact"]:
                attempts = attempt
                break
        used.append(attempts)
    landed = [a for a in used if a]
    return {"attempts": used, "landed": landed,
            "mean": (sum(landed) / len(landed)) if landed else None,
            "best": (min(landed) if landed else None)}


def summarise(rows: list[dict]) -> dict:
    total = len(rows)
    landings = Counter()
    for r in rows:
        if r["hits"]:
            landings[f"line {r['hits'][0][0] + 1}"] += 1
    return {
        "n": total,
        "exact": sum(r["exact"] for r in rows) / total,
        "seed": sum(bool(r["hits"]) for r in rows) / total,
        "first": sum(r["first"] for r in rows) / total,
        "inside": sum(bool(r["hits"]) and not r["first"] for r in rows) / total,
        "missing": sum(not r["hits"] for r in rows) / total,
        "repeat": sum(r["repeat"] for r in rows) / total,
        "invented": sum(r["invented"] for r in rows) / total,
        "ll": sum(r["ll"] for r in rows) / total,
        "twice": sum(len(r["hits"]) > 1 for r in rows) / total,
        "landings": landings,
    }


def landing_line(summary: dict) -> str:
    landed = sum(summary["landings"].values())
    if not landed:
        return ""
    parts = [f"{name} {100 * n / landed:.0f}%" for name, n in sorted(summary["landings"].items())]
    return ("first hit on " + ", ".join(parts)
            + f"; opens the poem {100 * summary['first']:.0f}%; "
            + f"says it twice {100 * summary['twice']:.0f}%")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="seed placement: opening word vs topic placement")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints_combo" / "best.pt")
    ap.add_argument("--words", default=DEFAULT_WORDS)
    ap.add_argument("--per-word", type=int, default=4, help="poems per word per placement")
    ap.add_argument("--best-of", type=int, default=5)
    ap.add_argument("--max-attempts", type=int, default=6, help="cap for the forced-topic loop")
    ap.add_argument("--samples", type=int, default=2, help="poems to print per word")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "seed_placement.md")
    args = ap.parse_args(argv)

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    model, tok, cfg, dev, payload = generate.load_model(args.checkpoint)
    print(f"loaded {rel(args.checkpoint)} (epoch {payload.get('epoch')}, "
          f"val {payload.get('val_loss'):.4f}) on {dev}")
    print(f"{len(words)} words x {args.per_word} poems x best-of {args.best_of}, strict 5-7-5\n")

    report = [f"# Seed placement probe - {args.checkpoint.parent.name}",
              "",
              f"epoch {payload.get('epoch')}, val {payload.get('val_loss'):.4f}; "
              f"{len(words)} words x {args.per_word} poems, best-of {args.best_of}, strict 5-7-5",
              ""]
    modes = {"first": True, "topic": False}
    agg = {name: [] for name in modes}
    per_word = {name: {} for name in modes}
    drawn: dict[str, dict[str, list[dict]]] = {name: {} for name in modes}
    forced: dict[str, dict] = {}

    for w in words:
        for name, seed_first in modes.items():
            rows = [draw(model, tok, w, seed_first, args.best_of, seed=i * 13 + 1, dev=dev)
                    for i in range(args.per_word)]
            agg[name].extend(rows)
            per_word[name][w] = summarise(rows)
            drawn[name][w] = rows
        forced[w] = forced_topic(model, tok, w, args.best_of, args.per_word, args.max_attempts, dev)
        s = per_word["topic"][w]
        print(f"{w:<7} first: exact {100 * per_word['first'][w]['exact']:3.0f}%  "
              f"topic: exact {100 * s['exact']:3.0f}%  says it {100 * s['seed']:3.0f}%  "
              f"inside {100 * s['inside']:3.0f}%  missing {100 * s['missing']:3.0f}%  "
              f"forced attempts {forced[w]['attempts']}")

    print()
    header = (f"{'placement':<10}{'exact':>8}{'says it':>9}{'first':>8}{'inside':>8}"
              f"{'missing':>9}{'repeat':>8}{'invent':>8}{'mean LL':>9}")
    print(header)
    print("-" * len(header))
    report += [f"| placement | poems | exact 5-7-5 | says it | opens with it | says it inside | "
               f"missing | repeat | invented | mean LL |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("first", "topic"):
        a = summarise(agg[name])
        print(f"{name:<10}{100 * a['exact']:>7.0f}%{100 * a['seed']:>8.0f}%{100 * a['first']:>7.0f}%"
              f"{100 * a['inside']:>7.0f}%{100 * a['missing']:>8.0f}%{100 * a['repeat']:>7.0f}%"
              f"{100 * a['invented']:>7.0f}%{a['ll']:>9.2f}")
        report.append(f"| {name} | {a['n']} | {100 * a['exact']:.0f}% | {100 * a['seed']:.0f}% | "
                      f"{100 * a['first']:.0f}% | {100 * a['inside']:.0f}% | "
                      f"{100 * a['missing']:.0f}% | {100 * a['repeat']:.0f}% | "
                      f"{100 * a['invented']:.0f}% | {a['ll']:.2f} |")
        where = landing_line(a)
        if where:
            print(f"  {name}: {where}")
            report.append(f"  - {name}: {where}")
    landed = sum(len(f["landed"]) for f in forced.values())
    total = sum(len(f["attempts"]) for f in forced.values())
    print(f"\nforced topic (word guaranteed + exact 5-7-5): landed {landed}/{total}, "
          f"mean {sum(f['mean'] for f in forced.values() if f['mean']) / max(1, sum(1 for f in forced.values() if f['mean'])):.1f} tries")
    report += ["", "## guaranteeing the word in topic mode", "",
               f"best-of-{args.best_of} draws until a poem says the word and holds exact 5-7-5 "
               f"(cap {args.max_attempts}): landed **{landed}/{total}**; "
               f"mean {sum(f['mean'] for f in forced.values() if f['mean']) / max(1, sum(1 for f in forced.values() if f['mean'])):.1f} draws",
               "", "| word | attempts per poem (None = gave up) |", "|---|---|"]
    for w, f in forced.items():
        report.append(f"| {w} | {f['attempts']} |")

    report += ["", "## samples", ""]
    for w in words:
        report += [f"### {w}", ""]
        for name in modes:
            report += [f"**{name}**", ""]
            for row in drawn[name][w][: args.samples]:
                tag = "exact" if row["exact"] else f"near {row['pat']}"
                where = row["hits"][0] if row["hits"] else "never says it"
                report += [f"- [{tag}, word at {where}] " + " / ".join(row["lines"])]
            report += [""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {rel(args.out)}")


if __name__ == "__main__":
    main()
