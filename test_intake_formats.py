"""Regression test for the AI-haiku intake: every format they actually download.

    python test_intake_formats.py

The batches arrive as txt (blank-line blocks or bare triplets, with `Theme 3:` and `#` headers),
json (list of objects, dict of lists, " / "-joined strings), jsonl and csv/tsv. Two bugs this
guards against, both found by running it:

- headers becoming a poem line (`Theme 1: Nature` was the first line of a poem, and every poem
  after it shifted by one);
- a jsonl record being dropped because its value was a bare 3-line string rather than an object.

It then checks the artifacts the real run wrote: every tier-A poem is exact 5-7-5, no furniture
line survived, and the counts agree with the stats file.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.meter_check import distance, pattern               # noqa: E402
from import_ai_haiku import HEADER_RE, parse_file, self_repeat  # noqa: E402

A = ["one red leaf falls", "the pond holds a quiet sky", "winter comes too soon"]
B = ["old gate swings open", "a moth circles the warm lamp", "the house keeps the dusk"]
C = ["the last bus is gone", "rain writes on the empty street", "a cat waits for me"]


def check(name: str, got: list[list[str]], want: int) -> None:
    assert len(got) == want, f"{name}: parsed {len(got)} poems, want {want}: {got}"
    for poem in got:
        assert len(poem) == 3, f"{name}: not 3 lines: {poem}"
        for line in poem:
            assert not HEADER_RE.match(line), f"{name}: furniture survived as a line: {line!r}"
            assert line.strip() == line and line, f"{name}: dirty line {line!r}"
    print(f"  ok  {name:<28} {len(got)} poems")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)

        (d / "blank_blocks.txt").write_text(
            "# batch 07\nTheme 1: Nature & Seasons\n\n"
            + "\n".join(A) + "\n\n" + "\n".join(B) + "\n\nBatch 08:\n\n" + "\n".join(C) + "\n",
            encoding="utf-8")
        check("txt, blank-line blocks", parse_file(d / "blank_blocks.txt"), 3)

        (d / "bare_triplets.txt").write_text(
            "Theme 2: Night\n" + "\n".join(A + B + C) + "\n", encoding="utf-8")
        check("txt, bare triplets", parse_file(d / "bare_triplets.txt"), 3)

        (d / "list_of_objects.json").write_text(json.dumps(
            [{"lines": A}, {"lines": B}, {"lines": C}]), encoding="utf-8")
        check("json, list of {lines}", parse_file(d / "list_of_objects.json"), 3)

        (d / "dict_of_lists.json").write_text(json.dumps(
            {"haikus": [{"haiku": "\n".join(A)}, {"haiku": "\n".join(B)},
                        {"haiku": "\n".join(C)}]}), encoding="utf-8")
        check("json, {haikus: [{haiku}]}", parse_file(d / "dict_of_lists.json"), 3)

        (d / "slashed.json").write_text(json.dumps(
            [{"poem": " / ".join(A)}, {"poem": " / ".join(B)}, {"poem": " / ".join(C)}]),
            encoding="utf-8")
        check("json, ' / '-joined poem", parse_file(d / "slashed.json"), 3)

        (d / "bare_strings.jsonl").write_text(
            "\n".join(json.dumps("\n".join(p)) for p in (A, B, C)) + "\n", encoding="utf-8")
        check("jsonl, bare 3-line strings", parse_file(d / "bare_strings.jsonl"), 3)

        (d / "objects.jsonl").write_text(
            "\n".join(json.dumps({"text": "\n".join(p)}) for p in (A, B, C)) + "\n",
            encoding="utf-8")
        check("jsonl, {text} records", parse_file(d / "objects.jsonl"), 3)

        (d / "rows.csv").write_text(
            "\n".join(",".join(p) for p in (A, B, C)) + "\n", encoding="utf-8")
        check("csv, three columns", parse_file(d / "rows.csv"), 3)

        (d / "rows.tsv").write_text(
            "\n".join("\t".join(p) for p in (A, B, C)) + "\n", encoding="utf-8")
        check("tsv, three columns", parse_file(d / "rows.tsv"), 3)

    # the recombination detector the vetter leans on
    assert self_repeat(["the clouds open wide", "the clouds sail over the fields",
                        "clouds drift over fields"]), "recombination poem not caught"
    assert not self_repeat(A), "clean poem flagged as recombination"
    print("  ok  self_repeat detector")

    # artifacts of the real run
    stats = json.loads((ROOT / "data" / "ai_intake_stats.json").read_text(encoding="utf-8"))
    tier_a = [p.splitlines() for p in
              (ROOT / "data" / "raw_ai" / "ai_intake.txt").read_text(encoding="utf-8").split("\n\n")
              if p.strip()]
    tier_b = [p.splitlines() for p in
              (ROOT / "data" / "raw_ai_near" / "ai_intake_near.txt").read_text(encoding="utf-8").split("\n\n")
              if p.strip()]
    assert len(tier_a) == stats["tier_a"], (len(tier_a), stats["tier_a"])
    assert len(tier_b) == stats["tier_b"], (len(tier_b), stats["tier_b"])
    bad = [p for p in tier_a if len(p) != 3 or distance(pattern(p)) != 0]
    assert not bad, f"tier A contains off-meter or malformed poems: {bad[:3]}"
    furniture = [(p, l) for p in tier_a + tier_b for l in p if HEADER_RE.match(l)]
    assert not furniture, f"headers leaked into the intake: {furniture[:3]}"
    print(f"  ok  real intake: {len(tier_a)} tier A (all exact 5-7-5), {len(tier_b)} tier B")

    print("all intake format checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
