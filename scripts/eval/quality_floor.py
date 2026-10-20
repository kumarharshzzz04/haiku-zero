"""Regression floor for decoder quality.

    python scripts/eval/quality_floor.py                # check the live checkpoint against logs/quality_floor.json
    python scripts/eval/quality_floor.py --write        # re-record the baseline
    python scripts/eval/quality_floor.py --self-test    # prove the comparator can fail

The decoder keeps several promises at once (exact 5-7-5, no repeats, no invented words, no junk,
seed present). Each is produced by a different rule, so a change in one place can quietly cost
another. This measures the same rates on a fixed set of seeds and exits 1 if any of them slips.
The baseline records its checkpoint, so a promotion has to re-record it on purpose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend import generate, meter_check  # noqa: E402
BASELINE = ROOT / "logs" / "quality_floor.json"

# Measured floors: exact/seed are 100%, repeat/nonword/junk are 0%. Thresholds leave room for
# sampling noise on a 24-poem set without making any of them meaningless (a repeat in 24 poems is
# 4%, so the ceiling is 6%; two invented words would already fail).
FLOORS = {"exact": ("min", 0.98), "seed": ("min", 0.95), "repeat": ("max", 0.06),
          "nonword": ("max", 0.04), "junk": ("max", 0.02)}


def measure(ckpt: Path, seeds: list[str], n: int) -> dict:
    model, tok, _cfg, dev, payload = generate.load_model(ckpt)
    rows = []
    for prompt in seeds:
        for i in range(n):
            lines, _pat, exact = generate.best_of_haiku(
                model, tok, prompt, n=4, strict=True, seed_first=True, device=dev,
                seed=i * 11 + 5)
            rows.append({
                "exact": bool(exact),
                "repeat": generate.repeated_content_rate(lines),
                "nonword": 1.0 - generate.attested_fraction(lines),
                "junk": any(generate.has_junk(l) for l in lines),
                "seed": generate.seed_adherence(lines, prompt) == 1.0,
                "text": " / ".join(lines),
            })
    total = len(rows)
    return {
        "n": total,
        "exact": sum(r["exact"] for r in rows) / total,
        "seed": sum(r["seed"] for r in rows) / total,
        "repeat": sum(r["repeat"] for r in rows) / total,
        "nonword": sum(r["nonword"] for r in rows) / total,
        "junk": sum(r["junk"] for r in rows) / total,
        "model": {"path": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                  "run": payload.get("run"), "epoch": payload.get("epoch"),
                  "val_loss": payload.get("val_loss")},
        "worst": min(rows, key=lambda r: (r["exact"], not r["junk"], -r["repeat"],
                                          -r["nonword"], r["seed"]))["text"],
    }


def check(metrics: dict, floors: dict = FLOORS) -> list[str]:
    """Return one human-readable line per failed promise (empty list == all good)."""
    failures = []
    for key, (kind, limit) in floors.items():
        value = metrics[key]
        bad = value < limit if kind == "min" else value > limit
        if bad:
            word = "below" if kind == "min" else "above"
            failures.append(f"{key}: {100 * value:.0f}% {word} the {100 * limit:.0f}% "
                            f"{'floor' if kind == 'min' else 'ceiling'}")
    return failures


def self_test() -> int:
    """The check is only useful if it can fail: prove both directions."""
    good = {"exact": 1.0, "seed": 1.0, "repeat": 0.0, "nonword": 0.0, "junk": 0.0}
    cases = {
        "clean": (good, 0),
        "repeat": ({**good, "repeat": 0.12}, 1),
        "invented": ({**good, "nonword": 0.09}, 1),
        "meter": ({**good, "exact": 0.90}, 1),
        "junk": ({**good, "junk": 0.09}, 1),
        "seed": ({**good, "seed": 0.80}, 1),
    }
    bad = 0
    for name, (metrics, want) in cases.items():
        got = len(check(metrics))
        ok = got >= want if want else got == 0
        print(f"  self-test {name:9s} failures={got} expected>={want} {'ok' if ok else 'BROKEN'}")
        bad += not ok
    return 1 if bad else 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="checkpoints")
    ap.add_argument("--seeds", default="autumn,breeze,old pond,rain")
    ap.add_argument("--n", type=int, default=6, help="poems per seed (default 24 in total)")
    ap.add_argument("--write", action="store_true", help="record the current numbers as the floor")
    ap.add_argument("--allow-model-change", action="store_true",
                    help="accept a different checkpoint's numbers (baseline still needs --write)")
    ap.add_argument("--self-test", action="store_true", help="test the comparator, no model needed")
    args = ap.parse_args(argv)

    if args.self_test:
        sys.exit(self_test())

    ckpt = ROOT / args.model / "best.pt"
    if not ckpt.exists():
        sys.exit(f"{ckpt} missing")
    seeds = [s for s in args.seeds.split(",")]
    metrics = measure(ckpt, seeds, args.n)
    model = metrics["model"]

    print(f"measured on {model['path']} (run {model['run']}, epoch {model['epoch']}, "
          f"val {model['val_loss']:.4f}), {metrics['n']} strict poems")
    print(f"{'promise':>10}{'measured':>10}{'floor':>10}{'':>3}")
    failures = check(metrics)
    for key, (kind, limit) in FLOORS.items():
        arrow = ">=" if kind == "min" else "<="
        flag = "" if all(not f.startswith(key + ":") for f in failures) else "  FAIL"
        print(f"{key:>10}{100 * metrics[key]:>9.0f}%{arrow}{100 * limit:>8.0f}%{flag}")
    print(f"  least typical poem: {metrics['worst']}")

    if args.write:
        payload = {**metrics, "floors": FLOORS, "seeds": seeds}
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {BASELINE.relative_to(ROOT)} (this is now the floor)")
        sys.exit(1 if failures else 0)

    if not BASELINE.exists():
        sys.exit(f"no baseline at {BASELINE.relative_to(ROOT)} - run with --write once")
    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))
    if recorded["model"]["path"] != model["path"] and not args.allow_model_change:
        sys.exit(f"baseline was measured on {recorded['model']['path']}, not {model['path']} - "
                 "re-record with --write if this promotion is intended")

    if failures:
        print("\nREGRESSION:")
        for line in failures:
            print("  -", line)
        sys.exit(1)
    print("\nquality floor holds")


if __name__ == "__main__":
    main()
