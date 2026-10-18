"""Sweep a few training configs and compare them by val loss + samples.

    python experiments.py                  # 5-epoch screen of 4 configs (~15-20 min on the 4050)
    python experiments.py --epochs 25      # full runs, hours - run this when you step away
    python experiments.py --runs clean-d0.15,meter-d0.10
    python experiments.py --device cpu     # slow, but works without the GPU

Everything lands in experiments/<run-name>/ (checkpoints + log.csv) and
experiments/results.json (val loss, minutes, samples, sorted best first).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from backend.config import Config
from backend.dataset import CLEAN_PATH, METER_PATH, build_loaders
from backend.generate import load_model, sample_haiku
from backend.train import train_run

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments"

RUNS = [
    {"name": "clean-d0.10", "corpus": "clean", "dropout": 0.10},
    {"name": "clean-d0.15", "corpus": "clean", "dropout": 0.15},
    {"name": "meter-d0.10", "corpus": "meter", "dropout": 0.10},
    {"name": "meter-d0.15", "corpus": "meter", "dropout": 0.15},
]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compare training configs")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--runs", default="", help="comma-separated names, default: all")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    wanted = [r for r in RUNS if not args.runs or r["name"] in args.runs.split(",")]
    if not wanted:
        raise SystemExit(f"no run matched; known: {[r['name'] for r in RUNS]}")
    corpora = {"clean": CLEAN_PATH, "meter": METER_PATH}
    results = []

    for run in wanted:
        cfg = Config(seed=args.seed, dropout=run["dropout"])
        if args.batch_size:
            cfg.batch_size = args.batch_size
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        started = time.time()
        print(f"\n=== {run['name']}: corpus {run['corpus']}, dropout {run['dropout']}, "
              f"{args.epochs} epochs", flush=True)

        train_loader, val_loader, tok, stats = build_loaders(
            cfg, corpus=corpora[run["corpus"]], batch_size=cfg.batch_size)
        ckpt = OUT / run["name"]
        best_val = train_run(cfg, train_loader, val_loader, tok, ckpt_dir=ckpt, epochs=args.epochs,
                             device=args.device, run_name=run["name"], log_path=ckpt / "log.csv")

        model, tok2, _, dev, _ = load_model(ckpt / "best.pt", None, args.device)
        samples = []
        for i in range(3):
            lines, _, _ = sample_haiku(model, tok2, cfg.preview_prompt, seed=i, device=str(dev), amp=None)
            samples.append(lines)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results.append({
            "run": run["name"], "corpus": run["corpus"], "dropout": run["dropout"],
            "epochs": args.epochs, "haiku_train": stats["haiku_train"],
            "val_loss": round(best_val, 4), "minutes": round((time.time() - started) / 60, 1),
            "samples": samples,
        })

    results.sort(key=lambda r: r["val_loss"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== sweep results, best val loss first")
    print(f"{'run':<14}{'corpus':<8}{'drop':<7}{'val_loss':<11}{'min':<7}first sample line")
    for r in results:
        first = r["samples"][0][0] if r["samples"] and r["samples"][0] else ""
        print(f"{r['run']:<14}{r['corpus']:<8}{r['dropout']:<7}{r['val_loss']:<11}{r['minutes']:<7}{first[:46]}")
    print(f"\nfull results + samples: {OUT / 'results.json'}")
    print("Winner is the lowest val_loss; a lower val loss with readable samples wins.")


if __name__ == "__main__":
    main()
