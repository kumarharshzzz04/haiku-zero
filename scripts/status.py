"""Pipeline status in one command - the factual half of PROGRESS.md.

    python scripts/status.py

Prints what exists, what is still missing, and the next thing to run.
Exit code is always 0; use the INFO/BLOCKED lines, not $?, to decide.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "haikus_clean.txt"
METER = ROOT / "data" / "haikus_meter.txt"
TOKENIZER = ROOT / "tokenizer.json"
CKPT = ROOT / "checkpoints"
LOG = ROOT / "logs" / "training" / "training_log.csv"
EXPERIMENTS = ROOT / "experiments" / "results.json"


def records_in(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() == "<eos>")


def report() -> list[str]:
    out: list[str] = []

    def say(label: str, ok: bool, detail: str = "") -> None:
        out.append(f"{'OK  ' if ok else 'MISS'} {label:<32}{detail}")

    cuda = torch.cuda.is_available()
    say("GPU", cuda, torch.cuda.get_device_name(0) if cuda else "activate pytorch_with_gpu first!")

    raw = sorted((ROOT / "data" / "raw").glob("*")) if (ROOT / "data" / "raw").exists() else []
    say("raw sources", bool(raw), ", ".join(p.name for p in raw) or "nothing in data/raw/")
    n_clean = records_in(CLEAN)
    say("haikus_clean.txt", n_clean > 0, f"{n_clean} haiku")
    n_meter = records_in(METER)
    say("haikus_meter.txt", n_meter > 0, f"{n_meter} haiku (5-7-5 +-1 subset)")

    vocab = 0
    if TOKENIZER.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOKENIZER))
        vocab = tok.get_vocab_size()
        say("tokenizer.json", True, f"vocab {vocab}, newline tokens {len(tok.encode(chr(10)).ids)}, "
                                    f"size {TOKENIZER.stat().st_size / 1024:.0f} KB")
    else:
        say("tokenizer.json", False, "run python -m backend.train_tokenizer")

    best, last = CKPT / "best.pt", CKPT / "last.pt"
    if best.exists():
        payload = torch.load(best, map_location="cpu", weights_only=False)
        say("checkpoints/best.pt", True,
            f"epoch {payload['epoch']}, step {payload['step']}, val {payload['val_loss']:.4f}, "
            f"run {payload.get('run')}")
    else:
        say("checkpoints/best.pt", False, "no trained model yet - run python -m backend.train")
    say("checkpoints/last.pt", last.exists(), "resume point" if last.exists() else "no partial run")

    if LOG.exists():
        rows = [r for r in LOG.read_text(encoding="utf-8").strip().splitlines()[1:] if r]
        last_row = rows[-1].split(",") if rows else []
        say("logs/training/training_log.csv", bool(rows),
            f"{len(rows)} epoch rows, last: epoch {last_row[0]} val {last_row[4]}" if last_row else "empty")
    else:
        say("logs/training/training_log.csv", False, "written during training")

    poetry = ROOT / "data" / "poetry_pretrain.txt"
    n_poetry = records_in(poetry)
    say("poetry_pretrain.txt", n_poetry > 0,
        f"{n_poetry} records, {poetry.stat().st_size / 1e6:.1f}M chars (stage-1 corpus)" if n_poetry
        else "optional stage-1 corpus: python scripts/data/fetch_poetry.py")
    stage1 = ROOT / "checkpoints_pretrain" / "best.pt"
    if stage1.exists():
        payload = torch.load(stage1, map_location="cpu", weights_only=False)
        say("checkpoints_pretrain/best.pt", True,
            f"stage-1 poetry pretrain, epoch {payload['epoch']}, val {payload['val_loss']:.4f}")
    else:
        say("checkpoints_pretrain/best.pt", False, "stage-1 weights (the two-stage recipe, RUNBOOK 3)")

    if EXPERIMENTS.exists():
        runs = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
        winner = runs[0] if runs else {}
        say("experiments/results.json", True,
            f"{len(runs)} runs, best {winner.get('run')} val {winner.get('val_loss')}")
    else:
        say("experiments/results.json", False, "optional: python scripts/experiments.py --epochs 25")

    frontend = ROOT / "frontend"
    say("frontend/ (shadcn layout)", (frontend / "components.json").exists(),
        "npm install && npm run dev -> :5177" if (frontend / "components.json").exists()
        else "missing components.json - not a shadcn project")
    say("frontend/dist", (frontend / "dist" / "index.html").exists(),
        ":8000 serves the built app" if (frontend / "dist" / "index.html").exists()
        else "optional: npm run build (dev server does not need it)")
    say("backend/server.py", (ROOT / "backend" / "server.py").exists(), "python -m backend.server")

    return out


def next_step(lines: list[str]) -> str:
    joined = "\n".join(lines)
    if "MISS checkpoints/best.pt" in joined:
        return "NEXT: python -m backend.train            (~10 min on the 4050)"
    if "MISS experiments" in joined:
        return "NEXT: python -m backend.generate --prompt \"autumn moonlight\" --count 3   " \
               "then python scripts/experiments.py --epochs 25"
    return "NEXT: compare experiments, write down the best val loss in PROGRESS.md"


def main() -> None:
    print(f"project : {ROOT}")
    lines = report()
    print("\n".join(lines))
    print("\n" + next_step(lines))


if __name__ == "__main__":
    main()
