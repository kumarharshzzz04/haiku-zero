"""Module 8 - training engine.

AMP fp16 on CUDA, AdamW + warmup/cosine over the *total* step budget, grad clipping
1.0, best/last checkpoints, CSV log, and a per-epoch haiku preview (the primary
debugging instrument, per PLAN.md).

Run
    python -m backend.train --max-steps 50     # GPU smoke test, ~1 minute
    python -m backend.train                    # full run (1-3 h on the RTX 4050)
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):  # allow: python backend/train.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import Config  # noqa: E402
from backend.dataset import CLEAN_PATH, METER_PATH, TOKENIZER_PATH, build_loaders  # noqa: E402
from backend.generate import sample_haiku  # noqa: E402
from backend.model import HaikuZeroLM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
LOG_PATH = ROOT / "training_log.csv"


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to ~0 across the whole run."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path: Path, model, optimizer, step: int, epoch: int, cfg: Config,
                    val_loss: float, run_name: str, tokenizer_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step, "epoch": epoch,
        "config": cfg.to_dict(),
        "val_loss": val_loss, "run": run_name,
        "tokenizer": str(tokenizer_path),
    }, path)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> float:
    model.eval()
    total, batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            _, loss = model(x, y)
        total += loss.item()
        batches += 1
    model.train()
    return total / max(1, batches)


def train_run(cfg: Config, train_loader, val_loader, tok, ckpt_dir: Path = CKPT_DIR,
              epochs: int | None = None, max_steps: int | None = None,
              device: str | None = None, amp: bool | None = None, resume: bool = False,
              run_name: str = "haikuzero", log_path: Path = LOG_PATH, preview: bool = True,
              tokenizer_path: Path = TOKENIZER_PATH, init_from: Path | None = None,
              verbose: bool = True) -> float:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = (cfg.amp and device.type == "cuda") if amp is None else amp
    epochs = epochs or cfg.epochs

    model = HaikuZeroLM(cfg).to(device)
    model.set_syllable_table(tok)                     # no-op unless cfg.meter_state is on
    if init_from is not None and Path(init_from).exists():
        payload = torch.load(init_from, map_location=device, weights_only=False)
        # weights only: fine-tuning, fresh schedule. strict=False is what lets a meter-state
        # model start from a checkpoint that predates the feature - the new embeddings are the
        # only missing tensors and they are learned in this run (a size mismatch still raises).
        missing, _ = model.load_state_dict(payload["model"], strict=False)
        extra = f" | {len(missing)} new tensors: {', '.join(sorted(missing))}" if missing else ""
        print(f"initialised from {init_from} (epoch {payload['epoch']}, "
              f"val {payload['val_loss']:.4f}){extra}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    total_steps = epochs * max(1, len(train_loader))
    if max_steps:
        total_steps = min(total_steps, max_steps)

    last_path, best_path = Path(ckpt_dir) / "last.pt", Path(ckpt_dir) / "best.pt"
    # keep the tokenizer beside the weights: a checkpoint without its tokenizer is unusable
    tok_copy = Path(ckpt_dir) / "tokenizer.json"
    if Path(tokenizer_path).exists() and not tok_copy.exists():
        tok_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tokenizer_path, tok_copy)
    step, start_epoch, best_val = 0, 1, float("inf")
    if resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        step, start_epoch, best_val = payload["step"], payload["epoch"] + 1, payload["val_loss"]
        print(f"resumed {last_path} at step {step} (epoch {start_epoch}), best val {best_val:.4f}")

    if verbose:
        gpu = f" - {torch.cuda.get_device_name(0)}" if device.type == "cuda" else " (no CUDA!)"
        state = " | meter state ON" if cfg.meter_state else ""
        print(f"device {device}{gpu} | amp {amp} | params {model.num_params() / 1e6:.2f}M{state}")
        print(f"{total_steps} total steps over {epochs} epochs at batch {cfg.batch_size}")

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("epoch,step,lr,train_loss,val_loss,elapsed_s\n", encoding="utf-8")

    started = time.time()
    running, seen, lr, stop = 0.0, 0, cfg.lr, False
    last_log_time, last_log_step = started, 0      # rolling rate, so tok/s + eta stay honest across epochs

    for epoch in range(start_epoch, epochs + 1):
        for x, y in train_loader:
            if max_steps and step >= max_steps:
                stop = True
                break
            lr = lr_at(step, cfg.lr, cfg.warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            running += loss.item()
            seen += 1
            if verbose and step % cfg.log_every == 0:
                now = time.time()
                rate = max(1, step - last_log_step) / max(1e-6, now - last_log_time)   # steps per second
                last_log_time, last_log_step = now, step
                print(f"epoch {epoch} step {step}/{total_steps} loss {running / seen:.4f} "
                      f"lr {lr:.2e} {rate * x.numel() / 1000:.1f}k tok/s "
                      f"eta {(total_steps - step) / rate / 60:.0f}m", flush=True)

        train_loss = running / max(1, seen)
        val_loss = evaluate(model, val_loader, device, amp) if len(val_loader) else train_loss
        save_checkpoint(last_path, model, optimizer, step, epoch, cfg, val_loss, run_name, tokenizer_path)
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, step, epoch, cfg, val_loss, run_name, tokenizer_path)
        with log_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([epoch, step, f"{lr:.3e}", f"{train_loss:.4f}",
                                    f"{val_loss:.4f}", f"{time.time() - started:.1f}"])
        print(f"epoch {epoch} done | train {train_loss:.4f} | val {val_loss:.4f}"
              f"{' *best*' if improved else ''}", flush=True)
        running, seen = 0.0, 0
        if preview:
            lines, _, _ = sample_haiku(model, tok, cfg.preview_prompt, cfg.temperature, cfg.top_p,
                                       cfg.repetition_penalty, 40, device=device, amp=amp)
            print("  preview over '" + cfg.preview_prompt + "':\n    " + "\n    ".join(lines), flush=True)
        if stop:
            print(f"stopped at max_steps={max_steps}")
            break

    print(f"done: {step} steps in {(time.time() - started) / 60:.1f} min | best val {best_val:.4f}")
    return best_val


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Module 8: train HAIKU ZERO")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="lower to 64 if you hit OOM on 6GB")
    ap.add_argument("--max-steps", type=int, default=None, help="cap total steps (smoke test)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/last.pt")
    ap.add_argument("--meter-data", action="store_true", help="train on the 5-7-5-ish subset instead")
    ap.add_argument("--corpus", type=Path, default=None, help="corpus file (default data/haikus_clean.txt)")
    ap.add_argument("--lr", type=float, default=None, help="peak LR, e.g. 5e-5 when fine-tuning")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from these weights (fine-tune), e.g. checkpoints/best.pt")
    ap.add_argument("--dropout", type=float, default=None, help="PLAN.md: raise to 0.15 if val climbs")
    ap.add_argument("--seed-prob", type=float, default=None,
                    help="share of rows trained as '<seed> word -> poem' (default 0.5)")
    ap.add_argument("--prefix-mask-prob", type=float, default=None,
                    help="share of rows trained as a continuation (default 0.5)")
    ap.add_argument("--meter-state", dest="meter_state", action="store_true",
                    help="feed syllables-so-far-in-line + line index to the model (RUNBOOK 4c)")
    ap.add_argument("--embed-dim", type=int, default=None, help="model width (multiple of num_heads)")
    ap.add_argument("--layers", type=int, default=None, help="number of transformer blocks")
    ap.add_argument("--ffn-dim", type=int, default=None, help="feed-forward width (4x embed is standard)")
    ap.add_argument("--seq-length", type=int, default=None, help="window size")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-name", default="haikuzero")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--log", type=Path, default=LOG_PATH)
    args = ap.parse_args(argv)

    cfg = Config()
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.seed_prob is not None:
        cfg.seed_prob = args.seed_prob
    if args.prefix_mask_prob is not None:
        cfg.prefix_mask_prob = args.prefix_mask_prob
    cfg.meter_state = args.meter_state
    if args.lr is not None:
        cfg.lr = args.lr
    for flag, field in (("embed_dim", "embed_dim"), ("layers", "num_layers"),
                        ("ffn_dim", "ffn_dim"), ("seq_length", "seq_length")):
        value = getattr(args, flag)
        if value is not None:
            setattr(cfg, field, value)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    corpus = args.corpus or (METER_PATH if args.meter_data else CLEAN_PATH)
    train_loader, val_loader, tok, _ = build_loaders(cfg, corpus=corpus, batch_size=cfg.batch_size)
    train_run(cfg, train_loader, val_loader, tok, ckpt_dir=args.ckpt_dir, epochs=args.epochs,
              max_steps=args.max_steps, device=args.device,
              amp=False if args.no_amp else None, resume=args.resume,
              run_name=args.run_name, log_path=args.log, init_from=args.init_from)


if __name__ == "__main__":
    main()
