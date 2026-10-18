"""Sweep decoder settings on a fixed checkpoint. No retraining involved.

    python inference_sweep.py                                  # temperature on the live model
    python inference_sweep.py --temperatures 0.7,0.8,0.9 --best-ofs 4,8
    python inference_sweep.py --models checkpoints --seeds autumn,breeze

Each row is the same 6 seeds x 8 poems: exact rate, repeat rate, non-word rate, mean logprob
and seconds per poem. Runs on CUDA when available.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import generate  # noqa: E402

ROOT = Path(__file__).resolve().parent


def run_config(model, tok, dev, seeds: list[str], n: int, temperature: float, top_p: float,
               penalty: float, best_of: int) -> dict:
    started = time.time()
    exact = repeat = nonword = logprob = total = 0
    for prompt in seeds:
        ctx = (generate.seed_context(tok, prompt) if prompt
               else [tok.token_to_id("<bos>")])
        for i in range(n):
            lines, _pat, ok = generate.best_of_haiku(
                model, tok, prompt, n=best_of, strict=True, seed_first=True, device=dev,
                seed=i * 7 + 1, temperature=temperature, top_p=top_p,
                repetition_penalty=penalty)
            total += 1
            exact += bool(ok)
            repeat += generate.repeated_content_rate(lines)
            nonword += 1.0 - generate.attested_fraction(lines)
            logprob += generate.score_text(model, tok, ctx, "\n".join(lines))
    elapsed = time.time() - started
    return {"exact": exact / total, "repeat": repeat / total, "nonword": nonword / total,
            "logprob": logprob / total, "sec": elapsed / total, "n": total}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="checkpoints")
    ap.add_argument("--seeds", default="autumn,breeze,old pond,rain,harbor,candle")
    ap.add_argument("--n", type=int, default=8, help="poems per seed per config")
    ap.add_argument("--temperatures", default="0.7,0.8,0.9,1.0")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--penalties", default="1.1,1.2,1.3")
    ap.add_argument("--best-ofs", default="4,8")
    args = ap.parse_args(argv)

    seeds = [s for s in args.seeds.split(",")]
    temps = [float(t) for t in args.temperatures.split(",")]
    pens = [float(p) for p in args.penalties.split(",")]
    bests = [int(b) for b in args.best_ofs.split(",")]
    base_pen = generate.DEFAULT_REPETITION_PENALTY

    for name in args.models.split(","):
        ckpt = ROOT / name / "best.pt"
        if not ckpt.exists():
            print(f"{ckpt} missing")
            continue
        model, tok, _cfg, dev, payload = generate.load_model(ckpt)
        print(f"\n{name} (epoch {payload.get('epoch')}, val {payload.get('val_loss'):.4f}, "
              f"device {dev}) - {len(seeds)} seeds x {args.n} poems per row")
        header = (f"{'temperature':>12}{'top_p':>7}{'penalty':>9}{'best_of':>9}{'exact':>8}"
                  f"{'repeat':>8}{'nonword':>9}{'logprob':>9}{'s/poem':>9}")
        print(header)
        print("-" * len(header))
        for temp in temps:
            r = run_config(model, tok, dev, seeds, args.n, temp, args.top_p, base_pen, bests[0])
            print(f"{temp:>12.2f}{args.top_p:>7.2f}{base_pen:>9.2f}{bests[0]:>9}"
                  f"{100*r['exact']:>7.0f}%{100*r['repeat']:>7.0f}%{100*r['nonword']:>8.0f}%"
                  f"{r['logprob']:>9.2f}{r['sec']:>9.2f}")
        for pen in pens:
            r = run_config(model, tok, dev, seeds, args.n, temps[0], args.top_p, pen, bests[0])
            print(f"{temps[0]:>12.2f}{args.top_p:>7.2f}{pen:>9.2f}{bests[0]:>9}"
                  f"{100*r['exact']:>7.0f}%{100*r['repeat']:>7.0f}%{100*r['nonword']:>8.0f}%"
                  f"{r['logprob']:>9.2f}{r['sec']:>9.2f}")
        for b in bests:
            r = run_config(model, tok, dev, seeds, args.n, temps[0], args.top_p, base_pen, b)
            print(f"{temps[0]:>12.2f}{args.top_p:>7.2f}{base_pen:>9.2f}{b:>9}"
                  f"{100*r['exact']:>7.0f}%{100*r['repeat']:>7.0f}%{100*r['nonword']:>8.0f}%"
                  f"{r['logprob']:>9.2f}{r['sec']:>9.2f}")


if __name__ == "__main__":
    main()
