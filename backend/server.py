"""Module 12 - local HTTP API + static host for the haiku front end.

Zero new dependencies: stdlib server, reusing the inference engine in backend.generate.

    python -m backend.server              # API + built front end on http://127.0.0.1:8000
    python -m backend.server --no-warm    # skip preloading the default model

Endpoints:
    GET  /api/health    -> device, torch version, loaded checkpoints
    GET  /api/models    -> checkpoints found on disk with epoch / val loss
    POST /api/generate  -> {prompt, model, count, strict, best_of, temperature, ...}
                           -> {poems: [{lines, pattern, exact, logprob, repeated_rate}]}
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

if __package__ in (None, ""):  # allow: python backend/server.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import generate, meter_check  # noqa: E402
from backend.config import Config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"

# Checkpoint folder -> label shown in the UI. Any checkpoints*/best.pt is picked up,
# so a new fine-tune appears in the dropdown without touching this file.
LABELS = {
    "checkpoints": "Live - poetry pretrain + haiku fine-tune, 15.8M params",
    "checkpoints_tweets_live": "Previous live - haiku only, 15.8M (invented 3% of words)",
    "checkpoints_pretrain": "Stage 1 only - 15.4M tokens of public-domain poetry",
    "checkpoints_mix": "Synthetic mix - live fine-tune + 180 authored haiku",
    "checkpoints_authentic": "Authentic - 40.9k non-tweet haiku",
    "checkpoints_balanced": "Balanced - 5.8M params",
    "checkpoints_classical": "Classical - old fine-tune (superseded)",
    "checkpoints_v3_live": "Legacy - 78% tweets (the tweet-y one)",
    "checkpoints_v3_clean": "v3 clean",
}
MAX_NEW_TOKENS = 96          # a 5-7-5 poem is ~30 tokens; headroom for the seed line
MAX_COUNT = 6
LOCK = threading.Lock()      # one forward pass at a time; the model is not thread safe
CACHE: dict[str, tuple] = {}


def model_registry() -> dict[str, Path]:
    found = {}
    for folder in sorted(ROOT.glob("checkpoints*")):
        ckpt = folder / "best.pt"
        if ckpt.exists():
            found[folder.name] = ckpt
    return found


def checkpoint_meta(ckpt: Path) -> dict:
    """epoch / val_loss straight from the payload; weights_only keeps this cheap and safe."""
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        return {"epoch": payload.get("epoch"), "val_loss": payload.get("val_loss")}
    except Exception as exc:                      # a listed model must not break the page
        return {"epoch": None, "val_loss": None, "meta_error": str(exc)}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:            # a checkpoint outside the project (tests, other drives)
        return str(path).replace("\\", "/")


def registry_payload() -> list[dict]:
    out = []
    for name, ckpt in model_registry().items():
        out.append({
            "id": name,
            "label": LABELS.get(name, name),
            "path": display_path(ckpt),
            **checkpoint_meta(ckpt),
        })
    return out


def get_model(model_id: str):
    registry = model_registry()
    if model_id not in registry:
        raise KeyError(f"unknown model {model_id!r}; have {sorted(registry)}")
    if model_id not in CACHE:
        with LOCK:
            if model_id not in CACHE:             # double check: two clicks, one load
                started = time.time()
                CACHE[model_id] = (*generate.load_model(registry[model_id]), time.time() - started)
    return CACHE[model_id]


def clamped(body: dict, key: str, default, low, high) -> float:
    try:
        value = float(body.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    return max(low, min(high, value))


def generate_poems(body: dict) -> dict:
    model_id = str(body.get("model") or "").strip() or next(iter(model_registry()))
    model, tok, cfg, dev, payload, load_s = get_model(model_id)

    prompt = generate.clean_seed(str(body.get("prompt") or ""))
    count = int(clamped(body, "count", 1, 1, MAX_COUNT))
    strict = bool(body.get("strict", True))
    best_of = int(clamped(body, "best_of", 4, 1, 16))
    seed = body.get("seed")
    seed = int(seed) if seed not in (None, "") else None
    seed_first = bool(body.get("seed_first", True))
    sampling = dict(
        temperature=clamped(body, "temperature", cfg.temperature, 0.1, 2.0),
        top_p=clamped(body, "top_p", cfg.top_p, 0.05, 1.0),
        # older checkpoints store 1.1 in their payload, which repeats words; take the higher of
        # the payload value and the measured default so a stale checkpoint cannot undo the fix
        repetition_penalty=clamped(body, "repetition_penalty",
                                   max(cfg.repetition_penalty, generate.DEFAULT_REPETITION_PENALTY),
                                   1.0, 2.0),
        max_new_tokens=int(clamped(body, "max_new_tokens", cfg.max_new_tokens, 8, MAX_NEW_TOKENS)),
    )

    ctx = generate.seed_context(tok, prompt) if prompt else [tok.token_to_id("<bos>")]
    poems, started = [], time.time()
    with LOCK:
        for i in range(count):
            lines, pat, strict_ok = generate.best_of_haiku(
                model, tok, prompt, n=best_of, strict=strict, seed_first=seed_first,
                # step by best_of: candidates are drawn from base..base+best_of, so +1 per poem
                # made consecutive poems share 3 of 4 candidates and return the same winner
                seed=None if seed is None else seed + i * max(1, best_of),
                amp=(dev.type == "cuda"), **sampling)
            if not lines:
                continue
            padded = tuple(pat) + (0,) * (3 - len(pat))
            poems.append({
                "lines": lines,
                "pattern": list(pat),
                "distance": meter_check.distance(padded),
                "exact": meter_check.distance(padded) == 0,
                "strict_ok": strict_ok,
                "logprob": round(generate.score_text(model, tok, ctx, "\n".join(lines)), 4),
                "repeated_rate": round(generate.repeated_content_rate(lines), 4),
                "unattested_rate": round(1.0 - generate.attested_fraction(lines), 4),
                "seed_hit": generate.seed_adherence(lines, prompt) == 1.0,
            })
    return {
        "model": model_id,
        "model_label": LABELS.get(model_id, model_id),
        "epoch": payload.get("epoch"),
        "val_loss": payload.get("val_loss"),
        "prompt": prompt,
        "strict": strict,
        "seed_first": seed_first,
        "best_of": best_of,
        "seed": seed,
        "poems": poems,
        "seconds": round(time.time() - started, 2),
        "load_seconds": round(load_s, 2) if isinstance(load_s, float) else None,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "haiku-zero/1.0"

    def log_message(self, fmt, *args):            # keep the console readable
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def send_json(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def send_file(self, path: Path):
        raw = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/api/health":
            return self.send_json({
                "ok": True,
                "device": str(generate.pick_device()),
                "torch": torch.__version__,
                "loaded": sorted(CACHE),
                "frontend_built": (DIST / "index.html").exists(),
            })
        if route == "/api/models":
            return self.send_json({"models": registry_payload(),
                                   "default": next(iter(model_registry()), None)})
        if route.startswith("/api/"):
            return self.send_json({"error": f"no route {route}"}, 404)

        if not (DIST / "index.html").exists():
            return self.send_json({
                "error": "front end not built",
                "hint": "the React UI is private and not published with this repo - "
                        "add frontend/ locally, then `npm run dev` (:5177) "
                        "or `npm run build` to serve it from here",
            }, 503)
        target = (DIST / route.lstrip("/")).resolve()
        if target.is_file() and DIST.resolve() in target.parents:
            return self.send_file(target)
        return self.send_file(DIST / "index.html")       # SPA fallback

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route != "/api/generate":
            return self.send_json({"error": f"no route {route}"}, 404)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self.send_json({"error": "bad Content-Length"}, 400)
        if length < 0 or length > 1_000_000:          # a bogus header must not hang the reader
            return self.send_json({"error": "bad Content-Length"}, 400)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            return self.send_json({"error": f"bad JSON: {exc}"}, 400)
        if not isinstance(body, dict):
            return self.send_json({"error": "body must be a JSON object"}, 400)
        try:
            return self.send_json(generate_poems(body))
        except KeyError as exc:
            return self.send_json({"error": str(exc)}, 400)
        except Exception as exc:                       # surface it in the UI, not the void
            sys.stderr.write(f"generate failed: {exc!r}\n")
            return self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Module 12: local API + front end host")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-warm", action="store_true", help="do not preload the default model")
    args = ap.parse_args(argv)

    registry = model_registry()
    if not registry:
        raise SystemExit("no checkpoints*/best.pt found - run backend.train first")
    print(f"models: {', '.join(sorted(registry))}")
    print(f"device: {generate.pick_device()}")
    print(f"front end: {'frontend/dist' if (DIST / 'index.html').exists() else 'none - the UI is private, see README'}")

    if not args.no_warm:
        def warm():
            try:
                get_model(next(iter(registry)))
                print("default model warm")
            except Exception as exc:
                print(f"warm-up failed: {exc!r}")
        threading.Thread(target=warm, daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}  (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
