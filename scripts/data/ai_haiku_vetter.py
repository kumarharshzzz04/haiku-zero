"""The smart vetter for AI haiku batches.

    python scripts/data/ai_haiku_vetter.py                       # scan data/ai_batches, score, write tiers
    python scripts/data/ai_haiku_vetter.py --refresh-model       # refit the soundness model first
    python scripts/data/ai_haiku_vetter.py --explain "line one | line two | line three"
    python scripts/data/ai_haiku_vetter.py --store 0.90          # move the tier-A cut

Why this is not a filter
------------------------
A filter answers yes/no with hand-written rules; this answers *how much* and says why. Every
poem gets 0-100 from three parts:

1. **Soundness** (fitted). A logistic model over measured features - meter distance, share of
   content words in the project's 145k-word public-domain vocabulary, bigram plausibility from
   the authentic haiku corpus, unseen-bigram rate, longest line, internal echo. It is trained on
   labels this repo can actually justify: real poems (the authentic corpus + our 180 authored
   ones) as positives, and **corruptions of those same poems** as negatives - two lines swapped,
   a content word replaced by a random one, a word deleted, a word repeated inside a line, words
   shuffled. So the model learns "is this poem structurally and semantically intact", not "does
   this sound like the 1902 corpus". Training on source identity is what made the first version
   score self-help cliches at 97/100; corruption labels fixed that.
   The held-out AUC is printed and stored, so the separation is a number, not a claim.

2. **Cliché penalty** (measured). Abstract filler - "life", "heart", "soul", "dream", "joy",
   "truth", "the world", "the mind" - is what AI batches emit when they have nothing to say.
   Penalty grows with the share of abstract content words, so "Joy is in my heart" (3 of 6) is
   punished and "One red leaf floats on the pond" is not.

3. **Batch-repetition penalty** (measured). AI batches recycle: batch 01 built 1,000 poems from
   a small pool of lines, so its lines appear dozens of times. Any line already used twice in
   this intake costs the poem, and a 3-word phrase shared with many other poems costs a little.

Outputs
    data/raw_ai/            tier A - training-ready: score over the cut AND exact 5-7-5
    data/raw_ai_near/       tier B - within +-1 syllable, kept out of training by default
    data/vetter_weights.json, docs/AI_HAIKU_INTAKE.md, data/ai_intake_stats.json

The score ranks, the meter gates: a poem that reads well but is not 5-7-5 never enters tier A,
because the corpus teaches form and an off-meter exemplar teaches the wrong form.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.meter_check import distance, pattern  # noqa: E402
from backend.prepare_data import collect, dedupe_key, normalize, reject_reason  # noqa: E402
from scripts.data.import_ai_haiku import (  # noqa: E402
    parse_file, self_repeat, repeated_content, content_words,
)

DEFAULT_DIR = ROOT / "data" / "ai_batches"
AI_OUT = ROOT / "data" / "raw_ai"
NEAR_OUT = ROOT / "data" / "raw_ai_near"
WEIGHTS_OUT = ROOT / "data" / "vetter_weights.json"
STATS_OUT = ROOT / "data" / "ai_intake_stats.json"
REPORT_OUT = ROOT / "docs" / "AI_HAIKU_INTAKE.md"
WORDLIST = ROOT / "data" / "wordlist.txt"
BIGRAM_SOURCE = ROOT / "data" / "haikus_clean.txt"
PRETRAIN = ROOT / "data" / "poetry_pretrain.txt"
VERB_CACHE = ROOT / "data" / "derived_verbs.txt"

TOKEN_RE = re.compile(r"[a-z']+")
FEATURE_NAMES = ["meter_distance", "exact", "attested", "mean_bigram", "unseen_bigram",
                 "cross_line_bigram", "line_length", "echo"]
SEASONS = {
    "winter": {"snow", "snows", "snowy", "frost", "frosty", "ice", "icy", "sleet", "blizzard",
               "hearth", "ember", "embers", "mittens", "icicle", "icicles", "freezing", "frozen"},
    "spring": {"bloom", "blooms", "blossom", "blossoms", "buds", "bud", "sprout", "sprouts",
               "thaw", "thaws", "crocus", "lilac", "lilacs", "seedling", "seedlings"},
    "summer": {"cicada", "cicadas", "cricket", "crickets", "firefly", "fireflies", "midsummer",
               "watermelon", "humid", "heatwave", "sunburn"},
    "autumn": {"acorn", "acorns", "harvest", "harvests", "pumpkin", "pumpkins", "persimmon",
               "persimmons", "chestnut", "chestnuts", "migrate", "migration"},
}
# abstract filler: the vocabulary of a poem that is not about anything
CLICHE = {"life", "lives", "heart", "hearts", "soul", "souls", "dream", "dreams", "joy", "joys",
          "truth", "truths", "mind", "minds", "time", "world", "worlds", "love", "loves", "hope",
          "hopes", "memory", "memories", "vision", "visions", "spirit", "spirits", "beauty",
          "wonder", "wonders", "peace", "pain", "destiny", "universe", "infinity", "eternity",
          "smile", "smiles", "smiling", "forever", "magic", "mystery", "silence", "stillness"}
CLICHE_WEIGHT = 0.45  # how hard abstract filler hits the final score
REPEAT_WEIGHT = 0.15


# ------------------------------------------------------------------ resources

def load_vocab() -> set[str]:
    if not WORDLIST.exists():
        return set()
    return {w.strip().lower() for w in WORDLIST.read_text(encoding="utf-8").split() if w.strip()}


def load_verbs() -> set[str]:
    """Verbs derived from the pretrain corpus (words that follow 'to ' often are infinitives)."""
    if VERB_CACHE.exists():
        cached = {w.strip() for w in VERB_CACHE.read_text(encoding="utf-8").split() if w.strip()}
        if len(cached) > 500:
            return cached
    verbs: Counter = Counter()
    if PRETRAIN.exists():
        with PRETRAIN.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                words = TOKEN_RE.findall(line.lower())
                for a, b in zip(words, words[1:]):
                    if a == "to" and len(b) > 2:
                        verbs[b] += 1
    common = {"is", "are", "was", "were", "be", "been", "has", "have", "had", "does", "do", "did",
              "goes", "go", "went", "comes", "come", "came", "falls", "fall", "fell", "rises",
              "rise", "rose", "stands", "stand", "stood", "hangs", "hang", "waits", "wait",
              "moves", "move", "moved", "sings", "sing", "sang", "sits", "sit", "sat", "runs",
              "run", "ran", "turns", "turn", "turned", "opens", "open", "closes", "close", "holds",
              "hold", "held", "keeps", "keep", "kept", "drifts", "drift", "floats", "float",
              "shines", "shine", "shone", "burns", "burn", "wakes", "wake", "woke", "sleeps",
              "sleep", "slept", "cools", "cool", "warms", "warm", "cracks", "crack", "breaks",
              "break", "broke", "grows", "grow", "grew", "dies", "die", "died", "casts", "cast",
              "wears", "wear", "wore", "leans", "lean", "leaned", "keeps", "sweeps", "swept"}
    got = {w for w, c in verbs.items() if c >= 5} | common
    if got:
        VERB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        VERB_CACHE.write_text("\n".join(sorted(got)), encoding="utf-8")
    return got


def load_bigrams() -> dict:
    uni: Counter = Counter()
    bi: Counter = Counter()
    if BIGRAM_SOURCE.exists():
        for block in BIGRAM_SOURCE.read_text(encoding="utf-8", errors="replace").split("\n\n"):
            for line in block.splitlines():
                words = TOKEN_RE.findall(line.lower())
                uni.update(words)
                bi.update(zip(words, words[1:]))
    return {"bi": bi, "uni": uni, "vocab_size": max(1, len(uni))}


RES: dict = {}


def resource(name: str):
    if name not in RES:
        RES[name] = {"vocab": load_vocab, "verbs": load_verbs, "bigrams": load_bigrams}[name]()
    return RES[name]


# ------------------------------------------------------------------ features

def features(lines: list[str]) -> np.ndarray:
    vocab, big = resource("vocab"), resource("bigrams")
    bi, uni, total = big["bi"], big["uni"], big["vocab_size"]
    pat, dist = pattern(lines), distance(pattern(lines))
    content = [w for l in lines[:3] for w in content_words(l)]
    attested = (sum(1 for w in content if w in vocab) / len(content)) if content else 0.0

    logps, unseen, pairs = [], 0, 0
    for line in lines[:3]:
        ws = TOKEN_RE.findall(line.lower())
        for a, b in zip(ws, ws[1:]):
            pairs += 1
            hits = bi[(a, b)]
            if not hits:
                unseen += 1
            logps.append(math.log((hits + 0.1) / (uni[a] + 0.1 * total)))
    mean_bigram = float(np.mean(logps)) if logps else -20.0

    # across the line breaks: does the poem flow from one line into the next? This is the feature
    # that catches a corrupted line order, which internal bigrams cannot see at all
    cross, cross_hits = [], 0
    for a_line, b_line in zip(lines[:3], lines[1:3]):
        a_words, b_words = TOKEN_RE.findall(a_line.lower()), TOKEN_RE.findall(b_line.lower())
        if a_words and b_words:
            hits = bi[(a_words[-1], b_words[0])]
            cross_hits += 1 if hits else 0
            cross.append(math.log((hits + 0.1) / (uni[a_words[-1]] + 0.1 * total)))
    cross_line = float(np.mean(cross)) if cross else -20.0

    echo = 1.0 if (repeated_content(lines, 1) or self_repeat(lines)) else 0.0
    return np.array([
        min(dist, 12) / 12.0,
        1.0 if dist == 0 else 0.0,
        attested,
        max(-12.0, min(0.0, mean_bigram)) / 12.0,
        unseen / max(1, pairs),
        max(-12.0, min(0.0, cross_line)) / 12.0,
        max(len(l) for l in lines[:3]) / 60.0,
        echo,
    ])


def cliche_rate(lines: list[str]) -> float:
    content = content_words(" ".join(lines[:3]))
    if not content:
        return 0.0
    return sum(1 for w in content if w in CLICHE) / len(content)


def season_clash(lines: list[str]) -> float:
    found = [{s for s, marks in SEASONS.items() if set(TOKEN_RE.findall(l.lower())) & marks}
             for l in lines[:3]]
    for a, b in (("winter", "spring"), ("winter", "summer"), ("spring", "summer"),
                 ("spring", "autumn"), ("summer", "autumn")):
        if any(a in f for f in found) and any(b in f for f in found):
            return 1.0
    return 0.0


# ------------------------------------------------------------------ model

def fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 4000, lr: float = 0.5,
                 l2: float = 0.01) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1] + 1)
    Zb = np.hstack([np.ones((len(Z), 1)), Z])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Zb @ w))
        w -= lr * (Zb.T @ (p - y) / len(y) + l2 * w)
    return np.concatenate([w[:1] - (mu / sd) @ w[1:], w[1:] / sd])


def soundness(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(params[0] + X @ params[1:])))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def corrupt(lines: list[str], rng: random.Random, vocab: list[str]) -> list[str] | None:
    """One of the five ways a real poem stops being one. Returns None if the edit did nothing."""
    out = list(lines[:3])
    kind = rng.randrange(5)
    if kind == 0:                                   # swap two lines: the poem loses its arc
        i, j = rng.sample(range(3), 2)
        out[i], out[j] = out[j], out[i]
    elif kind == 1:                                 # replace a content word with a random one
        cand = [(k, w) for k, l in enumerate(out) for w in content_words(l)]
        if not cand:
            return None
        k, _ = rng.choice(cand)
        words = out[k].split()
        idxs = [i for i, w in enumerate(words) if TOKEN_RE.findall(w.lower())
                and TOKEN_RE.findall(w.lower())[0] in content_words(out[k])]
        if not idxs:
            return None
        words[rng.choice(idxs)] = rng.choice(vocab)
        out[k] = " ".join(words)
    elif kind == 2:                                 # delete a content word
        cand = [(k, i, w) for k, l in enumerate(out) for i, w in enumerate(l.split())]
        if len(cand) < 4:
            return None
        k, i, _ = rng.choice(cand)
        words = out[k].split()
        if len(words) < 3:
            return None
        words.pop(i)
        out[k] = " ".join(words)
    elif kind == 3:                                 # repeat a word inside a line (stutter)
        k = rng.randrange(3)
        words = out[k].split()
        if len(words) < 2:
            return None
        words.insert(rng.randrange(len(words)), rng.choice(words))
        out[k] = " ".join(words)
    else:                                           # shuffle the words of one line
        k = rng.randrange(3)
        words = out[k].split()
        if len(words) < 3:
            return None
        rng.shuffle(words)
        out[k] = " ".join(words)
    return out


def training_set(rng: random.Random, per_poem: int = 2, vocab_sample: int = 2000):
    print("labelling: real poems vs corruptions of those poems...")
    authentic, _, _ = collect(ROOT / "data" / "raw_classical", lowercase=False)
    authored, _, _ = collect(ROOT / "data" / "raw_synthetic", lowercase=False)
    base = rng.sample(authentic, min(2500, len(authentic))) + authored
    vocab = sorted(resource("vocab"))
    rng.shuffle(base)
    vocab_pool = rng.sample(vocab, min(vocab_sample, len(vocab))) or ["stone"]
    Xp = np.array([features(p) for p in base])
    ys: list[float] = []
    Xs: list[np.ndarray] = []
    for poem in base:
        for _ in range(per_poem):
            broken = corrupt(poem, rng, vocab_pool)
            if broken and not reject_reason(normalize(broken, lowercase=False)):
                Xs.append(features(broken))
                ys.append(0.0)
    X = np.vstack([Xp, np.array(Xs)])
    y = np.array([1.0] * len(Xp) + ys)
    print(f"  positives {len(Xp)} real poems | negatives {len(ys)} corruptions")
    return X, y


def load_or_fit(refresh: bool, rng: random.Random):
    if WEIGHTS_OUT.exists() and not refresh:
        payload = json.loads(WEIGHTS_OUT.read_text(encoding="utf-8"))
        if payload.get("features") == FEATURE_NAMES:
            print(f"weights: {WEIGHTS_OUT.name} (held-out AUC {payload.get('auc_held_out'):.3f})")
            return np.array(payload["params"]), payload
    X, y = training_set(rng)
    idx = rng.sample(range(len(y)), len(y))
    cut = int(len(idx) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    params = fit_logistic(X[tr], y[tr])
    payload = {
        "params": params.tolist(), "features": FEATURE_NAMES,
        "auc_train": auc(soundness(params, X[tr]), y[tr]),
        "auc_held_out": auc(soundness(params, X[te]), y[te]),
        "positives": int((y == 1).sum()), "negatives": int((y == 0).sum()),
    }
    print(f"fitted: AUC {payload['auc_train']:.3f} train / {payload['auc_held_out']:.3f} held-out")
    WEIGHTS_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return params, payload


# ------------------------------------------------------------------ scoring

REFERENCE: dict = {"grid": []}   # percentile grid of the raw score over our authored 180,
                                  # so the output is "better than X% of the poems we wrote
                                  # and verified by hand" instead of an unanchored number


def build_reference(params: np.ndarray, authored: list[list[str]]) -> list[float]:
    if not authored:
        return []
    raw = sorted(raw_score(p, params, 0, 0.0) for p in authored)
    grid = [float(np.quantile(raw, q / 100.0)) for q in range(101)]
    REFERENCE["grid"] = grid
    return grid


def audit(parsed: list[tuple[Path, list[list[str]]]], known: set[str]) -> int:
    """Is this folder a haiku corpus at all? Writes nothing - point it at any download.

    The score answers "is this poem any good"; this answers the prior question the score cannot:
    does the file hold three-line poems in haiku form, or is it prose, free verse or a pile of
    duplicates? Run it before importing anything new.
    """
    print(f"\n{'file':34} {'rows':>6} {'3-line':>7} {'5-7-5':>7} {'±1':>6} {'off':>6} "
          f"{'dupe':>6} {'recomb':>7}  verdict")
    print("-" * 104)
    total = Counter()
    verdicts: list[str] = []
    for path, poems in parsed:
        c: Counter = Counter()
        for lines in poems:
            c["rows"] += 1
            lines = normalize(lines, lowercase=False)
            if reject_reason(lines):
                c["junk"] += 1
                continue
            c["three"] += 1
            if dedupe_key(lines) in known:
                c["dupe"] += 1
            elif self_repeat(lines):
                c["recomb"] += 1
            dist = distance(pattern(lines))
            if dist == 0:
                c["exact"] += 1
            elif dist <= 1:
                c["near"] += 1
            else:
                c["off"] += 1
        total.update(c)
        rows = c["rows"] or 1
        three = c["three"] or 1
        rate = c["exact"] / three
        verdict = ("haiku corpus" if rate >= 0.6 else
                   "mixed - form is the exception" if rate >= 0.2 else
                   "not haiku form")
        verdicts.append(f"{path.name}: {verdict} ({rate:.0%} exact)")
        print(f"{path.name:34} {c['rows']:6} {c['three']:7} {c['exact']:7} {c['near']:6} "
              f"{c['off']:6} {c['dupe']:6} {c['recomb']:7}  {verdict}")
    three = total["three"] or 1
    print("-" * 104)
    print(f"{len(parsed)} file(s), {total['rows']} rows: {total['three']} usable 3-line poems, "
          f"{total['exact'] / three:.1%} exact 5-7-5, {(total['exact'] + total['near']) / three:.1%} "
          f"within ±1, {total['off']} off form, {total['dupe']} already in the corpus, "
          f"{total['recomb']} recombination")
    print("audit only - nothing written. Drop the file in the AI folder to score and import it.")
    return 0


def percentile_of(raw: float) -> float:
    """Where this raw score sits among our own authored poems (0-100)."""
    grid = REFERENCE["grid"]
    if not grid:
        return min(100.0, max(0.0, raw * 100))
    return float(np.interp(raw, grid, np.arange(101)))


def raw_score(lines: list[str], params: np.ndarray, reuse: int, phrase_reuse: float) -> float:
    feats = features(lines)
    sound = float(soundness(params, feats[None, :])[0])
    cliche = cliche_rate(lines)
    cliche_penalty = CLICHE_WEIGHT * math.sqrt(min(1.0, cliche * 2.0))
    repeat = min(1.0, 0.5 * min(reuse, 2) + phrase_reuse)
    clash = season_clash(lines)
    return max(0.0, min(1.0, sound - cliche_penalty - REPEAT_WEIGHT * repeat - 0.10 * clash))


def score_poem(lines: list[str], params: np.ndarray, reuse: int, phrase_reuse: float) -> dict:
    feats = features(lines)
    raw = raw_score(lines, params, reuse, phrase_reuse)
    return {"score": percentile_of(raw) / 100.0, "raw": raw,
            "soundness": float(soundness(params, feats[None, :])[0]),
            "cliche": cliche_rate(lines), "repeat": min(1.0, 0.5 * min(reuse, 2) + phrase_reuse),
            "season_clash": season_clash(lines), "features": feats}


def write_raw(records: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join("\n".join(l) for l in records) + ("\n" if records else ""),
                    encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="score AI haiku batches before they touch training data")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--audit", action="store_true",
                    help="only report whether the folder is haiku at all (form, duplicates); writes nothing")
    ap.add_argument("--store", type=float, default=None,
                    help="tier-A cut as a percentile vs our authored 180 (default 25)")
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--explain", default=None, help="score one poem: lines separated by |")
    ap.add_argument("--refresh-model", action="store_true")
    args = ap.parse_args(argv)

    rng = random.Random(7)
    params, payload = load_or_fit(args.refresh_model, rng)
    authentic_all, _, _ = collect(ROOT / "data" / "raw_classical", lowercase=False)
    authored, _, _ = collect(ROOT / "data" / "raw_synthetic", lowercase=False)
    # the percentile grid is the scale: it must be stored with the weights, or a later run would
    # score poems against a reference it cannot reproduce
    grid = payload.get("reference_quantiles") or build_reference(params, authored)
    REFERENCE["grid"] = grid
    if not payload.get("reference_quantiles"):
        payload["reference_quantiles"] = grid
        WEIGHTS_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.explain:
        lines = [l.strip() for l in args.explain.split("|")]
        info = score_poem(lines, params, 0, 0.0)
        print(f"percentile {info['score'] * 100:.0f}/100 (beats {info['score'] * 100:.0f}% of our "
              f"curated 180)   " + " / ".join(lines))
        print(f"  soundness {info['soundness']:.3f} | cliche {info['cliche']:.2f} "
              f"(penalty {CLICHE_WEIGHT * math.sqrt(min(1.0, info['cliche'] * 2.0)):.2f}) | "
              f"repeat {info['repeat']:.2f} | season clash {info['season_clash']:.0f} | "
              f"meter {'/'.join(str(n) for n in pattern(lines))} (distance "
              f"{distance(pattern(lines))}, {'tier-A eligible' if distance(pattern(lines)) == 0 else 'not 5-7-5'})")
        return 0

    threshold = args.store if args.store is not None else 25.0     # percentile points

    folder = args.dir
    if not folder.exists():
        raise SystemExit(f"{folder} does not exist")
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    authentic = authentic_all
    known = {dedupe_key(p) for p in authentic} | {dedupe_key(p) for p in authored}

    parsed: list[tuple[Path, list[list[str]]]] = []
    for path in files:
        if path.suffix.lower() in (".txt", ".md", ".text", ".json", ".jsonl", ".csv", ".tsv",
                                   ".parquet"):
            got = parse_file(path)
            if got:
                parsed.append((path, got))

    if args.audit:
        return audit(parsed, known)

    print(f"tier-A cut: percentile {threshold:.0f} — the poem must beat {threshold:.0f}% of our own "
          f"curated 180, and be exact 5-7-5")

    # batch-level phrase statistics: what the batch repeats is what the model would memorise
    phrase_counts: Counter = Counter()
    for _, poems in parsed:
        for lines in poems:
            ws = TOKEN_RE.findall(" ".join(lines[:3]).lower())
            phrase_counts.update(tuple(ws[i:i + 3]) for i in range(len(ws) - 2))
    busiest = max(phrase_counts.values()) if phrase_counts else 1

    per_file: dict[str, dict] = {}
    rejected: Counter = Counter()
    tier_a: list[list[str]] = []
    tier_b: list[list[str]] = []
    seen_b: set[str] = set()
    accepted_lines: Counter = Counter()
    scored: list[tuple[float, list[str], str, dict]] = []

    tier_b_cut = max(5.0, threshold * 0.4)
    for path, poems in parsed:
        counts: Counter = Counter()
        for lines in poems:
            counts["rows"] += 1
            lines = normalize(lines, lowercase=False)
            reason = reject_reason(lines)
            if reason:
                counts[reason] += 1
                rejected[reason] += 1
                continue
            key = dedupe_key(lines)
            if key in known:
                counts["duplicate_of_corpus"] += 1
                rejected["duplicate_of_corpus"] += 1
                continue
            reuse = max((accepted_lines[l.lower()] for l in lines), default=0)
            ws = TOKEN_RE.findall(" ".join(lines).lower())
            grams = [tuple(ws[i:i + 3]) for i in range(len(ws) - 2)] or [()]
            # a phrase used twice is normal; a phrase used eight times is a template
            phrase_reuse = min(1.0, (max(phrase_counts[g] for g in grams) - 1) / 7.0)
            info = score_poem(lines, params, reuse, phrase_reuse)
            info["phrase_reuse"] = phrase_reuse
            pct = info["score"] * 100
            # the form gate is separate from the score on purpose: a poem can read beautifully and
            # still not be a haiku. Tier A is exact 5-7-5 only, because it goes into training as a
            # form exemplar; a good-sounding poem that is off-meter is demoted, never promoted.
            form = distance(pattern(lines))
            if pct >= threshold and reuse < 2 and form == 0:
                known.add(key)
                for l in lines:
                    accepted_lines[l.lower()] += 1
                tier_a.append(lines)
                counts["tier_a"] += 1
                scored.append((pct, lines, path.name, info))
            elif pct >= tier_b_cut and form <= 1:
                if key not in seen_b:
                    seen_b.add(key)
                    tier_b.append(lines)
                    counts["tier_b"] += 1
                    scored.append((pct, lines, path.name, info))
                else:
                    counts["duplicate_of_corpus"] += 1
                    rejected["duplicate_of_corpus"] += 1
            elif form > 1:
                counts["off_meter"] += 1
                rejected["off_meter"] += 1
            else:
                counts["below_threshold"] += 1
                rejected["below_threshold"] += 1
        per_file[path.name] = dict(counts)

    write_raw(tier_a, AI_OUT / "ai_intake.txt")
    write_raw(tier_b, NEAR_OUT / "ai_intake_near.txt")

    own_percentiles = sorted(percentile_of(raw_score(p, params, 0, 0.0)) for p in authored)
    stats = {
        "model": {"auc_train": payload.get("auc_train"), "auc_held_out": payload.get("auc_held_out"),
                  "positives": payload.get("positives"), "negatives": payload.get("negatives"),
                  "features": FEATURE_NAMES, "weights": {n: float(w) for n, w in zip(FEATURE_NAMES, params[1:])}},
        "scale": "percentile against the authored 180 (100 = better than all of them)",
        "reference_quantiles": grid,
        "threshold_percentile": threshold, "files": per_file,
        "tier_a": len(tier_a), "tier_b": len(tier_b), "rejected": dict(rejected),
        "own_percentile_median": own_percentiles[len(own_percentiles) // 2],
    }
    STATS_OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    scored.sort(key=lambda t: -t[0])
    median_own = stats["own_percentile_median"]
    report = [
        "# AI haiku intake — scored, not just filtered",
        "",
        "`ai_haiku_vetter.py` gives every poem a **percentile against our own curated 180**: \"this",
        "poem beats X% of the poems we wrote and hand-verified\". Three measured parts drive it:",
        "",
        "1. soundness — a logistic model fitted on real poems (authentic corpus + our authored 180)",
        "   against **corruptions of those same poems** (lines swapped, content word replaced or",
        "   deleted, word stuttered, line shuffled), so it measures \"is this poem intact\" rather than",
        f"   \"does this sound like 1902\". Held-out AUC **{payload.get('auc_held_out'):.3f}**.",
        f"2. cliché penalty — {CLICHE_WEIGHT:.0%} of the graded weight, from the share of abstract",
        "   filler words (\"joy\", \"heart\", \"life\", \"dream\"…).",
        f"3. repetition penalty — {REPEAT_WEIGHT:.0%}, from lines already reused in this intake and",
        "   3-word phrases the batch repeats, plus a small hit when two lines name clashing seasons.",
        "",
        f"- source folder: `{folder}` — {len(parsed)} file(s), "
        f"{sum(c.get('rows', 0) for c in per_file.values())} poems read",
        f"- **tier A: {len(tier_a)}** → `data/raw_ai/` (training-ready: passes the score cut **and** "
        "is exact 5-7-5 by `backend.meter_check`)",
        f"- tier B: {len(tier_b)} → `data/raw_ai_near/` (within ±1 syllable; not mixed by default)",
        f"- below the cut (percentile {threshold:.0f}): {rejected.get('below_threshold', 0)}",
        f"- dropped for form alone, however good the score ({rejected.get('off_meter', 0)}): more "
        "than 1 syllable off 5-7-5",
        "- the gate and the score are independent: the score ranks, the meter decides whether the "
        "result may enter the corpus at all",
        f"- calibration: our authored 180 span percentiles 1-100 with median **{median_own:.0f}**,",
        "  so \"tier A\" is literally \"beats the bottom quarter of what we already wrote\".",
        "",
        "## What the soundness model weighs",
        "",
        "| feature | weight | (positive = this feature makes a poem look real) |",
        "|---|---|---|",
    ]
    for name, weight in sorted(zip(FEATURE_NAMES, params[1:]), key=lambda kv: -abs(kv[1])):
        report.append(f"| {name} | {weight:+.3f} | |")
    report += ["", "## Per file", "",
               "| file | rows | tier A | tier B | below | off meter | duplicates of corpus | junk | mean percentile of accepted |",
               "|---|---|---|---|---|---|---|---|---|"]
    for name, c in per_file.items():
        junk = sum(c.get(k, 0) for k in ("not_3_lines", "no_letters", "line_too_long",
                                         "too_much_non_ascii", "url_or_mention"))
        mine = [s[0] for s in scored if s[2] == name]      # percentiles of the accepted rows
        mean = f"{sum(mine) / len(mine):.0f}" if mine else "—"
        report.append(f"| `{name}` | {c.get('rows', 0)} | {c.get('tier_a', 0)} | {c.get('tier_b', 0)} | "
                      f"{c.get('below_threshold', 0)} | {c.get('off_meter', 0)} | "
                      f"{c.get('duplicate_of_corpus', 0)} | {junk} | {mean} |")
    top = scored[: args.show]
    if top:
        report += ["", f"## The {len(top)} highest-scoring poems", ""]
        for i, (score, lines, src, info) in enumerate(top, 1):
            report.append(f"{i}. **p{score:.0f}** (sound {info['soundness'] * 100:.0f}, "
                          f"cliché {info['cliche']:.2f}, repeat {info['repeat']:.2f}) — "
                          + " / ".join(lines))
    edge = sorted((s for s in scored if abs(s[0] - threshold) < 1.0), key=lambda t: t[0])[: args.show]
    if edge:
        report += ["", f"## {len(edge)} poems sitting on the cut (the judgement calls)", ""]
        for score, lines, src, info in edge:
            report.append(f"- **p{score:.0f}** — " + " / ".join(lines))
    report += ["", "## Next", "",
               "```bash",
               "python scripts/data/build_synth_corpus.py --repeat 6 --ai-repeat 2    # rebuild the mix (x2: see RUNBOOK 2d)",
               "python -m backend.train --corpus data/haikus_mix.txt --init-from checkpoints/best.pt \\",
               "  --ckpt-dir checkpoints_combo --lr 6e-5 --embed-dim 384 --layers 8 --ffn-dim 1536 --epochs 12",
               "```",
               "", "Drop the next batch and re-run `python scripts/data/ai_haiku_vetter.py`. Move the cut with "
               "`--store 0.90`, refit with `--refresh-model`, interrogate one poem with "
               "`--explain \"line one | line two | line three\"`.", ""]
    REPORT_OUT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"\ntier A {len(tier_a)} -> {AI_OUT / 'ai_intake.txt'}")
    print(f"tier B {len(tier_b)} -> {NEAR_OUT / 'ai_intake_near.txt'}")
    print(f"rejected: {dict(rejected)}")
    print(f"wrote {REPORT_OUT.name}, {STATS_OUT.name}, {WEIGHTS_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
