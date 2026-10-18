"""Module 9 - inference engine.

Plain sampling runs temperature 0.8, top-p 0.9 and a repetition penalty, stopping at <eos> or
after three newlines. --strict switches to meter-constrained decoding (Module 10): tokens that
would break 5-7-5 are masked while sampling, so there is no retry loop.
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

import torch

if __package__ in (None, ""):  # allow: python backend/generate.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import meter_check  # noqa: E402
from backend.config import Config  # noqa: E402
from backend.dataset import load_tokenizer  # noqa: E402
from backend.model import HaikuZeroLM  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BEST_CKPT = ROOT / "checkpoints" / "best.pt"
TOKENIZER_PATH = ROOT / "tokenizer.json"
# 1.1 was too weak to stop a repeated content word (22% of poems). At 1.3: 0% repeats, 5-7-5
# unchanged. Older checkpoints carry 1.1 in their config, so the server takes the larger value.
DEFAULT_REPETITION_PENALTY = 1.3


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_seed(prompt: str) -> str:
    """Prompt -> the words a poem can say: lowercase letters and inner apostrophes.

    Priming copies the prompt into the poem, so it has to be clean before it gets there.
    """
    words = (w.strip("'") for w in re.findall(r"[a-z']+", prompt.lower()))
    return " ".join(w for w in words if w)


def seed_context(tok, prompt: str) -> list[int]:
    """Inference twin of the training seed format: [<bos>, <seed>, *prompt, newline].

    A prompt therefore means "write a poem about this", the job the model is trained for.
    """
    return ([tok.token_to_id("<bos>"), tok.token_to_id("<seed>")]
            + tok.encode(" " + prompt).ids + tok.encode("\n").ids)


def load_model(checkpoint: Path = BEST_CKPT, tokenizer_path: Path | None = None,
               device: str | None = None):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"{checkpoint} missing - run backend.train first")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sibling = checkpoint.parent / "tokenizer.json"      # self-contained checkpoint, preferred
    if tokenizer_path is None and sibling.exists():
        tokenizer_path = sibling
    else:
        tokenizer_path = Path(tokenizer_path or payload.get("tokenizer") or TOKENIZER_PATH)
    cfg = Config(**payload["config"])
    model = HaikuZeroLM(cfg)
    model.load_state_dict(payload["model"])
    dev = pick_device(device)
    model.to(dev).eval()
    tok = load_tokenizer(tokenizer_path)
    # meter-state checkpoints need their syllable table rebuilt from the tokenizer they ship with;
    # it is a buffer (not a weight), so a checkpoint can never disagree with its own tokenizer
    model.set_syllable_table(tok)
    return model, tok, cfg, dev, payload


@lru_cache(maxsize=2)
def junk_token_ids(tok) -> frozenset:
    """Token ids that would put a junk character into the poem, computed once per tokenizer.

    The strict path filters whole candidate lines; that is too slow for plain sampling (the
    whole vocab is a candidate), so this is the cheap static version: a token whose own text
    contains anything a haiku line should not (digits, punctuation, half a UTF-8 character)
    is banned outright. Newlines are exempt - they are how lines end.
    """
    return frozenset(
        i for i in range(tok.get_vocab_size())
        if has_junk(tok.decode([i]).replace("\n", ""))
    )


@torch.no_grad()
def sample_haiku(model: HaikuZeroLM, tok, prompt: str = "", temperature: float = 0.8,
                 top_p: float = 0.9, repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
                 max_new_tokens: int = 64, top_k: int = 64,
                 device=None, amp: bool | None = None, seed: int | None = None,
                 seed_first: bool = True):
    """Unconstrained sampling. Returns (up_to_3_lines, full_text, generated_ids).

    Structure is free (no meter), but junk characters are still masked out and `seed_first`
    still writes the seed as the poem's first word, so the switch means the same thing in
    both modes.
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    amp = dev.type == "cuda" if amp is None else amp
    eos_id = tok.token_to_id("<eos>")
    ids = seed_context(tok, prompt) if prompt else [tok.token_to_id("<bos>")]
    primed: list[int] = []
    if seed_first and prompt:
        # a whole paragraph is context, not the first line: prime only a line's worth of words
        if len(prompt.split()) <= 3 and meter_check.line_syllables(prompt) <= 5:
            primed = tok.encode(" " + prompt).ids
    ids = ids + primed                                          # the poem opens with your word
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    newline_id = tok.encode("\n").ids[0]
    banned_ids = sorted(junk_token_ids(tok))
    banned = torch.tensor(banned_ids or [-1], dtype=torch.long, device=dev)
    vocab = load_wordlist()      # gates fused words, same as strict mode

    generated: list[int] = list(primed)   # the seed counts as used, so the penalty can bite
    cur_ids: list[int] = list(primed)     # ids of the line being written, decoded as a whole
    used: set[str] = _spent(WORD_RE.findall(prompt.lower())) if primed else set()
    prev_last = ""                     # last word of the finished line, for the line-start stutter
    for _ in range(max_new_tokens):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits, _ = model(x[:, -model.cfg.seq_length:])
        logits = logits[0, -1].float().clone()
        logits[banned] = float("-inf")

        if repetition_penalty and repetition_penalty != 1.0 and generated:
            uniq = torch.tensor(sorted(set(generated)), device=logits.device)
            vals = logits[uniq]
            logits[uniq] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

        probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        full_sorted = sorted_probs.clone()          # rank -> probability, before top-p
        cutoff = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()

        # Free meter now filters at the token, like strict does. Without this the ranking
        # penalty fired only on finished poems, so a free poem said the seed word twice
        # ("blood-red wire ... heavy my blood") no matter what the reranker preferred.
        k = min(top_k, int(sorted_probs.numel()))
        cand_ids = sorted_idx[:k].tolist()
        cand_probs = tail_weighted(sorted_probs, full_sorted, k)
        line_text = tok.decode(cur_ids)
        spent = used | _spent(WORD_RE.findall(line_text.lower()))
        cand_texts = [tok.decode(cur_ids + [i]) for i in cand_ids]
        fits = allowed_candidates(cand_texts, line_text, None, spent, vocab,
                                  max_syllables=FREE_MAX_SYLLABLES)
        line_long = meter_check.line_syllables(line_text) >= FREE_MIN_SYLLABLES
        if not fits and line_text.strip():             # line is at its ceiling: close it here
            next_id = newline_id
        else:
            if not fits:                               # nothing written yet: sample anyway
                fits = list(range(len(cand_ids)))
            weights = [candidate_weight(cand_texts[i], line_text, spent, vocab, cand_probs[i],
                                        prev_last)
                       * (FREE_NEWLINE_BOOST
                          if line_long and "\n" in cand_texts[i] and "\n" not in line_text
                          else 1.0)
                       for i in fits]
            total = sum(weights)
            if total <= 0:                             # everything left is a flaw: sample anyway
                weights, total = [cand_probs[i] for i in fits], sum(cand_probs[i] for i in fits)
            roll = torch.rand(1, generator=generator).item() * total
            acc, next_id = 0.0, cand_ids[fits[0]]
            for idx, weight in zip(fits, weights):
                acc += weight
                if roll <= acc:
                    next_id = cand_ids[idx]
                    break

        generated.append(next_id)
        cur_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=dev)], dim=1)
        if next_id == eos_id:
            break
        line_done = tok.decode(cur_ids)
        if "\n" in line_done:                          # line finished: its words are spent now
            closed = line_done.split("\n", 1)[0]
            used |= _spent(WORD_RE.findall(closed.lower()))
            prev_last = last_word(closed)
            cur_ids = []
        if tok.decode(generated).count("\n") >= 3:      # only the poem's own newlines count,
            break                                        # the seed line has one too

    text = tok.decode(x[0, 1:].tolist())
    if prompt and "\n" in text:
        # drop the seed line: the poem starts after it (a primed seed word sits after it too,
        # so it stays part of the poem without any special case)
        text = text.split("\n", 1)[1]
    lines = [l.strip() for l in re.split(r"\n+", text) if l.strip()][:3]
    return lines, text, generated


WORD_RE = re.compile(r"[a-z']{2,}")
# Free meter used to keep writing until <eos>: with nothing bounding a line, a "free" poem came
# out as a single 20-syllable run-on. These are the same shape rules a haiku has, minus the
# exact target - a line may close anywhere between 4 and 10 syllables, but not later.
FREE_MAX_SYLLABLES = 10
FREE_MIN_SYLLABLES = 4
FREE_NEWLINE_BOOST = 2.5   # once a line is long enough, closing it is the preferred continuation
WHITELIST_PUNCT = set(" \t'\u2019-\u2010\u2011\u2012\u2013\u2014")
# runs of dashes / a dash at a word edge are the same filler trick as "?????" (e.g. "winter night------Suno")
PUNCT_RUN_RE = re.compile(r"[-\u2010-\u2015'\u2019]{2,}|(?:^|\s)[-\u2010-\u2015]|[-\u2010-\u2015](?:\s|$)")
# a quote mark floating between words ("time inde '") is a token-seam leftover, not punctuation
FLOATING_QUOTE_RE = re.compile(r"(?:^|\s)['\u2019](?=\s|$)")


def has_junk(text: str) -> bool:
    """True if `text` carries characters a haiku line should not contain.

    Zero-syllable punctuation is the trap: a line waiting for its last syllable can be filled
    with '?' without changing the count, which is how "soft night?????the?of?b" used to happen.
    """
    if PUNCT_RUN_RE.search(text) or FLOATING_QUOTE_RE.search(text):
        return True
    if "  " in text:                                 # "phone  haiku" is a seam glitch too
        return True
    # a word-final apostrophe ("... the string's boughs'") is the same seam leftover as a
    # floating one, just glued to a word: no line ends on an opening quote mark
    if text.rstrip("\n").rstrip().endswith(("'", "\u2019")):
        return True
    return any(ch not in WHITELIST_PUNCT and not ch.isalpha() for ch in text)


STOPWORDS = frozenset(
    "a an the and or of to in on at by for with from as is are was were be been i you he she "
    "it we they this that these those my your his her its our their not but so if nor than then "
    "there here when where which who whom whose what how why all any some no".split())


def _stem(word: str) -> str:
    """Crude stem, for repeat detection only: snow's -> snow, backs' -> back, snows -> snow."""
    word = word.strip("'")
    if word.endswith("'s"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("s"):
        word = word[:-1]
    return word


def _spent(words) -> set[str]:
    """Stems of the content words in `words` - what the poem has used up so far."""
    return {_stem(w) for w in words if w not in STOPWORDS}


def repeated_content_words(lines) -> list[str]:
    """Content words a poem says more than once, e.g. 'blood and blood are all'.

    Stopwords are exempt: repeating "the" is grammar, repeating "blood" is not.
    """
    words = [w for line in lines for w in WORD_RE.findall(str(line).lower())
             if w not in STOPWORDS]
    stems = [_stem(w) for w in words]
    return sorted({w for w, s in zip(words, stems) if stems.count(s) > 1})


def adds_repeat(text: str, used) -> bool:
    """True if `text` uses a content word already spent, in this line or an earlier one.

    Stems, not spellings: "blood's" counts as "blood" again.
    """
    return any(_stem(w) in used for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS)


def adds_repeat_now(piece: str, used) -> bool:
    """adds_repeat() for a candidate's new text only, not the line before it.

    Checking the whole line flags every continuation of a line that already used a word, which
    zeroes all candidates and falls back to the repeat it meant to filter.
    """
    if not piece or not used:
        return False
    return any(_stem(w) in used for w in WORD_RE.findall(piece.lower()) if w not in STOPWORDS)


def keeps_word_boundary(text: str, current: str, words: frozenset | None = None) -> bool:
    """False when a token glues a new word onto the previous one: "river" + "stream" = riverstream.

    The tokenizer has both "\\u0120stream" (a word opening) and "stream" (a word continuing), and this
    small model sometimes picks the continuation where a new word should start. A letter right after
    a space, or right after a word the corpus already completes ("river"), is that bug.
    """
    if not current or not text.startswith(current):
        return True
    piece = text[len(current):]
    if not piece or not piece[0].isalpha():
        return True
    if not current or current[-1].isspace():
        return False
    if words:
        last = current.rsplit(" ", 1)[-1].strip().lower()
        if len(last) >= 3 and last in words:      # "river" is a whole word already
            return False
    return True


APOSTROPHE_STUTTER_RE = re.compile(r"([a-z]{2,4})(?:['\u2019]\1)+")
WORD_STUTTER_RE = re.compile(r"([a-z]{3,4})\1")


def has_stutter(text: str) -> bool:
    """True when a word is said twice in a row - "autumn autumn day", "the the pond".

    Seeding invites this: the model primes on the word you typed and immediately repeats it.
    The same loop also shows up INSIDE one word at a token seam - "Tell'st'st'stwas" - which
    no word-level check could see, so the fragment repeat is caught too.
    """
    low = text.lower()
    words = WORD_RE.findall(low)
    if any(a == b for a, b in zip(words, words[1:])):
        return True
    return bool(APOSTROPHE_STUTTER_RE.search(low) or WORD_STUTTER_RE.search(low))


def allowed_candidates(candidate_texts: list[str], current: str, target: int | None,
                       used=frozenset(), words: frozenset | None = None,
                       max_syllables: int | None = None) -> list[int]:
    """Indices of candidate texts that keep the line within its target and add no junk.

    `candidate_texts` are the FULL line each candidate would produce (line so far + candidate),
    not the candidate's own piece. That matters for two reasons: a multi-byte character split
    across token boundaries only decodes correctly as a whole sequence, and syllable counts must
    be taken on the text the reader will see.

    A candidate containing a newline is allowed ONLY when the line reaches the target exactly -
    that single rule is what turns "5-7-5 by luck" into "5-7-5 by construction". `target=None`
    is free meter: only the junk and stutter checks apply, plus `max_syllables` when given, which
    is a ceiling on the line rather than a target (closing early is always allowed).

    Repeats and fused words are NOT decided here. They are weighted down in candidate_weight()
    instead: as tiers they used to outrank each other, so "blood of blood" beat a clean
    "river stream" (or the reverse) purely because of the order the tiers were consulted in.
    """
    allowed = []
    for i, text in enumerate(candidate_texts):
        head = text.split("\n", 1)[0]
        if text == current or has_junk(head) or has_stutter(head):
            continue
        if target is not None:
            if "\n" in text:
                if meter_check.line_syllables(head) != target:
                    continue
            elif meter_check.line_syllables(head) > target:
                continue
        elif max_syllables is not None and "\n" not in text:
            if meter_check.line_syllables(head) > max_syllables:
                continue
        allowed.append(i)
    return allowed


REPEAT_WEIGHT = 0.0        # repeats are dead weight: used only when every candidate repeats
# "river"+"stream" = riverstream, "blood"+"y" = bloody. The continuation is the model's top
# candidate after the seed, so a soft multiplier is not enough; 0.05 still produced "bloody".
SEAM_WEIGHT = 0.0
# "shavement", "windshavel": BPE pieces can compose a word the corpus never contained. Dead
# weight when anything else fits, never a rejection - a line still has to be able to finish.
UNATTESTED_WEIGHT = 0.0


TAIL_SCALE = 0.02          # rank outside the nucleus keeps 2% of its real probability


def tail_weighted(nucleus, full_sorted, k: int) -> list[float]:
    """Sampling weights for the top-k ranks: nucleus mass, plus a usable tail behind it.

    At temperature 0.8 the nucleus is often a single token with p=1.0. If a filter zeroes it,
    nothing is left to draw from and the sampler returns the token it tried to remove. Keeping
    a fraction of each rank's probability leaves somewhere to go.
    """
    return torch.maximum(nucleus[:k], full_sorted[:k] * TAIL_SCALE).tolist()


def first_word(text: str) -> str:
    words = WORD_RE.findall(text.lower())
    return words[0] if words else ""


def last_word(text: str) -> str:
    words = WORD_RE.findall(text.lower())
    return words[-1] if words else ""


def candidate_weight(text: str, current: str, used, words, weight: float,
                     prev_word: str = "") -> float:
    """Down-weight one allowed candidate that repeats a word or fuses two words together.

    Multipliers, not tiers: a poem is only stuck with a repeat when EVERY candidate repeats,
    and the same draw still prefers the better of two bad options instead of whichever tier
    happened to be checked first. Measurement that forced this: 44% of strict poems repeated
    a content word and 6% still did after ranking-only fixes.
    """
    head = text.split("\n", 1)[0]
    piece = head[len(current):] if head.startswith(current) else head
    if adds_repeat_now(piece, used):
        weight *= REPEAT_WEIGHT
    if not keeps_word_boundary(head, current, words):
        weight *= SEAM_WEIGHT
    if words and adds_invented_word(text, current, words):
        weight *= UNATTESTED_WEIGHT
    # "... the tree and / and the wound": the last word of one line opening the next is a
    # stutter, but has_stutter only sees one line at a time, so it is checked here.
    if prev_word and not current.strip() and first_word(head) == prev_word:
        weight = 0.0
    return weight


@torch.no_grad()
def sample_haiku_metered(model: HaikuZeroLM, tok, prompt: str = "", temperature: float = 0.8,
                         top_p: float = 0.9, repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
                         targets: tuple = (5, 7, 5), max_new_tokens: int = 64, top_k: int = 64,
                         device=None, amp: bool | None = None, seed: int | None = None,
                         seed_first: bool = True):
    """Meter-constrained decoding: each line is closed when it hits its syllable target.

    No retries. Candidates that would overshoot or add junk characters are masked out of the
    top-k, and if nothing fits the line is closed short (never over), which is reported as a
    non-exact line.

    `seed_first` writes the seed word as the poem's first word and lets the model continue it,
    which is how a human would be asked, and the only reliable way a model this small says the
    word you typed: reranking alone could not do it (measured 0/10 for 'autumn' before this).
    Returns (lines, pattern, all_lines_exact).
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    amp = dev.type == "cuda" if amp is None else amp
    eos_id = tok.token_to_id("<eos>")
    ids = seed_context(tok, prompt) if prompt else [tok.token_to_id("<bos>")]
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    newline_id = tok.encode("\n").ids[0]
    vocab = load_wordlist()      # vocabulary only, no reranking cost - it gates fused words

    lines: list[str] = []
    pattern: list[int] = []
    generated: list[int] = []
    used: set[str] = set()                     # content words the poem has already spent
    prev_last = ""                             # last word of the previous line (line-start stutter)
    line_ids: list[int] = []                   # ids of the line being written, decoded as a whole
    line_no, stall = 0, 0                      # the seed is context; the poem starts fresh
    stall_limit = 8          # tokens in a row that add no syllable: stop circling, close short

    if seed_first and prompt:
        primed = tok.encode(" " + prompt).ids
        if len(prompt.split()) <= targets[0] and meter_check.line_syllables(prompt) < targets[0]:
            line_ids = list(primed)            # "autumn ...": the seed word is the poem's own text
            ids = ids + primed
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            generated.extend(primed)           # the seed counts as used, so the penalty can bite;
            used |= _spent(WORD_RE.findall(prompt.lower()))

    def close_line(ids: list[int]) -> None:
        """Text of a finished line: decode the WHOLE id list, never token by token."""
        nonlocal prev_last
        closed = tok.decode(ids).split("\n", 1)[0].strip()
        lines.append(closed)
        pattern.append(meter_check.line_syllables(closed))
        used.update(_spent(WORD_RE.findall(closed.lower())))
        prev_last = last_word(closed)

    for _ in range(max_new_tokens):
        if line_no >= len(targets):
            break
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits, _ = model(x[:, -model.cfg.seq_length:])
        logits = logits[0, -1].float().clone()
        if repetition_penalty and repetition_penalty != 1.0 and generated:
            uniq = torch.tensor(sorted(set(generated)), device=logits.device)
            vals = logits[uniq]
            logits[uniq] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

        probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
        # nucleus filtering must happen in sorted space: masking the unsorted vector with
        # rank-based cutoffs zeroes random tokens instead of the tail
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        full_sorted = sorted_probs.clone()          # rank -> probability, before top-p
        cutoff = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        k = min(top_k, int(sorted_probs.numel()))
        cand_ids = sorted_idx[:k].tolist()
        cand_probs = tail_weighted(sorted_probs, full_sorted, k)

        target = targets[line_no]
        line_text = tok.decode(line_ids)
        cand_texts = [tok.decode(line_ids + [i]) for i in cand_ids]
        in_line = _spent(WORD_RE.findall(line_text.lower()))
        spent = used | in_line
        fits = allowed_candidates(cand_texts, line_text, target, spent, vocab)
        weights = [candidate_weight(cand_texts[i], line_text, spent, vocab, cand_probs[i],
                                    prev_last)
                   for i in fits]
        if not fits:                      # cannot reach the target from here: close the line short
            cand_ids, weights, fits = [newline_id], [1.0], [0]
        total = sum(weights)
        if total <= 0:
            weights, total, fits = [1.0], 1.0, [fits[0]]
        roll = torch.rand(1, generator=generator).item() * total
        acc, next_id = 0.0, cand_ids[fits[0]]
        for idx, weight in zip(fits, weights):
            acc += weight
            if roll <= acc:
                next_id = cand_ids[idx]
                break

        generated.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=dev)], dim=1)
        piece = tok.decode([next_id])
        if "\n" in piece:
            close_line(line_ids + [next_id])
            line_ids, line_no, stall = [], line_no + 1, 0
        elif next_id == eos_id:
            # strict mode promises three lines: <eos> ends the line, it does not stop the poem
            close_line(line_ids)
            line_ids, line_no, stall = [], line_no + 1, 0
        else:
            before = meter_check.line_syllables(line_text)
            line_ids.append(next_id)
            stall = 0 if meter_check.line_syllables(tok.decode(line_ids)) > before else stall + 1
            if stall >= stall_limit:
                close_line(line_ids)
                line_ids, line_no, stall = [], line_no + 1, 0

    if line_ids and tok.decode(line_ids).strip() and line_no < len(targets):
        close_line(line_ids)
    return lines[:3], tuple(pattern[:3]), tuple(pattern[:3]) == tuple(targets)


@torch.no_grad()
def score_text(model, tok, context_ids: list[int], text: str) -> float:
    """Mean per-token log-likelihood of `text` given the context. Higher = more fluent."""
    dev = next(model.parameters()).device
    budget = max(2, model.cfg.seq_length - len(context_ids))   # every candidate scored on the same budget
    text_ids = tok.encode(text).ids[:budget]
    if len(text_ids) < 2:
        return float("-inf")
    ids = context_ids + text_ids
    logits, _ = model(torch.tensor([ids], device=dev))
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    targets = torch.tensor(ids[len(context_ids):], device=dev)
    picked = logp[len(context_ids) - 1 : len(ids) - 1].gather(-1, targets.view(-1, 1)).squeeze(-1)
    if picked.numel() == 0:
        return float("-inf")
    return float(picked.mean())


def repeated_content_rate(lines) -> float:
    """Fraction of CONTENT words the poem spends twice (stems, so snow's == snow).

    This is the number the UI shows, the number the probes report and the one the decoder fixes
    target. An all-words rate used to sit next to it and count grammar too ("the ... the ..."),
    so a perfectly clean poem scored 29% and the repetition fix looked like it had done nothing;
    with one metric everywhere, the probe and the API can no longer disagree.
    """
    words = [w for line in lines for w in WORD_RE.findall(str(line).lower())
             if w not in STOPWORDS]
    if not words:
        return 0.0
    stems = [_stem(w) for w in words]
    return 1.0 - len(set(stems)) / len(stems)


SEED_BONUS = 0.6
# A 17-syllable poem has no room to say a content word twice; the poetry pretrain taught the model
# that it does ("blood and blood are all", 44% of poems vs 9% before the poetry stage). Big enough
# to outrank a marginally more fluent candidate, small enough that a poem is never rejected for it.
REPEAT_PENALTY = 2.5
# Bigger than any fluency/repeat/seed difference between candidates: a poem of words the corpus
# contains always beats one that invented a word. Without the tier, the 1.2*rate bonus could be
# outweighed by a slightly better logprob and the invented word shipped ("square for my mouth ons").
UNATTESTED_PENALTY = 2.0

WORD_WEIGHT = 1.2          # weight of the "is that even a word" signal (logprob is ~-3 per token)
SUFFIXES = ("ing", "ers", "er", "ed", "es", "est", "ly", "s", "d", "'s")
WORDLIST_PATH = ROOT / "data" / "wordlist.txt"
# every corpus the model is trained on, in both stages: stage-1 poetry counts, otherwise
# the metric calls "yonder" (a word PD verse taught it) an invention
WORDLIST_CORPORA = (ROOT / "data" / "haikus_clean.txt", ROOT / "data" / "haikus_classical.txt",
                    ROOT / "data" / "poetry_pretrain.txt")
# A wordlist entry that cannot be an English word at all: the tweet corpora glue whole lines
# together, so the list contained "slidesin'tohisemptysandals" and
# "weshenjuggalslomomengreskeytemskeytudlogueri". Measured: 489 of 145,515 entries (0.34%), and
# the only false positive is "rock'n'roll" - which is not a haiku word. Everything else is left
# ALONE on purpose, because no automatic test for "is this English" exists here: words missing
# from the poetry corpus include modern English ("zucchini", "gearbox", "acne"), and repetition
# does not separate them either ("breuck" 28x and "sundance" 52x in the haiku corpora, against
# "zucchini" 9x). The measured cost of the junk that survives is ~1 poem in 240 (RUNBOOK 2e).
WORDLIST_MAX_CHARS = 18
SOURCES_LINE = ("# sources: " + " ".join(Path(c).name for c in WORDLIST_CORPORA)
                + f" | drop entries longer than {WORDLIST_MAX_CHARS} chars or with 2+ apostrophes")


BANNED_PATH = ROOT / "data" / "banned_words.txt"
_BANNED_CACHE: dict[Path, tuple[tuple[int, int], frozenset]] = {}


def banned_words(path: Path | None = None) -> frozenset:
    """Words the sampler must not put in a poem, one per line in data/banned_words.txt.

    Prompted by 'blood' producing crucifixion imagery: a 15.8M model trained on PD verse
    has absorbed the Christian register of that corpus, and no amount of reranking removes
    it, because it is what the weights prefer. Deleting it from the weights means another
    full pretrain, so it is filtered at sampling time instead - cheap, reversible, and it
    costs nothing when the list is empty (an empty file just returns an empty set).

    Whole words only, so "cross" does not ban "crossing"; comments start with '#'.

    Cached by the file's (mtime, size), not forever: editing the list used to need a server
    restart, which made "add a word" quietly do nothing until the next restart.
    """
    path = Path(path or BANNED_PATH)      # resolved at call time so tests can redirect it
    if not path.exists():
        _BANNED_CACHE.pop(path, None)
        return frozenset()
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _BANNED_CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    words = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        word = raw.split("#", 1)[0].strip().lower()
        if word:
            words.add(word)
    frozen = frozenset(words)
    _BANNED_CACHE[path] = (stamp, frozen)
    return frozen


def clear_word_caches() -> None:
    """Forget the word lists read from disk (path swaps in tests; the server reloads itself)."""
    _BANNED_CACHE.clear()
    load_wordlist.cache_clear()


def has_banned(text: str, exempt: frozenset | set = frozenset()) -> bool:
    """True if `text` uses a banned word. `exempt` is normally the seed you typed."""
    banned = banned_words()
    if not banned:
        return False
    return any(w in banned and w not in exempt for w in WORD_RE.findall(text.lower()))


def blocked_words(prompt: str = "") -> frozenset:
    """Banned words minus the prompt's own words: seeding 'cross' is consent to say it."""
    banned = banned_words()
    if not banned:
        return frozenset()
    return banned - set(WORD_RE.findall(prompt.lower()))


@lru_cache(maxsize=1)
def load_wordlist(path: Path = WORDLIST_PATH) -> frozenset:
    """Every word the training corpora contain, cached to data/wordlist.txt.

    The model composes byte-level pieces, so it happily invents "shavement" and "windshavel"
    to fill a syllable. A word the corpus never contained is almost always that kind of filler,
    and nothing else in the score notices.

    The cache records which corpora produced it: a stale wordlist would quietly report every
    new word the model learned as an invention, which is the exact opposite of the truth.
    """
    path = Path(path)
    cached = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if not cached or cached[0] != SOURCES_LINE:
        words: set[str] = set()
        for corpus in WORDLIST_CORPORA:
            if Path(corpus).exists():
                words |= set(WORD_RE.findall(Path(corpus).read_text(encoding="utf-8").lower()))
        words = {w for w in words if not impossible_word(w)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SOURCES_LINE + "\n" + "\n".join(sorted(words)), encoding="utf-8")
        cached = path.read_text(encoding="utf-8").splitlines()
    return frozenset(cached[1:])


def impossible_word(word: str) -> bool:
    """True for wordlist entries no English word could be (line-join garbage from the tweets)."""
    return len(word) > WORDLIST_MAX_CHARS or word.count("'") > 1


def known_word(word: str, words: frozenset) -> bool:
    """True if the corpus contains `word`, or a regular inflection of a word it contains.

    Edge apostrophes are stripped first: WORD_RE keeps them ("boughs'"), and a possessive
    that landed at a line end used to read as an invented word.
    """
    word = word.strip("'\u2019")
    if word in words:
        return True
    return any(word.endswith(s) and len(word) - len(s) >= 2 and word[: -len(s)] in words
               for s in SUFFIXES)


def added_words(text: str, current: str) -> list[str]:
    """Words this candidate FINISHED: written after `current` and followed by a boundary.

    Only finished words count. "moon" + "light" is one word still being written, so checking
    the pieces separately would call "light" an invention even though "moonlight" is attested;
    a word that ends the text is skipped for the same reason. A newline counts as the boundary,
    so a word an invented line END would end on is caught - that is the "... my mouth ons" case.
    """
    return [m.group() for m in WORD_RE.finditer(text.lower())
            if m.end() > len(current) and m.end() < len(text)]


def adds_invented_word(text: str, current: str, words: frozenset) -> bool:
    """True if the candidate completes a word the training corpora never contained.

    This is the decoder twin of the `nonword` column in quality_probe.py: the reranker could
    only rank a poem containing "shavement" lower, so the poem still shipped with it. As a
    weight it is only a preference - every candidate being an invention still lets one through.
    """
    return any(not known_word(w, words) for w in added_words(text, current))


def attested_fraction(lines, words: frozenset | None = None) -> float:
    """Fraction of words the corpus actually contains (inflections allowed: snows -> snow)."""
    words = load_wordlist() if words is None else words
    tokens = WORD_RE.findall(" ".join(lines).lower())
    if not tokens:
        return 0.0
    return sum(known_word(t, words) for t in tokens) / len(tokens)


def seed_adherence(lines, prompt: str) -> float:
    """1.0 when the poem actually says the seed word (or an inflection of it), else 0.0.

    Needed because mean token log-likelihood alone rewards the most PREDICTABLE text, and
    "a poem about anything" is more predictable than "a poem about autumn" - so the reranker
    was actively picking the candidates that ignored the seed. Measured before this existed:
    seed 'autumn' reached the poem in 0/10 samples.
    """
    prompt = prompt.strip().lower()
    if not prompt:
        return 0.0
    text = " ".join(lines).lower()
    if prompt in text:
        return 1.0                                 # multi-word seeds match as a phrase
    head = prompt.split()[0]
    stem = head[:max(4, len(head) - 2)]            # autumn->autu, breeze->bree, snow->snow
    words = re.findall(r"[a-z']+", text)
    return 1.0 if any(w == head or w.startswith(stem) for w in words) else 0.0


def best_of_haiku(model, tok, prompt: str = "", n: int = 1, strict: bool = False,
                  seed_first: bool = True, word_weight: float = WORD_WEIGHT, **kw):
    """Generate n candidates, keep the best by fluency + real words + seed adherence + meter.

    Style is set by the data; this buys the best of what the model can already do, at n
    forward passes of cost. In strict mode non-exact candidates are ranked last, and a poem
    made of words the corpus contains outranks one that invented them.

    Candidates using a word from data/banned_words.txt are dropped rather than ranked, and the
    sampler keeps drawing (up to 4n) until one clean poem exists. That is rejection sampling,
    not retraining: it changes nothing in the weights, so it is undone by editing the file.
    """
    base_seed = kw.get("seed")
    prompt = clean_seed(prompt)          # "snow!" must not become the poem's first line

    words = load_wordlist() if word_weight else None
    blocked = blocked_words(prompt)

    def one(i: int):
        kw["seed"] = None if base_seed is None else base_seed + i
        if strict:
            return sample_haiku_metered(model, tok, prompt, seed_first=seed_first, **kw)
        lines, _, _ = sample_haiku(model, tok, prompt, seed_first=seed_first, **kw)
        return lines, meter_check.pattern(lines), meter_check.is_haiku(lines)

    ctx = seed_context(tok, prompt) if prompt else [tok.token_to_id("<bos>")]
    best = fallback = None
    limit = n if not blocked else max(n, 4 * n)
    for i in range(limit):
        lines, pat, exact = one(i)
        if not lines:
            continue
        attested = attested_fraction(lines, words) if words else 1.0
        score = (score_text(model, tok, ctx, "\n".join(lines))
                 - 0.5 * repeated_content_rate(lines)
                 + word_weight * attested
                 + SEED_BONUS * seed_adherence(lines, prompt)
                 - (1.0 if (strict and not exact) else 0.0))
        if words and attested < 1.0:
            score -= UNATTESTED_PENALTY      # tier: every real word outranks one invention
        entry = (score, lines, pat, exact)
        if blocked and any(w in blocked for w in WORD_RE.findall(" ".join(lines).lower())):
            if fallback is None:
                fallback = entry       # every draw said it: show the best of a bad set
            continue
        if repeated_content_words(lines):
            score -= REPEAT_PENALTY      # rank it down, never disqualify: the next candidate
        entry = (score, lines, pat, exact)   # can be a malformed word fused at a token seam
        if best is None or score > best[0]:
            best = entry
        if best is not None and i + 1 >= n:   # n candidates ranked already: done
            break
    chosen = best or fallback
    return (chosen[1], chosen[2], chosen[3]) if chosen else ([], (), None)


def attempt(model, tok, args):
    """One sample; --strict uses meter-constrained decoding instead of retrying."""
    return best_of_haiku(model, tok, args.prompt, n=args.best_of, strict=args.strict,
                         temperature=args.temperature, top_p=args.top_p,
                         repetition_penalty=args.repetition_penalty,
                         max_new_tokens=args.max_new_tokens, seed_first=args.seed_first,
                         seed=args.seed, amp=not args.no_amp)


def print_haiku(lines, pat, strict_ok=None, prompt: str = "") -> None:
    if prompt:
        print(f"  seed: {prompt}")
    for line in lines:
        print(f"    {line}")
    if pat:
        shown = tuple(pat) + (0,) * (3 - len(pat))
        tag = ("5-7-5 exact" if meter_check.distance(shown) == 0
               else f"near {shown[0]}-{shown[1]}-{shown[2]}")
        if strict_ok is False:
            tag += " (line closed short: no candidate could reach the target)"
        print(f"    [{tag}]")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Module 9: generate haiku from a checkpoint")
    ap.add_argument("--checkpoint", type=Path, default=BEST_CKPT)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--prompt", default=None, help="seed phrase; omit for interactive REPL")
    ap.add_argument("--count", type=int, default=1, help="how many haiku when --prompt is given")
    ap.add_argument("--strict", action="store_true",
                    help="meter-constrained decoding: close each line exactly at 5-7-5")
    ap.add_argument("--best-of", type=int, default=4,
                    help="generate N candidates, keep the most fluent (4 = measured sweet spot)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-seed-first", dest="seed_first", action="store_false",
                    help="strict mode: do NOT start the poem with the seed word (default: it does)")
    ap.add_argument("--temperature", type=float, default=Config().temperature)
    ap.add_argument("--top-p", type=float, default=Config().top_p)
    ap.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    ap.add_argument("--max-new-tokens", type=int, default=Config().max_new_tokens)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on CUDA")
    args = ap.parse_args(argv)

    model, tok, cfg, dev, payload = load_model(args.checkpoint, args.tokenizer, args.device)
    print(f"loaded {args.checkpoint} (epoch {payload.get('epoch')}, val {payload.get('val_loss'):.4f}) on {dev}\n")

    if args.prompt is not None:
        for i in range(args.count):
            if i:
                args.seed = None
            lines, pat, strict_ok = attempt(model, tok, args)
            print_haiku(lines, pat, strict_ok, args.prompt)
            print()
        return

    print("type a seed phrase, get a haiku (empty line or 'quit' to stop)")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("quit", "exit"):
            break
        args.prompt = prompt
        args.seed = None
        lines, pat, strict_ok = attempt(model, tok, args)
        print_haiku(lines, pat, strict_ok, prompt)
        print()


if __name__ == "__main__":
    main()
