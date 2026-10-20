# Data plan — what this model needs next

Written 2026-10-18, after the v2 corpus and both training stages were promoted. Every number
here is from a command in this repo; the throughput figures were measured on the RTX 4050
Laptop (6 GB) with the live checkpoint.

The short version: **100,000 generated haiku is affordable, but the model cannot use 100,000
of them yet.** At the safe synthetic ratio the current 39,697-poem corpus can absorb about
20,000. Growing the *real* corpus is the constraint, and the plan below is about which lever
to pull in what order.

---

## 1 · Where the model stands

| | |
|---|---|
| Stage-2 corpus | **39,697 poems**, ≈0.99M tokens |
| Stage-1 corpus | **54.35M tokens**, 1,954 of a 4,081-book catalogue |
| Parameters | 15.79M |
| Measured quality (strict, 150 poems) | 100% exact 5-7-5, 0% repeats, 0% invented, 0% junk |
| Generation speed | **0.21–0.36 s/poem** at best-of 1, **0.73–0.75 s/poem** at best-of 4 |

The decoder guarantees the form, so the weights only have to supply *language*. That is why
data — not capacity — is the lever.

## 2 · What has already been measured, and rules out two obvious ideas

**More parameters: worse.** A 33.65M model (512×10×2048) through the same two stages improved
validation loss both times (stage-2 val 3.8363 → 3.7759) and degraded every sample: invented
words 28% → 32%/29%, repeats 25% → 28%/32%, verbatim lines 1.1% → 2.2%. The 0.99M-token
corpus is the binding constraint, not the 15.8M weights.

**More self-generated data, alone: collapse.** Pure self-distillation drifts to **40% repeats**
(10 epochs) and **54%** (3 epochs). The same 900 pairs mixed with 3,000 authentic poems cut
invented words 30% → 18% with repeats only 19% → 25%. The authentic poems are load-bearing:
they are what stops the model memorising its own tics.

**Authored poems work, in small numbers.** 180 hand-written exact-5-7-5 poems mixed ×6 into
the corpus (42,004 rows, 12 epochs) moved val 3.9203 → 3.5646 with **zero verbatim copies** —
but val was flat after epoch one, so the lever is *more poems*, not more epochs.

## 3 · The three routes

### A · Self-generation at scale — cheap, useful up to ~20k

The machinery already exists: `scripts/self_distill.py --stage pairs` draws decoder-clean poems, one
per seed word, and keeps only poems that pass every rule the decoder enforces.

| | |
|---|---|
| Keep rate | **900 of 1,325 draws = 68%** (`no_seed` 406, `invented` 13, `repeat` 4, `banned` 2) |
| Draw cost | 0.21–0.36 s at best-of 1, 0.73–0.75 s at best-of 4 |
| 100k kept poems | **~12 h** at best-of 1 (+ rejects), **~20–30 h** at best-of 4 |
| Training cost | 3 epochs over 140k rows ≈ **2–3 min** of GPU |

The catch is §2. At 39,697 real poems, the largest synthetic share worth testing is around a
third: **100,000 synthetic + 39,697 real is 71% synthetic**, past the point where the measured
collapse begins. A 100k run is therefore 5× more generation than the corpus can absorb today;
the useful target is **10k–20k synthetic pairs**, which is 1–2 hours of generation.

```powershell
# the existing tool, at pilot scale (one poem per seed word, decoder-filtered)
python scripts/self_distill.py --stage pairs --pairs 2000 --best-of 4
# then train with the authentic corpus as the regulariser
python scripts/self_distill.py --stage train --epochs 3 --mix data/haikus_classical.txt:3000
```

Not built yet, and needed before any run above ~5k: `stage pairs` writes one fixed file and
starts from scratch each time. A large run wants shards, resume-after-interrupt, and a global
dedupe across shards (the current dedupe is in-memory, per run).

### B · Stronger external model — the intake already exists, the funnel is brutal

`scripts/data/ai_haiku_vetter.py` + `scripts/data/import_ai_haiku.py` already score poems against the authored 180 and
split them into tiers. Real numbers from the last intake:

| | |
|---|---|
| Poems read | 4,722 |
| **Tier A** (training-ready: score cut **and** exact 5-7-5) | **89** — 1.9% |
| Tier B (±1 syllable) | 289 |
| Below the score cut | 1,895 |
| Off meter by more than a syllable | 2,197 |

So **100,000 usable poems would need ~5.3M generations** — about 4× the whole 1.3M-record
stage-1 corpus, and an API bill that dwarfs the GPU time of route A. Either the prompt gets
much better at 5-7-5 (ask for *exact* meter, one poem per request, no titles, no commentary)
or the vetter has to accept tier B and repair the offending line with the meter counter.

The honest use of route B is a **small, high-quality set** — a few thousand poems at tier A —
not a bulk dump. Its held-out AUC is 0.787, good enough to rank, not to certify.

### C · More authentic haiku — the lever that costs no quality

Everything measured so far says the corpus is short, not the model. Concrete openings:

1. **The rest of the Gutenberg catalogue.** 4,081 books are listed, 1,954 contributed at the
   150M-char budget. Not haiku, but stage-1 vocabulary is where invented words come from
   (34% → 24% already).
2. **Public-domain haiku in translation.** R. H. Blyth-era collections and similar are
   public domain in many jurisdictions; each one is thousands of real, metered poems.
3. **More authored poems.** 180 → 400 was the original sketch; this is the only source whose
   content and form can both be guaranteed, and it costs writing time, not GPU time.

## 4 · The experiment to run first

Three arms, **the same number of training steps** (set with `--max-steps`, so the comparison is
the data mix and not the compute), all warm-started from `checkpoints/best.pt`:

| arm | corpus | synthetic share | question |
|---|---|---|---|
| A | 40k authentic | 0% | baseline |
| B | 40k authentic + 10k synthetic | 20% | does 20% synthetic help? |
| C | 40k authentic + 20k synthetic | 33% | where does it turn? |

Score each arm with the **filters-off probe** (candidate weights neutralised, best-of 1, free
meter) rather than the strict decoder, because strict saturates at 100% and cannot tell arms
apart. Report invented words, repeats, seed adherence, meter distance, and verbatim copies of
corpus lines.

```powershell
python scripts/self_distill.py --stage probe --models checkpoints,checkpoints_selfdistill
python scripts/eval/quality_floor.py          # must stay green for the promoted checkpoint
```

**Promotion rule:** an arm is promoted only if invented words *and* repeats are both no worse
than the baseline, and `scripts/eval/quality_floor.py` still exits 0. Anything that trades one for the other
is a finding for [PROGRESS.md](PROGRESS.md), not a new default.

## 5 · Order of work

1. Pilot: 2,000 self-generated pairs (~15 min), confirm the keep rate on this machine.
2. Run the three-arm A/B above (~1 h generation, ~10 min training).
3. If synthetic helps at ≤20%: build sharded/resumable generation, then the 20k set.
4. In parallel, the real lever: import more authentic collections and grow the authored set.
5. Only after the real corpus passes ~120k poems does a 100k synthetic set become usable at a
   third of the mix.

## 6 · Storage and cost

| Artefact | Size | Where |
|---|---|---|
| 20k synthetic pairs (JSONL) | ~2 MB | `data/`, not in git |
| 100k synthetic pairs | ~11 MB | `data/`, not in git |
| Trained checkpoint | 181 MB each | `checkpoints*/`, not in git |
| GPU time, 100k at best-of 4 | ~20–30 h | one 4050 |

The repo ships no corpora and no weights, on purpose: [`.gitignore`](../.gitignore) keeps them
out, and [README.md](../README.md) says so. What it does ship is `tokenizer.json`, because a
supplied checkpoint cannot run without it.
