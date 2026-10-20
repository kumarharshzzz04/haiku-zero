# HAIKU ZERO — PROGRESS

Living checklist. Update after every run (agent and human both). Facts come from `python scripts/status.py`,
never from memory. **Last updated:** 2026-10-18, the data phase landed (RUNBOOK §2c, §3b). The stage-1 poetry corpus was
budget-limited, so it was rebuilt from the rest of the catalogue — **1,954 books (was 611), 54.35M
tokens (3.5×), 271,966 unique words (2.1×)**, byte-identical on a verification re-run — trained through
both stages (stage 1 val **3.4573**, stage 2 val **3.8363**) and **promoted to `checkpoints/`**; the
previous live model is `checkpoints_v1_small/`. Raw-sample invented words **34% → 24%**, seed mention
85% → 93%, memorisation 1/180 → 2/180 lines, strict probes saturated either way. Before that, the meter experiments
(`Config.meter_state`, RUNBOOK §4c): giving the model syllables-so-far-in-line and line index, and
fine-tuning on the 5,930 exact-5-7-5 poems, lifts free-mode ±1 from **0% to 18-28%** — the data does
that, the feature itself is worth 2-6pp — and every form gain costs seed adherence (92% → 50-83%), so
the decoder still owns 5-7-5 and nothing was promoted. Before that: the typed word can be integrated
*inside* the poem (98% adherence, `scripts/eval/seed_placement_probe.py`), self-distillation moved invented words
30% → 18% into the weights, and 180 hand-written haiku were verified line by line and mixed into the
training data ×6 (lowest val loss in the repo, 3.5646, in `checkpoints_mix/`). The poetry
two-stage model stays live in
`checkpoints/`. Details in **SYNTH_CORPUS_REPORT.md**. See **Known limitations** below for what is
still open.

Legend: `[x]` done and verified · `[ ]` pending · `[~]` partially done · `[!]` blocked

## Phase checklist

### Phase 0 — environment
- [x] `pytorch_with_gpu` venv created (Python 3.11), activate scripts repointed after the rename
- [x] torch 2.14.1+cu130 installed, `cuda True`, `NVIDIA GeForce RTX 4050 Laptop GPU`
- [x] tokenizers 0.23.3, numpy installed
- [x] TensorFlow/WSL2 path rejected (no native-Windows TF GPU since 2.11) — see RUNBOOK §8

### Phase 1 — data + tokenizer
- [x] `data/raw/all_haiku.csv` in place (the 19 MB/144k-row source dataset)
- [x] Module 1 `backend/prepare_data.py` → 142,374 haiku kept (1,730 long lines, 12 URLs, 7 junk rows, 0 dupes)
- [x] `data/haikus_clean.txt` (9.9 MB) + `data/haikus_meter.txt` (108,605 meter-ish haiku) + `data/stats.json`
- [x] Module 2 `backend/train_tokenizer.py` → `tokenizer.json`, vocab 4096, byte-lossless, 16.6 tokens/haiku
- [x] Module 3 `backend/dataset.py` → 138,103 train / 4,271 val haiku, 2.90M train tokens, 45,319 windows, 354 steps/epoch

### Phase 2-3 — model
- [x] Module 4 `embedding.py` token + positional embeddings
- [x] Module 5 `multi_head_attention.py` hand-rolled causal attention
- [x] Module 6 `transformer_block.py` Pre-LN block, GELU MLP
- [x] Module 7 `model.py` → **5.80M params**, tied weights, float32 lm_head, initial loss 8.366 vs ln(4096)=8.32

### Phase 4 — train + inference
- [x] Module 8 `train.py` — AMP fp16 + GradScaler, warmup 100 → cosine over total steps, clip 1.0, best/last checkpoints, CSV log, per-epoch preview
- [x] Module 9 `generate.py` — plain sampling (temp 0.8 → top-p 0.9 → repetition penalty 1.1) + REPL
- [x] Module 10 `meter_check.py` — syllable counter, no external dependency
- [x] Meter-constrained decoding (`--strict`): masks overshooting candidates, newline only at target, no retry loop
- [x] Extra: `scripts/experiments.py` sweep (clean vs meter corpus × dropout 0.10/0.15)

### Verification already run
- [x] `tests/test_smoke.py` — all asserts pass in `pytorch_with_gpu`, exit 0 after every change (cleaning filters, tokenizer round-trip + byte-losslessness, `<bos>`/`<seed>` formats, prefix masking, packing masks, causal masking, tied weights, plain sampling, metered decoding + filter unit checks, meter counts, 2-step training + checkpoint reload)
- [x] GPU smoke: `--max-steps 50` on the real config, `device cuda`, loss 8.35 → 7.63
- [x] GPU timing: 399 steps/epoch, **~170k tokens/s → 25 epochs ≈ 8 min**
- [x] **Training v3 (live `checkpoints/best.pt`)**: 25 epochs, 9975 steps, train 3.5136, val **3.7771**, 8.0 min → val target (<4.5) met, train target (<3.5) missed by 0.014
- [x] Generation measured on v3: plain empty-prompt 3/10 exact 5-7-5; `--strict` 8/10 exact without a prompt, 3/10 with a seed phrase
- [x] **Best-of-n reranking** (`--best-of`, default 4; scores candidates by mean token log-likelihood, penalises repeated words, ranks non-exact last in `--strict`): measured on the live model, strict no-prompt: exact 8/10 → **10/10**, mean logprob **-2.68 → -2.27** (best-of 16 only reaches -2.17, so 4 is the sweet spot)

### Bugs found by evaluating (all fixed)
- [x] **No `<bos>` in training data** → poem starts were out-of-distribution, the model burned line 1 (empty first line, 2-line poems). Fixed: every poem is `[<bos> … <eos>]`.
- [x] **Seed phrases were continuations, not themes** → `snow` produced 2-syllable lines. Fixed: seed conditioning `[<bos> <seed> word \n poem <eos>]`, seed region excluded from the loss.
- [x] **Newline stop counted the seed line's newline** → generation stopped after 2 lines with a prompt. Fixed: only generated newlines count.
- [x] **Top-p cutoff applied to the unsorted probability vector** in the metered sampler → it zeroed random tokens and kept junk (`IFOG`, `I'N'!DAG?`). Fixed: mask in sorted space, then map back. Strict exactness went 0/10 → 8/10.
- [x] **Progress line tok/s + ETA reset per epoch** → reported `eta 10985m`. Fixed with a rolling rate.
- [x] **Checkpoints without their tokenizer** → adding `<seed>` made v1/v2 weights unusable. Fixed: `train.py` copies `tokenizer.json` into the checkpoint dir and `generate.py` prefers it. v1/v2 weights retired, their metrics kept in `logs/training/training_log_v1.csv` / `logs/training/training_log_v2.csv`.

## Remaining work

- [x] **Corpus winner picked**: meter-corpus model promoted to `checkpoints/` (strict 20/20 three-line,
  exact 5-7-5 19/20 empty / 16/20 seeded); clean-corpus run kept in `checkpoints_v3_clean/` for comparison
- [x] **Authored data lever taken (2026-10-17)**: 180 hand-written haiku (`data/raw_synthetic/`), all
  exact 5-7-5 by `backend/meter_check`, mixed into the authentic corpus ×6 → 42,004 poems, fine-tuned
  12 epochs from `checkpoints/best.pt` → **val 3.9203 → 3.5646** (`checkpoints_mix/`, best epoch 1,
  1488 steps, 2.5 min), 0 verbatim copies of an authored line in 75 generated lines. The decoder was
  never touched, so no probe column moved (all four already sat at 0 or 100%). Full write-up and
  30 verified samples: **SYNTH_CORPUS_REPORT.md**.
- [ ] Authored data is 180 poems, not 400: writing and syllabifying them is the bottleneck, and val is
  flat after the first epoch, so the next lever is more authored poems (then re-running the same
  fine-tune command), not more epochs.
- [x] **PLAN.md loss targets met by the live model**: train **3.4486** (<3.5), val **3.6813** (<4.5) at epoch 24.
  Val plateaued (3.6813 → 3.6816 at epoch 25), so more epochs are not the lever any more; `--dropout 0.15` or more data is
### Quality plan (what actually moves the needle, best first)

- [x] **1. More data — the only real lever for val loss.** 3.27M tokens for a 5.8M-param
  model was data-starved. **Taken 2026-10-18**: stage-1 corpus rebuilt to 54.35M tokens / 1,954 books
  (RUNBOOK §2c), both stages retrained (stage 1 val 3.4573, stage 2 val **3.8363** vs 3.9203) and
  promoted — raw-sample invented words **34% → 24%**, memorisation 1/180 → 2/180 lines (RUNBOOK §3b).
  `statworx/haiku` was imported long before this (28,642 poems) and is already inside the corpus.
- [~] **2. Style — data, not code (in progress).** Decision taken 2026-10-16: split the raw corpus by
  its own `source` column into `data/raw/` (full, 142,374 after cleaning) and `data/raw_classical/`
  (**30,658 authentic haiku**: gutenberg 4,992 + tempslibres 5,232 + sballas 8,125 + img2poems 12,309),
  then fine-tuned the live weights on the authentic subset (`--init-from --corpus --lr 6e-5`, 20 epochs,
  1.7 min) into `checkpoints_classical/`.
  **Measured trade (10-12 samples each, same seeds):**
  | model | strict exact | plain exact | 4-grams seen in twaiku | in authentic |
  |---|---|---|---|---|
  | live (tweet-heavy 142k) | 10/10 | 4/10 | 7% | 0% |
  | classical fine-tune (30.6k) | 9/10 | 0/10 | **4%** | **1%** |
  Verdict: style moves the right way (imagery becomes haiku-like, tweet overlap halves) but unconstrained
  meter degrades and grammar gets shakier — 0.62M tokens is too little to fine-tune 5.8M params without
  cost. **Not promoted**, both models ship (`checkpoints/` = best meter+fluency, `checkpoints_classical/` =
  authentic voice). Fixed size knob: `--checkpoint` picks either.
  Next fix: **more authentic haiku** (the only lever that raises style without cost), or a blended corpus
  (authentic + capped twaiku) trained from scratch — 8 min per attempt.
- [ ] **3. Inference knobs (free, no retrain)**: `--best-of` is in (done); temperature 0.7-0.9 and
  `--repetition-penalty 1.15` are worth a sweep; `--strict` already guarantees 3 lines.
- [ ] Optional: full 4-config sweep `python scripts/experiments.py --epochs 25` (~35 min)
- [ ] Do **not** bother: more epochs on this corpus (val flat), bigger model (would overfit 3M tokens harder).
- [x] **Browser demo (done 2026-10-16)**: `backend/server.py` (stdlib API, no new dependency) + `frontend/`
  (Vite + React + TS + Tailwind v4, shadcn-compatible) → poster view + sheet view with meter/logprob
  per poem; checkpoint dropdown auto-discovers all `checkpoints*/best.pt`. Run: RUNBOOK §5.
- [x] **Fit-for-purpose fixes after real use (2026-10-16, evening)** — three complaints, all measured:
  - **`?????` in poems — root cause found, fixed.** The glyphs were *real ASCII `?`* the corpus uses as
    line-end punctuation (4,829 of them, e.g. "midnight rain?"). Strict decoding forbids a newline until
    the line hits its syllable target, and `?` adds 0 syllables, so the model filled the gap with it:
    56% of strict samples from the authentic fine-tune contained runs like `soft night?????the?of?b`.
    Fixed at three layers: `generate.has_junk` rejects punctuation/digits as candidates, `has_stutter`
    rejects doubled words, and Module 1 now strips `?`, dot runs and edge punctuation from the targets.
    **Measured: 56% → 0%** of samples (`14/25` → `0/32`).
  - **The seed you type now reaches the poem.** Mean log-likelihood reranking was actively selecting
    seed-ignoring candidates ("a poem about anything" is more predictable than "about autumn"): 0/10
    for `autumn`. Added a seed-adherence bonus, then seed *leading* (`seed_first`, default on in strict
    mode: the word is written as the poem's first word and continued). **Measured: 0/10 → 100%**
    (and the UI labels it, so nothing is hidden).
  - **Word realism.** New `attested_fraction` rerank: poems of words the corpora contain outrank ones
    that invented them (`data/wordlist.txt`). **Measured: 9% → 5%** invented words on `checkpoints_balanced`.
- [x] **Corpus rebuilt around real haiku (2026-10-16, evening)**: imported `statworx/haiku`
  (28,642 non-tweet poems kept: bfbarry 7,037 · haiku_data_1 13,444 · haiku_data_2 8,032; its twaiku
  rows skipped), capped tweets at 35,000 (`data/raw_twaiku/` keeps the untouched original). Corpora:
  full **75,920** (1.43M tokens, 36% tweets, was 142,374 / 78%), authentic **40,924** (0.79M tokens,
  no tweets, was 30,658). Scripts: `scripts/data/import_external.py`, `scripts/data/prepare_corpora.py`. New tokenizer (vocab 4096,
  newline still one token, 24 tokens/haiku). Three models trained and probed: balanced 5.8M,
  authentic-only 5.8M, and **15.8M params on the balanced corpus — now promoted to `checkpoints/`**
  (the 78%-tweet model moved to `checkpoints_v3_live/` untouched). Model size became a flag
  (`--embed-dim/--layers/--ffn-dim/--seq-length`); 15.8M fits 6 GB at batch 64.
- [x] **Invented words: attacked from the data side and measured (2026-10-17).** Capacity was never the
  lever (5.8M → 15.8M changed nothing); vocabulary was. Built `data/poetry_pretrain.txt` — 375,681
  three-line records from 611 Project Gutenberg poetry books, **15.41M tokens, 120,488 unique words, all
  public domain** — then stage 1 pretrained 15 epochs on it (45.8 min, val 3.5145) and stage 2 fine-tuned
  25 epochs on the authentic haiku corpus (4.9 min, val 3.9203). **Promoted to `checkpoints/`.**
  Measured with seed leading OFF (the only honest seed test): invented words **3% → 1%** of words, seed
  adherence **58% → 100%**, and the haiku shape survived stage 2 (3.5 vs 3.4 words/line, 23% vs 27%
  within ±1 syllable, free meter). Stage 1 alone is the control: 6.5 words/line, 2% within ±1 — fluent
  verse, no haiku form, which is exactly what stage 2 supplies.
- [x] Next levers after this, in order: **more stage-1 text — taken** (611 → 1,954 books, 15.41M →
  54.35M tokens, RUNBOOK §3b). Still open, and now the top of the list: **a larger model** (the data is
  no longer the constraint: 15.8M params over 54M pretrain tokens is capacity-bound, not data-bound)
  and **`--best-of 8-16`** at inference (cost vs quality, unmeasured).
- [x] **A larger model — tested, and it is not the lever yet (2026-10-18).** 33.65M params beat 15.8M
  on both val losses but produced *worse* samples (more repeats, further from real haiku, more
  memorisation) — RUNBOOK §3c. The binding constraint is the 0.99M-token stage-2 corpus, so the next
  lever is stage-2 data: more authored poems (item above) or another authentic source, not parameters.
- [ ] Optional idea (not started): syllable-aware training signal (logit bias from the meter counter during training)
- [x] **Third decoder round — the leftovers on the list below (2026-10-17).**
  Invented words became a decoder weight and a reranker tier (`UNATTESTED_WEIGHT`,
  `UNATTESTED_PENALTY`, `adds_invented_word`), free meter got a length ceiling
  (`FREE_MAX_SYLLABLES`), a word-final apostrophe joined the junk rules, and `banned_words()` now
  caches on `(mtime, size)` so editing the list no longer needs a server restart.
  **Measured**: `scripts/eval/quality_probe.py --n 12` on the live model — strict and free both **invented words
  1% → 0%**, junk 0%, `?` 0%, exact 100% (strict), repeats 0%, seed 100%; `scripts/eval/repeat_probe.py` all four
  cells still **0% repeats / 1.00 seed uses**; free lines over 10 syllables **16% → 0%** (max 23 → 10);
  24 live API poems flagged 0; banned-list edit verified on the running server (6/18 poems said `snow`
  before, 0/18 after, no restart).

## Known limitations (measured, still open)

These are the things that are NOT fixed. Everything above them is filtered at sampling time; these
need data or training work, and saying so is cheaper than pretending otherwise.

1. **The filters live in the decoder, not in the weights.** `checkpoints/best.pt` would still write a
   repeat or an invented word if sampled raw (plain `sample_haiku`, no rerank). Any change to look at
   the model's real behaviour has to turn `--best-of 1` and the filters off. Measured how far
   self-distillation gets (`scripts/self_distill.py`, 900 decoder-clean pairs + 3,000 authentic poems, 3
   epochs): **invented words 30% → 18%** of raw poems move into the weights (≈4σ, n=240), adherence
   stays inside noise (83 → 86%), repeats get marginally worse (19 → 25%), and **meter not at all**
   (0% exact raw in every variant). The live model's raw numbers moved with the corpus: invented words
   in raw poems **34% → 24%** after the v2 pretrain (RUNBOOK §3b), still nowhere near 0. See RUNBOOK §4b. §4c pushed the meter question further: giving the
   model syllables-so-far-in-line and line index, and fine-tuning on the 5,930 exact-5-7-5 poems, lifts
   raw ±1 from **0% to 18-28%** (mean syllabic distance 9.70 → 3.4-4.8) — but the **data** does that;
   the feature itself is worth 2-6pp (6pp when ablated). Every form gain costs seed adherence in free
   mode (92% → 50% exact-corpus, → 83% with the distilled pairs mixed in), so the decoder is still the
   only thing that makes a poem 5-7-5 on demand.
2. **The vocabulary source is OCR'd public-domain poetry — mostly retired, measured.** `data/wordlist.txt`
   is now 145,026 words after dropping the 489 entries that cannot be English words at all
   (`impossible_word`: > 18 chars or 2+ apostrophes — the tweet corpora's line-join garbage). Two
   corrections from that pass: `gest` was a false example (a real archaic word, in 28 poetry books) and
   the real junk is the `ahkond` class (0 books, from the social-media corpora). No rule separates that
   class from modern English — a poetry-only rule drops 5.2% of the list including `zucchini` and
   `gearbox`, and repetition does not separate them either — so the list was deliberately **not**
   narrowed further: measured cost is ≈**1 poem in 240** (RUNBOOK §2e).
3. **Free meter is bounded but not shaped** — lines close between 0 and 10 syllables, so user-visible
   free poems often open on the seed word alone (1-3 syllables). The ceiling removes run-ons; it does
   not make free mode pretty. Strict mode is the one that looks like a haiku.
4. **Meaning is still thin.** Loss targets are met (train 3.4486, val 3.6813), so the remaining
   oddness ("wall on a mountain's should") is a data/scale problem, not a decoder bug: 3.27M haiku
   tokens for a 15.8M-param model. The poetry pretrain no longer is a data problem — v2 gives it 54.35M
   tokens of 1,954 books (RUNBOOK §3b) — so what is left is stage-2 haiku volume, model capacity and the
   fact that a 15.8M model cannot hold much semantics. The decoder still finds the 5-7-5, which is why
   strict quality saturates and weight-level probes are the only thing that moves.
5. **The syllable counter is heuristic** (±1 on unusual words, ~30 lines, no dictionary), which is why
   strict mode uses exact targets but the UI reports `distance` rather than claiming a perfect haiku.
6. **Reranking costs forwards passes**: `best-of 4` (default) quadruples a request. Server-side the
   model is also single-threaded (`LOCK`) and the first request after start pays the load.

## Log

| date | what ran | result |
|---|---|---|
| 2026-10-16 | data prep + tokenizer | 142,374 haiku, vocab 4096, byte-lossless |
| 2026-10-16 | `tests/test_smoke.py` (new env) | ALL SMOKE CHECKS PASSED |
| 2026-10-16 | `backend.train --max-steps 50` (GPU) | cuda 4050, loss 8.35 → 7.63, val 7.02 |
| 2026-10-16 | training v1 (25 ep, pre-fix) | train 3.7149, val 3.9244, 6.9 min · weights retired, metrics in `logs/training/training_log_v1.csv` |
| 2026-10-16 | training v2 (25 ep, `<bos>` + prefix masking) | train 3.4126, val 3.8084, 7.5 min · weights retired (stale tokenizer), metrics in `logs/training/training_log_v2.csv` |
| 2026-10-16 | training v3 (25 ep, + seed conditioning, new tokenizer) | train 3.5136, val **3.7771**, 8.0 min · **live in `checkpoints/`** |
| 2026-10-16 | strict meter on v3 (clean) | 8/10 exact (no prompt), 3/10 with seed |
| 2026-10-16 | meter-corpus run (25 ep) | train **3.4486**, val **3.6813** (own val set), 7.0 min — **promoted to `checkpoints/`**, both PLAN.md loss targets met |
| 2026-10-16 | strict meter, live model (after the eos fix) | **20/20 three lines**, 19/20 exact empty, 16/20 exact seeded |
| 2026-10-16 | `--best-of N` reranking added | strict exact 8/10 → 10/10, mean logprob −2.68 → −2.27 (default 4) |
| 2026-10-16 | corpus split by `source` column | full 142,374 · authentic 30,658 (`data/raw_classical/`) |
| 2026-10-16 | classical fine-tune (20 ep, lr 6e-5, 1.7 min) | val 4.236 on its own set · strict 9/10 · tweet overlap 7% → 4% · not promoted |
| 2026-10-16 | front end: Vite + React + TS + Tailwind v4 (shadcn layout) + `backend/server.py` | poster/sheet views live on :5177, API on :8000, 3 checkpoints auto-listed |
| 2026-10-16 | `tests/test_smoke.py` check [8] added (real HTTP socket + tiny checkpoint) | caught `relative_to` crash on out-of-project checkpoints (fixed), ALL SMOKE CHECKS PASSED |
| 2026-10-16 | diagnosed the `?????` complaint | real ASCII `?` from the corpus used as filler under strict meter; 56% of strict samples |
| 2026-10-16 | decoder fixes: junk/stutter filters, seed bonus + seed leading, word rerank | `?` 56% → **0%**, seed 0/10 → **100%**, invented words 9% → **5%** |
| 2026-10-16 | imported statworx/haiku + rebuilt corpora | full 75,920 (36% tweets), authentic 40,924 (none) |
| 2026-10-16 | trained balanced 5.8M (4.5 min) · authentic 5.8M (4.2 min) · big 15.8M (7.9 min) | val 4.135 / 4.363 / 4.079 (own splits); all probed: 0% junk, 100% exact, 100% seed, 4-5% invented |
| 2026-10-16 | promoted `checkpoints_big` → `checkpoints/`, old live → `checkpoints_v3_live/` | UI default now the balanced 15.8M model; API 3 poems in 2.9 s |
| 2026-10-17 | researched every large haiku source (3 web runs, downloads tested) | **no** large legally-usable literary haiku corpus exists; Kaggle 404s anonymous downloads; the vocabulary has to come from PD poetry — table in RUNBOOK §2b |
| 2026-10-17 | `scripts/data/fetch_poetry.py`: 962 PG books cached, corpus built | **375,681 records, 15.41M tokens, 120,488 unique words**, 611 books, `the` 5.63%, 0 U+FFFD |
| 2026-10-17 | stage 1 pretrain, 15 ep, 15.4M tokens (45.8 min) | val **3.5145** · free meter 6.5 words/line, 2% within ±1 — fluent verse, no haiku shape |
| 2026-10-17 | stage 2 fine-tune on authentic haiku, 25 ep (4.9 min) | val **3.9203** · **promoted to `checkpoints/`**, previous live → `checkpoints_tweets_live/` |
| 2026-10-17 | probes, seed leading OFF, 48 poems each | invented words 3% → **1%**, seed adherence 58% → **100%**, strict exact 100%, 0% junk in both modes |
| 2026-10-17 | bug found by the new corpus: prefix-mask could hide a whole 64-token window | poetry records reach the window size and haiku never did; masked run is now capped below one window, regression test in `tests/test_smoke.py` |
| 2026-10-17 | bug: `data/wordlist.txt` cached from the haiku corpora only | reported the poetry model's real English ("yonder", "mantling") as invented; cache now records its corpora and rebuilds (43,860 → 145,515 words) |
| 2026-10-17 | complaint: words repeat 2+ times after the poetry stage | measured **44%** of strict poems repeat a content word (previous live model 9%); fixed in the decoder, no retrain — seed counts as used, `repetition_penalty` 1.1 → **1.3**, `REPEAT_PENALTY` 2.5 on ranked candidates |
| 2026-10-17 | same fix, measured (`scripts/eval/repeat_probe.py`, 8 seeds × 4) | strict **44% → 6%**, previous model 9% → 3%, free meter 0%, exact 5-7-5 still **100%**, seed-word uses 1.06/poem |
| 2026-10-17 | fused tokens found while fixing it ("riverstreams", "duskycleannesses") | tokenizer has `Ġstream`/`stream`; a word-seam preference now rejects a letter glued after a complete word, as a preference so the meter still wins |
| 2026-10-17 | content filter: `data/banned_words.txt` + rejection sampling in `best_of_haiku` | `blood` returns crucifixion imagery because the PD corpus taught the register; no retrain — 84 poems over 14 seeds × both seed modes, **0 banned words**, 3.3 s per 3-poem request |
| 2026-10-17 | end-to-end in the browser (`backend.server` on :8000, built front end) | default model = poetry two-stage; 2 poems 1.76 s, exact 2/2, real vocabulary; UI verified |
| 2026-10-17 | poster readability report ("BLOOD" swamping the poem, haiku text too small) | reproduced at 1600 px, fixed: title capped to 0.19 of hero height, copy maxima 1.7rem→3.4rem / 1.2rem→2.3rem (body was 12 px at full width) / 1.2rem→2.7rem; `tsc -b` exit 0, rebuilt, re-verified at 1600 px and 430 px |

| 2026-10-17 | second decoder round: repeats were compared on the whole line, repeat/seam were ordered tiers, top-p left one candidate with weight, free meter had no filter | `scripts/eval/repeat_probe.py`: **0%** repeats and **1.00 seed uses/poem** in all four cells (both models, strict and free), exact still 100% |
| 2026-10-17 | seed sanitising + seam glitches | `snow!`/`snow3`/emoji no longer become the first line; floating quote, double space, `Tell'st'st'stwas` and `and / and` line scores rejected; `repeated_rate` now counts content words (a clean poem used to read 29%) |
| 2026-10-17 | API hardening + verification | negative/oversized `Content-Length` 400, non-object JSON 400, unknown model 400; 16/16 live poems clean (exact, no repeat, no seam glitch, no junk) in 14.3 s |
| 2026-10-17 | third decoder round: invented words as weight + reranker tier, free-meter ceiling, line-final apostrophe, banned-list hot reload, probe metric | invented words **1% → 0%** strict and free (`quality_probe --n 12`), free run-ons **16% → 0%** (max 23 → 10 syllables), repeats 0% in all four `repeat_probe` cells, exact/seed still 100% |
| 2026-10-17 | live verification of the round above | 24 API poems (12 strict + 12 free) flagged 0; banned-list edit took effect on the running server (6/18 → 0/18 poems say `snow`, no restart, file restored); browser: 2 poems 2.15 s, exact 2/2, repeats 0%, invented 0% |
| 2026-10-17 | front end restructured: cover → desk | wordmark `HAIKU` big + `ZERO` small in the poster's display face (`titleTail` prop, tail/title ratio measured 0.34, 132 px / 45 px at 1440), 1.9 s tween reveal (`forceProgress`) with a **Start writing** button (`cta` slot), then a sidebar holding every old console control + an ask panel for results. `tsc -b` + `vite build` exit 0; browser at 1440 and 430 px: reveal opacity 0 → 1 sampled, generate from the ask panel 2 poems in 1.84 s exact 2/2, sheet view 0% repeats / 0% invented, 0 console errors |
| 2026-10-17 | cover matched to the reference poster (word behind the branch) | `titleMaxFraction=0.42` lets `HAIKU` run under the copy; the supplied cut-out `hero-branch.png` vendored to `frontend/public/branch.png` (708×853 RGBA, 981 KB) and passed as `foregroundSrc` so the branch overlaps the letters; `scrim="strong"` keeps the copy readable over them; Japanese copy added (`五・七・五。`, `書かせながら、数える。`, `俳句 · ローカル`). Verified over CDP at 1000×620 and 1440×900: poster renders, 2 poem posters still generate (exact 2/2), sidebar + ask panel intact |
| 2026-10-17 | chat integrated: the supplied morph-orb component (`ai-thinking-orb-and-input.tsx`) replaces the ask panel as its own **write** stage | themed from black + success-green to the cover's palette (cream/blush/amber on umber), orb light gradient white → cream → rose, answer card 320×108 → **440×148 cream card** so a three-line haiku is not cropped, a `mask-image` with an invalid `var()` position removed, copy is the app's. `tsc -b` + `vite build` exit 0, `oxlint` 0 errors (2 pre-existing warnings in the poster component). Browser: pill → flight → dotted orb → card measured end to end at 1440×900 and 390×844, card 440×148 / 358×148, `Escape` returns to the pill and clears the input, `prefers-reduced-motion` reaches the card through cross-fades only, the **the poster →** button lands on the desk with 2 posters already generated, closed drawer is `inert`, 0 console + 0 page errors |
| 2026-10-17 | cover: copy too large over the photo, white frame around the card | two opt-in props instead of changing the shared styles: `fullBleed` (no frame padding / rounding / shadow — the scene reaches all four edges) and `copySize="compact"` (headline 45 → 29.5 px, body 33 → 19.4 px, subheadline 39 → 22.4 px at 1440), plus a 4-stop scrim and a text-shadow so the smaller type holds over the flowers, and the social line hidden under `sm` where it overlapped the footer. Poem posters untouched and measured to prove it: article `border-radius: 12px`, inset at x=356, `h2` 33.5 px / body 24.1 px. `tsc -b` + `vite build` exit 0, `oxlint` 0 errors; 1440×900 and 390×844 screenshots, article 1440×900 / 390 wide, 0 page + 0 console errors |
| 2026-10-17 | AI batches arrive in `haikus from ai/` (7 files: jsonl, txt, csv shapes) | 4,690 poems parsed; vetted by `scripts/data/ai_haiku_vetter.py` into **tier A 56 / tier B 245**; the batches recycle their own lines hard (batch 01 built 1,000 poems from a small pool — "a photo turns pale" appears 66 times) and most are not 5-7-5 (batch 04: 1 exact of 1,000) |
| 2026-10-17 | vetter v1 learned the wrong thing (fitted on authentic-vs-AI) | scored self-help clichés at 97/100. Rebuilt: soundness fitted on real poems vs **corruptions of those poems** (lines swapped, word replaced/deleted, stutter, shuffle) — held-out **AUC 0.787** — plus a cliché penalty and a batch-repetition penalty, output as a percentile against our authored 180. Cliché poem now 4, grammar error 1, our own leaf poem 97 |
| 2026-10-17 | mix builder learned the AI tier (`--ai-tier a\|b\|ab`, `--ai-repeat`) | `data/haikus_mix.txt` = 40,924 authentic + 180 authored ×6 + 301 vetted AI ×2 = **42,606 poems** (AI 1.4%) |
| 2026-10-17 | vetter had no form gate: 5 off-meter poems reached tier A ("Dissolving in morning light / Gone before we wake / Thoughts are flying birds", 7/5/4) | tier A now requires **exact 5-7-5** as well as the score cut; tier B is within ±1; anything further off is rejected as `off_meter` (2,197 rows), and the tier-B duplicate branch now increments the reject count too, so accepted + rejected = rows read. 8 files / 4,722 rows → tier A **89** (all exact, asserted by `tests/test_intake_formats.py`), tier B 289. Mix rebuilt: 40,924 + 180×6 + 378×2 = **42,760** |
| 2026-10-17 | combo fine-tune, AI tier AB ×6 (43,810 poems) | val 3.5203 on its own split (not comparable to other runs — split moved) — but `scripts/eval/synth_compare.py` then caught the cost: **5 of 60 generated lines (8%) were verbatim AI-batch lines** vs 0 for the live model. Kept as `checkpoints_combo_x6/` and rebuilt at ×2 — the ×2 model copies **0 of 75** generated lines, so that is the documented setting (RUNBOOK 2d) |
| 2026-10-17 | wrote 180 haiku by hand (`data/raw_synthetic/`, 5 themed files) | `scripts/data/synth_verify.py`: **180/180 exact 5-7-5**, 0 duplicate poems; 7 near-misses repaired against the counter |
| 2026-10-17 | quality passes on the authored set (repeated heads, fragment openers, stock lines) | 53 + 6 + 34 lines rewritten; 459 unique lines, 56 reused, 3 poems keep an intentional L1/L3 echo; still **180/180 exact** |
| 2026-10-17 | `scripts/data/build_synth_corpus.py` | 40,924 authentic + 180 × 6 authored = **42,004 poems**, synthetic share 2.6%; 30 samples in `data/synthetic_samples.md` |
| 2026-10-17 | fine-tune on the mix (`--init-from checkpoints/best.pt --lr 6e-5 --embed-dim 384 --layers 8 --ffn-dim 1536`, 12 ep) | **val 3.9203 → 3.5646** on the same split, best epoch 1, 1488 steps in 2.5 min → `checkpoints_mix/` |
| 2026-10-17 | first mix run died on a size mismatch | `Config` defaults are 256×6×1024 while the live checkpoint is 384×8×1536; the shape flags are required, not cosmetic |
| 2026-10-17 | memorisation check (`scripts/eval/synth_compare.py`, 75 generated lines) | authored copies **0**, authentic copies 1 — ×6 oversampling is not buying regurgitation |
| 2026-10-17 | defect found while verifying live: 3 poems in one API response were identical | per-poem seeds stepped by 1 while each poem draws `best_of` candidates, so poems shared 3 of 4; seeds now step by `best_of`, re-verified over HTTP (3 distinct poems) |
| 2026-10-17 | chat report: `/` between lines, only one haiku, input dead after clicking away | answers are line-per-row (blank line between poems), the card takes the whole run (2 poems by default) and its height follows the line count (76 + 25/line + 10/gap, capped 360; measured 440×236 at 1440, 358×236 at 390, no clipping). The click bug was real and not a glitch in the browser: `.mo-answer` is an `inset-0` layer rendered after the form, invisible at `--hOp: 0` but still hit-testable, so it covered the pill after the field blurred — it is now `pointer-events: none` except in `unfold`/`answered`, and clicking the pill refocuses the field. Verified with real mouse clicks (not `fill`): focus → click away → focus again → type → Enter; two haikus, 6 rows, 1 gap, `Escape`/**New haiku** back to a working pill, reduced-motion path 440×236, 0 page + 0 console errors |
| 2026-10-18 | seed placement probe (`scripts/eval/seed_placement_probe.py`, 12 words that occur in the corpus — `rain` 1,527 … `cloud` 194 — 4 poems each, best-of 5, strict) | opening the poem vs integrating the word: both **100% exact 5-7-5**; topic placement says the word **98%** at best-of 5 and **79%** at best-of 1 (so the model integrates it itself and the adherence rerank closes the rest), lands **inside** the poem 85% (line 1 53% / line 2 36% / line 3 11%), reads more fluently (−2.85 vs −3.09 mean logprob) and, raw, invents fewer words (11% vs 17%) and repeats fewer (1% vs 2%). Guaranteeing the word costs **1.0 extra draw** at best-of 5 / 1.3 at best-of 1 (48/48 and 96/96 landed), so the old "seed leading only" rule is now a preference, not a requirement. The live default (`checkpoints/`, epoch 6) measures the same way — topic **96%** says the word, 81% inside, 4% missing, both placements **100%** exact, 0% invented, logprob −3.00 vs −3.19, forcing it 48/48 on the first draw (`logs/seed_placement_live.md`) |
| 2026-10-18 | self-distillation: 900 decoder-clean pairs (`scripts/self_distill.py --stage pairs --best-of 6`, 1,325 draws → rejections `no_seed` 406, `invented` 13, `repeat` 4, `banned` 2) then 3 epochs with `--extra 3000` authentic, 6 s of GPU | raw probe (candidate weights neutralised, `word_weight 0`, best-of 1, free meter, 12 words × 20): invented words **30% → 18%** of poems (≈4σ), seed adherence 83 → 86% (inside noise), repeats 19 → 25% (marginal), meter **0% exact everywhere**. Pure distilled runs drift to **40%** (10 ep) / **54%** (3 ep) repeats, so the authentic mix is load-bearing. Under the decoder both models are identical (100% exact / 0% junk / 0% repeat / 0% invented); free topic-mode adherence 92 → 100%; 0 verbatim corpus copies. Live model unchanged, `checkpoints_selfdistill*/` kept for comparison — and 1,325 draws showed a *random* seed word lands only **69%** of the time vs 96-98% for common image words |
| 2026-10-18 | meter in the weights (`Config.meter_state` / `--meter-state`, RUNBOOK §4c): per-token syllable table plus accumulated-syllables and line-index embeddings, both derived inside `forward` from the ids, so training, strict decoding and free sampling need no call-site changes | provably inert for old checkpoints — 12 fixed-seed poems **byte-identical** after the change, `meter_state False` on load, `tests/test_smoke.py` exit 0. Five recipes fine-tuned from `checkpoints/`, measured with the filters-off probe (12 words × 20): live **0% ±1 / distance 9.70** · exact-5-7-5 corpus 15 ep **28% / 3.42** (control without the feature 24% / 3.73) · 50-epoch A/B 20% / 4.47 vs 18% / 4.80 · pairs+exact+authentic 6% / 6.85 (control 6% / 6.74) · zeroing the state embeddings after training costs 6pp. Form gains cost seed adherence in free mode (92% → 50% exact-only, → 83% with pairs); strict mode unaffected (100% exact, 0% junk/repeat/invented). Nothing promoted — the decoder still owns 5-7-5 |
| 2026-10-18 | stage-1 corpus v2 — the pretraining corpus was **budget-limited, not catalogue-limited** (defaults capped it at 42M chars / 1,200 books of a 4,081-book catalogue) | rebuilt at `--max-chars 150e6 --max-books 3000`: **1,954 contributing books (was 611)**, 1,349,665 records (3.6×), 150.1M chars, **54.35M BPE tokens (3.5×)**, 271,966 unique words (2.1×), 0 malformed records, 0 U+FFFD, `the` 5.49% (language filter intact; 36 non-English books skipped, 78,968 duplicates dropped). A verification re-run reproduced the file **byte-identical** (`cmp` clean, 168,201,929 bytes). Bug fixed on the way: a relative `--out` crashed the summary print *after* writing the corpus, so a successful run reported failure (`rel()` in `scripts/data/fetch_poetry.py`; same class as `scripts/eval/seed_placement_probe.py`, now also `scripts/data/import_external.py`). Stage 1 launched: 5 epochs × 54.35M tokens (266M tokens seen vs 231M in the v1 recipe) → `checkpoints_pretrain_v2/` |
| 2026-10-18 | stage 1 v2 finished (5 ep × 54.35M tokens, 33,170 steps, 60 min on the 4050) | val **3.4573**, best epoch 5, improved every epoch (3.7369 → 3.4659 → 3.4616 → 3.4579 → 3.4573); ~57k tok/s under load → `checkpoints_pretrain_v2/best.pt`. v1 was 24,154 steps in 45.8 min at val 3.5145 |
| 2026-10-18 | stage 2 v2 fine-tune (`--init-from checkpoints_pretrain_v2/best.pt`, same recipe as live: 25 ep, lr 1e-4, 384×8×1536) | val **3.8363** in 3,025 steps / 5.2 min (live was 3.9203). Same architecture (15.79M) and tokenizer, so the corpus is the only variable. **Promoted to `checkpoints/`**; previous live preserved as `checkpoints_v1_small/` |
| 2026-10-18 | v1 vs v2 on the same probes | strict `quality_probe` (150 samples each) **saturated on both** — 100% exact, 0% junk → useless as a discriminator. Raw probe (best-of 1, weights neutralised): invented words **34% → 24%**, seed mention 85% → **93%**, repeats 28% → 26%. Memorisation 1/180 → 2/180 lines; `seed_placement_probe` unchanged (100% exact / 100% topic / 0% missing). Free samples improved ("dust is cold and dry / A new wine lamp flares on / The night air is wet"). Verified through the API: `backend.server` loads `checkpoints` at val 3.8363, exact 5-7-5 in 0.9 s, `tests/test_smoke.py` exit 0 |
| 2026-10-18 | capacity test: 33.65M params (512×10×2048, batch 128) through the same two stages, plus a dropout-0.15 stage 2 (RUNBOOK §3c) | stage 1 val **3.4573 → 3.3713** (−0.086, same v2 split) and stage-2 val **3.8363 → 3.7759 / 3.7735**, yet the weights got **worse**: raw probe (240 poems/model) invented words 28% → 32%/29%, repeats 25% → 28%/32%, distance 9.78 → 10.29/9.99, verbatim lines 1.1% → 1.1%/2.2%. The 25-epoch run overfits (train/val gap 0.90 → 1.51; dropout 0.15 brings it to 0.54 without fixing quality). **Not promoted** — the limit is the 0.99M-token stage-2 corpus, not parameters. All probes ran on the 4050 (`device cuda`) |
| 2026-10-18 | decoder-knob sweep (`scripts/eval/inference_sweep.py`, new; RUNBOOK §4d) — the free lever, temperature 0.7-1.0, penalty 1.1-1.3, best-of 4/8, plus the same grid in free mode | every quality column is already at its floor/ceiling (**100% exact, 0% repeat, 0% nonword**) at every setting, so only fluency and cost move: temperature **0.7** logprob −3.06 vs 0.8 −3.18 (free mode −3.50 vs −3.68), `best_of 8` −2.96 but **1.77 s/poem vs 0.89**, penalty 1.2 vs 1.3 worth 0.05. No default changed — the checkpoint payload (0.8/1.3) is what the server serves and `{"temperature": 0.7}` per request already gets the gain; lowering the penalty was rejected because 1.3 is what fixed the 44% repeat bug |
| 2026-10-18 | quality gate (`scripts/eval/quality_floor.py`, new; RUNBOOK §7): the §4 promises as a regression check — 24 fixed strict poems, compared with `logs/quality_floor.json` | measured on the live model: **100% exact, 100% seed, 0% repeat, 0% nonword, 0% junk**; identical on two consecutive runs (CPU-seeded generator, so the sample set is reproducible). Exits **1** on any regression and refuses a changed checkpoint (`checkpoints_cap` → "baseline was measured on checkpoints/best.pt", exit 1) so a red gate cannot be silenced by swapping weights. `--self-test` proves the comparator fails all five ways (repeat 12%, nonword 9%, exact 90%, junk 9%, seed 80%) |
| 2026-10-18 | presentation pass: `README.md` (model table, two-stage story, 3 SVG charts, screenshots, measured-promise table), `scripts/make_charts.py`, `docs/shoot.mjs`; `setup.bat` / `run.bat` / `train.bat`; `.gitignore` rewritten | charts are generated from the CSVs (no plotting dependency) and screenshots from the running app over CDP; images 12 MB → 0.9 MB by using JPEG q88 for the photographic shots. Launchers verified: `run.bat` starts the server on :8000 (health ok, `loaded: ["checkpoints"]`, cuda), takes the "already listening" branch when a server is up, `setup.bat` exits 0 with `torch 2.14.1+cu130 | CUDA True`. `.gitignore` verified with `git check-ignore`: venv, `checkpoints*/`, corpora, `frontend/node_modules`, `frontend/dist` ignored; `logs/quality_floor.json`, `data/banned_words.txt`, README and `docs/images/` tracked. `tests/test_smoke.py` and `scripts/eval/quality_floor.py` still exit 0 after the comment-only polish |
| 2026-10-18 | wordlist hygiene (`impossible_word` in `backend/generate.py`, RUNBOOK §2e): drop entries no English word could be (line-join garbage from the tweet corpora) | 489 of 145,515 entries (0.34%) dropped — the only real word caught is `rock'n'roll`; cache header names the rule so the list rebuilt (145,515 → **145,026**, 0 impossible entries left), `tests/test_smoke.py` exit 0, `quality_probe` unchanged (strict 100% exact / 0% junk, free 92% seed / 0% nonword). Two doc corrections: `gest` is a real archaic word (28 poetry books), so the junk is the `ahkond` class (0 books). No automatic rule separates it from modern English (`zucchini` 9x vs `breuck` 28x in the haiku corpora), and a poetry-only rule would drop 5.2% of the list — so only the impossible class was filtered; measured remaining exposure ≈1 poem in 240 in both modes |

## How this file gets updated

1. Run `python scripts/status.py` — prints `OK`/`MISS` for every artifact plus the next command.
2. Paste that output into the chat (or edit this file directly): flip the matching `[ ]` to `[x]`,
   add a Log row with the real numbers, and bump the "Last updated" line.
3. Anything still `MISS` in `scripts/status.py` output must stay `[ ]` here — no marking ahead of evidence.
