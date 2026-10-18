# HAIKU ZERO — RUNBOOK

Native Windows, PyTorch + CUDA. No WSL, no admin.
Repo root: `C:\ai projects ikkatha\Haiku zero` (has spaces — quote paths).
Environment: **`pytorch_with_gpu`** — already created and verified (`torch 2.14.1+cu130`, CUDA True, RTX 4050).

## What takes how long

| Job | Time | Who runs it |
|---|---|---|
| Data cleaning + tokenizer | ~1 min | done already (re-runnable, §2) |
| `test_smoke.py` (CPU) | <1 min | done after every change |
| GPU smoke `--max-steps 50` | seconds | done once, re-run after any change |
| Full training (25 epochs) | **~10 min** | you |
| `experiments.py` screen (4 configs, 5 epochs) | **~6 min** | you |
| `experiments.py --epochs 25` (4 full runs) | **~45 min** | you, when stepping away |

## 1. Environment (already set up — only redo if you delete `pytorch_with_gpu`)

```powershell
cd "C:\ai projects ikkatha\Haiku zero"
pytorch_with_gpu\Scripts\Activate.ps1                 # or: pytorch_with_gpu\Scripts\activate.bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Rebuild from scratch if ever needed:

```powershell
& "C:\Users\kingh\AppData\Local\Programs\Python\Python311\python.exe" -m venv pytorch_with_gpu
pytorch_with_gpu\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu130   # ~2 GB, CUDA 13 build
pip install tokenizers numpy
```

## 2. Data + tokenizer

```powershell
python import_external.py            # (once, needs pyarrow) statworx/haiku -> data/raw/statworx.csv
python prepare_corpora.py            # parks raw twaiku, writes the capped sample, fills both source dirs
python -m backend.prepare_data --raw-dir data/raw           --out data/haikus_clean.txt
python -m backend.prepare_data --raw-dir data/raw_classical --out data/haikus_classical.txt `
    --meter-out data/haikus_classical_meter.txt --stats data/stats_classical.json
python -m backend.train_tokenizer   # -> tokenizer.json (vocab 4096, byte-lossless)
python -m backend.dataset           # prints split + window counts, trains nothing
```

**Sources.** `data/raw/` = full corpus (96,034 candidates → **75,920 haiku**, 1.43M tokens, 36%
of it tweets); `data/raw_classical/` = authentic only, no tweets (**40,924 haiku**, 0.79M tokens).
The untouched tweet file lives in `data/raw_twaiku/twaiku.csv` and only a 35,000-row deterministic
sample (`data/raw/twaiku_capped.csv`, seed 1337) trains — the original corpus was 78% tweets, which
is why the model sounded like a Twitter feed. `data/raw/statworx.csv` (28,642 poems kept of
49,024 rows: `bfbarry` 7,037, `haiku_data_1` 13,444, `haiku_data_2` 8,032) comes from
`statworx/haiku`; its `twaiku` rows are deliberately skipped. That dataset's card declares no
license, so it stays a local experiment — swap the import for a licensed corpus before shipping.

**Cleaning gotcha the corpus taught us:** the poems contain 4,829 literal `?` (e.g. "midnight rain?") —
real punctuation, but a *line-end signal* to a model that has to decide where lines end. Under strict
meter the model reproduced it as zero-syllable filler: "soft night?????the?of?b". `normalize()` now
strips `?`, collapses dot runs and trims edge punctuation, and generation refuses such characters
outright (`generate.has_junk`).

### 2b. More data: what actually exists (researched Oct 2026)

Current corpus is **1.43M BPE tokens** (0.79M authentic). That is the real bottleneck: invented words
("windshavel", "cicadases") did **not** improve when the model grew 5.8M -> 15.8M params, so it is a
data/vocabulary problem, not a capacity one.

**Measured form, not the card's claim.** `python ai_haiku_vetter.py --audit --dir data/external`
reads the parquet downloads directly and counts what is actually 5-7-5:

| file | rows | usable 3-line | exact 5-7-5 | within ±1 | already in our corpus |
|---|---|---|---|---|---|
| `davanstrien/Haiku_Dataset` | 144,122 | 142,002 | 45,633 (32%) | 30.7% | 27,373 |
| `statworx/haiku` | 48,996 | 48,855 | 11,722 (24%) | 22.0% | 28,892 |
| `pg_catalog.csv` (control, Gutenberg metadata) | 90,604 | 7,341 | 1 (0.0%) | 0.0% | 0 |

The control is the point: a book catalogue is obviously not haiku and the audit says so (1 exact in
7,341), so a low rate is the counter working. Inside the corpus we train on, exact 5-7-5 is
**10.1%** overall (statworx 13.4%, sballas 1.0%, img2poems 2.3%, gutenberg 1.5%, tempslibres 1.4%).
Two consequences: “haiku dataset” is a label, not a form, so no Hub download is a pure haiku corpus
and “add more haiku” is not a lever; and form has to be taught by the stage-2 fine-tune and
enforced at generation, which is what the meter filter already does.

**The genre ceiling, verified.** There is *no* large, legally-usable, literary English haiku corpus.
Every 100k-scale collection is social media; legal published haiku in English translation is on the
order of a few hundred poems (mostly still-copyrighted translators: Hass, Hamill, Hirshfield, Blyth).
So "more haiku" cannot fix the register — the vocabulary mass has to come from elsewhere.

Measured composition of what we already hold (`statworx/haiku`, 49,024 rows):
`twaiku` 19,431 · `haiku_data_1` 13,946 · `haiku_data_2` 8,371 · `bfbarry` 7,145 · misc 137.
The import keeps 28,643 of the 29,893 non-tweet rows (**96%**); 8,099 rows are duplicates *inside*
the file. There is nothing left to harvest there — that door is closed.

| Source | Size | License | Reachable? |
|---|---|---|---|
| `hjhalani30/haiku-dataset` (Kaggle) | >118,000 rows, has a `source` column | **CC BY 4.0** (declared) | **No** — Kaggle 404s anonymous downloads (tested). Needs a user token. Largest *licensed* non-tweet pool. |
| `bfbarry/haiku-dataset` (Kaggle, r/haiku) | ~11,000 | **CC 0** | Already inside `statworx` (7,145 rows) — no new data. |
| Beilenson, *Japanese Haiku* (1955) | ~220 genuine literary haiku | PD in US (non-renewal; unconfirmed in Copyright Office records) | **Yes** — Internet Archive plain text returns HTTP 200 (tested). Small but real literature. |
| Chamberlain, Basho epigram (1902) + *Japanese Poetry* (1910) | ~50 hokku | PD (pre-1929) | Yes via IA stream endpoints (the direct `/download/...djvu.txt` link 500s). |
| PoeTree (`en.zip`, Zenodo 10907309) | ~40,000 English poems, 388 MB, Gutenberg-derived | **none declared** | Yes, but prefer pulling the same poems straight from Project Gutenberg (unambiguously PD). |
| `taucris/haiku_333K` | 333,333 rows | MIT | Yes — but **synthetic** (another model's output). Teaches our model to imitate that model, not English. Skip. |
| `huanggab/reddit_haiku` | — | unknown | **Dead** — fails on the Hub (0 bytes); its content is r/haiku = `bfbarry` anyway. |
| Neiman `docmarionum1/haikurnn` | 25,886 | **none** | Skip — unlicensed, and its extra mass is `twaikugc.csv` (19.8 MB more tweets). |

**Re-checked 2026-10-17**, every dataset below queried through the Hub API and the
datasets-server rows endpoint (raw answers kept in `logs/hf_datasets_probe.json`):

| Source | What it actually is | Verdict |
|---|---|---|
| `davanstrien/Haiku_Dataset` (144,123) | CC-BY-4.0, but its `source` column is `twaiku` 111,721 + `img2poems` 13,538 + `sballas` 8,006 + `tempslibres` 5,126 + `gutenberg` 1,965 | **Nothing new** — four of the five are our own raw sources and the fifth is the tweet pile we cut. |
| `p1atdev/modern_haiku` (75.6k) | MIT, but the rows are **Japanese**: `朝霧の中に九段のともし哉` — Masaoka Shiki, with `kigo`/`season` fields. (The screenshot's `platdev/...` id does not exist; the real one is `p1atdev`.) | Useless for an English model — that is the language question answered. |
| `shigr3/haiku`, `PoetryMTEB/HaikuSeasonClassification` | CC-BY-4.0, Japanese (Basho, Buson, Shiki) with seasonal labels | Same: correct genre, wrong language. |
| `wentingzhao/haiku-analysis` (192k) | CC tag absent; the rows are QA/retrieval records about an unrelated `AHKCvJoyInterface` project — a name collision, not poems | Not haiku at all. |
| `taucris/haiku_333K` (333k) | MIT, one `haiku` column, and the rows are LLM output ("seeds crack open wide / green shoots pierce the waiting soil / life begins anew") | Skip, as decided — it teaches imitation of another model, not English. |
| `bathrobe/reddit_haiku_detox`, `bathrobe/safe-statworx-haiku` | Apache-2.0, columns identical to `statworx` (`source`, `text`, `text_phonemes`, ...) because they are re-slices of it plus r/haiku | Derived from what we hold; no new poems. |
| `davanstrien/haiku_dpo`, `haiku_prompts`, `haiku-topics`, `haiku-preferences`, `haiku-kto-raw-argilla` | Prompts ("Can you compose a haiku about the serenity of mountain peaks?"), topic words and preference scores | Not poems. |
| `dugward/english_haiku` (128), `huanggab/reddit_haiku` (61) | 128 rows transcribed with `\\` line separators and typos; the reddit one returns HTTP 500 on the rows endpoint | Too small / dead. |

Net: the Hub has no English literary haiku we do not already hold. `hjhalani30/haiku-dataset`
(Kaggle, CC BY 4.0, >118k) stays the one untapped, licensed pile, and it needs a user token —
worth assuming it overlaps `statworx`/`twaiku` until its `source` column is actually read.

**The plan this implies: two stages, not more haiku.** Stage 1 pretrain on a large public-domain
English poetry corpus (vocabulary + poetic register — this is what removes invented words); stage 2
fine-tune on the authentic haiku subset for form. `backend/train.py --init-from` already does stage 2.
Target: ~10-20M tokens of stage-1 text instead of 1.4M, all of it public domain.

### 2c. Stage-1 corpus: built and measured

```powershell
python fetch_poetry.py --catalog-only     # 4,081 English poetry books in the PG catalog
python fetch_poetry.py                    # cached downloads -> data/poetry_pretrain.txt
```

**Result: 375,681 three-line records, 15.41M BPE tokens, 120,488 unique words, from 611 books.**
That is a 10.8x vocabulary corpus over the 1.43M haiku tokens, every byte public domain. Books are
cached in `data/raw_poetry/` (205 MB, 962 files) so re-extraction costs no downloads; the output is
written to `.part` and renamed at the end, so a killed run can never leave a truncated corpus behind.

Selection is deliberately narrow: PG catalog rows that are `Type=Text`, `Language=en`, and classified
as poetry (bookshelf/subject, or LoCC PR1/PS1). NOT the PR/PS shelf fallback — that pulled 23,355 books
including *Moby-Dick*.

Extraction is filtered in three passes, each added because the sample showed the defect:

1. `verse_lines` — 10-72 chars, 2-13 words, no digits/markup/double-spaces, no stage directions, no
   long line closing on a full stop (a wrapped prose sentence).
2. `drop_prose_paragraphs` — deletes runs of 3+ consecutive 58+ char lines. Wrapped prose re-wraps at a
   fixed margin; verse does not. Cut wrapped prose from a third of the corpus to ~0.
3. `english_share` — percent of words that are `the`. Calibrated on the 1,047 stripped books: median
   6.0%, p10 4.1%, and every foreign/dialect book (Portuguese, French-Canadian, Japanese, German
   hymns) sits at or under ~1.9%. Threshold **2.5** sits in that gap; 9 books are skipped, not 1,007.

Measured on the final file: `the` 5.63% of words, 0 records containing U+FFFD, 0 records that are not
exactly 3 lines. Non-ASCII is legitimate typography (curly quotes, em dashes, `address'd`).

**v2 corpus, built 2026-10-18 — the budget was the limit, not the catalogue:**

```powershell
python fetch_poetry.py --out data/poetry_pretrain_v2.txt --max-chars 150e6 --max-books 3000
```

| | v1 (live stage 1) | v2 |
|---|---|---|
| books contributing | 611 | **1,954** |
| records (3-line) | 375,681 | **1,349,665** |
| characters | 47.1M | 150.1M |
| BPE tokens | 15.41M | **54.35M** (counted by the training loader) |
| unique words | 120,488 | 271,966 |
| `the` share | 5.63% | 5.49% |
| malformed records / U+FFFD | 0 / 0 | 0 / 0 |

The v1 run stopped on its own defaults (`--max-chars 42e6 --max-books 1200`) out of a 4,081-book
catalogue. Raising the budget gives **3.5x the data and 2.1x the vocabulary** at the same quality
checks: the `the`-share language filter still holds (36 non-English books skipped), 78,968 duplicate
blocks were deduped, and every record is exactly 3 lines with 0 U+FFFD. A verification re-run to a
second path reproduced the file **byte for byte** (168,201,929 bytes, `cmp` clean), which also
exercised a real bug fix: a relative `--out` crashed the *summary print* with `ValueError` after the
corpus was already written, so a successful run reported failure — `rel()` in `fetch_poetry.py`. The
same pattern (relative path + `relative_to`) had already bitten `seed_placement_probe.py` and the API,
so it is fixed there too (`import_external.py`, `seed_placement_probe.py`).

### 2d. AI batches: drop files in a folder, get vetted data out

New AI-written haiku arrive one batch at a time in `haikus from ai/`. Text, JSON, JSONL, CSV and
tab-separated all parse (jsonl rows need `lines` as a 3-item list, or a `haiku`/`poem`/`text`
string; txt is read as blank-line blocks, then as consecutive 3-line groups if the file has no
blank lines; `Theme 1:` style headers are skipped). Two commands:

```bash
python ai_haiku_vetter.py                 # score every poem, write the tiers + the report
python build_synth_corpus.py --repeat 6 --ai-repeat 2 --ai-tier ab
```

`ai_haiku_vetter.py` is the whole answer to "is this worthy". It scores each poem as a **percentile
against our own curated 180**: soundness (a logistic model fitted on real poems versus corruptions
of those poems, held-out AUC ~0.79) minus a cliché penalty (abstract filler such as "joy", "life",
"dream") minus a repetition penalty (lines and 3-word phrases the batch reuses).

**The score ranks; the meter gates.** Tier A needs *both* the score cut **and** exact 5-7-5 from
`backend.meter_check` — a poem that reads beautifully but is 6-5-5 is demoted to tier B, and more
than one syllable off is dropped however high it scored (2,197 rows so far). That separation
matters: scoring is what catches nonsense, meter is what makes it a haiku, and a corpus that
teaches form must not contain off-form exemplars. Every row is accounted for in the report —
accepted + rejected = rows read — so a new batch cannot silently lose poems.

- `--audit --dir <folder>` answers the question before the score: is this a haiku corpus at all?
  Prints rows, exact/±1/off-form counts, duplicates of our corpus and recombination per file, and
  writes nothing — point it at a fresh Hugging Face download (parquet included) before importing.
- `--explain "line one | line two | line three"` scores one poem and shows every component
- `--store 40` moves the cut, `--refresh-model` refits, `data/vetter_weights.json` holds the weights
- measured on the first eight batches (4,722 rows): tier A **89** (all exact 5-7-5), tier B
  **289**, 2,197 dropped for form, 1,895 below the score cut, 252 duplicates. The batches recycle
  their own lines hard (batch 01 built 1,000 poems from a small line pool: "a photo turns pale"
  appeared 66 times) and most are not 5-7-5 (batch 04: 1 exact of 1,000); the vetter keeps what
  survives both facts.
- parser shapes verified with fixtures: `csv` rows, `jsonl` with `lines` / `haiku` / `poem` /
  unknown-key-newline fields, txt with `#` comments, `Theme 1:` headers and run-together streams.
  Fixing those headers mattered — before the fix the header line sat *inside* the first poem, which
  pushed the qwen batch down to 20 tier-A poems; after it, 58.
- **`--ai-repeat 2`, measured, not guessed.** At `--ai-repeat 6` (the same AI poems × 6 in a
  43.8k corpus)
  `synth_compare.py` found **5 of 60 generated lines (8%) were verbatim AI-batch lines**, against 0
  for the live model; at ×2 the count is **0 of 75**. The ×6 checkpoint is kept as
  `checkpoints_combo_x6/` for the comparison; the recipe above is the one that does not regurgitate.

### 2e. Wordlist hygiene, measured

`data/wordlist.txt` is the decoder's "is that even a word" list (`load_wordlist`): the word rerank,
`UNATTESTED_WEIGHT`, `attested_fraction` and the `nonword` column all read it. It was every word the
three training corpora contain — which included line-join garbage from the tweet corpora
("slidesin'tohisemptysandals", "weshenjuggalslomomengreskeytemskeytudlogueri"). Two things were
measured before touching it:

- **The docs' example was wrong.** `gest` is not an OCR artifact — it is an archaic English word for a
tale in verse, present in **28** of the 1,175 cached poetry books. The real junk is the `ahkond` class
(**0** poetry books), and it comes from the social-media haiku corpora, not from the books.
- **No automatic "is this English" test exists here.** Words absent from the poetry vocabulary number
  **7,493** (5.2% of the list) and include modern English the PD corpus never used (`zucchini`,
  `gearbox`, `acne`, `advisor`). Repetition does not separate them either: in the haiku corpora
  `breuck` appears 28x and `sundance` 52x, against `zucchini` 9x and `gearbox` 1x.

So the rule is deliberately narrow — an entry is dropped only when it *cannot* be an English word
(`impossible_word`: longer than 18 characters, or 2+ apostrophes). **489 of 145,515 entries (0.34%)**
go, and the only real word caught is `rock'n'roll`. The rule is named in the cache header, so the list
rebuilds on the next load (145,515 → **145,026**, 0 impossible entries left) and `test_smoke.py`
passes; `quality_probe.py` is unchanged (0% junk / 100% exact strict, 92% seed free).

What the junk that survives actually costs, measured through the production path (best-of 4, reranker
on, 12 words × 20 poems, possessive-aware): **strict 1.7%** — and that one is the legitimate plural
"haikus" — **free 1.7%**, the same `ahkond` poem across seeds. About **1 poem in 240**, which is why
the list was not narrowed further: a poetry-only rule would have dropped 5.2% of the vocabulary,
silently redefining the `nonword` metric while deleting real words to catch a rounding error.

## 3. Training

```powershell
python -m backend.train --max-steps 50     # 1-minute GPU proof after any change
python -m backend.train                    # full run, ~5 min on the balanced corpus (~170k tok/s)
python -m backend.train --resume           # continue after a crash/interrupt
python -m backend.train --batch-size 64    # if you hit CUDA out of memory on 6 GB
python -m backend.train --dropout 0.15     # PLAN.md rule: val climbing while train falls
python -m backend.train --meter-state      # feed syllables-so-far/line index to the model (§4c)
python -m backend.train --seed-prob 0.7    # share of rows trained as '<seed> word -> poem'
python -m backend.train --prefix-mask-prob 0.3   # share trained as a continuation
python -m backend.train --meter-data       # train on the 5-7-5-ish subset instead
python -m backend.train --corpus data/haikus_classical.txt --ckpt-dir checkpoints_authentic `
    --log training_log_authentic.csv --epochs 40          # authentic-voice run
python -m backend.train --embed-dim 384 --layers 8 --ffn-dim 1536 --batch-size 64 `
    --epochs 20 --dropout 0.15 --ckpt-dir checkpoints_big --log training_log_big.csv
```

**The two-stage recipe that is now live** (~65 min total on the RTX 4050: 60 + 5.2). Stage 1 gives the model
English and a poetic register; stage 2 gives it haiku form. Same architecture, same tokenizer, same
3-line + `<eos>` format in both corpora, so the only new things are the corpus and `--init-from`:

```powershell
# stage 1 - vocabulary: 54.35M tokens of public-domain poetry, 60 min, best val 3.4573
python -m backend.train --corpus data/poetry_pretrain_v2.txt --epochs 5 --run-name poetry-pretrain-v2 `
    --ckpt-dir checkpoints_pretrain_v2 --log training_log_pretrain_v2.csv `
    --embed-dim 384 --layers 8 --ffn-dim 1536
# stage 2 - form: fine-tune on the authentic haiku corpus, 5.2 min, best val 3.8363
python -m backend.train --corpus data/haikus_classical.txt --epochs 25 --lr 1e-4 `
    --init-from checkpoints_pretrain_v2/best.pt --run-name poetry-then-haiku-v2 `
    --ckpt-dir checkpoints_v2 --log training_log_poetry_v2.csv `
    --embed-dim 384 --layers 8 --ffn-dim 1536
```

Stage 1 runs **5 epochs over 3.5x the tokens** (33,170 steps, ~60 min on the laptop 4050) instead of
15 epochs over the v1 corpus (24,154 steps, 45.8 min). That keeps the same total token budget — the
point of the v2 corpus is distinct vocabulary per step, not more gradient steps over the same words.

**v1 numbers, superseded but still selectable:** stage 1 on `data/poetry_pretrain.txt` (15.41M tokens,
15 epochs, 45.8 min, val 3.5145, `checkpoints_pretrain`); stage 2 from it (4.9 min, val 3.9203). That
pair is what the live `checkpoints/best.pt` was until 2026-10-18 and is preserved as
`checkpoints_v1_small/` so the comparison stays reproducible.

`checkpoints_pretrain/best.pt` is kept (selectable in the UI) because stage 1 alone is the diagnostic
that says whether a problem is vocabulary or form: it writes fluent long-lined verse and cannot do 5-7-5
(0% exact, 6.5 words/line). Cross-run val loss is **not** comparable across different corpora — use
`quality_probe.py`, which is why it exists.

Model size is now a flag (`--embed-dim/--layers/--ffn-dim/--seq-length`), so capacity experiments do
not need a config edit: 15.79M params fits in 6 GB at batch 64.

### 3b. The v2 promotion, measured

`checkpoints/best.pt` now holds the `poetry-then-haiku-v2` run; the previous live model is
`checkpoints_v1_small/`. Same architecture (15.79M params) and the same 4,096-token tokenizer in both,
so the corpus is the only variable. What the probes say:

| probe | v1 live | v2 live | reading |
|---|---|---|---|
| val loss (stage 2) | 3.9203 | **3.8363** | −0.084 on the same stage-2 corpus and split |
| `quality_probe` strict, best-of 4, 150 samples per model | 100% exact, 0% junk | 100% exact, 0% junk | **saturated** — says nothing about which model is better |
| `self_distill --stage probe` (raw, best-of 1, weights neutralised) | invent **34%** | invent **24%** | the decoder is not hiding it: this is the language model itself |
| same run, seed mention | 85% | **93%** | |
| same run, content-word repeat | 28% | **26%** | |
| verbatim lines from `haikus_clean.txt` (60 strict poems each) | 1/180 | 2/180 | memorisation stays negligible at 3.5x data |
| `seed_placement_probe` | 100% exact / 100% topic / 0% missing | 100% exact / 100% topic / 0% missing | no placement regression |

The honest summary: the strict decoder already saturates on both models, so the user-visible change is
small. The gain is in the weights — a third fewer invented words and better seed uptake when decoding
is left to the model. Free-mode samples read more like English than v1's ("dust is cold and dry / A new
wine lamp flares on / The night air is wet" vs "dust gurgles on the / field And fishing from a nook /
sat in a tank of"). Verified through the production path: `python -m backend.server --port 8123` loads
`checkpoints` at val 3.8363 and `/api/generate` returns an exact 5-7-5 poem in 0.9 s; `test_smoke.py`
passes with the promoted weights, and the old model stays loadable from `checkpoints_v1_small/`.

Every train run writes `best.pt` (best val — what generation loads), `last.pt` (resume) and its
`--log` CSV into `--ckpt-dir`, and prints one sampled haiku per epoch over `autumn moonlight`.
Targets: train loss < 3.5, val < 4.5, previews become readable 3-line poems.

### 3c. Capacity: measured, **not** the lever yet

With the corpus no longer the constraint, capacity was the obvious next hypothesis. Tested 2026-10-18:
**33.65M params** (512 × 10 × 2048, batch 128) through the identical two stages, then a regularised
stage 2 because the first one overfit.

```powershell
# stage 1 - 104 min, same 33,170 steps and token budget as the 15.8M run
python -m backend.train --corpus data/poetry_pretrain_v2.txt --epochs 5 --run-name poetry-pretrain-cap `
    --ckpt-dir checkpoints_pretrain_cap --log training_log_pretrain_cap.csv `
    --embed-dim 512 --layers 10 --ffn-dim 2048 --batch-size 128
# stage 2a - the live recipe, 9.3 min: overfits (best val 3.7759 at epoch 5, then climbs to 3.99)
python -m backend.train --corpus data/haikus_classical.txt --epochs 25 --lr 1e-4 `
    --init-from checkpoints_pretrain_cap/best.pt --run-name poetry-then-haiku-cap `
    --ckpt-dir checkpoints_cap --log training_log_cap.csv `
    --embed-dim 512 --layers 10 --ffn-dim 2048 --batch-size 128
# stage 2b - PLAN.md's overfit rule, 4.4 min: dropout 0.15, 12 epochs
python -m backend.train --corpus data/haikus_classical.txt --epochs 12 --lr 1e-4 --dropout 0.15 `
    --init-from checkpoints_pretrain_cap/best.pt --run-name poetry-then-haiku-cap-d15 `
    --ckpt-dir checkpoints_cap_d15 --log training_log_cap_d15.csv `
    --embed-dim 512 --layers 10 --ffn-dim 2048 --batch-size 128
```

| | live 15.8M | 33.65M | 33.65M + dropout 0.15 |
|---|---|---|---|
| stage-1 val (same v2 split) | 3.4573 | **3.3713** | — |
| stage-2 best val (same split) | 3.8363 | **3.7759** | 3.7735 |
| stage-2 train−val gap at final epoch | 0.90 | **1.51** | 0.54 |
| raw probe: invented words | **28%** | 32% | 29% |
| raw probe: seed mentioned | 88% | 84% | **91%** |
| raw probe: content repeat | **25%** | 28% | 32% |
| raw probe: distance to a real haiku | **9.78** | 10.29 | 9.99 |
| strict `quality_probe` | 100% exact, 0% junk | identical | identical |
| verbatim corpus lines | 1.1% | 1.1% | 2.2% |

Raw probe = `self_distill.py --stage probe --n 20` (240 poems per model, filters off, GPU).

**Reading:** capacity clearly buys pretrain loss (−0.086 on the same split) and stage-2 val too, but
**nothing improves in the samples**: repeats rise, output sits further from real haiku, and at 25
epochs it memorises (train gaps to 2.48 while val sits at 3.99). Dropout fixes the gap, not the quality
— 3.24/3.78, still 32% repeats. The limiting factor is the **0.99M-token stage-2 corpus**, which cannot
support 33.65M parameters; a bigger model just fits the haiku set harder. Nothing promoted;
`checkpoints_cap/` and `checkpoints_cap_d15/` stay selectable. Retest capacity only after the
authored-poem set grows.

## 4. Generation

Available models (each ships with its own `tokenizer.json`, so old checkpoints keep the tokenizer
they were trained with). Numbers are `python quality_probe.py --n 8` — 32 strict poems per model,
best-of 4, seed leading:

| path | what it is | junk | `?` | strict exact | seed hits | invented words |
|---|---|---|---|---|---|---|
| `checkpoints/best.pt` **(live)** | poetry pretrain v2 → haiku fine-tune, **15.8M params** (val 3.836, own split) | 0% | 0% | 100% | **100%** | **0%** |
| `checkpoints_v1_small/best.pt` | previous live: pretrain v1 (15.4M tokens) → haiku fine-tune (val 3.920) | 0% | 0% | 100% | 100% | 0% |
| `checkpoints_pretrain_v2/best.pt` | stage 1 only — 54.35M tokens of PD poetry, 5 ep | 0% | 0% | 100% | 100% | 1% |
| `checkpoints_pretrain/best.pt` | stage 1 only, v1 corpus — 15.41M tokens, 15 ep | 0% | 0% | 100% | 100% | 1% |
| `checkpoints_tweets_live/best.pt` | previous live: haiku only, 15.8M, 20 ep (val 4.079) | 0% | 0% | 100% | 58% | 3% |
| `checkpoints_authentic/best.pt` | authentic-only corpus, 5.8M, 40 ep | 0% | 0% | 100% | 100% | 4% |
| `checkpoints_balanced/best.pt` | balanced corpus, 5.8M, 25 ep | 0% | 0% | 100% | 100% | 5% |
| `checkpoints_v3_live/best.pt` | the old live model — 78% tweets | 0% | 0% | 100% | 100% | 4% |
| `checkpoints_classical/best.pt` | old 30.6k fine-tune (superseded) | 0% | 0% | 100% | 100% | 3% |
| `checkpoints_v3_clean/best.pt` | clean-corpus run, kept for comparison | 0% | 0% | 100% | 100% | 4% |

Measured with `--n 12 --seeds "autumn moonlight,breeze,old pond,"` (48 poems), **seed leading turned
off** for the seed column — that is the only setting where the number tests anything, since with seed
leading on the word is there by construction. This is the run that decided the promotion: the poetry
model obeys a typed seed **100%** of the time versus **58%** for the previous live model, and invents
**1%** of its words instead of 3%.

Structural quality is now mostly the decoder's doing (junk filter, exact meter, seed leading), so under
strict metering every checkpoint scores 100% exact — the table differentiates on *voice* and on the two
columns that still come from the weights: seed adherence and invented words. Val loss is **not**
comparable across runs (different splits and different corpora); `quality_probe.py` is the comparison
that is. The live row was re-measured after the third decoder round (below): `--n 12`, the same
`--seeds` string, **junk 0% / `?` 0% / exact 100% / seed 100% / repeats 0% / invented words 0%** in
the strict column and 0% invented in the free column. The other rows are the older run and predate
that round.

**Strict vs free is a real trade, pick per use.** Free meter is the honest picture of the weights
(no meter decoder and no best-of), measured as syllable patterns over 48 samples:

| model | mode | exact 5-7-5 | within ±1 syllable | words/line | invented words |
|---|---|---|---|---|---|
| `checkpoints/` (live) | free | 0% | 23% | 3.5 | 0% |
| `checkpoints_tweets_live/` | free | 2% | 27% | 3.4 | 2% |
| `checkpoints_pretrain/` | free | 0% | 2% | 6.5 | 1% |
| any | `--strict` | 100% | — | — | 0-3% |

Free meter is bounded as of the third decoder round: a line may close anywhere from 0 to **10**
syllables, but it may not run on (`FREE_MAX_SYLLABLES`, with a ×2.5 newline preference once a line
has `FREE_MIN_SYLLABLES` = 4). Same 8 seeds × 4 poems, live model, cap on vs off: max syllables/line
**23 → 10**, lines over 10 syllables **16% → 0%**, words/line 5.4 → 4.3. The rows above were measured
under the old unbounded free mode.

The live model kept the haiku shape through stage 2 (3.5 vs 3.4 words/line, 23% vs 27% within a
syllable) **and** gained the vocabulary — stage 1 alone is the control that proves the shape comes from
the fine-tune, not from pretraining (6.5 words/line, 2% within ±1).

Both modes ban junk characters and both lead with the seed word; they differ only in whether 5-7-5 is
enforced.

Two decoder changes this round, both measured:

- **`has_junk` / `has_stutter` / `junk_token_ids`**: a candidate line with punctuation, digits or a
  repeated word is never accepted, and plain sampling masks junk *tokens* outright (the static mask is
  built once per tokenizer). Before this, 56% of strict samples from the authentic fine-tune contained
  `?` filler ("soft night?????the?of?b") because a zero-syllable token was always "allowed"; now 0%.
  Measured on `checkpoints/`: strict 0% junk, free meter 5% → 0%.
- **seed leading** (`--no-seed-first` to turn off): the seed word is written as the poem's first word
  and the model continues it, in **both** modes (before this, free meter silently ignored the switch:
  measured 43% vs 100%). Reranking alone could not do it — a typed seed reached the poem in **0/10**
  samples before (`autumn`), 100% after. The word is in the poem by construction, which the UI labels.
- **seed placement measured: opening the poem vs integrating the word** (`seed_placement_probe.py`, 12
  words that occur in the corpus — `rain` 1,527 … `cloud` 194 — 4 poems each, best-of 5, strict, same
  draws for both placements):

  | placement | exact 5-7-5 | says the word | opens with it | word inside | missing | invented | mean logprob |
  |---|---|---|---|---|---|---|---|
  | `seed_first` (leading) | 100% | 100% (by construction) | 100% | 0% | 0% | 0% | −3.09 |
  | topic (`--no-seed-first`) | 100% | **98%** | 12% | **85%** | 2% | 0% | **−2.85** |

  The 0/10 above is what seed leading was built on; with the **adherence bonus** in the reranker the
  topic placement now lands the word by itself — **79%** of poems at `best-of 1` (no rerank selection,
  96 poems; worst word `snow` at 38%) and **98%** at best-of 5. Where it lands once it is inside: first
  line 53%, second 36%, third 11%. Two more differences favour topic placement: it reads more fluently
  (−2.85 vs −3.09 mean token logprob) and raw (`best-of 1`) it invents fewer words (**11%** vs 17% of
  poems) and repeats fewer (1% vs 2%) — continuing from a bare noun is the harder job. What seed
  leading still buys is the **guarantee**: topic mode missed the word in 2% of reranked poems (21%
  raw). When the word must appear, the fix is draws, not placement: looping best-of draws until the
  poem says the word *and* holds 5-7-5 landed **48/48 on the first draw** at best-of 5 and **96/96 at
  1.3 draws** at best-of 1, so the guarantee costs about one extra poem. The **live default**
  (`checkpoints/`, epoch 6) measures nearly the same — topic **96%** says the word, 81% inside, 4%
  missing, **100%** exact in both placements, 0% invented, logprob −3.00 vs −3.19, forcing it 48/48 on
  the first draw (`logs/seed_placement_live.md`) — so the recommendation does not depend on which
  checkpoint is selected.
- **word rerank** (`WORD_WEIGHT = 1.2`): candidates made of words the training corpora actually contain
  outrank ones that invented them (inflections allowed, `data/wordlist.txt` is built on first use).
  Measured on `checkpoints_balanced`: invented words 9% → 5% of words per poem, with visibly better picks.
  The list must include **every** corpus the model trained on (`WORDLIST_CORPORA`): after the poetry
  stage it was still built from the two haiku corpora, which reported the model's newly learned real
  English ("yonder", "mantling") as invented — 3% instead of the true 1%. The cache now stores the
  corpus list that produced it and rebuilds on any change (43,860 → 145,515 words); a stale list fails
  silently and in the wrong direction, so `test_smoke.py` covers it.
- **banned words** (`data/banned_words.txt`): a candidate that uses one is dropped instead of ranked and
  the sampler draws again (up to 4n) until a clean one exists. Prompted by `blood` returning crucifixion
  imagery — a 15.8M model trained on public-domain verse learned the Christian register of that corpus,
  and no reranker argues with a weight. Removing it from the weights would mean another pretrain, so it
  is a sampling-time filter: free when the file is empty, reversible by deleting a line, and it never
  touches a checkpoint. Whole words only (banning `cross` leaves `crossing`), and the seed you typed is
  exempt — seed `cross` and the poem may say `cross`. Measured after adding it: **84 poems, 14 seeds ×
  both seed modes, 0 banned words**, mean 3.3 s per 3-poem request on the CUDA box (retries add up to
  ~1 s when the model insists on the word, e.g. `temple` at 7.8 s once). The list hot-reloads: it is
  cached by the file's `(mtime, size)`, so editing `data/banned_words.txt` applies to the **running**
  server on the next request - verified live, 6/18 poems said `snow` before the edit and 0/18 after,
  with no restart. If every draw says the word, the best dirty poem is returned rather than an empty
  panel.

- **repetition gate** (`REPEAT_PENALTY`, `DEFAULT_REPETITION_PENALTY`): the poetry stage taught the
  model that a verse line may say a word twice — "blood and blood are all" — and it showed up in
  **44%** of strict poems (vs 9% before the poetry model). Three cheap changes, no retrain:
  the seed word now counts as *used*, so it cannot be repeated for free; `repetition_penalty`
  default went **1.1 → 1.3** (the server takes `max(checkpoint, 1.3)`, because old payloads store
  1.1); and a candidate that spends a content word twice takes a `REPEAT_PENALTY = 2.5` score hit
  rather than being disqualified — disqualifying it surfaced malformed candidates and made things
  worse. Measured with `repeat_probe.py` (8 seeds × 4 poems, stems so `snow/snow's/snows` count once):
  strict **44% → 6%**, previous live model **9% → 3%**, free meter **0%**, exact 5-7-5 still **100%**.
- **token-level filtering, second round**: 6% was not zero, and the free meter had no filter at all,
  so the ranking-only version still let a poem say the seed twice. Four decoder bugs, all measured:
  - *Repeats were compared on the whole candidate line*, so every continuation of a line that had
    already used a content word looked like a repeat. All candidate weights hit zero, the sampler
    fell back to "no alternative", and the repeat it was supposed to filter came back. Now only the
    candidate's NEW text is checked (`adds_repeat_now`), and stems are compared, so `blood's` counts
    as `blood`.
  - *Repeat and seam were ranked as tiers*, so a clean repeat lost to a fused word (or the reverse)
    purely by the order the tiers were consulted. They are multipliers now (`REPEAT_WEIGHT = 0.0`,
    `SEAM_WEIGHT = 0.0`): a flawed candidate is only sampled when every candidate is flawed.
  - *Top-p at 0.8 often leaves one candidate* (p=1.0, tail 0.0). Zeroing that one left nothing with
    weight, so the fallback handed the draw straight back. `tail_weighted()` keeps 2% of each rank's
    true probability, so the filter has somewhere to go while the nucleus token still dominates.
  - *Free meter had no token-level filter*, so `blood-red wire ... heavy my blood` was possible no
    matter what the reranker preferred. It now uses the same candidate filter (no meter rule).
  Measured with `repeat_probe.py` (8 seeds × 4 poems, both models, both modes): **0% repeats** and
  **exactly 1.00 seed-word uses per poem** in all four cells, exact 5-7-5 still 100%. `quality_probe`
  on the live model at that point: junk 0%, `?` 0%, invented words 1% strict / 0% free — the last
  number is the one the third round below takes to 0%.
- **seed cleaning** (`clean_seed`): priming used to paste the raw prompt into the poem, so `snow!`
  opened a poem with "snow!" and an emoji seed opened it with an emoji — junk the candidate filter
  never saw, because primed text never went through it. Prompts are now lowercased letters and inner
  apostrophes only, and a paragraph-length prompt is context, not a first line. The server and the
  poster UI report the cleaned prompt.
- **small seam glitches**: a quote mark floating between words (`time inde ’`), double spaces
  (`phone  haiku`), a fragment repeated inside one word (`Tell'st'st'stwas`) and a line opening with
  the word the previous line ended on (`... the tree and / and the wound`) are all rejected now.
  Possessive plurals (`the birds' song`) and contractions (`it's`) still pass.
- **`repeated_rate` counts content words** (`repeated_content_rate`), not every word: the old metric
  charged grammar ("the ... the ... and") and reported a perfectly clean poem as **29% repeats** in
  the UI, which made the decoder fix look like it had done nothing. `quality_probe.py` uses the same
  content-word metric now (it kept the all-word one for a while, which is why its column disagreed
  with the API).
- **word seam** (`keeps_word_boundary`): the tokenizer has both `Ġstream` (opens a word) and `stream`
  (continues one), and the model sometimes picked the continuation where a word should start, giving
  "riverstreams of rain", "duskycleannesses". A letter right after a space, or right after a word the
  corpus already completes ("river"), now loses to a candidate that keeps the seam — as a preference,
  so the meter still wins. The repeat fix needed this: the poems it preferred were the fused ones.

**Third decoder round** — the leftovers from the round above, all measured on the live model:

- **invented words are a weight now, not just a score** (`UNATTESTED_WEIGHT = 0.0`,
  `adds_invented_word`): the reranker could only rank a poem with "shavement" lower, so the poem
  still shipped with it (1% of words). A candidate that *finishes* a word the corpora never
  contained is dead weight while anything else fits — exactly like a repeat. Only finished words
  count, so "moon" + "light" is not read as the word "light" (`added_words`), and a newline counts
  as the boundary, which catches the invented word a line **ends** on ("square for my mouth ons").
- **full-attestation tier in the reranker** (`UNATTESTED_PENALTY = 2.0`): a poem made only of words
  the corpora contain now outranks one with a single invention, regardless of its logprob. The
  weight was previously a 1.2 × rate bonus, which a marginally better logprob could beat.
  Measured: invented words **1% → 0%** strict and free (`quality_probe.py --n 12`), junk 0%,
  `?` 0%, exact 100%, repeats 0%.
- **free meter is bounded** (`FREE_MAX_SYLLABLES`, `FREE_MIN_SYLLABLES`, `FREE_NEWLINE_BOOST`): free
  meant "don't enforce 5-7-5", not "never stop". A line now closes at the ceiling even when the
  model would rather keep writing. Max syllables/line 23 → 10, run-on lines 16% → 0% (same seeds,
  cap on vs off) with no change to strict mode.
- **a word-final apostrophe is junk**: `... the string's boughs'` ended a strict line — the same
  token-seam leftover as a floating quote, just glued to a word. `has_junk` rejects it, and
  `known_word` strips edge apostrophes so `boughs'` is not also counted as an invented word.
- **`data/banned_words.txt` hot-reloads** (mtime + size cache) — see the bullet above; the old
  "restart the server" step is gone.
- **`quality_probe.py` reports the content-word repeat metric**, the same number the API and UI show.

### 4b. Can the decoder's job move into the weights? Measured (`self_distill.py`)

Every quality rule in §4 lives in the decoder, so the obvious next question is whether training can
absorb them. Tested: draw poems the way the live pipeline does (`--strict`, reranker on, seed as
context), keep only those passing every rule (exact 5-7-5, seed present, no repeated content word,
every word attested, no junk, no banned word), then fine-tune on them in the training seed format
(`[<bos> <seed> word \n poem <eos>]`, prefix masked — `dataset.encode_records`' shape).

```powershell
python self_distill.py --stage pairs --pairs 900 --best-of 6    # ~25 min, 1,325 draws -> 900 pairs
python self_distill.py --stage train --epochs 3 --lr 5e-5 --mix data/haikus_classical.txt:3000
python self_distill.py --stage probe --models checkpoints,checkpoints_selfdistill_mix --n 20
```

1,325 draws → **900 clean pairs**; the rejections are the interesting part: `no_seed` **406**,
`invented` 13, `repeat` 4, `banned` 2. A *random* corpus word comes back in the poem only ~69% of
the time, against 96-98% for the common image words in `seed_placement_probe.py` — adherence is a
property of the word, not just of the reranker.

`--stage probe` is the honest test (PROGRESS.md limitation #1): `REPEAT_WEIGHT`, `SEAM_WEIGHT` and
`UNATTESTED_WEIGHT` at 1.0, `word_weight 0`, `SEED_BONUS 0`, `--best-of 1`, free meter — nothing is
ranked, chosen or rejected for quality, so what is left is the weights. 12 words × 20 poems each:

| model | exact | ±1 | repeats | invented | seed |
|---|---|---|---|---|---|
| `checkpoints` (live) | 0% | 0% | 19% | **30%** | 83% |
| `checkpoints_selfdistill_mix` (900 distilled + 3,000 authentic, 3 epochs) | 0% | 0% | 25% | **18%** | 86% |

Read it as: **vocabulary does move** (invented words 30% → 18% of raw poems, ≈4σ at n=240), **adherence
does not** (83 → 86%, inside noise), **meter does not at all** (0% exact in every variant — form is
still the decoder's), and repeating a content word gets no better (19 → 25%, marginal). A pure run on
the distilled pairs alone drifts hard on repeats — **40%** at 10 epochs and **54%** at 3 (n=72),
against 20% for the live model — and `--mix <corpus>:<count>` is the fix: the distilled set is
low-entropy, real poems in the same shape pull it back to 25% while keeping the vocabulary gain. Under the decoder (`quality_probe.py`, best-of 4, strict) the two models
are indistinguishable — 100% exact, 0% junk, 0% repeat, 0% invented — and in *free* topic mode the
distilled model's adherence is 100% against 92%. No verbatim corpus copying either (`synth_compare.py`:
authored 0, AI 0, authentic 0).

Verdict: a partial win, not a replacement. Keep the live model; `checkpoints_selfdistill/` (10 epochs)
and `checkpoints_selfdistill_mix/` (the recipe above) stay for comparison. The binding constraint is the
pair set, not compute — training the whole thing takes 6 seconds, so the lever is more pairs (and pairs
whose seeds are hard words, where the decoder currently fails 30% of the time).

### 4c. Meter in the weights: the state inputs, measured

§4 keeps form in the decoder because the model cannot count. This phase gave it a way to: two small
embeddings feed the model **syllables accumulated in the current line** and **which line this is**
(`Config.meter_state`, `--meter-state`). Both are derived inside `forward` from the ids themselves —
a per-token syllable table built from the tokenizer plus the reset tokens (`\n`, `<eos>`, `<bos>`), so
training, strict decoding and free sampling all get the feature with **no call-site changes**.

```powershell
python -m backend.train --corpus data/haikus_exact.txt --init-from checkpoints/best.pt `
    --epochs 15 --lr 8e-5 --meter-state --seed-prob 0.7 `
    --embed-dim 384 --layers 8 --ffn-dim 1536 --batch-size 64 `
    --ckpt-dir checkpoints_meter --log training_log_meter.csv --run-name meter-state
python self_distill.py --stage probe --models checkpoints,checkpoints_meter --n 20      # filters off
python self_distill.py --stage probe --models checkpoints_meter --ablate-meter --n 20   # state zeroed

# the balanced recipe: distilled pairs (own seeds) + exact form + authentic voice
python self_distill.py --stage train --epochs 8 --lr 4e-5 --meter-state `
    --mix data/haikus_exact.txt:5930 --mix data/haikus_classical.txt:6000 `
    --ckpt-dir checkpoints_meter_pairs --log training_log_meter_pairs.csv
```

`data/haikus_exact.txt` is the 5,930 exact-5-7-5 poems inside the mix (13.9%; a further 3,442 are
within ±1). Everything below is fine-tuned from `checkpoints/` at 384×8×1536, and "raw" means the
`self_distill.py --stage probe` numbers: candidate weights neutralised, `word_weight 0`, `best-of 1`,
free meter with the 10-syllable ceiling and the newline boost switched off, 12 words × 20 poems.

| recipe | raw ±1 | raw mean syllabic distance | raw seed | production free exact | production free seed |
|---|---|---|---|---|---|
| live `checkpoints/`, no fine-tune | 0% | **9.70** | 85% | 0% | 92% |
| exact corpus, 15 ep, **state ON** | **28%** | **3.42** | 72% | 6% | 50% |
| exact corpus, 15 ep, state OFF (control) | 24% | 3.73 | 73% | — | — |
| exact corpus, 50 ep, state ON | 20% | 4.47 | 65% | — | — |
| exact corpus, 50 ep, state OFF (control) | 18% | 4.80 | 62% | — | — |
| pairs + exact + authentic, 8 ep, state ON | 6% | 6.85 | 82% | 4% | 83% |
| pairs + exact + authentic, 8 ep, state OFF | 6% | 6.74 | 88% | — | — |

What the table says:

- **The data is the lever, not the architecture.** Fine-tuning on exact-5-7-5 poems takes the raw
  mean syllabic distance from 9.70 to 3.4-4.8 and ±1 from 0% to 18-28% with no feature at all. Line
  lengths are learnable from the line-break distribution; the model never needed to be told how to count.
- **The feature is real but small.** Across three matched comparisons it wins by 2-6pp of ±1 and
  0.3-0.5 syllables (15 ep: 24% → 28%; 50 ep: 18% → 20%), and zeroing its two embeddings *after*
  training costs 6pp (28% → 22%) — so the model does consult them. It is not the 20pp that data buys.
- **Form and instruction-following compete at this size.** Every form gain costs seed adherence in
  free mode: 92% → 50% for the exact-corpus run, 92% → 83% when the distilled pairs are in the mix
  (pairs carry their own seed and are adherence-positive). Strict mode is unaffected — 100% exact,
  0% junk, 0% repeat, 0% invented in all of them.
- **Nothing is promoted.** No variant beats the live model overall: each trades a form number for an
  adherence number, and the decoder still guarantees 5-7-5 for free. `checkpoints_meter*` are kept
  for these numbers. The form-first model also *reads* worse — with `checkpoints_meter/best.pt` loaded
  through `generate.load_model` (the function the server uses) a strict seed `sea` gives
  "sea lam jar gem goes / Out of reach of the ocean / The sky is quieter": real words, correct 5-7-5,
  and almost no sense. Meter was never the binding constraint on quality.

**The feature is provably inert for existing checkpoints.** `meter_state` defaults to False, so old
payloads build the old architecture; both embeddings are additive and the branch is skipped. Verified
by regenerating 12 fixed-seed poems from `checkpoints/` in both modes and diffing against a snapshot
taken before the code change: **byte-identical**, with `meter_state False` reported on load
(`logs/baseline_before_meter.txt`). `test_smoke.py` passes, and the strict probes are unchanged.

```powershell
python repeat_probe.py                     # repeat / exact / seed-use rate, live vs previous model
python repeat_probe.py --free              # same, free meter
python -m backend.generate --prompt "autumn moonlight" --count 3
python -m backend.generate --prompt "snow" --strict     # always 3 lines, 5-7-5 by construction
python -m backend.generate                               # interactive REPL
python -m backend.generate --temperature 0.9 --top-p 0.95
python -m backend.generate --checkpoint checkpoints_v3_clean/best.pt --prompt "snow"
python -m backend.generate --prompt "old pond" --strict --best-of 8
```

`--best-of N` (default 4) generates N candidates and keeps the best one — mean token log-likelihood,
minus a repeated-word penalty, plus the real-word and seed-adherence bonuses, and in `--strict` mode
non-exact candidates rank last. Measured: strict exactness 8/10 → 10/10 and mean logprob −2.68 → −2.27
on the old live model (best-of 16 only reaches −2.17, so 4 is the cheap sweet spot). Raise it to 8-16
if you want more of the word-realism rerank to bite (it costs one forward pass per candidate).

### 4d. Decoder knobs, swept (`inference_sweep.py`)

```powershell
python inference_sweep.py                          # temperature 0.7-1.0, penalty 1.1-1.3, best-of 4/8
python inference_sweep.py --models checkpoints_cap --n 8
```

Live model, 6 seeds × 8 strict poems per row, GPU:

| temperature | penalty | best_of | exact | repeat | nonword | logprob | s/poem |
|---|---|---|---|---|---|---|---|
| **0.70** | 1.30 | 4 | 100% | 0% | 0% | **−3.06** | 0.93 |
| 0.80 (default) | 1.30 | 4 | 100% | 0% | 0% | −3.18 | 0.98 |
| 0.90 | 1.30 | 4 | 100% | 0% | 0% | −3.39 | 0.85 |
| 1.00 | 1.30 | 4 | 100% | 0% | 0% | −3.59 | 0.79 |
| 0.70 | 1.10 / 1.20 | 4 | 100% | 0% | 0% | −3.04 / −3.05 | 0.88 |
| 0.70 | 1.30 | **8** | 100% | 0% | 0% | **−2.96** | **1.77** |

Free mode (no best-of — where the repetition penalty actually governs), 6 seeds × 12:

| temperature | penalty | repeat | nonword | distinct | logprob |
|---|---|---|---|---|---|
| **0.70** | **1.20** | 0% | 0% | 100% | **−3.50** |
| 0.70 | 1.30 | 0% | 0% | 100% | −3.56 |
| 0.80 | 1.20 | 0% | 0% | 100% | −3.59 |
| 0.80 (default) | 1.30 | 0% | 0% | 100% | −3.68 |

**Verdict: no default changed, because none of them is worth a code change.** Every quality column is already
at its floor or ceiling (`exact` 100%, `repeat` 0%, `nonword` 0%) at every setting — the reranker and
filters own those, not the sampler. The only column that moves is fluency: temperature 0.7 buys ~0.1
logprob, `best_of 8` buys ~0.1 for **double the time**, and penalty 1.2 vs 1.3 buys 0.05–0.06. The
checkpoint payload stores its own temperature/penalty (the live one: 0.8 / 1.3), so the server serves
those unless a request overrides them — `POST /api/generate {"temperature": 0.7}` already works and is
the recommended setting; lowering `DEFAULT_REPETITION_PENALTY` was rejected because 1.3 is what fixed
the 44% repeat bug and the gain is inside noise.

`--strict` is now **constrained decoding, not retrying**: candidates that would overshoot a line's
syllable target, add junk characters or stutter a word are masked out of the top-k, and a newline is
only reachable when the line hits the target exactly. If nothing can reach the target the line closes
short (never over) and the printout says so. No 10-attempt loop, no wasted sampling.

## 5. Front end (local "Sakura Desk" desk UI)

React + TypeScript + Tailwind v4 in `frontend/` (shadcn-compatible layout: `components.json`,
`src/components/ui/`, `src/lib/utils.ts`, `@/*` alias), talking to `backend/server.py` — stdlib only,
**no new Python dependency**.

```powershell
# terminal 1 - API + model (port 8000, warms the default checkpoint on start)
python -m backend.server

# terminal 2 - Vite dev server, hot reload (port 5177, proxies /api to 8000)
cd frontend
npm install          # once
npm run dev
```

Open http://127.0.0.1:5177. No dev server needed either: `npm run build` in `frontend/` once and
`python -m backend.server` serves the built app itself on http://127.0.0.1:8000 (`/api/*` still wins).

Three stages, as of 2026-10-17:

1. **Cover** — full-viewport sakura poster whose wordmark is `HAIKU` big with `ZERO` set small on the
   baseline (`titleTail`), at `titleMaxFraction={0.42}` so it runs under the copy like the original
   `SAKURA` layout, with the supplied cut-out branch downloaded to `frontend/public/branch.png` and
   passed as `foregroundSrc` — the branch sits at z-30 over the wordmark (z-20), which is the whole
   point of the design. `fullBleed` drops the page frame (no inset padding, no `rounded-xl`, no
   shadow), so the photograph reaches every edge; the copy uses `copySize="compact"`, which is
   deliberately smaller than a poem poster's (at 1440: headline 29.5 px, body 19.4 px, subheadline
   22.4 px — the poem's three lines stay at 33.5 / 24.1 px, because there the text is the content).
   `scrim="strong"` darkens the copy band so the smaller type stays readable over the flowers — four
   gradient stops, not three, since a from/via/to ramp is already a wash by the headline — plus a
   text-shadow, and the social line hides under `sm` where it collided with the footer. The copy
   itself carries Japanese (`五・七・五。`, `書かせながら、数える。`, `俳句 · ローカル`, keywords `五七五` /
   `手元`). The reveal is a 1.9 s tween of `forceProgress` 0 → 1, so the letters slide in, the copy
   rises and the **Start writing** button (`cta` slot) fades in — no scrollbar. Respects
   `prefers-reduced-motion` (jumps straight to the settled state).
2. **Write** — the chat, `src/components/ui/ai-thinking-orb-and-input.tsx`. A glass pill sits 96 px
   above the bottom edge; a word and Enter send it flying up (1.5 s: the pill shrinks to a ball, the
   ball grows and becomes a dotted sphere), the labels cycle (`Counting` → `Metering` → `Weighing` →
   `Composing`) while the request runs, the sphere sweeps cream → rose, condenses into a petal pulse and
   unfolds into the answer card. The card is **cream with dark text**, 440 wide, and its height follows
   the answer (`76 + 25 px per line + 10 px per stanza gap`, capped at 360): one haiku 151, two 236.
   Each haiku line is its own row — no `` / `` separators — and a blank line separates two poems. The
   card carries **every poem the request returned** (2 with the default settings), the same poems the
   desk shows under **the poster →**.
   `Escape` returns to the pill from any phase; **New haiku** resets it; **the poster →** (top right,
   appears once an answer lands) opens the desk with the same request already generated. `?debug` adds a
   speed slider and Replay. `prefers-reduced-motion` swaps the whole flight for cross-fades.
   The stage is full-viewport, so the sidebar becomes an overlay drawer behind the `panel` button (it is
   `inert` while closed, so its controls stay out of the tab order).
3. **Desk** — the same poster component, now with two panels: a **sidebar** holding every control
   (seed phrase, checkpoint, poems, best-of, temperature, seed, meter, start-with-word, scene, view,
   Generate, random seed, presets, errors) and the main column with the results. The chat's answer also
   fills the desk, so a submitted word is already generated when you arrive; the top bar's **ask** button
   goes back. The sidebar is sticky and full height on desktop, an off-canvas drawer behind a `panel`
   button under `lg`. The old `SAKURA DESK` display title is gone: the display face is the cover's
   wordmark, not the console chrome.

Controls: seed prompt (plus presets), checkpoint dropdown that **auto-discovers every
`checkpoints*/best.pt`** with its epoch/val loss, poems, best-of, temperature, optional seed,
strict/free meter toggle, and four Unsplash scenes. Two result views:

- **Poster** — the sakura editorial component, one sticky scroll panel per poem; seed (or the poem's
  longest word) is the huge title, the three lines map to headline / body / subheadline, footers carry
  model, `5 · 7 · 5` tag, logprob and date.
- **Sheet** — plain list with meter, logprob and repeat rate per poem. **Judge output in this view**;
  the poster is for feel only.

API: `GET /api/health`, `GET /api/models`, `POST /api/generate`
(`{prompt, model, count, strict, best_of, temperature, seed}` -> `{poems: [{lines, pattern, exact,
logprob, repeated_rate}], seconds}`). Generation is serialised by a lock and checkpoints stay cached
after the first load, so the second click is fast.

Component note: `src/components/ui/ai-thinking-orb-and-input.tsx` is the supplied morph-orb chat with
deliberate changes — the demo's `../../index.css` import is gone (the app imports it once in
`main.tsx`), its stylesheet lives next to it as `ai-thinking-orb.css` and was re-skinned from black +
success-green to the cover's palette (cream `#f6eee8`, blush `#e7a3ae`, amber `#d6946e`, umber
`#1a1114`), the orb's hard-coded white → green light gradient became cream → rose, the card grew from
320×108 to 440×148 so a three-line haiku fits uncropped (and lost a `mask-image` whose `var()` was an
invalid gradient position, which is why the original could not clip), and the copy is the app's
(`HAIKU` / `New haiku` / `Give it one word…`). The answer layer is `pointer-events: none` until it is
actually on screen: it sits above the pill, and as a plain `inset-0` box it swallowed every click on the
input once the field had lost focus, so the page had to be reloaded to write again. Clicking anywhere on
the pill also focuses the field. No new dependencies: it is canvas + CSS only.

Component note: `src/components/ui/sakura-editorial-poster.tsx` is the supplied design component with
deliberate changes — `cn` imported from `@/lib/utils` (shadcn convention) instead of a local copy,
`bg-gradient-to-*` renamed to the Tailwind v4 `bg-linear-to-*`, the demo `design-layer.com` assets
replaced with the four verified Unsplash URLs in `src/lib/sakura-scenes.ts` (kept out of the
component file so it stays a pure component module for fast refresh), and the two readability fixes
below. Its default `foregroundSrc` is `null` (Unsplash has no transparent branch cut-out); pass a PNG
URL to bring the branch back (the cover uses the original `hero-branch.png`, vendored to
`frontend/public/branch.png` — 708×853 RGBA — so nothing is hotlinked at runtime). Four optional props
were added for the cover: `titleTail` (the small second word, measured at the probe size and folded into
the same width fit as `title`, at `TITLE_TAIL_RATIO` of its size), `titleMaxFraction` (height cap,
default 0.19 for poem posters), `cta` (a node in the copy block), `scrim: "soft" | "strong"`,
`fullBleed` (frame padding, rounding and shadow off) and `copySize: "default" | "compact"`.
All default to the old behaviour, so poem posters render exactly as before.

**Title sizing (fixed after a real-use report of "BLOOD" swallowing the whole poster).**
`SakuraFitTitle` originally sized the title from `wrap.clientWidth / probe.scrollWidth` — fit to the
row width — which is correct for a long title and absurd for a short one: a five-letter word was
stretched to the full width, ~750 px tall at 1600 px, and it buried the poem underneath it. The fitted
size is now also capped at `TITLE_MAX_HERO_FRACTION` (0.19) of the hero's height, so a short title
stays a title (~110 px cap height at 1600 px wide) while a long one still shrinks to fit. The hero is
observed too, so it re-fits on resize.

**Poem text sizing.** The three haiku lines are mapped headline / body / subheadline, and their
`clamp()` **maxima** were the real problem — `2.6cqw` capped at `1.2rem` meant a 1888 px-wide poster
rendered raw 12–19 px text while the title took the whole frame. Maxima are now 3.4rem / 2.3rem /
2.7rem (line 2, the body, was the worst offender at 12 px); the minima keep small screens legible,
and the body's `max-w` went 62% → 74% so the bigger text does not wrap.

## 6. Experiments (compare configs, keep the winner)

```powershell
python experiments.py                                    # 4 configs x 5 epochs, ~15-20 min
python experiments.py --epochs 25                         # the real thing, ~2 h
python experiments.py --runs clean-d0.15,meter-d0.10      # subset
```

Each run gets `experiments/<name>/{best.pt,last.pt,log.csv}`; the sweep prints a table sorted by val
loss and writes `experiments/results.json` with val loss, minutes and 3 samples per run.

## 7. Checks

```powershell
python test_smoke.py          # pipeline + decoder rules + HTTP API on a tiny fixture, CPU, <1 min
python quality_probe.py       # fixed probes on every checkpoints*/best.pt: junk, meter, seed, non-words
python quality_probe.py --models checkpoints --n 20 --word-weight 0    # measure the rerank itself
python quality_probe.py --models checkpoints --no-seed-first           # then 'seed' is a real test
python inference_sweep.py --n 8      # temperature / penalty / best-of: quality vs seconds per poem
python quality_floor.py              # gate: exit 1 if a measured quality promise regressed
python quality_floor.py --self-test  # prove the gate can fail (no model needed, seconds)
python seed_placement_probe.py --words sea,moon,river                  # where the typed word ends up
python self_distill.py --stage probe --models checkpoints,checkpoints_selfdistill_mix --n 20   # filters off
python self_distill.py --stage probe --models checkpoints_meter --ablate-meter --n 20   # state zeroed
```

`quality_floor.py` is the gate the probes exist for: every promise in §4 (100% exact, 0% repeats,
0% invented words, 0% junk, 100% seed) is a *rate*, and any decoder change can quietly cost one.
It measures the same fixed 24 strict poems (fixed seeds, fixed generation seeds, GPU or CPU), compares
them with `logs/quality_floor.json`, prints the least typical poem and **exits 1** on any regression.
The baseline records the checkpoint it was measured on, so a promotion fails the gate until the floor
is re-recorded with `--write` — a red result needs a decision, not a different set of weights.
Remaining noise is priced in: a repeat in 24 poems is 4%, so the ceiling is 6%.

Cross-run val loss is not comparable (each run has its own split), so **the probe harness is the way to
compare models**. `--word-weight 0` isolates the real-word rerank, `--no-seed-first` isolates whether
the model would have mentioned the seed on its own (it usually would not).

## 8. Deviations from PLAN.md (each deliberate)

| PLAN.md said | Built | Why |
|---|---|---|
| TensorFlow 2.x + Keras 3 in WSL2 | PyTorch on native Windows | TF has had no native-Windows GPU support since 2.11, so WSL2 was mandatory for TF; PyTorch uses the 4050 directly with no admin/reboot. Same architecture and hyperparameters. |
| `keras.mixed_precision` | `torch.amp` autocast fp16 + GradScaler | PyTorch equivalent, with dynamic loss scaling (what keeps fp16 stable on 6 GB). |
| lowercase everything | case preserved, `--lowercase` available | Lowercasing destroys proper nouns; flip the flag for the plan's behavior. |
| custom BPE, `ByteLevel` | same + `initial_alphabet=ByteLevel.alphabet()` | Without it, bytes unseen in training are silently *dropped* at encode time and generation loses characters (verified: `"3.14/2"`, `"été café"`, `"jazz?"` fail without, pass with). |
| Module 10 = retry until 5-7-5 | constrained decoding + local syllable counter | Exact meter by construction, no retry loop, no extra dependency. |
| one corpus file | `haikus_clean.txt` + `haikus_meter.txt` | Free second experiment; `--meter-data` / `experiments.py` compare them. |
| — | GPT-2 init + residual scaling, tied weights, float32 head | Standard small-model stabilizers; 5.80M params, in line with the ~6M target. |

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` False | CPU wheel installed — reinstall with the `cu130` index (§1). |
| CUDA out of memory | `--batch-size 64` (or 32). |
| Loss stuck near 8.32 | tokenizer/corpus mismatch: rerun §2 in order. |
| Preview never improves | check the val column in `training_log.csv`; try `--dropout 0.15`, more epochs, or the meter corpus. |
| Everything on CPU | activate `pytorch_with_gpu` first; the `device cuda` line in §3 is the proof. |
| Want the plan's original TF path | not possible on this GPU without WSL2 (`wsl --install`, admin + reboot). |
| Front end says "Could not reach the API" | `python -m backend.server` is not running, or is on another port: the Vite proxy targets `127.0.0.1:8000`. |
| Poems carry `?????` or `. . . .` | fixed by `generate.has_junk` (strict mode). If it returns, a checkpoint is not being fed through the metered path. |
| Poems ignore the word I typed | seed leading (default) puts it in the poem by construction. Leaving it off is fine at the default `--best-of 4` (measured 98% adherence); if the word must appear, the adherence loop costs ~1 extra draw (`seed_placement_probe.py`). |
| Poems invent words ("shavement") | raise `--best-of` (8-16) so the real-word rerank has more candidates; the deeper lever is more data — 1.8M tokens is thin for a 15.8M-param model. |
| `front end not built` from :8000 | run `npm run build` in `frontend/`, or just use the dev server on :5177. |
