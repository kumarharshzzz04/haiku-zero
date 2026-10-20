# AI haiku intake — scored, not just filtered

`scripts/data/ai_haiku_vetter.py` gives every poem a **percentile against our own curated 180**: "this
poem beats X% of the poems we wrote and hand-verified". Three measured parts drive it:

1. soundness — a logistic model fitted on real poems (authentic corpus + our authored 180)
   against **corruptions of those same poems** (lines swapped, content word replaced or
   deleted, word stuttered, line shuffled), so it measures "is this poem intact" rather than
   "does this sound like 1902". Held-out AUC **0.787**.
2. cliché penalty — 45% of the graded weight, from the share of abstract
   filler words ("joy", "heart", "life", "dream"…).
3. repetition penalty — 15%, from lines already reused in this intake and
   3-word phrases the batch repeats, plus a small hit when two lines name clashing seasons.

- source folder: `data/ai_batches` — 8 file(s), 4722 poems read
- **tier A: 89** → `data/raw_ai/` (training-ready: passes the score cut **and** is exact 5-7-5 by `backend.meter_check`)
- tier B: 289 → `data/raw_ai_near/` (within ±1 syllable; not mixed by default)
- below the cut (percentile 25): 1895
- dropped for form alone, however good the score (2197): more than 1 syllable off 5-7-5
- the gate and the score are independent: the score ranks, the meter decides whether the result may enter the corpus at all
- calibration: our authored 180 span percentiles 1-100 with median **50**,
  so "tier A" is literally "beats the bottom quarter of what we already wrote".

## What the soundness model weighs

| feature | weight | (positive = this feature makes a poem look real) |
|---|---|---|
| unseen_bigram | -13.294 | |
| exact | +2.193 | |
| cross_line_bigram | +1.243 | |
| line_length | -0.609 | |
| attested | -0.442 | |
| echo | +0.180 | |
| mean_bigram | +0.095 | |
| meter_distance | +0.071 | |

## Per file

| file | rows | tier A | tier B | below | off meter | duplicates of corpus | junk | mean percentile of accepted |
|---|---|---|---|---|---|---|---|---|
| `haiku_zero_corpus_batch_01.jsonl` | 1000 | 31 | 217 | 746 | 6 | 0 | 0 | 31 |
| `haiku_zero_corpus_batch_01.txt` | 1000 | 0 | 0 | 746 | 6 | 248 | 0 | — |
| `haiku_zero_corpus_batch_02.jsonl` | 72 | 2 | 4 | 56 | 10 | 0 | 0 | 22 |
| `haiku_zero_corpus_batch_03.txt` | 1000 | 0 | 0 | 8 | 992 | 0 | 0 | — |
| `haiku_zero_corpus_batch_04.txt` | 1000 | 0 | 0 | 2 | 998 | 0 | 0 | — |
| `haiku_zero_corpus_batch_05.txt` | 114 | 2 | 4 | 64 | 44 | 0 | 0 | 33 |
| `haiku_zero_corpus_batch_06.txt` | 37 | 1 | 0 | 26 | 10 | 0 | 0 | 37 |
| `batch 1.txt` | 499 | 53 | 64 | 247 | 131 | 4 | 0 | 34 |

## The 12 highest-scoring poems

1. **p95** (sound 87, cliché 0.00, repeat 0.00) — Mountains stand so high / Peaks are covered in the snow / Touching the white clouds
2. **p94** (sound 87, cliché 0.00, repeat 0.29) — Space is vast and cold / We are floating on a rock / Looking at the stars
3. **p94** (sound 86, cliché 0.00, repeat 0.29) — The cold wind blows hard / White snow falls down on the ground / Winter is here now
4. **p94** (sound 87, cliché 0.00, repeat 0.43) — Summer heat is thick / Air is heavy and so still / Seeking out the shade
5. **p89** (sound 87, cliché 0.00, repeat 0.79) — Space is vast and deep / We are floating on a rock / Looking at the stars
6. **p82** (sound 86, cliché 0.00, repeat 1.00) — Sand on the white beach / Palm trees sway in the warm breeze / I rest in the shade
7. **p81** (sound 70, cliché 0.00, repeat 0.00) — Moss grows on the rocks / Soft and green and damp to touch / Old and quiet place
8. **p79** (sound 72, cliché 0.00, repeat 0.29) — Weekend is here now / No alarm to wake me up / Sleep in late today
9. **p78** (sound 81, cliché 0.00, repeat 1.00) — The porch waits for you / The song comes back at dusk / I still know that song
10. **p78** (sound 79, cliché 0.00, repeat 1.00) — Spring warms the bare earth / The warm wind crosses the fields / The wind smells of snow
11. **p78** (sound 69, cliché 0.00, repeat 0.29) — Blanket wraps me tight / I close my eyes and drift off / Dreaming of the sea
12. **p77** (sound 79, cliché 0.00, repeat 1.00) — The porch waits for you / The room is quiet at noon / The years pass like rain

## 12 poems sitting on the cut (the judgement calls)

- **p24** — Your words rest in dust / The old house smells of cedar / The old gate swings wide
- **p24** — The moon climbs the roof / A moth circles the warm light / The dark holds its breath
- **p24** — The years pass like rain / I hear your steps in the hall / Your scarf holds the cold
- **p24** — The old gate swings wide / The old house smells of cedar / The lamp glows as then
- **p24** — One light in the hall / One light burns late in the hall / The key rusts in rain
- **p24** — The seed sleeps in earth / The geese call over the marsh / Spring warms the bare earth
- **p24** — Your cup by the sink / A ring of tea stains the wood / The house keeps the dusk
- **p24** — A moth taps the glass / A moth circles the warm light / A moth taps the glass
- **p25** — A bee rests on thyme / The reeds bow down to the stream / The moon lights the pond
- **p25** — I still know that song / The photo curls at the edge / The lamp glows as then
- **p25** — Poet writes a verse / Words dance on the paper white / Feelings come alive
- **p25** — The seed sleeps in earth / The plum tree blooms by the gate / The last rose bends low

## Next

```bash
python scripts/data/build_synth_corpus.py --repeat 6 --ai-repeat 2    # rebuild the mix (x2: see RUNBOOK 2d)
python -m backend.train --corpus data/haikus_mix.txt --init-from checkpoints/best.pt \
  --ckpt-dir checkpoints_combo --lr 6e-5 --embed-dim 384 --layers 8 --ffn-dim 1536 --epochs 12
```

Drop the next batch and re-run `python scripts/data/ai_haiku_vetter.py`. Move the cut with `--store 0.90`, refit with `--refresh-model`, interrogate one poem with `--explain "line one | line two | line three"`.

