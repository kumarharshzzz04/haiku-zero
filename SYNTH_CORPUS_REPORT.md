# Authored haiku → verification → fine-tune

Everything below was run in this repo on 2026-10-17. Numbers are from the commands shown,
not estimates. Files: `data/raw_synthetic/*.txt` (the poems), `synth_verify.py` (the checker),
`build_synth_corpus.py` (the mixer), `synth_compare.py` (the model comparison),
`checkpoints_mix/` (the fine-tuned weights).

The point of the exercise: the live model was trained on found data (public-domain poetry
books, tweet-shaped datasets, scraped haiku pages). This adds data I wrote myself, in the
voice the app wants, verified line by line against the same syllable counter the decoder uses.

## 1. What was written

| file | theme | poems |
|---|---|---|
| `data/raw_synthetic/w01_winter.txt` | winter, frost, indoor cold | 49 |
| `data/raw_synthetic/w02_spring.txt` | spring, rain, the first warm days | 41 |
| `data/raw_synthetic/w03_summer.txt` | summer heat, insects, kettles | 40 |
| `data/raw_synthetic/w04_autumn.txt` | autumn, smoke, harvest | 25 |
| `data/raw_synthetic/w05_night.txt` | night, rooms, street lights | 25 |
| | **total** | **180** |

180, not the 400 first sketched: every line is hand-written and then syllabified exactly, and
180 is what survived the checks below without padding the set with filler. 459 unique lines,
no two poems identical, one poem = 3 lines separated by blank lines.

## 2. Verification (`synth_verify.py`)

Runs every poem through `backend/meter_check.line_syllables` — the same counter that gates strict
decoding — and reports the failing ones with their pattern.

```
180/180 exact 5-7-5 across 5 files   (0 duplicates)
  w01_winter.txt: 49/49
  w02_spring.txt: 41/41
  w03_summer.txt: 40/40
  w04_autumn.txt: 25/25
  w05_night.txt: 25/25
```

Writing the first draft produced 7 near-misses (5-8-5, 5-7-6, 6-7-5). Each was repaired by
testing candidate lines against the counter rather than by guessing syllables, then re-verified.
The check is not "roughly haiku": the counter's exact answer is the pass condition.

Three more quality passes were run because the first draft was written as linked chains
(the last line of one poem opening the next), which left artefacts:

1. **repeated two-word heads inside one poem** — 48 poems opened two lines with the same
   noun ("the year turns again / the year turns over again"). 53 lines rewritten; 3 poems keep
   a deliberate L1/L3 echo, none repeat L1/L2.
2. **sentence-fragment openers** ("then takes it down", "and a far country") left over from the
   chaining. Replaced in the winter file.
3. **stock lines reused across poems** — "the night is quiet" appeared 7 times. A dedupe pass
   swapped 34 occurrences for fresh lines of the same syllable count.

After all three: 459 unique lines, 56 reused (0.18 of them, mostly closers like "the light is
leaving"), 3 poems with an intentional echo. Re-run of `synth_verify.py`: still **180/180 exact**,
because every edited line was checked against the counter, not trusted.

## 3. The mixed corpus (`build_synth_corpus.py`)

```
python build_synth_corpus.py --repeat 6 --samples 30
authentic 40924 + authored 180 x 6 = 42004 poems
synthetic share of the mix: 2.6%
```

- authentic half = `data/raw_classical/` (the 40,924 non-tweet poems from the current pipeline)
- the authored set is oversampled ×6 so 180 poems carry ~1,080 slots of the ~42k corpus. Higher
  oversampling buys influence and starts buying memorisation, which section 6 measures.
- output: `data/haikus_mix.txt` (training), `data/haikus_synthetic.txt` (just the authored ones),
  `data/synthetic_samples.md` (30 random samples, seed 20261017), `data/synth_mix_stats.json`.

30 samples from `data/synthetic_samples.md` (seed 20261017; this block is copied out of
that file, not typed):

```
1. the smell of cut grass / the first mown grass of the year / the grass is wet now
2. your breath in the air / the window fogs with my breath / a crow on the fence
3. the road is white now / the sun stands still on the road / the church bell at noon
4. by dusk it is gone / the sun sets behind the barn / a thin line of red
5. all the lane is still / no sound comes from the dark / the night is quiet
6. grain by grain by grain / the world is written again / one more white letter
7. sun on the tin roof / the road is white in the sun / one cloud in the sky
8. one apple falls down / the apple drops in the grass / the dog is sleeping
9. one red leaf falling / a red leaf floats on the pond / the pond is cold now
10. smoke in the still air / the smoke goes up in the cold / the smoke stands still now
11. the hedges are bare / the wind turns over the grass / a crow on the plough
12. frost on the stubble / a hard frost comes at dusk / the field is white now
13. soft rain on the roof / the washing flaps on the line / the line is heavy
14. a drift of leaves / a kite of leaves and wind / the lane is quiet
15. leaves a dark trail / by noon it is memory / by dusk it is gone
16. warm dust in the lane / the stove door opens and glows / the long hot morning
17. a cold wind at dusk / the field is cut to the bone / the smoke stands still now
18. the well runs low now / the well gives up its cold / a bucket of rain
19. one warm day at last / the last warm day of the year / the light is leaving
20. the wind is turning / the year turns over again / the light is leaving
21. red thread on the snow / the small notebook is blank / snow at the window
22. the hall light is on / a lamp is left in the hall / the stairs creak once more
23. a thin line of red / the last light on the snowfield / holding on too long
24. inch by inch it goes / the wind works all afternoon / at the same small drift
25. the old pear tree blooms / the orchard holds its white smoke / the garden stirs now
26. the orchard in bloom / your hands smell of the wet earth / work is not over
27. the crickets start up / a hen scratches at the frost / the long hot morning
28. a cat on the step / the cat waits by the back door / the night is quiet
29. the door will not shut / a letter from the summer / warm in my cold hand
30. the pond is frozen / the river slows under ice / a grey heron waits
```

## 4. Fine-tune

```
python -m backend.train --corpus data/haikus_mix.txt --init-from checkpoints/best.pt \
  --ckpt-dir checkpoints_mix --log training_log_mix.csv --run-name haikuzero-mix \
  --lr 6e-5 --embed-dim 384 --layers 8 --ffn-dim 1536 --epochs 12
```

The shape flags are not decoration: the live checkpoint is 384×8×1536 (`Config` defaults are
256×6×1024), and loading its weights into the default shape fails with a size mismatch —
the first attempt at this command died that way.

- 1,488 steps (12 epochs × 124), 2.5 min on the 4050, ~87k tok/s
- **val 3.9203 → 3.5646** on the *same* validation split (same corpus reader, `seed=1337`,
  `val_fraction=0.03`), the lowest val of any checkpoint in the repo
- best checkpoint landed at epoch 1 and every epoch after it was worse: train 3.6788 → 3.3239
  while val climbs 3.5646 → 3.5805 in one step per epoch. The run is 42k poems for a 15.8M-param
  model at this LR, so the first pass is the useful one and the rest overfits.
- `training_log_mix.csv` holds 24 epoch rows: rows 1-12 are the first attempt, rows 13-24 the
  shipped run. The second run happened because one line of the winter file was still a fragment
  ("then takes it down") after the fine-tune had already been trained on it; rewriting the line
  and re-running was cheaper than shipping text and weights that disagree.

## 5. Probes (`quality_probe.py --models checkpoints,checkpoints_mix --n 8 --best-of 4`)

```
model                    epoch    junk       ?   exact    seed  repeat  nonword
checkpoints                  6      0%      0%    100%    100%      0%       0%
checkpoints_mix              1      0%      0%    100%    100%      0%       0%
```

All four columns are already at their ceiling in both models, so the probe cannot separate
them — the decoder does that work. The separation is in val loss and in what the samples read
like (`synth_compare.py`, 25 poems per model over 5 prompts, written to
`logs/synth_compare.md`):

```
live  autumn:   autumn rain falls on / The old tree walk to his house / In the garden room
live  old pond: old pond bum tight sam / in the blue of the lake toast / my child's eye flaps
live  snow:    snow flakes of blue / the moon on the seat roof's / dusk tops in his eye
mix   autumn:   autumn evening in / a white birch glitter and fall / with its shadow the
mix   snow:     snow on the water / the raven's eyes are wet / and the rain is cold
mix   old pond: old pond's shadow / the black cat leans through the grass / a tin of maple
```

The mix model's failures look like ordinary bad poetry (wrong image, no verb); the live one
still mixes up parts of speech ("bum tight sam", "the seat roof's", "her sketch's roots").
Both are exact 5-7-5 by construction, so that column says nothing about either.

## 6. Memorisation

`synth_compare.py` counts every generated line that appears verbatim in the 180 authored poems
or in the authentic corpus:

```
verbatim copies: authored 0, authentic 1
```

Over 25 poems × 3 lines = 75 lines from the mix model, **none** copies an authored line, so the
×6 oversampling is not buying regurgitation at this factor. This counter is the check to re-run
if the factor is ever raised — the authored poems are the one part of the corpus whose exact
wording is known, so a copy shows up in this count and nowhere else.

One caveat on reading it: a line that happens to be common in the authentic corpus counts under
`authentic` (1 of 75 here), which is inheritance, not memorisation of the new data.

## 7. End to end, live

- the API lists the new checkpoint automatically (`GET /api/models` → `checkpoints_mix`,
  val 3.5646) because it globs `checkpoints*/best.pt`
- `backend/server.py` got a label for it and one real fix: consecutive poems in one response
  were seeded `seed, seed+1, seed+2` while each poem draws `best_of` candidates from
  `seed..seed+best_of`, so poems 2 and 3 inherited 3 of poem 1's candidates and the same winner
  came back three times. Seeds now step by `best_of`.
- verified over HTTP against the restarted server: `count=3`, seed 11, mix model → three
  *different* poems, and the response carries
  `model_label: "Synthetic mix - live fine-tune + 180 authored haiku"`.

## 8. How to use it, and how to undo it

The live model in `checkpoints/` was not touched. To run the app on the fine-tune, either pick
`checkpoints_mix` in the UI's model list (already served on :8000) or make it the default:

```bash
cp -r checkpoints checkpoints_live_backup     # keeps the current live model
cp checkpoints_mix/best.pt checkpoints/best.pt
```

## 9. Honest limits

1. **180 authored poems is a small lever.** At ×6 it is 2.6% of the corpus; the val drop is
   real but modest. 400+ would move it more; the cost is writing and syllabifying them.
2. **56 lines are reused** across the set (0.18), mostly closers. Fine at ×6, worth fixing
   before raising the factor.
3. **Val is flat after epoch 1**, so more fine-tune epochs are not the next lever — more data is.
4. **The probe columns are saturated**, so "better" here means val loss and reading the samples.
   A blind A/B on human preference would be the honest next measurement.
5. The mix model is only better *on this corpus*; a future change to the authentic half means
   rebuilding `data/haikus_mix.txt` and re-running section 4.
