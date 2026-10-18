# Seed placement probe - checkpoints_v2

epoch 6, val 3.8363; 12 words x 4 poems, best-of 5, strict 5-7-5

| placement | poems | exact 5-7-5 | says it | opens with it | says it inside | missing | repeat | invented | mean LL |
|---|---|---|---|---|---|---|---|---|---|
| first | 48 | 100% | 100% | 100% | 0% | 0% | 0% | 0% | -3.24 |
  - first: first hit on line 1 100%; opens the poem 100%; says it twice 0%
| topic | 48 | 100% | 98% | 12% | 85% | 2% | 0% | 0% | -2.96 |
  - topic: first hit on line 1 60%, line 2 36%, line 3 4%; opens the poem 12%; says it twice 0%

## guaranteeing the word in topic mode

best-of-5 draws until a poem says the word and holds exact 5-7-5 (cap 6): landed **48/48**; mean 1.1 draws

| word | attempts per poem (None = gave up) |
|---|---|
| rain | [1, 1, 1, 1] |
| night | [1, 1, 1, 1] |
| moon | [1, 1, 1, 1] |
| wind | [1, 1, 1, 1] |
| light | [1, 1, 1, 1] |
| snow | [2, 1, 1, 1] |
| sea | [1, 1, 1, 1] |
| river | [1, 1, 1, 1] |
| stone | [1, 1, 1, 1] |
| bird | [1, 1, 1, 3] |
| fire | [1, 1, 1, 1] |
| cloud | [1, 1, 1, 1] |

## samples

### rain

**first**


**topic**


### night

**first**


**topic**


### moon

**first**


**topic**


### wind

**first**


**topic**


### light

**first**


**topic**


### snow

**first**


**topic**


### sea

**first**


**topic**


### river

**first**


**topic**


### stone

**first**


**topic**


### bird

**first**


**topic**


### fire

**first**


**topic**


### cloud

**first**


**topic**


