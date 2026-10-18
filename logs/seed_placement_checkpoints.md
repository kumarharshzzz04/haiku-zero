# Seed placement probe - checkpoints

epoch 6, val 3.9203; 12 words x 4 poems, best-of 5, strict 5-7-5

| placement | poems | exact 5-7-5 | says it | opens with it | says it inside | missing | repeat | invented | mean LL |
|---|---|---|---|---|---|---|---|---|---|
| first | 48 | 100% | 100% | 98% | 2% | 0% | 0% | 0% | -3.19 |
  - first: first hit on line 1 100%; opens the poem 98%; says it twice 2%
| topic | 48 | 100% | 96% | 15% | 81% | 4% | 0% | 0% | -3.00 |
  - topic: first hit on line 1 52%, line 2 37%, line 3 11%; opens the poem 15%; says it twice 0%

## guaranteeing the word in topic mode

best-of-5 draws until a poem says the word and holds exact 5-7-5 (cap 6): landed **48/48**; mean 1.0 draws

| word | attempts per poem (None = gave up) |
|---|---|
| rain | [1, 1, 1, 1] |
| night | [1, 1, 1, 1] |
| moon | [1, 1, 1, 1] |
| wind | [1, 1, 1, 1] |
| light | [1, 1, 1, 1] |
| snow | [1, 1, 1, 1] |
| sea | [1, 1, 1, 1] |
| river | [1, 1, 1, 1] |
| stone | [1, 3, 1, 1] |
| bird | [1, 1, 1, 1] |
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


