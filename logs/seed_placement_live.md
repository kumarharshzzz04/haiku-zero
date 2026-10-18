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

- [exact, word at (0, 0)] rain on the sand and / a lamp that is burning bright / slips from his eyes in

**topic**

- [exact, word at (2, 0)] Parfum tree rings out / The sound of a candle's / Rain on the head of

### night

**first**

- [exact, word at (0, 0)] night after day of / a red chocolate wax out / of the dent on its

**topic**

- [exact, word at (1, 2)] sudden rain from the / moon at night in darkness to / the long day of light

### moon

**first**

- [exact, word at (0, 0)] moon and stars are gone / and on the road there comes / a shadow of snow

**topic**

- [exact, word at (1, 0)] In the blue sky and / a moon at sunset in a / scent of dusk I see

### wind

**first**

- [exact, word at (0, 0)] wind pours through the night / the sound of a water flute / and his teeth are white

**topic**

- [exact, word at (0, 1)] slow wind blows out of / the ocean's bough and claps / through the leaves each

### light

**first**

- [exact, word at (0, 0)] light is on the side / of a dark pile of grey / on my eyes and hair

**topic**

- [exact, word at (0, 1)] camp light on the sky / soaring oak leaves soared to / flee away in a

### snow

**first**

- [exact, word at (0, 0)] snow clouds flashing down / into the sky the moon dips / fills in water and

**topic**

- [exact, word at never says it] Spring rain is so cold / The Sycamore flute can tell / I guess you are dead

### sea

**first**

- [exact, word at (0, 0)] sea tide side of reach / a sour moon sets down the lake / on the ocean surge

**topic**

- [exact, word at (1, 1)] Relent is a star / A bright sea of sky and earth / This place is full of

### river

**first**

- [exact, word at (0, 0)] river to sea coal / the eagle eyes on the shore / turtle over the

**topic**

- [exact, word at (0, 3)] Home from the river / We follow to earth in sight / Dancing down in the

### stone

**first**

- [exact, word at (0, 0)] stone wall tops of ice / sharp in a shell from the stair / the ocean moves

**topic**

- [exact, word at (0, 1)] one stone branch is stained / on the right side of a field / i hope that it must

### bird

**first**

- [exact, word at (0, 0)] bird sings through the trees / A single pulse of life brings / Eleven leaves

**topic**

- [exact, word at (0, 0)] a bird sings in the / morning sunshine of gardens / in a bar's name

### fire

**first**

- [exact, word at (0, 0)] fire in the sunlight / the scent of ice cream tiles / in the blue pitch bag

**topic**

- [exact, word at (1, 0)] thunder and rain and / a fire of smoke dips the flame / into the blue light

### cloud

**first**

- [exact, word at (0, 0)] cloud of a rainbow / faint in the air and swirl through / a bird's song of

**topic**

- [exact, word at (1, 1)] Rain falls from the sky / The cloud sinks down on a path / On its shadow's

