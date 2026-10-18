# Seed placement probe - checkpoints_combo

epoch 3, val 3.5866; 12 words x 8 poems, best-of 1, strict 5-7-5

| placement | poems | exact 5-7-5 | says it | opens with it | says it inside | missing | repeat | invented | mean LL |
|---|---|---|---|---|---|---|---|---|---|
| first | 96 | 100% | 100% | 100% | 0% | 0% | 2% | 17% | -3.49 |
  - first: first hit on line 1 100%; opens the poem 100%; says it twice 0%
| topic | 96 | 100% | 79% | 3% | 76% | 21% | 1% | 11% | -3.34 |
  - topic: first hit on line 1 46%, line 2 42%, line 3 12%; opens the poem 3%; says it twice 0%

## guaranteeing the word in topic mode

best-of-1 draws until a poem says the word and holds exact 5-7-5 (cap 6): landed **96/96**; mean 1.3 draws

| word | attempts per poem (None = gave up) |
|---|---|
| rain | [1, 1, 1, 1, 2, 1, 1, 2] |
| night | [1, 1, 4, 2, 1, 1, 1, 4] |
| moon | [1, 3, 1, 3, 1, 2, 1, 1] |
| wind | [1, 1, 1, 1, 1, 1, 1, 1] |
| light | [1, 1, 1, 1, 2, 1, 1, 1] |
| snow | [2, 2, 1, 3, 2, 2, 2, 1] |
| sea | [1, 3, 1, 1, 1, 1, 1, 1] |
| river | [1, 1, 1, 1, 1, 1, 2, 1] |
| stone | [1, 1, 1, 1, 1, 1, 1, 1] |
| bird | [2, 2, 1, 1, 1, 1, 2, 2] |
| fire | [1, 2, 1, 1, 1, 1, 1, 1] |
| cloud | [1, 1, 1, 1, 1, 1, 1, 1] |

## samples

### rain

**first**

- [exact, word at (0, 0)] rain falls on the stove / the cold is still on my tongue / the sun has ris'd

**topic**

- [exact, word at (2, 3)] snow on the shore of / the farm with my hair on the / moon of autumn rain

### night

**first**

- [exact, word at (0, 0)] night heat is cold now / The light goes over the stairs / The moon comes down

**topic**

- [exact, word at (0, 1)] spring night rain dark and / the ket's ears shriek at the / long light of day on

### moon

**first**

- [exact, word at (0, 0)] moon at the edge of / a star to come in like rain / the old dog dips the

**topic**

- [exact, word at (0, 0)] Moon on the edge of / The Obsolete I love you / Dream in your eyes and

### wind

**first**

- [exact, word at (0, 0)] wind clouds drift over / the stream's sound from the lake / mountain leaves increed

**topic**

- [exact, word at (2, 1)] trees sway on the high / a leaf is coming in now / the wind has a footstwred

### light

**first**

- [exact, word at (0, 0)] light and darkness come / the last bright morning sun sets / and the old trees lie

**topic**

- [exact, word at never says it] End of summer rain / The cat is quiet at night / It's time to come

### snow

**first**

- [exact, word at (0, 0)] snow is on the top / the road runs in and now it / is in a little

**topic**

- [exact, word at never says it] Morning fog is now / On the ice patch of a cloud / A moth slips from its

### sea

**first**

- [exact, word at (0, 0)] sea foam rising up / from the high steep to the lake / a roo say was i

**topic**

- [exact, word at never says it] spring wind blows slowly / a puff of rain drifts from the / rust onto my handwrpped

### river

**first**

- [exact, word at (0, 0)] river to sea in / the town is out of the path / my mother's eyes

**topic**

- [exact, word at never says it] pale moon shines / into the lake's shadow / autumn mist on the

### stone

**first**

- [exact, word at (0, 0)] stone grave wall fishing / a pig's head peek the same / autumn sunset haze

**topic**

- [exact, word at (0, 2)] dust on stone wall bou / the last white hous's trick of / blind eye and mud pit

### bird

**first**

- [exact, word at (0, 0)] bird calls out the train / a lamp turns to mine eyes espe / the sky in silence

**topic**

- [exact, word at (0, 2)] Poet is a bird / An inch of leaves from the / Grey evening haze soft

### fire

**first**

- [exact, word at (0, 0)] fire and smoke are still / as if you had to say a / unseen fare not look

**topic**

- [exact, word at (0, 2)] thunder and fire and / pink as if by a word of / a puff of rain or

### cloud

**first**

- [exact, word at (0, 0)] cloud rings in the sky / the world is asleep in dark / a leaf falls on the

**topic**

- [exact, word at (1, 4)] Star lights drift over / The horizon shift by cloud / Clean and float away

