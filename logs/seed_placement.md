# Seed placement probe - checkpoints_combo

epoch 3, val 3.5866; 12 words x 4 poems, best-of 5, strict 5-7-5

| placement | poems | exact 5-7-5 | says it | opens with it | says it inside | missing | repeat | invented | mean LL |
|---|---|---|---|---|---|---|---|---|---|
| first | 48 | 100% | 100% | 100% | 0% | 0% | 0% | 0% | -3.09 |
  - first: first hit on line 1 100%; opens the poem 100%; says it twice 0%
| topic | 48 | 100% | 98% | 12% | 85% | 2% | 0% | 0% | -2.85 |
  - topic: first hit on line 1 53%, line 2 36%, line 3 11%; opens the poem 12%; says it twice 0%

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
| stone | [1, 1, 1, 1] |
| bird | [1, 1, 1, 1] |
| fire | [1, 1, 1, 1] |
| cloud | [1, 1, 1, 1] |

## samples

### rain

**first**

- [exact, word at (0, 0)] rain falls on the hill / the lamp turns to snow again / there is a flower
- [exact, word at (0, 0)] rain falls on the snow / the smell of water fills the / white stairs open wide

**topic**

- [exact, word at (2, 1)] Dawn on the puddle / The snow falls over my head / Spring rain is warm now
- [exact, word at (0, 1)] Autumn rain and dusk / A bone of red light in the / Stars are so green now

### night

**first**

- [exact, word at (0, 0)] night heat is cold now / The light goes over the stairs / The moon comes down
- [exact, word at (0, 0)] night is asleep in / one lamp of light on the roof / the empty door keeps

**topic**

- [exact, word at (0, 3)] sound of the night sky / listening to a small line / a red man's voice
- [exact, word at (0, 0)] Night falls on the stove / A moth flutters in the dusk / Mist's first glow of

### moon

**first**

- [exact, word at (0, 0)] moon at the edge of / a star to come in like rain / the old dog dips the
- [exact, word at (0, 0)] moon in the west there / is a glimpse of the sky with / a gypsy girl's

**topic**

- [exact, word at (0, 0)] Moon on the edge of / The Obsolete I love you / Dream in your eyes and
- [exact, word at (1, 1)] I'm lost in my / harvest moon And I want to / bring a frog in the

### wind

**first**

- [exact, word at (0, 0)] wind blow the rain on / the night in the leaves of / a tree's shadow
- [exact, word at (0, 0)] wind on the roof wall / a leaf in the dark room is / the spring rain in the

**topic**

- [exact, word at (0, 1)] Strong wind blows the heart / Bless me the friend's fiddle / Again I reach the
- [exact, word at (0, 1)] autumn wind chimes / the lane's cry and the sound / of a neighbor's

### light

**first**

- [exact, word at (0, 0)] light clouds drift over / the fire is heavy with rain / the moon moves bright
- [exact, word at (0, 0)] light on the mountain / all day long i hear you say / i will tell you ah

**topic**

- [exact, word at never says it] End of summer rain / The cat is quiet at night / It's time to come
- [exact, word at (1, 0)] In the fog of a / light Temple of Liber Zeus / Break in the dark wood

### snow

**first**

- [exact, word at (0, 0)] snow falls on the hill / the room is warm now and dark / this night that comes
- [exact, word at (0, 0)] snow falls on the wall / the long shadow of a tree / takes its own way

**topic**

- [exact, word at (1, 1)] spring rain is cold now / the snow is heavy and white / i love to sleep in
- [exact, word at (0, 0)] Snow rain return to / Again in the dark of night / The wind blows cold and

### sea

**first**

- [exact, word at (0, 0)] sea of the south wind / a wild bird in the puddle / snow falls down on the
- [exact, word at (0, 0)] sea and shore at noon / a far stream rises from the / south of blue sky gleams

**topic**

- [exact, word at (2, 3)] and i know where i / shook my head and went out the / din of the sea and
- [exact, word at (1, 2)] A star dissolves / In the sea of my life's / The ocean is black

### river

**first**

- [exact, word at (0, 0)] river to sea in / the town is out of the path / my mother's eyes
- [exact, word at (0, 0)] river lake of blue / slowly drifting down the stream / and a bit of stone

**topic**

- [exact, word at (1, 1)] In the dark city / A black river fills the path / The hills are apace
- [exact, word at (2, 2)] Birds fall down from the / field of the Lake Away and / drink the river from

### stone

**first**

- [exact, word at (0, 0)] stone wall in the dark / the smell of pine chocolate / on a tin roof peck
- [exact, word at (0, 0)] stone wall to hold it / the door open wide and warm / and the bars are cold

**topic**

- [exact, word at (1, 2)] Home and house and there / The old stone wall will never / Touch my feet again
- [exact, word at (2, 3)] The last bone stands still / Far off in the sky and out / A white line of stone

### bird

**first**

- [exact, word at (0, 0)] bird sings low in the / temple by ebony now / a bee hang on the
- [exact, word at (0, 0)] bird song in the field / the wind smell of a sparrow / from below the pond

**topic**

- [exact, word at (0, 0)] a bird in the woods / the piano sings as he pass / as it tak's a
- [exact, word at (0, 1)] the bird on the street / a berry in a puddle / maybe we'll stop home

### fire

**first**

- [exact, word at (0, 0)] fire and smoke that is / a tin of light from the earth / on my fore hand's
- [exact, word at (0, 0)] fire and lightning shall / be piled on the earth to a / mountain of light that

**topic**

- [exact, word at (1, 1)] In the kitchen room / A white fire flutters through the / Storm drops from the path
- [exact, word at (1, 1)] My head is cut now / It's a fire in my mouth / Another snow canoe

### cloud

**first**

- [exact, word at (0, 0)] cloud rings in the sky / the world is asleep in dark / a leaf falls on the
- [exact, word at (0, 0)] cloud of star light on / the moon shines in the sky / a white leaf falls down

**topic**

- [exact, word at (1, 4)] Star lights drift over / The horizon shift by cloud / Clean and float away
- [exact, word at (1, 0)] the sky is still in / a cloud of light shines on / an autumn star tops

