# voxy.lexique

French heteronyms derived from Lexique, to merge under voxy’s hand-made table.

Lexique (New, Pallier et al., [http://www.lexique.org](http://www.lexique.org)) lists about 140 000 French
word forms with their pronunciation, lemma, part of speech and frequency. Its
homographs that are not homophones (*est* /ɛ/ or /ɛst/, *couvent* /kuv/ or /kuvɑ̃/,
*président*…) are words a speech model can misread. voxy ships a short hand-made
table (`voxy.pronounce.heteronym_readings`) and no Lexique data, so derive more
from your own copy and let the hand-made readings win:

```pycon
>>> from voxy.lexique import lexique_heteronyms
>>> from voxy.pronounce import heteronym_readings
>>> derived = lexique_heteronyms("Lexique383.tsv", min_frequency=1)
>>> table = {**derived, **heteronym_readings("fr")}
>>> derived["couvent"]
{'noun': 'kuvɑ̃', 'verb': 'kuv'}
```

The derived table is not complete. Lexique stores one pronunciation per spelling,
part of speech and lemma, and no liaison or variant readings, so heteronyms that
differ only there are missing from it: *tous*, *os*, *six*, *dix*, *christ*,
*lacs*, *fait*. It extends the hand-made table; it does not replace it.

The result has the shape `voxy.pronounce` takes (`{word: {reading name: IPA}}`,
IPA in the symbols of its phoneme model). Pronunciations that differ only by
Lexique’s transcription noise are one reading and its variants (`verb~e`); a
reading is named by its part of speech, with the names of
`voxy.speechtext.FRENCH_READING_NAMES`, or by its lemma when two readings share a
part of speech:

```pycon
>>> rows = [
...     {"ortho": "est", "phon": "E", "lemme": "être", "cgram": "AUX", "freqfilms2": "100", "freqlivres": "90"},
...     {"ortho": "est", "phon": "e", "lemme": "être", "cgram": "VER", "freqfilms2": "9", "freqlivres": "9"},
...     {"ortho": "est", "phon": "Est", "lemme": "est", "cgram": "NOM", "freqfilms2": "20", "freqlivres": "30"},
...     {"ortho": "chat", "phon": "Sa", "lemme": "chat", "cgram": "NOM", "freqfilms2": "40", "freqlivres": "30"},
... ]
>>> heteronyms_from_rows(rows)
{'est': {'verb': 'ɛ', 'verb~e': 'e', 'noun': 'ɛst'}}
```

### Module Attributes

| [`LEXIQUE_TO_IPA`](#voxy.lexique.LEXIQUE_TO_IPA)      | Lexique's phonetic code -> IPA in the vocabulary of voxy's phoneme model (`facebook/wav2vec2-lv-60-espeak-cv-ft`); Lexique's other letters are that IPA.                                                                                             |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`TRANSCRIPTION_NOISE`](#voxy.lexique.TRANSCRIPTION_NOISE) | two pronunciations equal after this are one reading (open/closed vowels, schwa).                                                                                                                                                                     |
| [`LEXIQUE_TO_UPOS`](#voxy.lexique.LEXIQUE_TO_UPOS)     | Lexique's part of speech (`cgram`, whole or before its `:`) -> universal POS.                                                                                                                                                                        |
| [`LEXIQUE_POS_NAMES`](#voxy.lexique.LEXIQUE_POS_NAMES)   | Lexique's part of speech -> a reading name (`voxy.speechtext.FRENCH_READING_NAMES`).                                                                                                                                                                 |
| [`VERB_NAME`](#voxy.lexique.VERB_NAME)           | A pronunciation with a verb row is named `verb` when its heaviest part of speech is one of these: Lexique tags some verb forms as nouns or adjectives too (*violent*, *insolent*), while an adverb (*plus*) is genuinely heavier than its verb rows. |
| [`UNTAGGED_NAME`](#voxy.lexique.UNTAGGED_NAME)       | The name of a reading whose part of speech Lexique leaves empty.                                                                                                                                                                                     |
| [`ORTHO`](#voxy.lexique.ORTHO)               | Columns read from a Lexique 3 file.                                                                                                                                                                                                                  |
| [`PHON`](#voxy.lexique.PHON)                | Columns read from a Lexique 3 file.                                                                                                                                                                                                                  |
| [`LEMMA`](#voxy.lexique.LEMMA)               | Columns read from a Lexique 3 file.                                                                                                                                                                                                                  |
| [`CGRAM`](#voxy.lexique.CGRAM)               | Columns read from a Lexique 3 file.                                                                                                                                                                                                                  |

### Functions

| [`heteronyms_from_rows`](#voxy.lexique.heteronyms_from_rows)(rows, \*[, ...])              | `{word: {reading name: IPA}}` for the spellings Lexique reads in more than one way.                                                                                 |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`lexique_heteronyms`](#voxy.lexique.lexique_heteronyms)(path, \*[, min_frequency, ...]) | The heteronyms of a Lexique 3 file (tab-separated, e.g. `Lexique383.tsv`), as [`heteronyms_from_rows()`](#voxy.lexique.heteronyms_from_rows) derives them. |
| [`lexique_to_ipa`](#voxy.lexique.lexique_to_ipa)(phon, \*[, symbols])                | A Lexique pronunciation in IPA.                                                                                                                                     |
| [`phon_key`](#voxy.lexique.phon_key)(phon, \*[, noise])                        | What a Lexique pronunciation is compared as, transcription noise removed.                                                                                           |

### voxy.lexique.CGRAM *= 'cgram'*

Columns read from a Lexique 3 file.

### voxy.lexique.LEMMA *= 'lemme'*

Columns read from a Lexique 3 file.

### voxy.lexique.LEXIQUE_POS_NAMES *= {'ADJ': 'adjective', 'ADJ:dem': 'det', 'ADJ:ind': 'det', 'ADJ:int': 'det', 'ADJ:num': 'numeral', 'ADJ:pos': 'det', 'ADV': 'adverb', 'ART': 'det', 'AUX': 'verb', 'CON': 'conjunction', 'NOM': 'noun', 'ONO': 'interjection', 'PRE': 'preposition', 'PRO': 'pronoun', 'VER': 'verb'}*

Lexique’s part of speech -> a reading name (`voxy.speechtext.FRENCH_READING_NAMES`).

### voxy.lexique.LEXIQUE_TO_IPA *= {'1': 'œ̃', '2': 'ø', '5': 'ɛ̃', '8': 'y', '9': 'œ', '@': 'ɑ̃', 'E': 'ɛ', 'G': 'ŋ', 'N': 'ɲ', 'O': 'ɔ', 'R': 'ʁ', 'S': 'ʃ', 'Z': 'ʒ', 'g': 'ɡ', '§': 'ɔ̃', '°': 'ə'}*

Lexique’s phonetic code -> IPA in the vocabulary of voxy’s phoneme model
(`facebook/wav2vec2-lv-60-espeak-cv-ft`); Lexique’s other letters are that IPA.

### voxy.lexique.LEXIQUE_TO_UPOS *= {'ADJ': 'ADJ', 'ADJ:dem': 'DET', 'ADJ:ind': 'DET', 'ADJ:int': 'DET', 'ADJ:num': 'NUM', 'ADJ:pos': 'DET', 'ADV': 'ADV', 'ART': 'DET', 'AUX': 'AUX', 'CON': 'CCONJ', 'NOM': 'NOUN', 'ONO': 'INTJ', 'PRE': 'ADP', 'PRO': 'PRON', 'VER': 'VERB'}*

Lexique’s part of speech (`cgram`, whole or before its `:`) -> universal POS.

### voxy.lexique.ORTHO *= 'ortho'*

Columns read from a Lexique 3 file.

### voxy.lexique.PHON *= 'phon'*

Columns read from a Lexique 3 file.

### voxy.lexique.TRANSCRIPTION_NOISE *= {'8': 'y', '9': '2', 'E': 'e', 'O': 'o', '°': ''}*

two
pronunciations equal after this are one reading (open/closed vowels, schwa).

* **Type:**
  Lexique letters it writes inconsistently -> the letter they are compared as

### voxy.lexique.UNTAGGED_NAME *= 'word'*

The name of a reading whose part of speech Lexique leaves empty.

### voxy.lexique.VERB_NAME *= 'verb'*

A pronunciation with a verb row is named `verb` when its heaviest part of speech
is one of these: Lexique tags some verb forms as nouns or adjectives too (*violent*,
*insolent*), while an adverb (*plus*) is genuinely heavier than its verb rows.

### voxy.lexique.heteronyms_from_rows(rows, , min_frequency=0.0, pos_names={'ADJ': 'adjective', 'ADJ:dem': 'det', 'ADJ:ind': 'det', 'ADJ:int': 'det', 'ADJ:num': 'numeral', 'ADJ:pos': 'det', 'ADV': 'adverb', 'ART': 'det', 'AUX': 'verb', 'CON': 'conjunction', 'NOM': 'noun', 'ONO': 'interjection', 'PRE': 'preposition', 'PRO': 'pronoun', 'VER': 'verb'}, symbols={'1': 'œ̃', '2': 'ø', '5': 'ɛ̃', '8': 'y', '9': 'œ', '@': 'ɑ̃', 'E': 'ɛ', 'G': 'ŋ', 'N': 'ɲ', 'O': 'ɔ', 'R': 'ʁ', 'S': 'ʃ', 'Z': 'ʒ', 'g': 'ɡ', '§': 'ɔ̃', '°': 'ə'}, noise={'8': 'y', '9': '2', 'E': 'e', 'O': 'o', '°': ''})

`{word: {reading name: IPA}}` for the spellings Lexique reads in more than one way.

Pronunciations equal under [`phon_key()`](#voxy.lexique.phon_key) are one reading, its heaviest
pronunciation the reading and the others `<name>~<IPA>` variants. Readings come
heaviest first (weight: the sum of their rows’ frequencies) and are named by their
heaviest part of speech (`verb` if any row is a verb); readings that share one
are named by lemma, or `<pos>:<IPA>` when their lemmas coincide too. Words whose
heaviest reading is under `min_frequency` (per million) are left out.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### voxy.lexique.lexique_heteronyms(path, , min_frequency=0.0, pos_names={'ADJ': 'adjective', 'ADJ:dem': 'det', 'ADJ:ind': 'det', 'ADJ:int': 'det', 'ADJ:num': 'numeral', 'ADJ:pos': 'det', 'ADV': 'adverb', 'ART': 'det', 'AUX': 'verb', 'CON': 'conjunction', 'NOM': 'noun', 'ONO': 'interjection', 'PRE': 'preposition', 'PRO': 'pronoun', 'VER': 'verb'}, symbols={'1': 'œ̃', '2': 'ø', '5': 'ɛ̃', '8': 'y', '9': 'œ', '@': 'ɑ̃', 'E': 'ɛ', 'G': 'ŋ', 'N': 'ɲ', 'O': 'ɔ', 'R': 'ʁ', 'S': 'ʃ', 'Z': 'ʒ', 'g': 'ɡ', '§': 'ɔ̃', '°': 'ə'}, noise={'8': 'y', '9': '2', 'E': 'e', 'O': 'o', '°': ''})

The heteronyms of a Lexique 3 file (tab-separated, e.g. `Lexique383.tsv`), as
[`heteronyms_from_rows()`](#voxy.lexique.heteronyms_from_rows) derives them.

Raises `ValueError` if the file lacks one of `REQUIRED_COLUMNS`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### voxy.lexique.lexique_to_ipa(phon, , symbols={'1': 'œ̃', '2': 'ø', '5': 'ɛ̃', '8': 'y', '9': 'œ', '@': 'ɑ̃', 'E': 'ɛ', 'G': 'ŋ', 'N': 'ɲ', 'O': 'ɔ', 'R': 'ʁ', 'S': 'ʃ', 'Z': 'ʒ', 'g': 'ɡ', '§': 'ɔ̃', '°': 'ə'})

A Lexique pronunciation in IPA.

```pycon
>>> lexique_to_ipa("kuv@"), lexique_to_ipa("pRezid@"), lexique_to_ipa("Est")
('kuvɑ̃', 'pʁezidɑ̃', 'ɛst')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.lexique.phon_key(phon, , noise={'8': 'y', '9': '2', 'E': 'e', 'O': 'o', '°': ''})

What a Lexique pronunciation is compared as, transcription noise removed.

```pycon
>>> phon_key("vjOl°"), phon_key("vjol")
('vjol', 'vjol')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
