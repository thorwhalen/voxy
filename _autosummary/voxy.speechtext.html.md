# voxy.speechtext

Display text vs speech text, and spotting words a model may misread.

A text a person reads and the text a model should be given are not always the same.
Measured case (French, a Qwen3-TTS clone): a verse line starting with a capitalised
verb, “Est parti” after a line “Mon frere”, was read as the noun /est/ (“l’Est”) in 4
of 4 takes; lowercased (“est parti”) it was read correctly in 4 of 4. The capital, not
the line break, is the trigger: line-initial “Est” tokenises like the noun in “l’Est”.
ASR checks cannot catch this (Whisper hears “est” either way), so the fix is upstream:

1. [`prepare_speech_text()`](#voxy.speechtext.prepare_speech_text) turns display text into speech text: it lowercases a
   *verse-initial* capital that does not start a sentence (keeping deliberate capitals:
   words also capitalised mid-line, or named in `keep`), then applies a per-language
   respelling table and proper-name aliases, and returns every change it made.
2. [`flag_risks()`](#voxy.speechtext.flag_risks) lists the words of a text a model may misread (heteronyms you
   supply, e.g. from Lexique; more detectors plug in through `extra`), saying which
   ones sit where the capital trigger bites. [`pos_detector()`](#voxy.speechtext.pos_detector) is one such
   detector: it tags the text and, for heteronyms whose part of speech picks the
   reading (French “est”: verb /ɛ/, noun /ɛst/), names the `expected` reading;
   [`llm_detector()`](#voxy.speechtext.llm_detector) asks a language model for the risky words and their readings.

Engines differ in what they accept: for several local models text is the only lever
(they take no phonemes or SSML), so this is a text-level step; it plugs into
`LocalCloneSpeechModel(text_prep=...)`. Verifying the audio itself (phoneme
recognition of the flagged words) is [`voxy.pronounce`](voxy.pronounce.html.md#module-voxy.pronounce).

```pycon
>>> text = "Mon frere\nEst parti\nLoin d'ici."
>>> prepare_speech_text(text).speech
"Mon frere\nest parti\nloin d'ici."
>>> prepare_speech_text(text, keep={"Est", "Loin"}).speech == text
True
```

### Module Attributes

| [`SENTENCE_END`](#voxy.speechtext.SENTENCE_END)     | the next line's capital then starts a new one.                                                                           |
|-------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [`Tagger`](#voxy.speechtext.Tagger)           | `text -> (char offset, token, universal POS tag)` for every token.                                                       |
| [`DFLT_SPACY_MODEL`](#voxy.speechtext.DFLT_SPACY_MODEL) | The spaCy model [`spacy_tagger()`](#voxy.speechtext.spacy_tagger) loads by default (French).               |
| [`LLM_RISK_PROMPT`](#voxy.speechtext.LLM_RISK_PROMPT)  | The prompt [`llm_detector()`](#voxy.speechtext.llm_detector) sends; `$language` and `$text` are filled in. |

### Functions

| [`flag_risks`](#voxy.speechtext.flag_risks)(text, \*[, heteronyms, extra])   | Words of `text` a model may misread, in text order.                                                                                  |
|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [`llm_detector`](#voxy.speechtext.llm_detector)(\*[, chat, language, prompt])  | A [`flag_risks()`](#voxy.speechtext.flag_risks) `extra` detector that asks a language model for risky words.         |
| [`pos_detector`](#voxy.speechtext.pos_detector)([rules, tagger])               | A [`flag_risks()`](#voxy.speechtext.flag_risks) `extra` detector: heteronyms whose part of speech picks the reading. |
| [`prepare_speech_text`](#voxy.speechtext.prepare_speech_text)(text, \*[, ...])        | Display text in, speech text out, with a record of every change.                                                                     |
| [`spacy_tagger`](#voxy.speechtext.spacy_tagger)([model])                       | A [`Tagger`](#voxy.speechtext.Tagger) over a spaCy pipeline, loaded on first use.                                |

### Classes

| [`Change`](#voxy.speechtext.Change)(line, original, replacement, reason)   | One edit [`prepare_speech_text()`](#voxy.speechtext.prepare_speech_text) made (`line` counts from 1).   |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| [`Risk`](#voxy.speechtext.Risk)(line, word, reason[, readings, ...])     | A word a model may misread: where it is, why it is flagged, its known readings.                                |
| [`SpeechText`](#voxy.speechtext.SpeechText)(display, speech[, changes])        | The text to show (`display`), the text to speak (`speech`) and what changed.                                   |

### *class* voxy.speechtext.Change(line, original, replacement, reason)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One edit [`prepare_speech_text()`](#voxy.speechtext.prepare_speech_text) made (`line` counts from 1).

### voxy.speechtext.DFLT_SPACY_MODEL *= 'fr_core_news_sm'*

The spaCy model [`spacy_tagger()`](#voxy.speechtext.spacy_tagger) loads by default (French).

### voxy.speechtext.LLM_RISK_PROMPT *= 'You help a text-to-speech system avoid misreadings. Below is a $language text, its lines numbered. List the words a speech model may misread: heteronyms (words with more than one pronunciation, such as French "est" as a verb or as a noun), names, abbreviations. For each, give the line, the word exactly as written, short names of its possible readings, and the reading this context needs.\\n\\nAnswer with a JSON list only, one object per word:\\n[{"line": 2, "word": "est", "readings": ["verb", "noun"], "expected": "verb", "why": "..."}]\\nUse "expected": null when the context does not decide. Answer [] if nothing is risky.\\n\\n$text\\n'*

The prompt [`llm_detector()`](#voxy.speechtext.llm_detector) sends; `$language` and `$text` are filled in.

### *class* voxy.speechtext.Risk(line, word, reason, readings=(), line_initial_capital=False, expected=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A word a model may misread: where it is, why it is flagged, its known readings.

#### expected *: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)* *= None*

The reading the context picks (a name among `readings`), when a detector knows it.

### voxy.speechtext.SENTENCE_END *= re.compile('[.!?…][\\'\\\\"”’»)\\\\]]\*\\\\s\*$')*

the next line’s capital then starts a new one.

* **Type:**
  What closes a sentence

### *class* voxy.speechtext.SpeechText(display, speech, changes=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The text to show (`display`), the text to speak (`speech`) and what changed.

### voxy.speechtext.Tagger

`text -> (char offset, token, universal POS tag)` for every token.

* **Type:**
  A tagger

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]

### voxy.speechtext.flag_risks(text, , heteronyms=(), extra=())

Words of `text` a model may misread, in text order.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text that will be spoken (run it through [`prepare_speech_text()`](#voxy.speechtext.prepare_speech_text)
    first to see what is left).
  * **heteronyms** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`Collection`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Words with more than one reading, as a collection or as
    `{word: readings}` (e.g. from Lexique). Matched case-insensitively.
  * **extra** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Risk`](#voxy.speechtext.Risk)]]]) – More detectors, each `text -> risks` (a part-of-speech rule, an LLM
    proposer). Their risks are merged in by line.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Risk`](#voxy.speechtext.Risk)]

A heteronym that is capitalised at a line start where no sentence begins is flagged
with `line_initial_capital=True`: that is where the capital trigger bites.

### voxy.speechtext.llm_detector(, chat=None, language='French', prompt='You help a text-to-speech system avoid misreadings. Below is a $language text, its lines numbered. List the words a speech model may misread: heteronyms (words with more than one pronunciation, such as French "est" as a verb or as a noun), names, abbreviations. For each, give the line, the word exactly as written, short names of its possible readings, and the reading this context needs.\\n\\nAnswer with a JSON list only, one object per word:\\n[{"line": 2, "word": "est", "readings": ["verb", "noun"], "expected": "verb", "why": "..."}]\\nUse "expected": null when the context does not decide. Answer [] if nothing is risky.\\n\\n$text\\n')

A [`flag_risks()`](#voxy.speechtext.flag_risks) `extra` detector that asks a language model for risky words.

The model’s proposals are kept only when the word is really on the line it names,
and an `expected` reading only when it is one of the proposed `readings`; a
reply that is not a JSON list proposes nothing.

* **Parameters:**
  * **chat** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `prompt -> reply text`; default `aix.chat` (`pip install 'voxy[aix]'`,
    keys and model from aix’s own config). Each call costs a model request.
  * **language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text’s language, named in the prompt.
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A `string.Template` with `$language` and `$text` (numbered lines).
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Risk`](#voxy.speechtext.Risk)]]

```pycon
>>> fake = lambda prompt: '[{"line": 1, "word": "est", "readings": ["verb", "noun"], "expected": "verb"}]'
>>> [(r.line, r.word, r.expected) for r in llm_detector(chat=fake)("Il est parti")]
[(1, 'est', 'verb')]
```

### voxy.speechtext.pos_detector(rules={'affections': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'affluent': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'as': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'content': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'convergent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'couvent': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'coïncident': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'divergent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'est': {'ADJ': 'noun', 'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'excellent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'exceptions': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'expédient': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'ferment': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'fier': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'influent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'insolent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'intentions': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'inventions': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'lis': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'mentions': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'négligent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'options': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'parent': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'portions': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'président': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'relations': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'résident': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'sens': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'somnolent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'talent': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'tous': {'DET': 'det', 'PRON': 'pronoun'}, 'urgent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'violent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'vis': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'éditions': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'émergent': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}, 'équivalent': {'AUX': 'verb', 'NOUN': 'noun', 'PROPN': 'noun', 'VERB': 'verb'}, 'évident': {'ADJ': 'adjective', 'AUX': 'verb', 'VERB': 'verb'}}, , tagger=None)

A [`flag_risks()`](#voxy.speechtext.flag_risks) `extra` detector: heteronyms whose part of speech picks the reading.

Run it on the *speech* text ([`prepare_speech_text()`](#voxy.speechtext.prepare_speech_text)): a stray verse-initial
capital fools the tagger the same way it fools the speech model.

* **Parameters:**
  * **rules** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – `{word: {UPOS tag: reading name}}`, words matched case-insensitively
    (default `FRENCH_POS_HETERONYMS`).
  * **tagger** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `text -> (offset, token, UPOS)`; default [`spacy_tagger()`](#voxy.speechtext.spacy_tagger)
    (French), loaded on first use.
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Risk`](#voxy.speechtext.Risk)]]

```pycon
>>> fake = lambda text: [(0, "Il", "PRON"), (3, "est", "AUX"), (7, "parti", "VERB")]
>>> [(r.word, r.expected) for r in pos_detector(tagger=fake)("Il est parti")]
[('est', 'verb')]
```

### voxy.speechtext.prepare_speech_text(text, , lowercase_verse_initial=True, keep=(), respellings=None, aliases=None, transforms=())

Display text in, speech text out, with a record of every change.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text as a reader sees it (lines may be verse lines).
  * **lowercase_verse_initial** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Lowercase a line-initial capital that does not start
    a sentence (the first line of a paragraph, or a line after one ending in
    `. ! ? …`, keeps its capital).
  * **keep** ([`Collection`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Words whose capital is deliberate and stays (compared exactly). Words
    also capitalised elsewhere in a line are kept automatically.
  * **respellings** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Word -> same-sound spelling, whole words, case-insensitive (e.g.
    a language’s ambiguous forms written unambiguously).
  * **aliases** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Proper name -> pronounceable spelling, whole words, case-insensitive.
  * **transforms** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Extra `text -> text` steps applied last, over the whole text.
* **Return type:**
  [`SpeechText`](#voxy.speechtext.SpeechText)
* **Returns:**
  A [`SpeechText`](#voxy.speechtext.SpeechText).

### voxy.speechtext.spacy_tagger(model='fr_core_news_sm')

A [`Tagger`](#voxy.speechtext.Tagger) over a spaCy pipeline, loaded on first use.

Needs `pip install 'voxy[pos]'` and the model itself
(`python -m spacy download fr_core_news_sm`).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]
