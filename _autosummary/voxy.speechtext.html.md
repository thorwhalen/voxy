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
   ones sit where the capital trigger bites.

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

| [`SENTENCE_END`](#voxy.speechtext.SENTENCE_END)   | the next line's capital then starts a new one.   |
|-----------------------------------------------------------------|--------------------------------------------------|

### Functions

| [`flag_risks`](#voxy.speechtext.flag_risks)(text, \*[, heteronyms, extra])   | Words of `text` a model may misread, in text order.              |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| [`prepare_speech_text`](#voxy.speechtext.prepare_speech_text)(text, \*[, ...])        | Display text in, speech text out, with a record of every change. |

### Classes

| [`Change`](#voxy.speechtext.Change)(line, original, replacement, reason)   | One edit [`prepare_speech_text()`](#voxy.speechtext.prepare_speech_text) made (`line` counts from 1).   |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| [`Risk`](#voxy.speechtext.Risk)(line, word, reason[, readings, ...])     | A word a model may misread: where it is, why it is flagged, its known readings.                                |
| [`SpeechText`](#voxy.speechtext.SpeechText)(display, speech[, changes])        | The text to show (`display`), the text to speak (`speech`) and what changed.                                   |

### *class* voxy.speechtext.Change(line, original, replacement, reason)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One edit [`prepare_speech_text()`](#voxy.speechtext.prepare_speech_text) made (`line` counts from 1).

### *class* voxy.speechtext.Risk(line, word, reason, readings=(), line_initial_capital=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A word a model may misread: where it is, why it is flagged, its known readings.

### voxy.speechtext.SENTENCE_END *= re.compile('[.!?…][\\'\\\\"”’»)\\\\]]\*\\\\s\*$')*

the next line’s capital then starts a new one.

* **Type:**
  What closes a sentence

### *class* voxy.speechtext.SpeechText(display, speech, changes=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The text to show (`display`), the text to speak (`speech`) and what changed.

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
