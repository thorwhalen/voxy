# voxy.pronounce

Checking, in the audio, which reading of a flagged word a render used.

[`voxy.speechtext.flag_risks()`](voxy.speechtext.html.md#voxy.speechtext.flag_risks) lists the words a model may misread (heteronyms
such as French “est” /ɛ/ vs “Est” /ɛst/). This module listens for the answer: it finds
the word in a render (word timestamps from ASR), asks a phoneme recogniser for its
frame-wise phoneme log-probabilities, and scores each candidate reading by its CTC
likelihood. The reading the audio supports best is the one the model said; a render
whose word came out as the wrong one can be re-rendered.

ASR word times are loose, so a word cropped alone carries bits of its neighbours, and a
neighbour’s /s/ or /t/ then reads as the longer reading. [`audit()`](#voxy.pronounce.audit) (the call for a
finished render) therefore scores each reading as the whole phrase around it, the
neighbours written out by espeak ([`phrase_readings()`](#voxy.pronounce.phrase_readings), [`espeak_phonemizer()`](#voxy.pronounce.espeak_phonemizer));
[`verify_words()`](#voxy.pronounce.verify_words) and [`render_verified()`](#voxy.pronounce.render_verified) still crop.

The recogniser is `facebook/wav2vec2-lv-60-espeak-cv-ft`, run in its own torch env
([`PHONEME_ENV`](#voxy.pronounce.PHONEME_ENV), built by `voxy.envs`, with a bundled espeak-ng). Its
`vocab.json` maps espeak IPA symbols to ids, so readings are written in IPA and
tokenised against that vocabulary. Everything above the recogniser is pure numpy and
takes the recogniser (`emitter`) and the phonemizer as arguments, so it is tested
without torch.

```pycon
>>> vocab = ["<pad>", "ɛ", "s", "t"]
>>> emissions = log_probs_from_path([1, 0, 0], vocab_size=4)  # hears /ɛ/ only
>>> best_reading(emissions, vocab, {"verb": "ɛ", "noun": "ɛst"})
'verb'
```

### Module Attributes

| [`DFLT_PHONEME_MODEL`](#voxy.pronounce.DFLT_PHONEME_MODEL)   | wav2vec2 fine-tuned to emit espeak IPA (CTC, 16 kHz).                                                                          |
|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| [`PHONEME_ENV`](#voxy.pronounce.PHONEME_ENV)          | CPU torch plus transformers (the processor's vocab.json is all it needs).                                                      |
| [`DFLT_CROP_PADDING`](#voxy.pronounce.DFLT_CROP_PADDING)    | ASR word edges are loose.                                                                                                      |
| [`DFLT_MARGIN`](#voxy.pronounce.DFLT_MARGIN)          | A reading must beat the runner-up by this much (mean log-likelihood per target symbol) for a verdict to count as decided.      |
| [`DFLT_FILLER_PENALTY`](#voxy.pronounce.DFLT_FILLER_PENALTY)  | What spotting a word in its context charges per frame left to filler (nats).                                                   |
| [`DFLT_CONTEXT_MARGIN`](#voxy.pronounce.DFLT_CONTEXT_MARGIN)  | A reading must beat the runner-up by this many nats when scored as a whole phrase.                                             |
| [`ESPEAK_LANGUAGES`](#voxy.pronounce.ESPEAK_LANGUAGES)     | espeak voices for [`audit()`](#voxy.pronounce.audit)'s `language` codes (others are passed as they are). |
| [`Phonemizer`](#voxy.pronounce.Phonemizer)           | writes out the words around a flagged one.                                                                                     |
| [`DFLT_CONTEXT_WORDS`](#voxy.pronounce.DFLT_CONTEXT_WORDS)   | Words of context on each side of a flagged word when [`audit()`](#voxy.pronounce.audit) scores it.       |
| [`DFLT_SPOT_MARGIN`](#voxy.pronounce.DFLT_SPOT_MARGIN)     | A reading must beat the runner-up by this many nats when spotted in context.                                                   |
| [`VARIANT_SEPARATOR`](#voxy.pronounce.VARIANT_SEPARATOR)    | In a reading name, what separates a reading from one of its variants (`verb~liaison`).                                         |
| [`BLANK_SYMBOL`](#voxy.pronounce.BLANK_SYMBOL)         | The CTC blank of the recogniser's vocabulary (wav2vec2 uses its pad token).                                                    |
| [`Emitter`](#voxy.pronounce.Emitter)              | `(audio, start, end) -> (log_probs [frames x vocab], vocab)`.                                                                  |
| [`DFLT_MAX_ATTEMPTS`](#voxy.pronounce.DFLT_MAX_ATTEMPTS)    | Renders of one line at most (the first take included).                                                                         |
| [`Check`](#voxy.pronounce.Check)                | one word to check.                                                                                                             |
| [`HETERONYMS_DIR`](#voxy.pronounce.HETERONYMS_DIR)       | The heteronym tables voxy ships, one `<language>.json` per language.                                                           |

### Functions

| [`align_words`](#voxy.pronounce.align_words)(text, words)                           | `{index of a word of text: index of the ASR word heard for it}`.                                                                                                                             |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`audit`](#voxy.pronounce.audit)(text, audio, \*[, readings, language, ...])  | Which reading did each heteronym of `text` get in the finished `audio`?                                                                                                                      |
| [`best_reading`](#voxy.pronounce.best_reading)(log_probs, vocab, readings, \*)       | The name of the reading `log_probs` support best.                                                                                                                                            |
| [`checks_by_line`](#voxy.pronounce.checks_by_line)(risks, \*, readings[, expected])    | [`checks_from_risks()`](#voxy.pronounce.checks_from_risks), grouped by the line each risk is on.                                                                                   |
| [`checks_from_risks`](#voxy.pronounce.checks_from_risks)(risks, \*, readings[, expected]) | `verify_words` checks for the flagged words that have known IPA readings.                                                                                                                    |
| [`ctc_log_likelihood`](#voxy.pronounce.ctc_log_likelihood)(log_probs, target, \*[, blank]) | `log P(target | log_probs)` under CTC (the forward algorithm, in log space).                                                                                                                 |
| [`espeak_phonemizer`](#voxy.pronounce.espeak_phonemizer)(\*[, model, threads, rootdir])   | The default [`Phonemizer`](#voxy.pronounce.Phonemizer): espeak-ng (bundled by `espeakng-loader`), run in the recogniser's env, so its IPA is the recogniser's own alphabet. |
| [`failing`](#voxy.pronounce.failing)(verdicts)                                  | The verdicts whose word was decidedly read the wrong way (`ok is False`).                                                                                                                    |
| [`heteronym_readings`](#voxy.pronounce.heteronym_readings)([language])                     | voxy's own heteronym table for `language`: `{word: {reading name: IPA}}`.                                                                                                                    |
| [`known_symbols`](#voxy.pronounce.known_symbols)(ipa, vocab)                          | The symbols of `ipa` the recogniser knows, longest first; stress marks, spaces and anything else it never emits are dropped.                                                                 |
| [`load_readings`](#voxy.pronounce.load_readings)(path)                                | Heteronym readings `{word: {reading name: IPA}}` from a caller's file.                                                                                                                       |
| [`log_probs_from_path`](#voxy.pronounce.log_probs_from_path)(path, \*, vocab_size[, ...])   | Log-probabilities of a recogniser that hears `path` (one id per frame).                                                                                                                      |
| [`phoneme_emitter`](#voxy.pronounce.phoneme_emitter)(\*[, model, threads, rootdir])     | The default [`Emitter`](#voxy.pronounce.Emitter): the wav2vec2 phoneme recogniser in its own env.                                                                        |
| [`phrase_readings`](#voxy.pronounce.phrase_readings)(log_probs, vocab, readings, \*)    | Each reading's CTC log-likelihood as the whole phrase `before + reading + after`.                                                                                                            |
| [`reading_name`](#voxy.pronounce.reading_name)(name)                                 | The reading a name stands for: `"verb~liaison"` is a variant of `"verb"`.                                                                                                                    |
| [`render_verified`](#voxy.pronounce.render_verified)(lines, checks, \*, render, ...)    | Render each line, and re-render the ones whose flagged words came out wrong.                                                                                                                 |
| [`score_readings`](#voxy.pronounce.score_readings)(log_probs, vocab, readings, \*)     | Each reading's CTC log-likelihood per target symbol (higher is better).                                                                                                                      |
| [`speech_renderer`](#voxy.pronounce.speech_renderer)(model[, voice, outdir, ...])       | A `render` for [`render_verified()`](#voxy.pronounce.render_verified) over a [`voxy.base.SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel).       |
| [`spot_log_likelihood`](#voxy.pronounce.spot_log_likelihood)(log_probs, target, \*[, ...])  | `log P` of `target` said somewhere in `log_probs`, the rest being filler.                                                                                                                    |
| [`spot_readings`](#voxy.pronounce.spot_readings)(log_probs, vocab, readings, \*)      | Each reading's [`spot_log_likelihood()`](#voxy.pronounce.spot_log_likelihood), less that of filler alone (higher is better).                                                         |
| [`timed_words`](#voxy.pronounce.timed_words)(segments)                              | The timed words of ASR segments (`voxy.refprep.transcribe`'s output).                                                                                                                        |
| [`tokenize_phonemes`](#voxy.pronounce.tokenize_phonemes)(reading, vocab)                  | Vocabulary ids of an IPA `reading`, longest symbol first; spaces are ignored.                                                                                                                |
| [`verify_words`](#voxy.pronounce.verify_words)(audio, checks, \*, words, emitter)    | Verdicts for each `(word, readings, expected)` at each place ASR found it.                                                                                                                   |
| [`word_spans`](#voxy.pronounce.word_spans)(words, word)                            | `(start, end)` of every ASR word matching `word` (case and punctuation aside).                                                                                                               |

### Classes

| [`Finding`](#voxy.pronounce.Finding)(risk[, verdict])                 | One flagged word of a render: the risk, and what the audio said about it.                                                               |
|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [`LineTakes`](#voxy.pronounce.LineTakes)(line, text[, takes, checks])   | Every take of one line (1-based `line`, as [`voxy.speechtext.Risk`](voxy.speechtext.html.md#voxy.speechtext.Risk)). |
| [`Take`](#voxy.pronounce.Take)(attempt, audio[, verdicts])         | One render of a line and what the audio says about its flagged words.                                                                   |
| [`Verdict`](#voxy.pronounce.Verdict)(word, start, end, scores[, ...]) | What the audio says about one flagged word.                                                                                             |

### voxy.pronounce.BLANK_SYMBOL *= '<pad>'*

The CTC blank of the recogniser’s vocabulary (wav2vec2 uses its pad token).

### voxy.pronounce.Check

one word to check.

* **Type:**
  `(word, {reading name: IPA}, expected reading or None)`

alias of [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], str | None]

### voxy.pronounce.DFLT_CONTEXT_MARGIN *= 2.0*

A reading must beat the runner-up by this many nats when scored as a whole phrase.

### voxy.pronounce.DFLT_CONTEXT_WORDS *= 1*

Words of context on each side of a flagged word when [`audit()`](#voxy.pronounce.audit) scores it.

### voxy.pronounce.DFLT_CROP_PADDING *= 0.08*

ASR word edges are loose.

* **Type:**
  Seconds kept on each side of a word’s ASR timestamps

### voxy.pronounce.DFLT_FILLER_PENALTY *= 1.0*

What spotting a word in its context charges per frame left to filler (nats).

### voxy.pronounce.DFLT_MARGIN *= 0.5*

A reading must beat the runner-up by this much (mean log-likelihood per target
symbol) for a verdict to count as decided.

### voxy.pronounce.DFLT_MAX_ATTEMPTS *= 3*

Renders of one line at most (the first take included).

### voxy.pronounce.DFLT_PHONEME_MODEL *= 'facebook/wav2vec2-lv-60-espeak-cv-ft'*

wav2vec2 fine-tuned to emit espeak IPA (CTC, 16 kHz).

* **Type:**
  The phoneme recogniser

### voxy.pronounce.DFLT_SPOT_MARGIN *= 2.0*

A reading must beat the runner-up by this many nats when spotted in context.

### voxy.pronounce.ESPEAK_LANGUAGES *= {'en': 'en-us', 'fr': 'fr-fr'}*

espeak voices for [`audit()`](#voxy.pronounce.audit)’s `language` codes (others are passed as they are).

### voxy.pronounce.Emitter

`(audio, start, end) -> (log_probs [frames x vocab], vocab)`.

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]

### *class* voxy.pronounce.Finding(risk, verdict=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One flagged word of a render: the risk, and what the audio said about it.

`verdict` is None when the word could not be located in the audio (ASR heard it
a different number of times than the text has it): listen to that one.

#### *property* ok *: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None)*

As [`Verdict.ok`](#voxy.pronounce.Verdict.ok); None when unchecked or undecided.

### voxy.pronounce.HETERONYMS_DIR *= PosixPath('/home/runner/work/voxy/voxy/voxy/data/heteronyms')*

The heteronym tables voxy ships, one `<language>.json` per language.

### *class* voxy.pronounce.LineTakes(line, text, takes=tuple(), checks=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Every take of one line (1-based `line`, as [`voxy.speechtext.Risk`](voxy.speechtext.html.md#voxy.speechtext.Risk)).

A blank line has no take.

#### *property* chosen *: [Take](#voxy.pronounce.Take) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The first take with no failing word, else the one with the fewest.

#### *property* unconfirmed *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

Checked words with an expected reading that the chosen take did not confirm.

A word is confirmed when the audio decidedly gave the expected reading at least
once. It stays unconfirmed when ASR never found it, or when the recogniser
could not decide.

#### *property* verified *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

No checked word misread, and every one with an expected reading confirmed.

### voxy.pronounce.PHONEME_ENV *= EnvSpec(name='phonemes', requirements=('torch', 'transformers', 'soundfile', 'numpy', 'librosa', 'phonemizer', 'espeakng-loader'), python='3.11', torch_backend='cpu', env_vars={})*

CPU torch plus transformers (the processor’s vocab.json is all it needs).

* **Type:**
  Its env

### voxy.pronounce.Phonemizer

writes out the words around a flagged one.

* **Type:**
  `(texts, language) -> IPA of each`

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### *class* voxy.pronounce.Take(attempt, audio, verdicts=())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One render of a line and what the audio says about its flagged words.

#### *property* failing *: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[Verdict](#voxy.pronounce.Verdict)]*

The verdicts of this take whose word was decidedly misread.

### voxy.pronounce.VARIANT_SEPARATOR *= '~'*

In a reading name, what separates a reading from one of its variants (`verb~liaison`).

### *class* voxy.pronounce.Verdict(word, start, end, scores, expected=None, margin=0.5)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What the audio says about one flagged word.

#### *property* decided *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the best reading beats the runner-up reading by at least `margin`.

Variants of one reading do not compete with each other.

#### *property* heard *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The reading (variant included) the audio supports best.

#### *property* ok *: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None)*

True/False when an `expected` reading was given and the audio decided.

#### *property* reading *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The reading the audio supports best, variants folded in (`verb~e` is `verb`).

#### *property* reading_scores *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [float](https://docs.python.org/3/builtins/functions.html#float)]*

`scores` with each reading’s variants (`name~variant`) merged by their best.

### voxy.pronounce.align_words(text, words)

`{index of a word of text: index of the ASR word heard for it}`.

Words are matched by aligning the two sequences (`difflib`), so a word ASR missed,
merged or added elsewhere does not shift the others; a text word with no exact
counterpart is left out.

```pycon
>>> align_words("Il est là", [{"text": " Il"}, {"text": " et"}, {"text": " est"}, {"text": " là."}])
{0: 0, 1: 2, 2: 3}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### voxy.pronounce.audit(text, audio, \*, readings=None, language='fr', expected=None, detectors=(), emitter=None, words_of=<function \_transcribed_words>, scoring='phrase', phonemize=None, context=1, filler_penalty=1.0, padding=0.08, margin=None)

Which reading did each heteronym of `text` get in the finished `audio`?

ASR (WER) checks cannot answer this: Whisper writes “est” whether the model said
/ɛ/ or /ɛst/. This flags the words of `text` that have several readings
([`voxy.speechtext.flag_risks()`](voxy.speechtext.html.md#voxy.speechtext.flag_risks)), finds each in the audio (ASR word times; the
n-th time the text has a word is the n-th time ASR heard it) and scores its
readings with the phoneme recogniser.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – What the model was given (the speech text, not the display text).
  * **audio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)) – The render.
  * **readings** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `{word: {reading name: IPA}}`; default [`heteronym_readings()`](#voxy.pronounce.heteronym_readings)
    for `language`.
  * **expected** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`object`](https://docs.python.org/3/builtins/functions.html#object)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The reading each word should have: `{word: reading}`, or
    `risk -> reading` to decide per occurrence (`risk.line`, `risk.word`).
    Otherwise what a `detectors` entry found for that word on that line
    (e.g. [`voxy.speechtext.pos_detector()`](voxy.speechtext.html.md#voxy.speechtext.pos_detector)), else None (just report).
  * **detectors** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)]]) – `text -> risks` proposers of expected readings.
  * **emitter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The phoneme recogniser; default [`phoneme_emitter()`](#voxy.pronounce.phoneme_emitter).
  * **words_of** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]]) – `audio -> timed words`; default ASR (`voxy.refprep.transcribe`).
  * **scoring** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – How a reading is scored in the audio. `"phrase"` (default): the
    `context` words on each side are written out with `phonemize` and each
    reading is scored as the whole phrase ([`phrase_readings()`](#voxy.pronounce.phrase_readings)); the
    neighbours are the same in every candidate, so only the word decides.
    `"spot"`: the reading is spotted in that window, the rest left to filler
    ([`spot_readings()`](#voxy.pronounce.spot_readings); no phonemizer, but a neighbour’s /s/ or /t/ can be
    credited to the longer reading). `"crop"`: the word alone, cut at its ASR
    times (`padding` each side), scored by [`score_readings()`](#voxy.pronounce.score_readings); ASR word
    times are often too loose for this.
  * **phonemize** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – For `"phrase"`; default [`espeak_phonemizer()`](#voxy.pronounce.espeak_phonemizer).
  * **context** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Words kept on each side of the flagged one.
  * **margin** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – See [`Verdict.decided`](#voxy.pronounce.Verdict.decided); default [`DFLT_CONTEXT_MARGIN`](#voxy.pronounce.DFLT_CONTEXT_MARGIN),
    [`DFLT_SPOT_MARGIN`](#voxy.pronounce.DFLT_SPOT_MARGIN) or [`DFLT_MARGIN`](#voxy.pronounce.DFLT_MARGIN) by `scoring`.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Finding`](#voxy.pronounce.Finding)]

Returns the findings in text order; `[f for f in findings if f.ok is False]` are
the words decidedly misread (re-render those), `f.ok is None` ones were not
decided or not found.

### voxy.pronounce.best_reading(log_probs, vocab, readings, , blank=None)

The name of the reading `log_probs` support best.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.pronounce.checks_by_line(risks, , readings, expected=None)

[`checks_from_risks()`](#voxy.pronounce.checks_from_risks), grouped by the line each risk is on.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]]

### voxy.pronounce.checks_from_risks(risks, , readings, expected=None)

`verify_words` checks for the flagged words that have known IPA readings.

`readings` maps a word (case-insensitively) to `{name: IPA}`, e.g. from a
Lexique-derived heteronym list the caller supplies; `expected` names the
reading each word should have, and otherwise the risks’ own `expected` (what a
detector such as `voxy.speechtext.pos_detector` found) is used. Words with no
entry are skipped. Each word is checked once: `verify_words` finds every place
it is spoken, so a word whose risks name *different* readings gets none here
(group the risks by line with [`checks_by_line()`](#voxy.pronounce.checks_by_line) to check each one).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]

### voxy.pronounce.ctc_log_likelihood(log_probs, target, , blank=0)

`log P(target | log_probs)` under CTC (the forward algorithm, in log space).

`log_probs` is `frames x vocab` (log-softmax). `-inf` when the target cannot
be aligned (more symbols than frames allow).

```pycon
>>> lp = log_probs_from_path([1, 2], vocab_size=3)
>>> ctc_log_likelihood(lp, [1, 2]) > ctc_log_likelihood(lp, [2, 1])
True
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### voxy.pronounce.espeak_phonemizer(, model='facebook/wav2vec2-lv-60-espeak-cv-ft', threads=None, rootdir=None)

The default [`Phonemizer`](#voxy.pronounce.Phonemizer): espeak-ng (bundled by `espeakng-loader`), run in
the recogniser’s env, so its IPA is the recogniser’s own alphabet.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### voxy.pronounce.failing(verdicts)

The verdicts whose word was decidedly read the wrong way (`ok is False`).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Verdict`](#voxy.pronounce.Verdict)]

### voxy.pronounce.heteronym_readings(language='fr')

voxy’s own heteronym table for `language`: `{word: {reading name: IPA}}`.

A short, hand-made list (French: *est*, *fils*, *plus*, *tous*, the  *-ent* verbs…);
merge a fuller one (e.g. derived from Lexique) read with [`load_readings()`](#voxy.pronounce.load_readings).

```pycon
>>> heteronym_readings("fr")["est"]["noun"]
'ɛst'
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### voxy.pronounce.known_symbols(ipa, vocab)

The symbols of `ipa` the recogniser knows, longest first; stress marks, spaces
and anything else it never emits are dropped.

```pycon
>>> known_symbols("ˈtʁist ɛ", ["<pad>", "t", "ʁ", "i", "s", "ɛ"])
['t', 'ʁ', 'i', 's', 't', 'ɛ']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.pronounce.load_readings(path)

Heteronym readings `{word: {reading name: IPA}}` from a caller’s file.

A `.json` file holds that mapping; any other file is tab-separated
`word<TAB>reading name<TAB>IPA`, one reading per line (`#` lines and blank
lines skipped), e.g. a list derived from Lexique, which voxy does not ship.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### voxy.pronounce.log_probs_from_path(path, , vocab_size, confidence=0.9)

Log-probabilities of a recogniser that hears `path` (one id per frame).

A test and doctest helper: each frame gives `confidence` to its id and shares
the rest evenly.

```pycon
>>> log_probs_from_path([0], vocab_size=2).round(2).tolist()
[[-0.11, -2.3]]
```

* **Return type:**
  `ndarray`

### voxy.pronounce.phoneme_emitter(, model='facebook/wav2vec2-lv-60-espeak-cv-ft', threads=None, rootdir=None)

The default [`Emitter`](#voxy.pronounce.Emitter): the wav2vec2 phoneme recogniser in its own env.

The env (torch, transformers) is built on first use under the data root, and the
model weights download from the Hugging Face hub then.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]

### voxy.pronounce.phrase_readings(log_probs, vocab, readings, , before='', after='', blank=None)

Each reading’s CTC log-likelihood as the whole phrase `before + reading + after`.

The words around the flagged one are written out (IPA) and are the same in every
candidate, so the comparison isolates the word: its neighbours’ sounds cannot be
credited to a longer reading, as they are when the word is cropped or spotted alone.

```pycon
>>> vocab = ["<pad>", "ɛ", "s", "t", "l", "ə"]
>>> lp = log_probs_from_path([2, 3, 1, 4, 5], vocab_size=6)  # "(trist)e" s t ɛ l ə
>>> s = phrase_readings(lp, vocab, {"verb": "ɛ", "noun": "ɛst"}, before="st", after="lə")
>>> s["verb"] > s["noun"]
True
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### voxy.pronounce.reading_name(name)

The reading a name stands for: `"verb~liaison"` is a variant of `"verb"`.

```pycon
>>> reading_name("verb~liaison"), reading_name("noun")
('verb', 'noun')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.pronounce.render_verified(lines, checks, \*, render, emitter, words_of=<function \_transcribed_words>, max_attempts=3, padding=0.08, margin=0.5)

Render each line, and re-render the ones whose flagged words came out wrong.

A line with checks is verified ([`verify_words()`](#voxy.pronounce.verify_words)) after each render and
rendered again while a word is decidedly misread ([`failing()`](#voxy.pronounce.failing)), up to
`max_attempts` renders; a line without checks is rendered once, and a blank
line not at all (line numbers still count it, as `flag_risks` does).
[`LineTakes.chosen`](#voxy.pronounce.LineTakes.chosen) is the take to keep; [`LineTakes.verified`](#voxy.pronounce.LineTakes.verified) also
requires each expected reading to be confirmed (a word ASR did not find, or the
recogniser could not decide, leaves the line unverified rather than retried).

* **Parameters:**
  * **lines** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The lines to speak, in order (line numbers start at 1).
  * **checks** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]]) – By line number, the words to check ([`checks_by_line()`](#voxy.pronounce.checks_by_line)).
  * **render** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)]) – `(text, line, attempt) -> audio path`; each call should give a
    new take (a new seed, or a service’s own variation).
    [`speech_renderer()`](#voxy.pronounce.speech_renderer) adapts a [`voxy.base.SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel).
  * **emitter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – The phoneme recogniser ([`phoneme_emitter()`](#voxy.pronounce.phoneme_emitter)).
  * **words_of** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]]) – `audio -> timed words`; the default transcribes with
    `voxy.refprep.transcribe`.
  * **max_attempts** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Renders of one line at most.
  * **padding** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – As in [`verify_words()`](#voxy.pronounce.verify_words).
  * **margin** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – As in [`verify_words()`](#voxy.pronounce.verify_words).
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`LineTakes`](#voxy.pronounce.LineTakes)]

```pycon
>>> heard = {"est": "ɛ"}  # a fake recogniser: what each word sounds like
>>> def emitter(audio, start, end):
...     vocab = ["<pad>", "ɛ", "s", "t"]
...     path = [vocab.index(s) for s in heard["est"]]
...     return log_probs_from_path(path, vocab_size=4), vocab
>>> def render(text, line, attempt):
...     heard["est"] = "ɛst" if attempt else "ɛ"  # the second take is right
...     return f"take-{line}-{attempt}.wav"
>>> words = lambda audio: [{"start": 0.0, "end": 0.3, "text": "Est"}]
>>> checks = {1: [("Est", {"verb": "ɛ", "noun": "ɛst"}, "noun")]}
>>> [result] = render_verified(["L'Est"], checks, render=render, emitter=emitter,
...                            words_of=words)
>>> [t.audio for t in result.takes], result.chosen.attempt, result.verified
(['take-1-0.wav', 'take-1-1.wav'], 1, True)
```

### voxy.pronounce.score_readings(log_probs, vocab, readings, , blank=None)

Each reading’s CTC log-likelihood per target symbol (higher is better).

Normalising by length keeps a longer reading (“ɛst”) from losing to a shorter one
(“ɛ”) just for having more symbols to pay for. `blank` defaults to the id of
[`BLANK_SYMBOL`](#voxy.pronounce.BLANK_SYMBOL) in `vocab` (0 if it has none).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### voxy.pronounce.speech_renderer(model, voice=None, , outdir=None, take_kwarg=None, \*\*synthesize_kwargs)

A `render` for [`render_verified()`](#voxy.pronounce.render_verified) over a [`voxy.base.SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel).

Each take is saved as `<outdir>/<line>-<attempt>.<format>` (default outdir: a
new folder per renderer under `voxy_cache_dir("takes")`, so runs never
overwrite each other’s takes). With `take_kwarg` (e.g. `"seed"`), the attempt
number is passed to `model.synthesize` under that name, so a seeded model gives
a new take; without it, the model is simply called again.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.pronounce.spot_log_likelihood(log_probs, target, , blank=0, filler_penalty=None)

`log P` of `target` said somewhere in `log_probs`, the rest being filler.

Keyword spotting under CTC: frames before and after the target are explained by a
filler that takes each frame’s best symbol at a cost of `filler_penalty` nats.
Scoring a word inside a window that holds its neighbours, a reading wins by
matching phonemes that are really there, not by the window’s edges: a crop cut
from loose ASR word times otherwise favours the longer reading (/ɛst/ over /ɛ/).

```pycon
>>> vocab = ["<pad>", "ɛ", "s", "t", "l"]
>>> lp = log_probs_from_path([4, 1, 0, 4], vocab_size=5)  # "l ɛ l": the verb
>>> verb, noun = (spot_log_likelihood(lp, tokenize_phonemes(r, vocab)) for r in ("ɛ", "ɛst"))
>>> verb > noun
True
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### voxy.pronounce.spot_readings(log_probs, vocab, readings, , blank=None, filler_penalty=None)

Each reading’s [`spot_log_likelihood()`](#voxy.pronounce.spot_log_likelihood), less that of filler alone (higher is better).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### voxy.pronounce.timed_words(segments)

The timed words of ASR segments (`voxy.refprep.transcribe`’s output).

Words without times are dropped; a segment with no timed word counts as one word.

```pycon
>>> timed_words([{"words": [{"start": 0.0, "end": 0.2, "text": "a"},
...                         {"start": None, "end": None, "text": "b"}]},
...              {"start": 1.0, "end": 2.0, "text": "c", "words": []}])
[{'start': 0.0, 'end': 0.2, 'text': 'a'}, {'start': 1.0, 'end': 2.0, 'text': 'c'}]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]

### voxy.pronounce.tokenize_phonemes(reading, vocab)

Vocabulary ids of an IPA `reading`, longest symbol first; spaces are ignored.

A reading may also be given as a sequence of symbols. Raises `ValueError` on a
symbol the recogniser does not know, since it could never be heard.

```pycon
>>> tokenize_phonemes("ɛ̃s", ["<pad>", "ɛ", "ɛ̃", "s"])
[2, 3]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]

### voxy.pronounce.verify_words(audio, checks, , words, emitter, padding=0.08, margin=0.5)

Verdicts for each `(word, readings, expected)` at each place ASR found it.

* **Parameters:**
  * **audio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)) – The render.
  * **checks** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – The words to check, each with its candidate readings
    (`{name: IPA}`) and the reading it should have (or None to just
    report). [`checks_from_risks()`](#voxy.pronounce.checks_from_risks) builds them from `flag_risks`.
  * **words** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – Timed words of the render (`transcribe(audio)`’s segments’ words).
  * **emitter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – `(audio, start, end) -> (log_probs, vocab)`; the default worker
    is [`phoneme_emitter()`](#voxy.pronounce.phoneme_emitter).
  * **padding** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Seconds added to each side of the ASR span.
  * **margin** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – See [`Verdict.decided`](#voxy.pronounce.Verdict.decided).
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Verdict`](#voxy.pronounce.Verdict)]

### voxy.pronounce.word_spans(words, word)

`(start, end)` of every ASR word matching `word` (case and punctuation aside).

`words` are timed words as `voxy.refprep.transcribe` gives them
(`{"start", "end", "text", ...}`).

```pycon
>>> word_spans([{"start": 0.1, "end": 0.3, "text": " Est,"}], "est")
[(0.1, 0.3)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
