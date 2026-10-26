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
finished render), [`verify_words()`](#voxy.pronounce.verify_words) and [`render_verified()`](#voxy.pronounce.render_verified) (the calls while
rendering) therefore score each reading as the whole phrase around it by default, the
neighbours written out by espeak ([`phrase_readings()`](#voxy.pronounce.phrase_readings), [`espeak_phonemizer()`](#voxy.pronounce.espeak_phonemizer)).
They share one seam, a [`Scorer`](#voxy.pronounce.Scorer): [`phrase_scorer()`](#voxy.pronounce.phrase_scorer), [`spot_scorer()`](#voxy.pronounce.spot_scorer) and
[`crop_scorer()`](#voxy.pronounce.crop_scorer) make one, each carrying the margin it is calibrated for, and
[`reading_scorer()`](#voxy.pronounce.reading_scorer) makes one by name ([`SCORERS`](#voxy.pronounce.SCORERS)).

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

| [`DFLT_PHONEME_MODEL`](#voxy.pronounce.DFLT_PHONEME_MODEL)   | wav2vec2 fine-tuned to emit espeak IPA (CTC, 16 kHz).                                                                                                                                                                                                   |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`PHONEME_ENV`](#voxy.pronounce.PHONEME_ENV)          | CPU torch plus transformers (the processor's vocab.json is all it needs).                                                                                                                                                                               |
| [`DFLT_CROP_PADDING`](#voxy.pronounce.DFLT_CROP_PADDING)    | ASR word edges are loose.                                                                                                                                                                                                                               |
| [`DFLT_MARGIN`](#voxy.pronounce.DFLT_MARGIN)          | The margin of [`crop_scorer()`](#voxy.pronounce.crop_scorer) (mean log-likelihood per target symbol).                                                                                                                                   |
| [`DFLT_CONTEXT_MARGIN`](#voxy.pronounce.DFLT_CONTEXT_MARGIN)  | The margin of [`phrase_scorer()`](#voxy.pronounce.phrase_scorer) (total nats).                                                                                                                                                            |
| [`DFLT_FILLER_PENALTY`](#voxy.pronounce.DFLT_FILLER_PENALTY)  | What spotting a word in its context charges per frame left to filler (nats).                                                                                                                                                                            |
| [`DFLT_LANGUAGE`](#voxy.pronounce.DFLT_LANGUAGE)        | The language of the texts checked, unless told otherwise.                                                                                                                                                                                               |
| [`ESPEAK_LANGUAGES`](#voxy.pronounce.ESPEAK_LANGUAGES)     | espeak voices for [`audit()`](#voxy.pronounce.audit)'s `language` codes (others are passed as they are).                                                                                                                          |
| [`Phonemizer`](#voxy.pronounce.Phonemizer)           | writes out the words around a flagged one.                                                                                                                                                                                                              |
| [`DFLT_CONTEXT_WORDS`](#voxy.pronounce.DFLT_CONTEXT_WORDS)   | Words of context on each side of a flagged word when [`audit()`](#voxy.pronounce.audit) scores it.                                                                                                                                |
| [`DFLT_SPOT_MARGIN`](#voxy.pronounce.DFLT_SPOT_MARGIN)     | The margin of [`spot_scorer()`](#voxy.pronounce.spot_scorer) (total nats).                                                                                                                                                              |
| [`DFLT_SCORING`](#voxy.pronounce.DFLT_SCORING)         | How readings are scored unless told otherwise (a name in [`SCORERS`](#voxy.pronounce.SCORERS)).                                                                                                                                     |
| [`VARIANT_SEPARATOR`](#voxy.pronounce.VARIANT_SEPARATOR)    | In a reading name, what separates a reading from one of its variants (`verb~liaison`).                                                                                                                                                                  |
| [`EDGE_PUNCTUATION`](#voxy.pronounce.EDGE_PUNCTUATION)     | Punctuation stripped from the edges of an ASR or text word before it is matched or spoken.                                                                                                                                                              |
| [`BLANK_SYMBOL`](#voxy.pronounce.BLANK_SYMBOL)         | The CTC blank of the recogniser's vocabulary (wav2vec2 uses its pad token).                                                                                                                                                                             |
| [`Emitter`](#voxy.pronounce.Emitter)              | `(audio, start, end) -> (log_probs [frames x vocab], vocab)`.                                                                                                                                                                                           |
| [`SCORERS`](#voxy.pronounce.SCORERS)              | The ways a reading can be scored in the audio, by name (see [`reading_scorer()`](#voxy.pronounce.reading_scorer)); each scorer carries the margin it is calibrated for (see [`DFLT_MARGIN`](#voxy.pronounce.DFLT_MARGIN)). |
| [`DFLT_MAX_ATTEMPTS`](#voxy.pronounce.DFLT_MAX_ATTEMPTS)    | Renders of one line at most (the first take included).                                                                                                                                                                                                  |
| [`Check`](#voxy.pronounce.Check)                | one word to check.                                                                                                                                                                                                                                      |
| [`HETERONYMS_DIR`](#voxy.pronounce.HETERONYMS_DIR)       | The heteronym tables voxy ships, one `<language>.json` per language.                                                                                                                                                                                    |

### Functions

| [`align_words`](#voxy.pronounce.align_words)(text, words, \*[, elisions])           | `{index of a word of text: index of the ASR word heard for it}`.                                                                                                                                                                     |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`audit`](#voxy.pronounce.audit)(text, audio, \*[, readings, language, ...])  | Which reading did each heteronym of `text` get in the finished `audio`?                                                                                                                                                              |
| [`best_reading`](#voxy.pronounce.best_reading)(log_probs, vocab, readings, \*)       | The name of the reading `log_probs` support best.                                                                                                                                                                                    |
| [`checks_by_line`](#voxy.pronounce.checks_by_line)(risks, \*, readings[, expected])    | [`checks_from_risks()`](#voxy.pronounce.checks_from_risks), grouped by the line each risk is on.                                                                                                                           |
| [`checks_from_risks`](#voxy.pronounce.checks_from_risks)(risks, \*, readings[, expected]) | `verify_words` checks for the flagged words that have known IPA readings.                                                                                                                                                            |
| [`crop_scorer`](#voxy.pronounce.crop_scorer)(\*, emitter[, padding, margin])        | The word alone, cut at its ASR times (`padding` each side), scored by [`score_readings()`](#voxy.pronounce.score_readings); ASR word times are often too loose for this.                                                |
| [`ctc_log_likelihood`](#voxy.pronounce.ctc_log_likelihood)(log_probs, target, \*[, blank]) | `log P(target | log_probs)` under CTC (the forward algorithm, in log space).                                                                                                                                                         |
| [`espeak_phonemizer`](#voxy.pronounce.espeak_phonemizer)(\*[, model, threads, rootdir])   | The default [`Phonemizer`](#voxy.pronounce.Phonemizer): espeak-ng (bundled by `espeakng-loader`), run in the recogniser's env, so its IPA is the recogniser's own alphabet.                                         |
| [`failing`](#voxy.pronounce.failing)(verdicts)                                  | The verdicts whose word was decidedly read the wrong way (`ok is False`).                                                                                                                                                            |
| [`heteronym_readings`](#voxy.pronounce.heteronym_readings)([language])                     | voxy's own heteronym table for `language`: `{word: {reading name: IPA}}`.                                                                                                                                                            |
| [`known_symbols`](#voxy.pronounce.known_symbols)(ipa, vocab)                          | The symbols of `ipa` the recogniser knows, longest first; stress marks, spaces and anything else it never emits are dropped.                                                                                                         |
| [`load_readings`](#voxy.pronounce.load_readings)(path)                                | Heteronym readings `{word: {reading name: IPA}}` from a caller's file.                                                                                                                                                               |
| [`log_probs_from_path`](#voxy.pronounce.log_probs_from_path)(path, \*, vocab_size[, ...])   | Log-probabilities of a recogniser that hears `path` (one id per frame).                                                                                                                                                              |
| [`phoneme_emitter`](#voxy.pronounce.phoneme_emitter)(\*[, model, threads, rootdir])     | The default [`Emitter`](#voxy.pronounce.Emitter): the wav2vec2 phoneme recogniser in its own env.                                                                                                                |
| [`phrase_readings`](#voxy.pronounce.phrase_readings)(log_probs, vocab, readings, \*)    | Each reading's CTC log-likelihood as the whole phrase `before + reading + after`.                                                                                                                                                    |
| [`phrase_scorer`](#voxy.pronounce.phrase_scorer)(\*, emitter[, phonemize, ...])       | Each reading as the whole phrase around the word ([`phrase_readings()`](#voxy.pronounce.phrase_readings)), the `context` words on each side written out by `phonemize` in `language`.                                    |
| [`reading_name`](#voxy.pronounce.reading_name)(name)                                 | The reading a name stands for: `"verb~liaison"` is a variant of `"verb"`.                                                                                                                                                            |
| [`reading_scorer`](#voxy.pronounce.reading_scorer)([scoring])                          | The [`Scorer`](#voxy.pronounce.Scorer) named `scoring` in [`SCORERS`](#voxy.pronounce.SCORERS), made with `options`.                                                                        |
| [`render_verified`](#voxy.pronounce.render_verified)(lines, checks, \*, render[, ...])  | Render each line, and re-render the ones whose flagged words came out wrong.                                                                                                                                                         |
| [`score_readings`](#voxy.pronounce.score_readings)(log_probs, vocab, readings, \*)     | Each reading's CTC log-likelihood per target symbol (higher is better).                                                                                                                                                              |
| [`speech_renderer`](#voxy.pronounce.speech_renderer)(model[, voice, outdir, ...])       | A `render` for [`render_verified()`](#voxy.pronounce.render_verified) over a [`voxy.base.SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel).                                               |
| [`split_elisions`](#voxy.pronounce.split_elisions)(words, \*[, elisions])              | Timed words split at each elision of `elisions`, as `_text_words()` splits the text, so [`align_words()`](#voxy.pronounce.align_words) can match them.                                                               |
| [`spoken_context`](#voxy.pronounce.spoken_context)(words, \*[, elisions])              | `(text, tail)`: what `words` say, for a phonemizer, and the IPA of an elided clitic they end with.                                                                                                                                   |
| [`spot_log_likelihood`](#voxy.pronounce.spot_log_likelihood)(log_probs, target, \*[, ...])  | `log P` of `target` said somewhere in `log_probs`, the rest being filler.                                                                                                                                                            |
| [`spot_readings`](#voxy.pronounce.spot_readings)(log_probs, vocab, readings, \*)      | Each reading's [`spot_log_likelihood()`](#voxy.pronounce.spot_log_likelihood), less that of filler alone (higher is better).                                                                                                 |
| [`spot_scorer`](#voxy.pronounce.spot_scorer)(\*, emitter[, context, padding, ...])  | The reading spotted in the window of `context` words on each side, the rest left to filler ([`spot_readings()`](#voxy.pronounce.spot_readings)); a neighbour's /s/ or /t/ can still be credited to the longer reading. |
| [`timed_words`](#voxy.pronounce.timed_words)(segments)                              | The timed words of ASR segments (`voxy.refprep.transcribe`'s output).                                                                                                                                                                |
| [`tokenize_phonemes`](#voxy.pronounce.tokenize_phonemes)(reading, vocab)                  | Vocabulary ids of an IPA `reading`, longest symbol first; spaces are ignored.                                                                                                                                                        |
| [`verify_words`](#voxy.pronounce.verify_words)(audio, checks, \*, words[, ...])      | Verdicts for each `(word, readings, expected)` at each place ASR found it.                                                                                                                                                           |
| [`word_indices`](#voxy.pronounce.word_indices)(words, word)                          | Indices of the ASR words matching `word` (case and punctuation aside).                                                                                                                                                               |
| [`word_spans`](#voxy.pronounce.word_spans)(words, word)                            | `(start, end)` of every ASR word matching `word` (case and punctuation aside).                                                                                                                                                       |

### Classes

| [`Finding`](#voxy.pronounce.Finding)(risk[, verdict])                 | One flagged word of a render: the risk, and what the audio said about it.                                                               |
|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [`LineTakes`](#voxy.pronounce.LineTakes)(line, text[, takes, checks])   | Every take of one line (1-based `line`, as [`voxy.speechtext.Risk`](voxy.speechtext.html.md#voxy.speechtext.Risk)). |
| [`Scorer`](#voxy.pronounce.Scorer)(\*args, \*\*kwargs)               | Scores the readings of timed ASR word `words[i]` in `audio` (higher is better).                                                         |
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

The margin of [`phrase_scorer()`](#voxy.pronounce.phrase_scorer) (total nats).

### voxy.pronounce.DFLT_CONTEXT_WORDS *= 1*

Words of context on each side of a flagged word when [`audit()`](#voxy.pronounce.audit) scores it.

### voxy.pronounce.DFLT_CROP_PADDING *= 0.08*

ASR word edges are loose.

* **Type:**
  Seconds kept on each side of a word’s ASR timestamps

### voxy.pronounce.DFLT_FILLER_PENALTY *= 1.0*

What spotting a word in its context charges per frame left to filler (nats).

### voxy.pronounce.DFLT_LANGUAGE *= 'fr'*

The language of the texts checked, unless told otherwise.

### voxy.pronounce.DFLT_MARGIN *= 0.5*

The margin of [`crop_scorer()`](#voxy.pronounce.crop_scorer) (mean log-likelihood per target symbol).

### voxy.pronounce.DFLT_MAX_ATTEMPTS *= 3*

Renders of one line at most (the first take included).

### voxy.pronounce.DFLT_PHONEME_MODEL *= 'facebook/wav2vec2-lv-60-espeak-cv-ft'*

wav2vec2 fine-tuned to emit espeak IPA (CTC, 16 kHz).

* **Type:**
  The phoneme recogniser

### voxy.pronounce.DFLT_SCORING *= 'phrase'*

How readings are scored unless told otherwise (a name in [`SCORERS`](#voxy.pronounce.SCORERS)).

### voxy.pronounce.DFLT_SPOT_MARGIN *= 2.0*

The margin of [`spot_scorer()`](#voxy.pronounce.spot_scorer) (total nats).

### voxy.pronounce.EDGE_PUNCTUATION *= '.,;:!?«»"()…“”'*

Punctuation stripped from the edges of an ASR or text word before it is matched or spoken.

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

### voxy.pronounce.SCORERS *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[..., [Scorer](#voxy.pronounce.Scorer)]]* *= {'crop': <function crop_scorer>, 'phrase': <function phrase_scorer>, 'spot': <function spot_scorer>}*

The ways a reading can be scored in the audio, by name (see [`reading_scorer()`](#voxy.pronounce.reading_scorer));
each scorer carries the margin it is calibrated for (see [`DFLT_MARGIN`](#voxy.pronounce.DFLT_MARGIN)).

### *class* voxy.pronounce.Scorer(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Scores the readings of timed ASR word `words[i]` in `audio` (higher is better).

`margin` is the margin its scores are calibrated for, on their own scale.

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

`margin` defaults to [`DFLT_MARGIN`](#voxy.pronounce.DFLT_MARGIN), the crop scale; [`verify_words()`](#voxy.pronounce.verify_words)
and [`audit()`](#voxy.pronounce.audit) give each verdict their scorer’s margin (phrase and spot scores
are totals in nats).

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

### voxy.pronounce.align_words(text, words, , elisions=())

`{index of a word of text: index of the ASR word heard for it}`.

Words are matched by aligning the two sequences (`difflib`), so a word ASR missed,
merged or added elsewhere does not shift the others; a text word with no exact
counterpart is left out. A word elided onto one of `elisions` counts as two
(`_text_words()`).

```pycon
>>> align_words("Il est là", [{"text": " Il"}, {"text": " et"}, {"text": " est"}, {"text": " là."}])
{0: 0, 1: 2, 2: 3}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### voxy.pronounce.audit(text, audio, \*, readings=None, language='fr', elisions=None, clitics=None, expected=None, detectors=(), emitter=None, words_of=<function \_transcribed_words>, scoring='phrase', phonemize=None, context=1, filler_penalty=1.0, padding=0.08, margin=None, scorer=None)

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
  * **elisions** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `{clitic: IPA}` of the language’s elisions (default
    `ELISIONS[language]`): text and ASR words are split there to be aligned.
  * **clitics** ([`Collection`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Collection)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The clitics a heteronym is still flagged after (default
    `NOUN_CLITICS[language]`): “l’Est” is the risk `Est`, clitic `l'`.
  * **expected** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`object`](https://docs.python.org/3/builtins/functions.html#object)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The reading each word should have: `{word: reading}`, or
    `risk -> reading` to decide per occurrence (`risk.line`, `risk.word`,
    `risk.clitic`). A mapping names an elided form by its clitic
    (`{"est": "verb", "l'est": "noun"}`); `"est"` alone does not cover
    “l’Est”. Otherwise what a `detectors` entry found for that word on that
    line (e.g. [`voxy.speechtext.pos_detector()`](voxy.speechtext.html.md#voxy.speechtext.pos_detector)), else None (just report).
  * **detectors** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)]]) – `text -> risks` proposers of expected readings.
  * **emitter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The phoneme recogniser; default [`phoneme_emitter()`](#voxy.pronounce.phoneme_emitter).
  * **words_of** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]]) – `audio -> timed words`; default ASR (`voxy.refprep.transcribe`).
  * **scoring** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – How a reading is scored in the audio ([`SCORERS`](#voxy.pronounce.SCORERS)).
    `"phrase"` (default): as the whole phrase around the word, its
    `context` neighbours written out with `phonemize`
    ([`phrase_scorer()`](#voxy.pronounce.phrase_scorer)). `"spot"`: spotted in that window
    ([`spot_scorer()`](#voxy.pronounce.spot_scorer)). `"crop"`: the word alone, cut at its ASR times
    ([`crop_scorer()`](#voxy.pronounce.crop_scorer)). Only the options that scoring takes are used.
  * **phonemize** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – For `"phrase"`; default [`espeak_phonemizer()`](#voxy.pronounce.espeak_phonemizer).
  * **context** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Words kept on each side of the flagged one.
  * **margin** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – See [`Verdict.decided`](#voxy.pronounce.Verdict.decided); default `scorer.margin`.
  * **scorer** ([`Scorer`](#voxy.pronounce.Scorer) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A ready [`Scorer`](#voxy.pronounce.Scorer); wins over `scoring` and its options.
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
entry are skipped, and so are words elided onto a clitic (a risk with a `clitic`,
“l’Est”; see thorwhalen/voxy#48), whose differing reading then leaves the word’s open: `verify_words`
matches spellings, not places. Each word is checked once: `verify_words` finds every place
it is spoken, so a word whose risks name *different* readings gets none here
(group the risks by line with [`checks_by_line()`](#voxy.pronounce.checks_by_line) to check each one).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]

### voxy.pronounce.crop_scorer(, emitter, padding=0.08, margin=0.5)

The word alone, cut at its ASR times (`padding` each side), scored by
[`score_readings()`](#voxy.pronounce.score_readings); ASR word times are often too loose for this.

* **Return type:**
  [`Scorer`](#voxy.pronounce.Scorer)

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

### voxy.pronounce.phrase_scorer(, emitter, phonemize=None, language='fr', elisions=None, context=1, padding=0.08, margin=2.0)

Each reading as the whole phrase around the word ([`phrase_readings()`](#voxy.pronounce.phrase_readings)), the
`context` words on each side written out by `phonemize` in `language`.

The neighbours are the same in every candidate, so only the word decides.
`phonemize` defaults to [`espeak_phonemizer()`](#voxy.pronounce.espeak_phonemizer) (its worker starts on the first
call); it must give one IPA per text, or a `ValueError` is raised. An elided
clitic next to the word (`l'` before `Est`) is written from `elisions`
(default `ELISIONS[language]`, see [`spoken_context()`](#voxy.pronounce.spoken_context)).

* **Return type:**
  [`Scorer`](#voxy.pronounce.Scorer)

### voxy.pronounce.reading_name(name)

The reading a name stands for: `"verb~liaison"` is a variant of `"verb"`.

```pycon
>>> reading_name("verb~liaison"), reading_name("noun")
('verb', 'noun')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.pronounce.reading_scorer(scoring='phrase', \*\*options)

The [`Scorer`](#voxy.pronounce.Scorer) named `scoring` in [`SCORERS`](#voxy.pronounce.SCORERS), made with `options`.

```pycon
>>> reading_scorer("crop", emitter=None).margin
0.5
```

* **Return type:**
  [`Scorer`](#voxy.pronounce.Scorer)

### voxy.pronounce.render_verified(lines, checks, \*, render, emitter=None, words_of=<function \_transcribed_words>, max_attempts=3, scorer='phrase', margin=None)

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
  * **emitter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The phoneme recogniser for a `scorer` given by name; default
    [`phoneme_emitter()`](#voxy.pronounce.phoneme_emitter).
  * **words_of** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]]) – `audio -> timed words`; the default transcribes with
    `voxy.refprep.transcribe`.
  * **max_attempts** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Renders of one line at most.
  * **scorer** ([`Scorer`](#voxy.pronounce.Scorer) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – As in [`verify_words()`](#voxy.pronounce.verify_words) (default: phrase scoring, its
    neighbours written out by the real espeak worker, so offline tests
    inject `phrase_scorer(emitter=..., phonemize=fake)` or use `"crop"`).
  * **margin** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – As in [`verify_words()`](#voxy.pronounce.verify_words) (default: phrase scoring, its
    neighbours written out by the real espeak worker, so offline tests
    inject `phrase_scorer(emitter=..., phonemize=fake)` or use `"crop"`).
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`LineTakes`](#voxy.pronounce.LineTakes)]

```pycon
>>> heard = {"est": "ɛ"}  # a fake recogniser: what each word sounds like
>>> def emitter(audio, start, end):
...     vocab = ["<pad>", "ɛ", "s", "t", "l"]
...     path = [vocab.index(s) for s in "l" + heard["est"]]
...     return log_probs_from_path(path, vocab_size=5), vocab
>>> def render(text, line, attempt):
...     heard["est"] = "ɛst" if attempt else "ɛ"  # the second take is right
...     return f"take-{line}-{attempt}.wav"
>>> words = lambda audio: [{"start": 0.0, "end": 0.1, "text": "L'"},
...                        {"start": 0.1, "end": 0.3, "text": "Est"}]
>>> phonemize = lambda texts, language: ["l" for _ in texts]  # a fake espeak
>>> scorer = phrase_scorer(emitter=emitter, phonemize=phonemize)
>>> checks = {1: [("Est", {"verb": "ɛ", "noun": "ɛst"}, "noun")]}
>>> [result] = render_verified(["L'Est"], checks, render=render, words_of=words,
...                            scorer=scorer)
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

### voxy.pronounce.split_elisions(words, , elisions=())

Timed words split at each elision of `elisions`, as `_text_words()` splits
the text, so [`align_words()`](#voxy.pronounce.align_words) can match them.

Whisper’s `" l" | "'est"` becomes `" l'" | "est"` (the apostrophe moved back);
a word heard whole, `" l'est"`, becomes `" l'"` (empty, at its start) and
`"est"` (its whole span). Other apostrophes (an English opening quote) are left
alone.

```pycon
>>> words = [{"text": " à", "start": 0, "end": 1}, {"text": " l", "start": 1, "end": 2},
...          {"text": "'est", "start": 2, "end": 3}, {"text": " l'est", "start": 3, "end": 4},
...          {"text": " 'War", "start": 4, "end": 5}]
>>> [(w["text"], w["start"], w["end"]) for w in split_elisions(words, elisions={"l'": "l"})]
[(' à', 0, 1), (" l'", 1, 2), ('est', 2, 3), (" l'", 3, 3), ('est', 3, 4), (" 'War", 4, 5)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.pronounce.spoken_context(words, , elisions=None)

`(text, tail)`: what `words` say, for a phonemizer, and the IPA of an elided
clitic they end with.

A clitic of `elisions` (`{clitic: IPA}`, e.g. `ELISIONS["fr"]`) that ASR split
from its word is joined back to it, whichever side the apostrophe went
(`L'` | `homme`, or Whisper’s `l` | `'aube`). One left at the end (its word
is the flagged one, or outside the window) is written from `elisions` instead of
being left to the phonemizer, which reads a standalone `l'` as /ɛl/. Anything
not in `elisions` (an English quote, `James'`) is left as it is.

```pycon
>>> fr = {"l'": "l", "d'": "d"}
>>> spoken_context([{"text": " L'"}, {"text": " homme,"}, {"text": " d’"}], elisions=fr)
("L'homme", 'd')
>>> spoken_context([{"text": " avant"}, {"text": " l"}, {"text": "'aube."}], elisions=fr)
("avant l'aube", '')
>>> spoken_context([{"text": " read"}, {"text": " 'War"}], elisions=fr)
("read 'War", '')
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

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

### voxy.pronounce.spot_scorer(, emitter, context=1, padding=0.08, filler_penalty=1.0, margin=2.0)

The reading spotted in the window of `context` words on each side, the rest
left to filler ([`spot_readings()`](#voxy.pronounce.spot_readings)); a neighbour’s /s/ or /t/ can still be
credited to the longer reading.

* **Return type:**
  [`Scorer`](#voxy.pronounce.Scorer)

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

### voxy.pronounce.verify_words(audio, checks, , words, emitter=None, scorer='phrase', margin=None)

Verdicts for each `(word, readings, expected)` at each place ASR found it.

* **Parameters:**
  * **audio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)) – The render.
  * **checks** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – The words to check, each with its candidate readings
    (`{name: IPA}`) and the reading it should have (or None to just
    report). [`checks_from_risks()`](#voxy.pronounce.checks_from_risks) builds them from `flag_risks`.
  * **words** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – Timed words of the render (`transcribe(audio)`’s segments’ words).
  * **emitter** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(audio, start, end) -> (log_probs, vocab)` for a `scorer`
    given by name; default [`phoneme_emitter()`](#voxy.pronounce.phoneme_emitter).
  * **scorer** ([`Scorer`](#voxy.pronounce.Scorer) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – How each reading is scored: a [`Scorer`](#voxy.pronounce.Scorer), or a name in
    [`SCORERS`](#voxy.pronounce.SCORERS) (default `"phrase"`: the whole phrase around the word,
    its neighbours written out by the real espeak worker, so offline tests
    inject `phrase_scorer(emitter=..., phonemize=fake)` or use `"crop"`).
  * **margin** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – See [`Verdict.decided`](#voxy.pronounce.Verdict.decided); default `scorer.margin`.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Verdict`](#voxy.pronounce.Verdict)]

### voxy.pronounce.word_indices(words, word)

Indices of the ASR words matching `word` (case and punctuation aside).

```pycon
>>> word_indices([{"text": " Il"}, {"text": " est,"}, {"text": "Est"}], "est")
[1, 2]
>>> word_indices([{"text": " est…"}], "est")
[0]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]

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
