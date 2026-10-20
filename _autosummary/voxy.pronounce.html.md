# voxy.pronounce

Checking, in the audio, which reading of a flagged word a render used.

[`voxy.speechtext.flag_risks()`](voxy.speechtext.html.md#voxy.speechtext.flag_risks) lists the words a model may misread (heteronyms
such as French “est” /ɛ/ vs “Est” /ɛst/). This module listens for the answer: it crops
the word out of a render (word timestamps from ASR), asks a phoneme recogniser for its
frame-wise phoneme log-probabilities, and scores each candidate reading by its CTC
likelihood. The reading the audio supports best is the one the model said; a render
whose word came out as the wrong one can be re-rendered.

The recogniser is `facebook/wav2vec2-lv-60-espeak-cv-ft`, run in its own torch env
([`PHONEME_ENV`](#voxy.pronounce.PHONEME_ENV), built by `voxy.envs`). Its `vocab.json` maps espeak IPA
symbols to ids, so candidate readings are written in IPA and tokenised against that
vocabulary: no phonemizer is needed. Everything above the recogniser is pure numpy and
takes the recogniser as an argument (`emitter`), so it is tested without torch.

```pycon
>>> vocab = ["<pad>", "ɛ", "s", "t"]
>>> emissions = log_probs_from_path([1, 0, 0], vocab_size=4)  # hears /ɛ/ only
>>> best_reading(emissions, vocab, {"verb": "ɛ", "noun": "ɛst"})
'verb'
```

### Module Attributes

| [`DFLT_PHONEME_MODEL`](#voxy.pronounce.DFLT_PHONEME_MODEL)   | wav2vec2 fine-tuned to emit espeak IPA (CTC, 16 kHz).                                                                     |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [`PHONEME_ENV`](#voxy.pronounce.PHONEME_ENV)          | CPU torch plus transformers (the processor's vocab.json is all it needs).                                                 |
| [`DFLT_CROP_PADDING`](#voxy.pronounce.DFLT_CROP_PADDING)    | ASR word edges are loose.                                                                                                 |
| [`DFLT_MARGIN`](#voxy.pronounce.DFLT_MARGIN)          | A reading must beat the runner-up by this much (mean log-likelihood per target symbol) for a verdict to count as decided. |
| [`BLANK_SYMBOL`](#voxy.pronounce.BLANK_SYMBOL)         | The CTC blank of the recogniser's vocabulary (wav2vec2 uses its pad token).                                               |
| [`Emitter`](#voxy.pronounce.Emitter)              | `(audio, start, end) -> (log_probs [frames x vocab], vocab)`.                                                             |

### Functions

| [`best_reading`](#voxy.pronounce.best_reading)(log_probs, vocab, readings, \*)       | The name of the reading `log_probs` support best.                                                                     |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [`checks_from_risks`](#voxy.pronounce.checks_from_risks)(risks, \*, readings[, expected]) | `verify_words` checks for the flagged words that have known IPA readings.                                             |
| [`ctc_log_likelihood`](#voxy.pronounce.ctc_log_likelihood)(log_probs, target, \*[, blank]) | `log P(target | log_probs)` under CTC (the forward algorithm, in log space).                                          |
| [`failing`](#voxy.pronounce.failing)(verdicts)                                  | The verdicts whose word was decidedly read the wrong way (`ok is False`).                                             |
| [`log_probs_from_path`](#voxy.pronounce.log_probs_from_path)(path, \*, vocab_size[, ...])   | Log-probabilities of a recogniser that hears `path` (one id per frame).                                               |
| [`phoneme_emitter`](#voxy.pronounce.phoneme_emitter)(\*[, model, threads, rootdir])     | The default [`Emitter`](#voxy.pronounce.Emitter): the wav2vec2 phoneme recogniser in its own env. |
| [`score_readings`](#voxy.pronounce.score_readings)(log_probs, vocab, readings, \*)     | Each reading's CTC log-likelihood per target symbol (higher is better).                                               |
| [`tokenize_phonemes`](#voxy.pronounce.tokenize_phonemes)(reading, vocab)                  | Vocabulary ids of an IPA `reading`, longest symbol first; spaces are ignored.                                         |
| [`verify_words`](#voxy.pronounce.verify_words)(audio, checks, \*, words, emitter)    | Verdicts for each `(word, readings, expected)` at each place ASR found it.                                            |
| [`word_spans`](#voxy.pronounce.word_spans)(words, word)                            | `(start, end)` of every ASR word matching `word` (case and punctuation aside).                                        |

### Classes

| [`Verdict`](#voxy.pronounce.Verdict)(word, start, end, scores[, ...])   | What the audio says about one flagged word.   |
|---------------------------------------------------------------------------------------------|-----------------------------------------------|

### voxy.pronounce.BLANK_SYMBOL *= '<pad>'*

The CTC blank of the recogniser’s vocabulary (wav2vec2 uses its pad token).

### voxy.pronounce.DFLT_CROP_PADDING *= 0.08*

ASR word edges are loose.

* **Type:**
  Seconds kept on each side of a word’s ASR timestamps

### voxy.pronounce.DFLT_MARGIN *= 0.5*

A reading must beat the runner-up by this much (mean log-likelihood per target
symbol) for a verdict to count as decided.

### voxy.pronounce.DFLT_PHONEME_MODEL *= 'facebook/wav2vec2-lv-60-espeak-cv-ft'*

wav2vec2 fine-tuned to emit espeak IPA (CTC, 16 kHz).

* **Type:**
  The phoneme recogniser

### voxy.pronounce.Emitter

`(audio, start, end) -> (log_probs [frames x vocab], vocab)`.

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]

### voxy.pronounce.PHONEME_ENV *= EnvSpec(name='phonemes', requirements=('torch', 'transformers', 'soundfile', 'numpy', 'librosa'), python='3.11', torch_backend='cpu', env_vars={})*

CPU torch plus transformers (the processor’s vocab.json is all it needs).

* **Type:**
  Its env

### *class* voxy.pronounce.Verdict(word, start, end, scores, expected=None, margin=0.5)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What the audio says about one flagged word.

#### *property* decided *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the best reading beats the runner-up by at least `margin`.

#### *property* heard *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The reading the audio supports best.

#### *property* ok *: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None)*

True/False when an `expected` reading was given and the audio decided.

### voxy.pronounce.best_reading(log_probs, vocab, readings, , blank=None)

The name of the reading `log_probs` support best.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.pronounce.checks_from_risks(risks, , readings, expected=None)

`verify_words` checks for the flagged words that have known IPA readings.

`readings` maps a word (case-insensitively) to `{name: IPA}`, e.g. from a
Lexique-derived heteronym list the caller supplies; `expected` names the
reading each word should have. Words with no entry are skipped. Each word is
checked once: `verify_words` finds every place it is spoken.

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

### voxy.pronounce.failing(verdicts)

The verdicts whose word was decidedly read the wrong way (`ok is False`).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Verdict`](#voxy.pronounce.Verdict)]

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

### voxy.pronounce.score_readings(log_probs, vocab, readings, , blank=None)

Each reading’s CTC log-likelihood per target symbol (higher is better).

Normalising by length keeps a longer reading (“ɛst”) from losing to a shorter one
(“ɛ”) just for having more symbols to pay for. `blank` defaults to the id of
[`BLANK_SYMBOL`](#voxy.pronounce.BLANK_SYMBOL) in `vocab` (0 if it has none).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

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
