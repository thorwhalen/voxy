# voxy.align

Where each word of a known text is in a recording, to the character: forced alignment.

ASR gives word times for what it heard; when the text is known (a script, lyrics, the
words around a spot to edit), forced alignment places exactly those words, with each
character’s span. That is what a splice needs at a word’s onset, and what a check
needs when ASR heard a different word than the one spoken.

```pycon
>>> from voxy.align import force_align
>>> spans = force_align("line.wav", "Mon frère est parti")
>>> spans[2]["word"], spans[2]["start"], spans[2]["chars"][0]
```

The default aligner is torchaudio’s `MMS_FA` (multilingual, Latin letters; accents are
dropped for matching) in its own environment, built on first use (or `python -m voxy
setup align`). Its weights are non-commercial (see `workers/align_adapter.py`).
Align short stretches (a line, a word and its neighbours, under about 30 s): one pass
over a long recording drifts, sung audio most.

`run` is the seam: `(audio_path, words) -> [span or None, ...]`.

```pycon
>>> fake = lambda audio, words: [{"start": i, "end": i + 0.5, "score": 0.9, "chars": []}
...                              for i, _ in enumerate(words)]
>>> [s["word"] for s in force_align(b"RIFF....", "Bonjour  le monde", run=fake)]
['Bonjour', 'le', 'monde']
```

### Module Attributes

| [`ALIGN_ENV`](#voxy.align.ALIGN_ENV)            | torch and torchaudio bounded to releases that still ship `forced_align`.   |
|-----------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`DFLT_DECISIVE_MARGIN`](#voxy.align.DFLT_DECISIVE_MARGIN) | tight crops can favour a slightly shorter text by fractions of a point.    |

### Functions

| [`close_alignment`](#voxy.align.close_alignment)()                               | Stop the aligner workers (free their memory).                                                                                                                                                                                                                                                                                            |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`force_align`](#voxy.align.force_align)(audio, text, \*[, run])             | Each word of `text`, in order, with its `start`, `end` (s), `score` (mean character confidence) and `chars` (`char`, `start`, `end`, `score`).                                                                                                                                                                                           |
| [`which_said`](#voxy.align.which_said)(audio, candidates, \*[, start, ...]) | The `candidates` (texts) ranked by how well the audio supports each, best first: `[{"text", "score", "margin", "decisive"}]`: `score` the aligner's log-likelihood (minus its CTC loss; `-inf` for a text too long for the crop), `margin` the lead over the next candidate, `decisive` whether that lead is at least `decisive_margin`. |
| [`worker_run`](#voxy.align.worker_run)(\*[, model, threads, rootdir])       | The default `run`: the aligner adapter in its own environment.                                                                                                                                                                                                                                                                           |
| [`worker_score`](#voxy.align.worker_score)(\*[, model, threads, rootdir])     | The default scorer for `which_said`: the aligner adapter's `score`.                                                                                                                                                                                                                                                                      |

### voxy.align.ALIGN_ENV *= EnvSpec(name='align', requirements=('torch>=2.6,<2.12', 'torchaudio>=2.6,<2.12', 'soundfile', 'numpy'), python='3.11', torch_backend='cpu', env_vars={'TORCH_HOME': '{models}'})*

torch and torchaudio bounded to releases that still ship `forced_align`.

### voxy.align.DFLT_DECISIVE_MARGIN *= 1.0*

tight crops can favour a slightly
shorter text by fractions of a point.

* **Type:**
  A lead (in log-likelihood) under this is a tie

### voxy.align.close_alignment()

Stop the aligner workers (free their memory).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.align.force_align(audio, text, , run=None)

Each word of `text`, in order, with its `start`, `end` (s), `score` (mean
character confidence) and `chars` (`char`, `start`, `end`, `score`).

* **Parameters:**
  * **audio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – A WAV (or any format soundfile reads) path, or WAV bytes.
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The words spoken, as a string (split on whitespace) or a list.
  * **run** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(audio_path, words) -> [span or None, ...]`; default `worker_run()`.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

A word with no letter the aligner knows (a number, a symbol) is placed as one
“any sound” token. Low scores (under about 0.5) mean the text is not what is said
there. Audio over 30 s is refused by the default aligner: align in stretches. A
run that returns None for a word gives `start`/`end` None.

### voxy.align.which_said(audio, candidates, , start=None, end=None, run=None, decisive_margin=1.0)

The `candidates` (texts) ranked by how well the audio supports each, best first:
`[{"text", "score", "margin", "decisive"}]`: `score` the aligner’s
log-likelihood (minus its CTC loss; `-inf` for a text too long for the crop),
`margin` the lead over the next candidate, `decisive` whether that lead is at
least `decisive_margin`.

A forced choice, so it cannot hallucinate as transcribing a short crop can: use it
to check an edit (“witch” or the old word?) on a crop around it (`start`/`end`,
s). Cut the crop at silences and include the neighbouring words: a crop that ends
inside a word, or holds one word alone, favours shorter texts by small margins,
which `decisive` flags. Candidates are spelled in letters (numbers spelled out).

```pycon
>>> fake = lambda audio, texts, start=None, end=None: [-5.0, -2.0]
>>> [(c["text"], c["margin"], c["decisive"])
...  for c in which_said(b"RIFF", ["bitch", "witch"], run=fake)]
[('witch', 3.0, True), ('bitch', None, None)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.align.worker_run(, model='MMS_FA', threads=None, rootdir=None)

The default `run`: the aligner adapter in its own environment.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]

### voxy.align.worker_score(, model='MMS_FA', threads=None, rootdir=None)

The default scorer for `which_said`: the aligner adapter’s `score`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]
