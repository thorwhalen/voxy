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
over a long recording drifts, sung audio most. For a whole recording (an episode),
`align_long` finds the stretches itself: a rough speech-recognition pass gives anchor
words, and each stretch between anchors is aligned exactly.

`run` is the seam: `(audio_path, words, *, start, end) -> [span or None, ...]`.

```pycon
>>> fake = lambda audio, words, **crop: [{"start": i, "end": i + 0.5, "score": 0.9, "chars": []}
...                                      for i, _ in enumerate(words)]
>>> [s["word"] for s in force_align(b"RIFF....", "Bonjour  le monde", run=fake)]
['Bonjour', 'le', 'monde']
```

### Module Attributes

| [`ALIGN_ENV`](#voxy.align.ALIGN_ENV)              | torch and torchaudio bounded to releases that still ship `forced_align`.                                                                                                |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`Run`](#voxy.align.Run)                    | `(audio_path, words, *, start, end) -> [span or None, ...]`                                                                                                             |
| [`DFLT_DECISIVE_MARGIN`](#voxy.align.DFLT_DECISIVE_MARGIN)   | tight crops can favour a slightly shorter text by fractions of a point.                                                                                                 |
| [`DFLT_MAX_STRETCH_S`](#voxy.align.DFLT_MAX_STRETCH_S)     | under its 30 s limit, with margin.                                                                                                                                      |
| [`DFLT_MIN_STRETCH_SCORE`](#voxy.align.DFLT_MIN_STRETCH_SCORE) | A stretch whose mean word score is under this is reported as unreliable.                                                                                                |
| [`DFLT_MIN_CUT_GAP_S`](#voxy.align.DFLT_MIN_CUT_GAP_S)     | A cut prefers a pause at least this long (s) between two anchored words, so a rough pass's timing error (Whisper's is a few tenths) cannot put the cut inside a word.   |
| [`DFLT_EDGE_S`](#voxy.align.DFLT_EDGE_S)            | How far past its outer anchored words (s) a stretch reaches into a long pause, the recording's lead-in or its tail: a long music break or intro is not aligned against. |
| [`DFLT_CROP_PAD_S`](#voxy.align.DFLT_CROP_PAD_S)        | Each stretch's crop reaches this far (s) past its bounds, for the same reason.                                                                                          |
| [`ALIGNER_MAX_S`](#voxy.align.ALIGNER_MAX_S)          | a crop must stay under it.                                                                                                                                              |
| [`ALIGN_RATE`](#voxy.align.ALIGN_RATE)             | The rate the recording is converted to once, before it is cut (the aligner's own).                                                                                      |
| [`Heard`](#voxy.align.Heard)                  | timed words heard.                                                                                                                                                      |

### Functions

| [`align_long`](#voxy.align.align_long)(audio, text, \*[, heard, run, ...])    | [`force_align()`](#voxy.align.force_align) for a recording of any length (a whole episode).                                                                                                                                                                                                                          |
|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`close_alignment`](#voxy.align.close_alignment)()                                 | Stop the aligner workers (free their memory).                                                                                                                                                                                                                                                                                            |
| [`force_align`](#voxy.align.force_align)(audio, text, \*[, start, end, run])   | Each word of `text`, in order, with its `start`, `end` (s), `score` (mean character confidence) and `chars` (`char`, `start`, `end`, `score`).                                                                                                                                                                                           |
| [`plan_stretches`](#voxy.align.plan_stretches)(n_words, anchor_times, ...[, ...]) | Cut `n_words` text words, `[0, duration)` s of audio, into stretches.                                                                                                                                                                                                                                                                    |
| [`which_said`](#voxy.align.which_said)(audio, candidates, \*[, start, ...])   | The `candidates` (texts) ranked by how well the audio supports each, best first: `[{"text", "score", "margin", "decisive"}]`: `score` the aligner's log-likelihood (minus its CTC loss; `-inf` for a text too long for the crop), `margin` the lead over the next candidate, `decisive` whether that lead is at least `decisive_margin`. |
| [`worker_run`](#voxy.align.worker_run)(\*[, model, threads, rootdir])         | The default `run`: the aligner adapter in its own environment.                                                                                                                                                                                                                                                                           |
| [`worker_score`](#voxy.align.worker_score)(\*[, model, threads, rootdir])       | The default scorer for `which_said`: the aligner adapter's `score`.                                                                                                                                                                                                                                                                      |

### voxy.align.ALIGNER_MAX_S *= 30.0*

a crop must stay under it.

* **Type:**
  What the aligner refuses (s)

### voxy.align.ALIGN_ENV *= EnvSpec(name='align', requirements=('torch>=2.6,<2.12', 'torchaudio>=2.6,<2.12', 'soundfile', 'numpy'), python='3.11', torch_backend='cpu', env_vars={'TORCH_HOME': '{models}'})*

torch and torchaudio bounded to releases that still ship `forced_align`.

### voxy.align.ALIGN_RATE *= 16000*

The rate the recording is converted to once, before it is cut (the aligner’s own).

### voxy.align.DFLT_CROP_PAD_S *= 0.3*

Each stretch’s crop reaches this far (s) past its bounds, for the same reason.

### voxy.align.DFLT_DECISIVE_MARGIN *= 1.0*

tight crops can favour a slightly
shorter text by fractions of a point.

* **Type:**
  A lead (in log-likelihood) under this is a tie

### voxy.align.DFLT_EDGE_S *= 1.0*

How far past its outer anchored words (s) a stretch reaches into a long pause, the
recording’s lead-in or its tail: a long music break or intro is not aligned against.

### voxy.align.DFLT_MAX_STRETCH_S *= 25.0*

under its 30 s limit, with margin.

* **Type:**
  The longest stretch handed to the aligner (s)

### voxy.align.DFLT_MIN_CUT_GAP_S *= 0.25*

A cut prefers a pause at least this long (s) between two anchored words, so a rough
pass’s timing error (Whisper’s is a few tenths) cannot put the cut inside a word.

### voxy.align.DFLT_MIN_STRETCH_SCORE *= 0.5*

A stretch whose mean word score is under this is reported as unreliable.

### voxy.align.Heard

timed words heard.

* **Type:**
  `audio_path -> [{"text" (or "word"), "start", "end"}, ...]`

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### voxy.align.Run

`(audio_path, words, *, start, end) -> [span or None, ...]`

alias of [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[…], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[dict | None]]

### voxy.align.align_long(audio, text, , heard=None, run=None, language=None, max_stretch_s=25.0, min_score=0.5, pad_s=0.3)

[`force_align()`](#voxy.align.force_align) for a recording of any length (a whole episode).

1. A rough pass of speech recognition (`heard`: default voxy’s ASR) gives
   approximate times for the words it heard.
2. The known text is matched to those words as two sequences
   (`mixing.transcript.align_words`, as `voxy.pronounce` does); matched words
   are anchors, and the text between two anchors belongs to the audio between them.
3. The recording is cut between anchored words into stretches of at most
   `max_stretch_s` ([`plan_stretches()`](#voxy.align.plan_stretches)), and each is aligned exactly with
   [`force_align()`](#voxy.align.force_align), on a crop reaching `pad_s` past its bounds so a rough time
   > a few tenths off cannot clip a word. The recording is converted once to a 16 kHz
   > mono WAV (ffmpeg) and each crop is cut from it, so an hour is not decoded once
   > per stretch.

Each word comes back as [`force_align()`](#voxy.align.force_align) gives it (`word`, `start`, `end`,
`score`, `chars`), on the recording’s clock, plus `stretch` (its index) and
`reliable`: False when its stretch’s mean score is under `min_score`, or when no
cut could keep the stretch short enough (its times are then None: reported, not
guessed).

```pycon
>>> heard = lambda path: [{"text": w, "start": 2.0 * i, "end": 2.0 * i + 1}
...                       for i, w in enumerate("one two three four".split())]
>>> fake = lambda audio, words, *, start, end: [
...     {"start": i, "end": i + 0.5, "score": 0.9, "chars": []}
...     for i, _ in enumerate(words)]
>>> import io, wave
>>> buf = io.BytesIO()
>>> with wave.open(buf, "wb") as w:
...     w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(bytes(2 * 16000 * 8))
>>> out = align_long(buf.getvalue(), "One two, three four.", heard=heard, run=fake, max_stretch_s=4)
>>> [(w["word"], w["stretch"], w["reliable"]) for w in out]
[('One', 0, True), ('two,', 0, True), ('three', 1, True), ('four.', 2, True)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.align.close_alignment()

Stop the aligner workers (free their memory).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.align.force_align(audio, text, , start=None, end=None, run=None)

Each word of `text`, in order, with its `start`, `end` (s), `score` (mean
character confidence) and `chars` (`char`, `start`, `end`, `score`).

* **Parameters:**
  * **audio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – A WAV (or any format soundfile reads) path, or WAV bytes.
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The words spoken, as a string (split on whitespace) or a list.
  * **start** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The stretch of `audio` (s) the words are in (default: all of
    it); times stay on the recording’s clock.
  * **end** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The stretch of `audio` (s) the words are in (default: all of
    it); times stay on the recording’s clock.
  * **run** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(audio_path, words, *, start, end) -> [span or None, ...]`; default
    `worker_run()`.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

A word with no letter the aligner knows (a number, a symbol) is placed as one
“any sound” token. Low scores (under about 0.5) mean the text is not what is said
there. Audio over 30 s is refused by the default aligner: align in stretches. A
run that returns None for a word gives `start`/`end` None.

### voxy.align.plan_stretches(n_words, anchor_times, duration, , max_stretch_s=25.0, min_gap_s=0.25, edge_s=1.0, transparent=frozenset({}))

Cut `n_words` text words, `[0, duration)` s of audio, into stretches.

`anchor_times` maps a text word’s index to the `(start, end)` heard for it. A
cut falls only between two anchored text words with nothing between them but
`transparent` words (punctuation no recogniser hears), so every word between two
cuts is known to be in that stretch of audio. A stretch ends `edge_s` after its
last anchored word and the next starts `edge_s` before its first, or both meet
in the middle of a short pause; likewise the first stretch starts `edge_s` before
an anchored first word and the last ends `edge_s` after an anchored last word, so
an intro, an outro or a long break is not part of any stretch. Greedy: each stretch
reaches the farthest cut within `max_stretch_s` in a pause of at least
`min_gap_s` (else the widest pause in reach). A stretch no cut can keep that short
is marked `too_long` (the default aligner refuses it), never cut on a guess.

```pycon
>>> times = {0: (0.0, 0.4), 1: (0.5, 0.9), 2: (10.0, 10.4), 3: (10.5, 10.9), 4: (20.0, 20.4)}
>>> [(s["words"], s["start"], s["end"]) for s in plan_stretches(5, times, 21.0, max_stretch_s=12)]
[((0, 4), 0.0, 11.9), ((4, 5), 19.0, 21.0)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

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
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]]

### voxy.align.worker_score(, model='MMS_FA', threads=None, rootdir=None)

The default scorer for `which_said`: the aligner adapter’s `score`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]
