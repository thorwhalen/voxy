# voxy.speakers

Keep one speaker: find their voice in a recording and mark everyone else’s words.

Recordings of a person are rarely just that person: an interviewer asks the
questions, someone says “yes” or “uh-huh” in the middle of an answer. A clone or a
fine-tune trained on those clips learns the other voices too. This module
compares the voice in short sliding windows with a **voice print** (the mean
ECAPA speaker embedding of the target) and marks every transcribed word, and
every untranscribed stretch of someone else’s voice, that does not match it:

```pycon
>>> from voxy import refprep, speakers
>>> print_ = speakers.voice_print("clean_answer.wav")
>>> refprep.prepare_reference(["interview.m4a"], voice="ada",
...                           speaker=speakers.speaker_filter(print_))
```

`plan_clips` then never puts a marked word, or the pause around one, inside a
clip. With no voice print, `speaker_filter()` takes each recording’s
**dominant** voice (an iterated centroid over its loud windows), which is right
for a talk or an interview where the target does most of the talking.

The embeddings come from SpeechBrain’s ECAPA-TDNN (Apache-2.0, VoxCeleb weights)
running in voxy’s isolated `speaker` environment (`SPEAKER_ENV`, built on
first use, see `voxy.envs`); everything after the embeddings is numpy, so the
logic below is testable with made-up vectors.

### Module Attributes

| [`DFLT_MATCH_SIMILARITY`](#voxy.speakers.DFLT_MATCH_SIMILARITY)    | Cosine similarity to the voice print below which a window is someone else.                                                                                                                                                                                             |
|---------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DFLT_SILENCE_MARGIN_DB`](#voxy.speakers.DFLT_SILENCE_MARGIN_DB)   | neither voice.                                                                                                                                                                                                                                                         |
| [`DFLT_MIN_FOREIGN_SECONDS`](#voxy.speakers.DFLT_MIN_FOREIGN_SECONDS) | Untranscribed foreign stretches shorter than this are ignored (one window's wobble).                                                                                                                                                                                   |
| [`DFLT_FOREIGN_PAD_SECONDS`](#voxy.speakers.DFLT_FOREIGN_PAD_SECONDS) | Margin added around a foreign stretch, so a cut never clips its edge into a clip.                                                                                                                                                                                      |
| [`DFLT_WORD_OVERLAP`](#voxy.speakers.DFLT_WORD_OVERLAP)        | A word with this share of its length inside a foreign stretch is not kept.                                                                                                                                                                                             |
| [`DFLT_DIP_SIMILARITY`](#voxy.speakers.DFLT_DIP_SIMILARITY)      | Inside a transcribed turn, a stretch is someone else's only below this (smoothed) match: one-second windows of old, noisy audio dip well below `DFLT_MATCH_SIMILARITY` on the target's own voice, so a turn that matches as a whole is only cut where the dip is deep. |
| [`DFLT_SMOOTH_WINDOWS`](#voxy.speakers.DFLT_SMOOTH_WINDOWS)      | Window similarities are averaged over this many neighbours before finding stretches.                                                                                                                                                                                   |
| [`DFLT_MIN_TURN_SECONDS`](#voxy.speakers.DFLT_MIN_TURN_SECONDS)    | never kept.                                                                                                                                                                                                                                                            |

### Functions

| [`close_speaker`](#voxy.speakers.close_speaker)()                                   | Stop the embedding workers (free their memory).                                       |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [`dominant_centroid`](#voxy.speakers.dominant_centroid)(vectors, \*[, iterations, ...]) | The voice heard most: re-centre on the windows most like the centroid, repeatedly.    |
| [`embed_windows`](#voxy.speakers.embed_windows)(audio, \*[, spans, window, ...])    | ECAPA embeddings of `window`-second windows every `hop` s of `audio`.                 |
| [`foreign_spans`](#voxy.speakers.foreign_spans)(windows, similarity, \*[, ...])     | Stretches where a loud window does not match the voice print, padded by `pad`.        |
| [`loud_mask`](#voxy.speakers.loud_mask)(windows, samples, sample_rate, \*)      | Windows with speech in them: loud frames within `margin_db` of the speech level.      |
| [`mark_speaker`](#voxy.speakers.mark_speaker)(segments, windows, similarity, \*)   | Annotate each timed word with `speaker_similarity` and `keep`; add gap markers.       |
| [`smooth`](#voxy.speakers.smooth)(similarity[, windows])                     | Moving average of `similarity` over `windows` neighbours (edges: the shorter mean).   |
| [`speaker_filter`](#voxy.speakers.speaker_filter)([target, threshold, ...])          | A `refprep.prepare_reference(speaker=...)` step: `(wav_path, segments) -> segments`.  |
| [`turn_similarity`](#voxy.speakers.turn_similarity)(segments, windows, ...[, loud])   | Match of each transcribed segment (a turn) to `voice_print`, from its mean window.    |
| [`voice_print`](#voxy.speakers.voice_print)(audio, \*[, spans, embed])            | The voice print of `audio` (one recording or several): the dominant voice's centroid. |

### Classes

| [`Windows`](#voxy.speakers.Windows)(times, vectors)   | Speaker embeddings of sliding windows: `times` (n, 2) seconds, unit `vectors` (n, d).   |
|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|

### voxy.speakers.DFLT_DIP_SIMILARITY *= 0.15*

Inside a transcribed turn, a stretch is someone else’s only below this (smoothed) match:
one-second windows of old, noisy audio dip well below `DFLT_MATCH_SIMILARITY` on the
target’s own voice, so a turn that matches as a whole is only cut where the dip is deep.

### voxy.speakers.DFLT_FOREIGN_PAD_SECONDS *= 0.15*

Margin added around a foreign stretch, so a cut never clips its edge into a clip.

### voxy.speakers.DFLT_MATCH_SIMILARITY *= 0.45*

Cosine similarity to the voice print below which a window is someone else.

### voxy.speakers.DFLT_MIN_FOREIGN_SECONDS *= 0.2*

Untranscribed foreign stretches shorter than this are ignored (one window’s wobble).

### voxy.speakers.DFLT_MIN_TURN_SECONDS *= 1.0*

never kept.

* **Type:**
  Turns shorter than this are too short to attribute (”Yes.”, ”Uh-huh”)

### voxy.speakers.DFLT_SILENCE_MARGIN_DB *= 25.0*

neither voice.

* **Type:**
  Windows this far below the recording’s speech level are silence

### voxy.speakers.DFLT_SMOOTH_WINDOWS *= 3*

Window similarities are averaged over this many neighbours before finding stretches.

### voxy.speakers.DFLT_WORD_OVERLAP *= 0.25*

A word with this share of its length inside a foreign stretch is not kept.

### *class* voxy.speakers.Windows(times, vectors)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Speaker embeddings of sliding windows: `times` (n, 2) seconds, unit `vectors` (n, d).

#### similarity(voice_print)

Cosine similarity of each window to `voice_print`.

* **Return type:**
  `ndarray`

### voxy.speakers.close_speaker()

Stop the embedding workers (free their memory).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.speakers.dominant_centroid(vectors, , iterations=6, quantile=0.5)

The voice heard most: re-centre on the windows most like the centroid, repeatedly.

Clustering fragments a noisy voice into several clusters; this converges on
the majority voice instead (it needs that voice to be more than half the windows
the first mean is computed from).

```pycon
>>> rng = np.random.default_rng(0)
>>> a, b = np.eye(4)[0], np.eye(4)[1]
>>> vs = np.array([_unit(a + 0.1 * rng.standard_normal(4)) for _ in range(30)]
...             + [_unit(b + 0.1 * rng.standard_normal(4)) for _ in range(10)])
>>> bool(dominant_centroid(vs) @ a > 0.95)
True
```

* **Return type:**
  `ndarray`

### voxy.speakers.embed_windows(audio, , spans=None, window=1.0, hop=0.1, model='speechbrain/spkrec-ecapa-voxceleb', threads=None, rootdir=None)

ECAPA embeddings of `window`-second windows every `hop` s of `audio`.

`spans` restricts the windows to those `(start, end)` stretches (default:
the whole file). Any sample rate; the worker resamples to 16 kHz.

* **Return type:**
  [`Windows`](#voxy.speakers.Windows)

### voxy.speakers.foreign_spans(windows, similarity, , threshold=0.45, loud=None, min_seconds=0.2, pad=0.15)

Stretches where a loud window does not match the voice print, padded by `pad`.

`threshold` is one similarity, or one per window.

Each window speaks for the `hop`-wide slot around its centre, so a short “yes”
marks about the length of the dip it causes, not a whole window.

```pycon
>>> t = np.array([[i * 0.1, i * 0.1 + 1] for i in range(20)])
>>> sim = np.array([0.8] * 8 + [0.1] * 4 + [0.8] * 8)
>>> [[round(x, 2) for x in s] for s in foreign_spans(Windows(t, None), sim, pad=0)]
[[1.25, 1.65]]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### voxy.speakers.loud_mask(windows, samples, sample_rate, , margin_db=25.0)

Windows with speech in them: loud frames within `margin_db` of the speech level.

```pycon
>>> w = Windows(np.array([[0, 1], [1, 2]]), np.zeros((2, 3)))
>>> x = np.concatenate([0.3 * np.ones(8000), np.zeros(8000)])
>>> loud_mask(w, x, 8000).tolist()
[True, False]
```

* **Return type:**
  `ndarray`

### voxy.speakers.mark_speaker(segments, windows, similarity, , threshold=0.45, loud=None, turns=None, dip_threshold=0.15, min_turn_seconds=1.0, word_overlap=0.25, pad=0.15, min_seconds=0.2)

Annotate each timed word with `speaker_similarity` and `keep`; add gap markers.

Without `turns`, a word is kept when the windows centred inside it match the
voice print on average. With `turns` (one match per segment, see
[`turn_similarity()`](#voxy.speakers.turn_similarity)), a word is kept when its whole segment matches and lasts
at least `min_turn_seconds`, and its `speaker_similarity` is the segment’s;
inside a segment only a deep dip (below `dip_threshold`) counts as someone else,
and a one-word sentence (“Yes.”) must match the voice print on its own.
Either way a word is dropped when `word_overlap` of it lies in a foreign
stretch. Every foreign stretch is also added as an empty word with
`keep=False` (an untranscribed “uh-huh” has no word of its own), padded by up
to `pad` and trimmed back to the kept words around it, so `refprep.plan_clips`
never puts it inside a clip. A segment without word timings is judged as one word.

```pycon
>>> t = np.array([[i * 0.1, i * 0.1 + 1] for i in range(40)])
>>> sim = np.array([0.8] * 15 + [0.1] * 5 + [0.8] * 20)
>>> w = lambda s, e, x: {"start": s, "end": e, "text": x}
>>> segs = [{"start": 0, "end": 4, "text": "a b",
...          "words": [w(0.5, 1.4, " a"), w(2.5, 3.8, " b")]}]
>>> out = mark_speaker(segs, Windows(t, None), sim)
>>> [(x["text"], x["keep"]) for x in out[0]["words"]], out[1]["words"][0]["end"]
([(' a', True), (' b', True)], 2.5)
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.speakers.smooth(similarity, windows=3)

Moving average of `similarity` over `windows` neighbours (edges: the shorter mean).

```pycon
>>> smooth(np.array([1.0, 1.0, 0.0, 1.0, 1.0])).round(2).tolist()
[1.0, 0.67, 0.67, 0.67, 1.0]
```

* **Return type:**
  `ndarray`

### voxy.speakers.speaker_filter(target=None, , threshold=0.45, dip_threshold=0.15, smooth_windows=3, min_turn_seconds=1.0, embed=None, \*\*embed_kwargs)

A `refprep.prepare_reference(speaker=...)` step: `(wav_path, segments) -> segments`.

Decides per transcribed segment (a turn: [`turn_similarity()`](#voxy.speakers.turn_similarity)), then cuts
deep dips inside kept turns and other voices between them ([`mark_speaker()`](#voxy.speakers.mark_speaker)).

* **Parameters:**
  * **target** (`ndarray` | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The voice print to keep (from [`voice_print()`](#voxy.speakers.voice_print)); None takes each
    recording’s dominant voice.
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Cosine similarity a turn (or a stretch between turns) needs to
    count as the target.
  * **dip_threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Smoothed similarity below which a stretch inside a kept turn
    is someone else.
  * **smooth_windows** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Neighbouring windows averaged before finding stretches.
  * **min_turn_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Shorter turns are never kept.
  * **embed** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Windows`](#voxy.speakers.Windows)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(path, **embed_kwargs) -> Windows` (default [`embed_windows()`](#voxy.speakers.embed_windows)).
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### voxy.speakers.turn_similarity(segments, windows, voice_print, , loud=None)

Match of each transcribed segment (a turn) to `voice_print`, from its mean window.

Averaging the embeddings over a whole turn is far steadier than any one window
on noisy audio: it is what tells an interviewer’s question from an answer. None
for a segment with no (loud) window centred in it.

```pycon
>>> t = np.array([[i * 0.1, i * 0.1 + 1] for i in range(40)])
>>> v = np.array([np.eye(2)[0]] * 20 + [np.eye(2)[1]] * 20)
>>> turn_similarity([{"start": 0, "end": 2}, {"start": 2.6, "end": 4.4}],
...                 Windows(t, v), np.eye(2)[0])
[1.0, 0.0]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]

### voxy.speakers.voice_print(audio, , spans=None, embed=None, \*\*embed_kwargs)

The voice print of `audio` (one recording or several): the dominant voice’s centroid.

Pass recordings (or `spans` of one) where the target speaks alone, and this is
simply their voice; over mixed recordings it is whoever speaks most.

* **Return type:**
  `ndarray`
