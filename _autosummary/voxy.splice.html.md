# voxy.splice

Correct words in a recording, in the speaker’s voice: find, render, splice, check.

```pycon
>>> from voxy.splice import replace_words
>>> fixed = replace_words("narration.wav", "Anna Smyth", "Anna Smith",
...                       voice="ada")
>>> fixed.passed, fixed.save("narration-fixed.wav")
```

Replace the whole coarticulated phrase, not one word: a surname spliced alone was
heard with the wrong first consonant, because the first name before it anticipates
the old surname’s lips; replacing first name and surname worked (voxy#38).

Steps, each a seam (keyword argument of `replace_words`):

- `words`: the recording’s timed ASR words (default: `transcriber(audio)`,
  `voxy.refprep.transcribe`); `old` is found among them (`find_phrase`).
- `render(text) -> (samples, sample_rate)`: the new phrase in the voice
  (default `tts_render(voice)`: `voxy.text_to_speech`; a speech-to-speech
  conversion of a spoken correction, or a local clone, plug in the same way).
- The cut lands at the quietest moment near each end of the phrase
  (`cut_points`): in the gap when there is one, and a little past ASR’s word
  times when there is none, since those are often early (a final /z/ left behind
  was heard as “Smith’s”). The render is trimmed, resampled, loudness-matched to
  what it replaces (`level`: by default the RMS of the loud frames on both
  sides, `speech_level`, so pauses do not count), and crossfaded. With
  `keep_pauses` (default), the pauses the cut removed are put back around the
  render from the cut’s own edges, so they keep their length and their background.
- `background(samples, sample_rate, length)`: what is mixed under the render so
  the recording’s floor does not drop out while the new words play (default
  `quiet_background`: the recording’s quietest frames, or nothing when even
  those are not quiet, as in wall-to-wall audio; `None` for no background).
- `verify`: checks of the result (default `heard_check`: the new words are
  transcribed where they were put, and the old ones are gone; `voice_check`
  compares the new phrase’s voice print with the rest of the recording’s,
  `voxy.speakers`).

```pycon
>>> import numpy as np
>>> sr = 8000
>>> tone = lambda s, amp: amp * np.sin(np.arange(int(s * sr)) / 3).astype(np.float32)
>>> gap = np.zeros(sr // 2)
>>> speech = np.concatenate([tone(1, .3), gap, tone(1, .3), gap, tone(.5, .3)])
>>> words = [{"text": "Hello", "start": 0.0, "end": 1.0},
...          {"text": "Smyth", "start": 1.5, "end": 2.5},
...          {"text": "there.", "start": 3.0, "end": 3.5}]
>>> fixed = replace_words(speech, "smyth", "Smith", words=words, sample_rate=sr,
...                       render=lambda text: (tone(0.8, 0.9), sr), verify=())
>>> fixed.replaced, round(fixed.start, 3), round(fixed.end, 3), fixed.passed
((1.375, 2.625), 1.48, 2.28, None)
```

### Module Attributes

| [`DFLT_MAX_MARGIN_MS`](#voxy.splice.DFLT_MAX_MARGIN_MS)        | How far a cut may sit from the phrase when the neighbouring gap is wider.                                                                              |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DFLT_SNAP_MS`](#voxy.splice.DFLT_SNAP_MS)              | How far past an ASR word boundary the cut may look for a quiet moment.                                                                                 |
| [`DFLT_QUIET_TOLERANCE_DB`](#voxy.splice.DFLT_QUIET_TOLERANCE_DB)   | Frames within this many dB of the quietest count as equally quiet.                                                                                     |
| [`DFLT_LOUD_MARGIN_DB`](#voxy.splice.DFLT_LOUD_MARGIN_DB)       | Frames within this many dB of the loudest count as speech when matching levels.                                                                        |
| [`DFLT_CHECK_CONTEXT_S`](#voxy.splice.DFLT_CHECK_CONTEXT_S)      | Audio heard around the splice when checking it (ASR needs some context).                                                                               |
| [`DFLT_PAUSE_DROP_DB`](#voxy.splice.DFLT_PAUSE_DROP_DB)        | A frame this far below the loudest of a stretch counts as pause, not speech.                                                                           |
| [`DFLT_BACKGROUND_SHARE`](#voxy.splice.DFLT_BACKGROUND_SHARE)     | Share of the recording's frames, the quietest, that its background is built from.                                                                      |
| [`DFLT_BACKGROUND_DROP_DB`](#voxy.splice.DFLT_BACKGROUND_DROP_DB)   | Those frames count as background only this far below the speech level, or further.                                                                     |
| [`DFLT_BACKGROUND_SPREAD_DB`](#voxy.splice.DFLT_BACKGROUND_SPREAD_DB) |                                                                                                                                                        |
| [`DFLT_PAUSE_ABOVE_FLOOR_DB`](#voxy.splice.DFLT_PAUSE_ABOVE_FLOOR_DB) | A frame within this much of the recording's floor counts as pause when padding: soft consonants (/f/, /s/, /h/, 25-35 dB below the vowel) stay speech. |
| [`FLOOR_PERCENTILE`](#voxy.splice.FLOOR_PERCENTILE)          | this percentile of its frame levels.                                                                                                                   |
| [`DFLT_RENDER_FORMATS`](#voxy.splice.DFLT_RENDER_FORMATS)       | ElevenLabs returns mp3 by default; raw PCM decodes without ffmpeg.                                                                                     |

### Functions

| [`cap_peak`](#voxy.splice.cap_peak)(samples[, peak])                          | `samples`, scaled down if needed so that no sample exceeds `peak`.                                                                                                                                |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`crossfade_join`](#voxy.splice.crossfade_join)(head, middle, tail, \*, fade)       | `(head + middle + tail, where middle starts in it)`, with linear crossfades of up to `fade` samples at each join (shorter where a piece is too short; none at an edge of the audio).              |
| [`cut_points`](#voxy.splice.cut_points)(samples, sample_rate, words, ...)       | Where to cut (s): the quietest moment before `words[first]` and after `words[last]`, searched in the gap to the neighbouring word (up to `max_margin`) and at least `snap` past the ASR boundary. |
| [`edge_pauses`](#voxy.splice.edge_pauses)(samples, sample_rate, \*[, ...])       | Seconds of pause at the start and at the end of `samples`: frames at or below `quiet_db` when given (e.g. the recording's floor plus a margin), else more than `drop_db` below its loudest frame. |
| [`find_phrase`](#voxy.splice.find_phrase)(words, phrase, \*[, occurrence])       | Indices `(first, last)` of the `occurrence`-th run of ASR words saying `phrase`.                                                                                                                  |
| [`heard_check`](#voxy.splice.heard_check)(\*[, transcriber, context, heard_as])  | A check: are the new words transcribed where they were spliced in, and the old gone?                                                                                                              |
| [`match_loudness`](#voxy.splice.match_loudness)(samples, reference, ...[, ...])     | `samples` scaled to `reference`'s `level`, its peak capped at `peak`.                                                                                                                             |
| [`quiet_background`](#voxy.splice.quiet_background)(samples, sample_rate, length, \*) | `length` samples of the recording's background, or `None` when it has none quiet enough to trust.                                                                                                 |
| [`quietest`](#voxy.splice.quietest)(samples, sample_rate, lo, hi, \*[, ...])  | The time (s) of the quietest frame in `[lo, hi]`; of equally quiet frames (within `tolerance_db`), the one nearest the middle of the window.                                                      |
| [`recording_floor`](#voxy.splice.recording_floor)(samples, sample_rate)              | The level (dB) of a recording's background: a low percentile of its frames.                                                                                                                       |
| [`replace_words`](#voxy.splice.replace_words)(audio, old, new, \*[, voice, ...])   | Replace the spoken phrase `old` with `new`, rendered in `voice`.                                                                                                                                  |
| [`speech_level`](#voxy.splice.speech_level)(samples, sample_rate, \*[, ...])      | RMS of the loud frames (within `margin_db` of the loudest), so that pauses do not lower it; plain RMS when there is no full frame.                                                                |
| [`tts_render`](#voxy.splice.tts_render)([voice, backend, voices, ...])          | A `render` speaking the text with `voxy.text_to_speech` in `voice`.                                                                                                                               |
| [`voice_check`](#voxy.splice.voice_check)(\*[, embed, min_similarity, window])   | A check: does the new phrase sound like the rest of the recording's speaker?                                                                                                                      |

### Classes

| [`Replacement`](#voxy.splice.Replacement)(speech, old, new, start, end, ...)   | A corrected recording, where the new phrase is, and what the checks found.   |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|

### voxy.splice.DFLT_BACKGROUND_DROP_DB *= 20.0*

Those frames count as background only this far below the speech level, or further.

### voxy.splice.DFLT_BACKGROUND_SHARE *= 0.05*

Share of the recording’s frames, the quietest, that its background is built from.

### voxy.splice.DFLT_BACKGROUND_SPREAD_DB *= 6.0*

… and its quiet frames must be steady (within this spread), as noise is and
speech or music is not.

### voxy.splice.DFLT_CHECK_CONTEXT_S *= 1.0*

Audio heard around the splice when checking it (ASR needs some context).

### voxy.splice.DFLT_LOUD_MARGIN_DB *= 20.0*

Frames within this many dB of the loudest count as speech when matching levels.

### voxy.splice.DFLT_MAX_MARGIN_MS *= 250*

How far a cut may sit from the phrase when the neighbouring gap is wider.

### voxy.splice.DFLT_PAUSE_ABOVE_FLOOR_DB *= 10.0*

A frame within this much of the recording’s floor counts as pause when padding:
soft consonants (/f/, /s/, /h/, 25-35 dB below the vowel) stay speech.

### voxy.splice.DFLT_PAUSE_DROP_DB *= 25.0*

A frame this far below the loudest of a stretch counts as pause, not speech.

### voxy.splice.DFLT_QUIET_TOLERANCE_DB *= 1.0*

Frames within this many dB of the quietest count as equally quiet.

### voxy.splice.DFLT_RENDER_FORMATS *= {'elevenlabs': 'pcm_24000'}*

ElevenLabs returns mp3 by default; raw PCM decodes without ffmpeg.

### voxy.splice.DFLT_SNAP_MS *= 80*

How far past an ASR word boundary the cut may look for a quiet moment.

### voxy.splice.FLOOR_PERCENTILE *= 5*

this percentile of its frame levels.

* **Type:**
  The recording’s floor

### *class* voxy.splice.Replacement(speech, old, new, start, end, replaced, checks=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A corrected recording, where the new phrase is, and what the checks found.

#### *property* audio *: [bytes](https://docs.python.org/3/builtins/stdtypes.html#bytes)*

The WAV bytes.

#### *property* passed *: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None)*

True if every check passed; None when no check ran.

#### save(path)

Write the WAV to `path`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.splice.cap_peak(samples, peak=0.99)

`samples`, scaled down if needed so that no sample exceeds `peak`.

```pycon
>>> cap_peak(np.array([0.5, -2.0]), 1.0).tolist()
[0.25, -1.0]
```

* **Return type:**
  `ndarray`

### voxy.splice.crossfade_join(head, middle, tail, , fade)

`(head + middle + tail, where middle starts in it)`, with linear crossfades
of up to `fade` samples at each join (shorter where a piece is too short; none
at an edge of the audio).

```pycon
>>> joined, at = crossfade_join(np.ones(10), np.ones(6), np.ones(10), fade=2)
>>> len(joined), at
(22, 8)
>>> joined, at = crossfade_join(np.ones(0), np.ones(6), np.ones(10), fade=2)
>>> len(joined), at
(14, 0)
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### voxy.splice.cut_points(samples, sample_rate, words, first, last, , max_margin=0.25, snap=0.08)

Where to cut (s): the quietest moment before `words[first]` and after
`words[last]`, searched in the gap to the neighbouring word (up to
`max_margin`) and at least `snap` past the ASR boundary. A phrase at an
edge of the audio is cut at that edge.

```pycon
>>> sr, tone = 8000, np.ones(8000)
>>> x = np.concatenate([tone, np.zeros(1600), tone, np.zeros(8000), tone])
>>> ws = [{"start": 0, "end": 1}, {"start": 1.2, "end": 2.2}, {"start": 3.2, "end": 4.2}]
>>> cut_points(x, sr, ws, 1, 1), cut_points(x, sr, ws, 0, 0)[0], cut_points(x, sr, ws, 2, 2)[1]
((1.095, 2.325), 0.0, 4.2)
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### voxy.splice.edge_pauses(samples, sample_rate, , quiet_db=None, drop_db=25.0)

Seconds of pause at the start and at the end of `samples`: frames at or below
`quiet_db` when given (e.g. the recording’s floor plus a margin), else more than
`drop_db` below its loudest frame.

```pycon
>>> x = np.concatenate([np.zeros(800), 0.5 * np.ones(1600), np.zeros(2400)])
>>> edge_pauses(x, 8000)
(0.1, 0.3)
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### voxy.splice.find_phrase(words, phrase, , occurrence=0)

Indices `(first, last)` of the `occurrence`-th run of ASR words saying `phrase`.

```pycon
>>> ws = [{"text": " Anna"}, {"text": " Smyth,"}, {"text": " Anna"}, {"text": " Smyth."}]
>>> find_phrase(ws, "anna smyth"), find_phrase(ws, "Anna … Smyth", occurrence=1)
((0, 1), (2, 3))
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### voxy.splice.heard_check(, transcriber=None, context=1.0, heard_as=None)

A check: are the new words transcribed where they were spliced in, and the old gone?

`transcriber(wav_path) -> segments` (default `voxy.refprep.transcribe`) runs
on the splice plus `context` seconds each side. `heard_as` is the new phrase
as ASR writes it, when that differs from the text rendered (“Dr. Patterson”
for “Doctor Patterson”); default: the replacement’s `new`. The old phrase
counts as heard only as a whole run of its words (“anna jones”, not “anna”),
and not at all when the new phrase contains it.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int), [`Replacement`](#voxy.splice.Replacement)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.splice.match_loudness(samples, reference, sample_rate, \*, level=<function speech_level>, peak=0.99)

`samples` scaled to `reference`’s `level`, its peak capped at `peak`.

```pycon
>>> x = match_loudness(0.1 * np.ones(160), np.r_[0.5 * np.ones(160), np.zeros(800)], 8000)
>>> round(float(x.max()), 3)
0.5
```

* **Return type:**
  `ndarray`

### voxy.splice.quiet_background(samples, sample_rate, length, , share=0.05, drop_db=20.0, spread_db=6.0)

`length` samples of the recording’s background, or `None` when it has none
quiet enough to trust.

The background is made of the quietest `share` of its 10 ms frames, in time
order, overlap-added (`_overlap_add`) and cycled to `length`. It is used only
when every one of those frames is at least `drop_db` below `speech_level` and
they are steady (within `spread_db` of each other): otherwise they would carry
speech (a recording with no pause) or music (wall-to-wall audio), and nothing is
better than that under the new words.
O(len(samples)).

```pycon
>>> rng = np.random.default_rng(0)
>>> x = np.concatenate([np.full(4000, 0.5), 0.005 * rng.standard_normal(2400)])
>>> bg = quiet_background(x, 8000, 5000)
>>> len(bg), round(float(np.sqrt(np.mean(bg**2))), 4)  # the floor's quietest frames
(5000, 0.0044)
>>> print(quiet_background(np.full(8000, 0.5), 8000, 5000))  # wall-to-wall
None
```

* **Return type:**
  `ndarray` | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.splice.quietest(samples, sample_rate, lo, hi, , tolerance_db=1.0, fallback=None)

The time (s) of the quietest frame in `[lo, hi]`; of equally quiet frames
(within `tolerance_db`), the one nearest the middle of the window. When no
frame lies there (an empty window, or one past the last full frame):
`fallback`, default `lo`.

```pycon
>>> x = np.concatenate([np.ones(800), np.zeros(800), np.ones(800)])
>>> quietest(x, 8000, 0.0, 0.3)
0.145
>>> quietest(x, 8000, 0.3, 0.3), quietest(x, 8000, 0.3, 0.35, fallback=0.31)
(0.3, 0.31)
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### voxy.splice.recording_floor(samples, sample_rate)

The level (dB) of a recording’s background: a low percentile of its frames.

```pycon
>>> rng = np.random.default_rng(0)
>>> x = np.concatenate([0.5 * np.ones(8000), 0.01 * rng.standard_normal(8000)])
>>> round(recording_floor(x, 8000))
-41
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### voxy.splice.replace_words(audio, old, new, \*, voice=None, render=None, words=None, transcriber=None, sample_rate=None, occurrence=0, crossfade_ms=15, max_margin_ms=250, snap_ms=80, level=<function speech_level>, keep_pauses=True, background=<function quiet_background>, verify=None)

Replace the spoken phrase `old` with `new`, rendered in `voice`.

* **Parameters:**
  * **audio** – A WAV path or bytes (channels averaged), or raw mono samples (with
    `sample_rate=`; integer PCM is scaled to `[-1, 1)`).
  * **old** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The phrase as spoken (several words: see the module docstring).
  * **new** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – What it should say.
  * **voice** – The speaker’s voice, for the default `render` (a library name, a
    backend voice, a profile). `voice` or `render` is required.
  * **render** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `text -> (samples, sample_rate)`; default `tts_render(voice)`.
    A render that is silent raises `ValueError`.
  * **words** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Timed ASR words of `audio`; default `transcriber(audio)`.
  * **transcriber** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `wav_path -> segments` (default `voxy.refprep.transcribe`),
    for `words` and the default check.
  * **occurrence** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Which occurrence of `old` (0: the first).
  * **crossfade_ms** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Length of each crossfade.
  * **max_margin_ms** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – The furthest a cut may sit from the phrase, in a gap.
  * **snap_ms** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – How far past an ASR boundary a cut may look for quiet.
  * **level** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – `(samples, sample_rate) -> level` the render is matched on;
    default `speech_level` (e.g. `partial(speech_level, margin_db=10)`).
  * **keep_pauses** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Put back, around the render, the pauses at the edges of the
    cut (taken from the cut itself, background included), so the pauses
    keep their length; `False` for the tighter splice.
  * **background** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)], `ndarray` | [`None`](https://docs.python.org/3/builtins/constants.html#None)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(samples, sample_rate, length) -> samples or None`: what to
    mix under the render (`None` from it: nothing); default
    `quiet_background`, which gives nothing for audio with no quiet
    frames (wall-to-wall speech or music); `None` for no background.
  * **verify** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int), [`Replacement`](#voxy.splice.Replacement)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)] | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int), [`Replacement`](#voxy.splice.Replacement)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A check or checks (`(samples, sample_rate, replacement) -> dict`
    with a `passed` key); default `heard_check`; `()` for none.
* **Return type:**
  [`Replacement`](#voxy.splice.Replacement)

### voxy.splice.speech_level(samples, sample_rate, , margin_db=20.0)

RMS of the loud frames (within `margin_db` of the loudest), so that pauses
do not lower it; plain RMS when there is no full frame.

```pycon
>>> x = np.concatenate([0.5 * np.ones(800), np.zeros(8000)])
>>> round(speech_level(x, 8000), 3), round(float(np.sqrt(np.mean(x**2))), 3)
(0.5, 0.151)
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### voxy.splice.tts_render(voice=None, , backend=None, voices=None, use_library=True, model=None, \*\*tts_kwargs)

A `render` speaking the text with `voxy.text_to_speech` in `voice`.

The voice is resolved once, here (`voxy.facade.resolve_voice`, with
`voices`, `use_library` and `model`’s backend), and every render
synthesizes with what it resolved to. Backends whose default output is
compressed get a WAV/PCM format (`DFLT_RENDER_FORMATS`) unless
`output_format=` is given; other `tts_kwargs` go to `text_to_speech`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]]

### voxy.splice.voice_check(, embed=None, min_similarity=None, window=None)

A check: does the new phrase sound like the rest of the recording’s speaker?

Cosine similarity of the voice print of the new phrase and of the rest
(`voxy.speakers.voice_print`, ECAPA in its own environment unless `embed`
is given). `min_similarity` defaults to `speakers.DFLT_MATCH_SIMILARITY`
(below it, a window counts as someone else); `window` (s) defaults to the
speakers’ window, shortened to the phrase. Fails, with a `reason`, when the
new phrase is the whole recording.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int), [`Replacement`](#voxy.splice.Replacement)], [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
