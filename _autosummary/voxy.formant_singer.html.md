# voxy.formant_singer

A formant synthesiser that sings “la” on a list of notes: voices that belong to no one.

A zero-shot singing model learns a timbre from a sung prompt clip. When no consenting
singer is at hand (a default voice, a test, a demo), this makes one: glottal-like
harmonics shaped by /l/ then /a/ formants, a portamento between notes, a delayed
vibrato and a little breath noise. Because the clip is rendered from known notes,
its notes and words are exact by construction, which is what an SVS prompt needs.
It is buzzy, and words sung in it come out less clearly than with a real voice.

```pycon
>>> from voxy.score import notes_from_pitches
>>> samples = sing_la(notes_from_pitches([(60, 1)], bpm=120, start=0.1), sample_rate=8000)
>>> len(samples), float(abs(samples).max()) <= 1.0
(8000, True)
```

### Module Attributes

| [`SYNTH_VERSION`](#voxy.formant_singer.SYNTH_VERSION)     | Bump when the synthesis changes, so cached renders are made again.         |
|--------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`FORMANT_PRESETS`](#voxy.formant_singer.FORMANT_PRESETS)   | `(centre Hz, bandwidth Hz)` per formant, for /a/ and /l/, per preset.      |
| [`PORTAMENTO_S`](#voxy.formant_singer.PORTAMENTO_S)      | Glide between notes (time constant of a one-pole smoother, run both ways). |
| [`CONSONANT_S`](#voxy.formant_singer.CONSONANT_S)       | /l/ turns into /a/ over this long at each onset.                           |
| [`NYQUIST_MARGIN_HZ`](#voxy.formant_singer.NYQUIST_MARGIN_HZ) | Harmonics stop this far below Nyquist.                                     |

### Functions

| [`sing_la`](#voxy.formant_singer.sing_la)(notes, \*[, sample_rate, preset, ...])   | Mono float32 samples of "la" sung on `notes` (silence before the first note, `tail_s` after the last), peaking at 0.8.   |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|

### voxy.formant_singer.CONSONANT_S *= 0.08*

/l/ turns into /a/ over this long at each onset.

### voxy.formant_singer.FORMANT_PRESETS *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Sequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[float](https://docs.python.org/3/builtins/functions.html#float), [float](https://docs.python.org/3/builtins/functions.html#float)]]]]* *= {'high': {'a': ((850, 80), (1220, 90), (2810, 120), (3800, 150), (4950, 200)), 'l': ((380, 60), (1100, 100), (2700, 150), (3700, 180), (4900, 220))}, 'low': {'a': ((730, 80), (1090, 90), (2440, 120), (3400, 150), (4500, 200)), 'l': ((360, 60), (1000, 100), (2500, 150), (3300, 180), (4400, 220))}}*

`(centre Hz, bandwidth Hz)` per formant, for /a/ and /l/, per preset.

### voxy.formant_singer.NYQUIST_MARGIN_HZ *= 200*

Harmonics stop this far below Nyquist.

### voxy.formant_singer.PORTAMENTO_S *= 0.035*

Glide between notes (time constant of a one-pole smoother, run both ways).

### voxy.formant_singer.SYNTH_VERSION *= 1*

Bump when the synthesis changes, so cached renders are made again.

### voxy.formant_singer.sing_la(notes, , sample_rate=24000, preset='high', tail_s=0.4, seed=0)

Mono float32 samples of “la” sung on `notes` (silence before the first note,
`tail_s` after the last), peaking at 0.8.

* **Return type:**
  `ndarray`
