# voxy.refprep

Prepare a speaker’s recordings for cloning and fine-tuning: clean, cut, transcribe, score.

```pycon
>>> from voxy import refprep
>>> clips = refprep.prepare_reference(["tape1.m4a", "tape2.mp3"], voice="ada")
>>> ref = refprep.best_references(clips)[0]
>>> ref["path"], ref["text"], ref["score"]
```

The steps, each a function you can call alone or replace:

1. `condition` (ffmpeg): any format to mono PCM WAV at `sample_rate`, with a
   high-pass, optional denoising (`denoise=`: ‘afftdn’, ‘anlmdn’, or None) and
   loudness normalisation (EBU R128, `loudness_lufs`).
2. `transcribe`: timed segments and words from a local Whisper (scribed’s
   faster-whisper backend, in its own environment, see `voxy.envs`).
3. `plan_clips`: cut points at sentence/segment boundaries, so each clip is
   `min_seconds`-`max_seconds` long and carries its exact text.
4. `score_clip`: duration, signal-to-noise estimate, clipping, level, ASR
   confidence and speaking rate, folded into a `score` in [0, 1].

`prepare_reference` runs them all and saves the clips plus `manifest.jsonl` in
the voice’s samples store (`voxy.stores.samples_store`), so any backend (a
reference clip) and fine-tuning (`voxy.finetune`, the whole set) read the same data.

### Module Attributes

| [`DENOISE_FILTERS`](#voxy.refprep.DENOISE_FILTERS)   | no denoising).   |
|--------------------------------------------------------------------|------------------|

### Functions

| [`best_references`](#voxy.refprep.best_references)(manifest, \*[, n, ...])            | The `n` best clips for zero-shot reference (length in range, best score first).         |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`close_asr`](#voxy.refprep.close_asr)()                                        | Stop the transcription workers (free their memory).                                     |
| [`condition`](#voxy.refprep.condition)(src, dst, \*[, sample_rate, start, ...]) | Convert `src` (any format ffmpeg reads) to clean mono 16-bit WAV at `dst`.              |
| [`conditioning_filters`](#voxy.refprep.conditioning_filters)(\*[, denoise, ...])           | The ffmpeg audio filter chain for `condition`.                                          |
| [`load_manifest`](#voxy.refprep.load_manifest)(samples)                             | The manifest saved by `prepare_reference` in a samples store.                           |
| [`partial_transcribe`](#voxy.refprep.partial_transcribe)([rootdir])                      | The default transcriber, bound to a data root.                                          |
| [`plan_clips`](#voxy.refprep.plan_clips)(segments, \*[, min_seconds, ...])       | Clips `[{start, end, text, words}]` cut between words, ending sentences where possible. |
| [`prepare_reference`](#voxy.refprep.prepare_reference)(sources, \*[, voice, ...])       | Condition, transcribe, cut and score `sources`; save clips and manifest; return it.     |
| [`score_clip`](#voxy.refprep.score_clip)(samples, sample_rate, \*[, words, ...]) | Quality measures of a clip, and their combination `score` in [0, 1].                    |
| [`summarize`](#voxy.refprep.summarize)(manifest)                                | Totals for a manifest: clips, minutes, minutes above score thresholds.                  |
| [`transcribe`](#voxy.refprep.transcribe)(audio, \*[, language, model, ...])      | Timed segments `[{start, end, text, confidence, words: [...]}, ...]` of `audio`.        |
| [`transcribe_text`](#voxy.refprep.transcribe_text)(audio, \*\*kwargs)                 | The transcript of `audio` as one string.                                                |

### voxy.refprep.DENOISE_FILTERS *= {'afftdn': ['afftdn=nr=12:nf=-50:tn=1'], 'anlmdn': ['anlmdn=s=0.0005'], None: []}*

no denoising).

* **Type:**
  ffmpeg filters for each `denoise` strategy (None

### voxy.refprep.best_references(manifest, , n=3, min_seconds=5.0, max_seconds=12.0, min_score=0.0)

The `n` best clips for zero-shot reference (length in range, best score first).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

```pycon
>>> m = [{"duration": 8, "score": 0.7}, {"duration": 2, "score": 0.9},
...      {"duration": 10, "score": 0.8}]
>>> [c["score"] for c in best_references(m, n=2)]
[0.8, 0.7]
```

### voxy.refprep.close_asr()

Stop the transcription workers (free their memory).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.refprep.condition(src, dst, \*, sample_rate=24000, start=None, duration=None, filters=None, run=<function run>)

Convert `src` (any format ffmpeg reads) to clean mono 16-bit WAV at `dst`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.refprep.conditioning_filters(, denoise='afftdn', loudness_lufs=-20.0, highpass_hz=70)

The ffmpeg audio filter chain for `condition`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

```pycon
>>> conditioning_filters(denoise=None, loudness_lufs=-20, highpass_hz=70)
['highpass=f=70', 'loudnorm=I=-20:TP=-1.5:LRA=11']
```

### voxy.refprep.load_manifest(samples)

The manifest saved by `prepare_reference` in a samples store.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.refprep.partial_transcribe(rootdir=None)

The default transcriber, bound to a data root.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### voxy.refprep.plan_clips(segments, , min_seconds=3.0, max_seconds=12.0, pad_seconds=0.12, total_seconds=None)

Clips `[{start, end, text, words}]` cut between words, ending sentences where possible.

Words accumulate until the clip passes `min_seconds` and a word ends a
sentence (or the next word would push it past `max_seconds`). Cuts fall in
the pause between two words, padded by up to `pad_seconds`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

```pycon
>>> w = lambda s, e, t: {"start": s, "end": e, "text": t, "confidence": 0.9}
>>> segs = [{"words": [w(0, 1, "One"), w(1.1, 2, "two."), w(3, 4, "Three"),
...                    w(4.1, 5, "four.")], "start": 0, "end": 5, "text": ""}]
>>> [c["text"] for c in plan_clips(segs, min_seconds=1.5, max_seconds=3)]
['One two.', 'Three four.']
```

### voxy.refprep.prepare_reference(sources, \*, voice=None, samples=None, sample_rate=24000, denoise='afftdn', loudness_lufs=-20.0, min_seconds=3.0, max_seconds=12.0, language=None, transcriber=None, conditioner=None, rootdir=None, log=<built-in function print>)

Condition, transcribe, cut and score `sources`; save clips and manifest; return it.

* **Parameters:**
  * **sources** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)]) – Recordings, any format ffmpeg reads.
  * **voice** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice name: clips go to `samples_store(voice)`.
  * **samples** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A store to write to instead (`filename -> bytes`).
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Conditioning (see `condition`).
  * **denoise** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Conditioning (see `condition`).
  * **loudness_lufs** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Conditioning (see `condition`).
  * **min_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Clip length bounds.
  * **max_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Clip length bounds.
  * **language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Force the ASR language (default: detected).
  * **transcriber** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(wav_path, language=) -> segments` (default `transcribe`).
  * **conditioner** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(src, dst, sample_rate=, filters=) -> path` (default `condition`).
* **Returns:**
  one dict per clip (`key`, `path` if on disk, `text`,
  `source`, `start`, `end`, and the `score_clip` measures), best first.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.refprep.score_clip(samples, sample_rate, , words=(), text='', min_seconds=3.0, max_seconds=12.0)

Quality measures of a clip, and their combination `score` in [0, 1].

- `snr_db`: loud-frame level minus quiet-frame level (speech vs floor).
- `clipped`: fraction of samples at full scale.
- `level_db`: level of the loud frames.
- `asr_confidence`: mean Whisper word probability (low: mumbling, noise, music).
- `chars_per_second`: speaking rate (outliers: bad alignment).

```pycon
>>> rng = np.random.default_rng(0)
>>> speechy = np.concatenate([0.3 * rng.standard_normal(40000), 0.001 * rng.standard_normal(8000)])
>>> s = score_clip(speechy, 8000, text="x" * 70)
>>> s["snr_db"] > 30, 0 < s["score"] <= 1
(True, True)
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.refprep.summarize(manifest)

Totals for a manifest: clips, minutes, minutes above score thresholds.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

```pycon
>>> summarize([{"duration": 60, "score": 0.9}, {"duration": 30, "score": 0.4}])
{'clips': 2, 'minutes': 1.5, 'minutes_score_0.5+': 1.0, 'minutes_score_0.7+': 1.0}
```

### voxy.refprep.transcribe(audio, , language=None, model='large-v3-turbo', threads=None, rootdir=None)

Timed segments `[{start, end, text, confidence, words: [...]}, ...]` of `audio`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.refprep.transcribe_text(audio, \*\*kwargs)

The transcript of `audio` as one string.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
