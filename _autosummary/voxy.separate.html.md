# voxy.separate

Split a recording into the voice and the rest (music, noise) before using the voice.

A song or a noisy interview has to lose its backing before its voice can be cloned,
transcribed or edited. `separate` runs Demucs (MIT; `htdemucs_ft` by default, the
fine-tuned hybrid transformer) in its own environment, built on first use under the
data root like the other local models, and returns the two stems as WAV bytes:
the voice, and `rest`, the input minus the voice (so the two add back up to it).

```pycon
>>> from voxy.separate import separate
>>> stems = separate("song.mp3")
>>> open("vocals.wav", "wb").write(stems["vocals"])
```

`run` is the seam (`(audio_path, out_dir, *, stem, sample_rate, \*\*options) ->
{stem: path}`); tests and other separators plug in there.

```pycon
>>> import tempfile, pathlib
>>> def fake(audio, out_dir, **options):
...     paths = {n: pathlib.Path(out_dir) / f"{n}.wav" for n in ("vocals", "rest")}
...     for p in paths.values():
...         p.write_bytes(b"RIFF")
...     return paths
>>> sorted(separate(b"RIFF....", run=fake))
['rest', 'vocals']
```

### Module Attributes

| [`DFLT_STEM`](#voxy.separate.DFLT_STEM)      | The stem kept as "the voice"; Demucs also has drums, bass and other.            |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`FLOAT_SUBTYPE`](#voxy.separate.FLOAT_SUBTYPE)  | float, so no level is lost.                                                     |
| [`MODEL_SOURCES`](#voxy.separate.MODEL_SOURCES)  | The stems of the pretrained models whose sources are known before loading them. |
| [`SEPARATION_ENV`](#voxy.separate.SEPARATION_ENV) | torch is bounded to the releases Demucs 4.0.1 was checked with.                 |

### Functions

| [`check_stem`](#voxy.separate.check_stem)(stem, model)                       | Raise `ValueError` if `model` is known (`MODEL_SOURCES`) and lacks `stem`.                                                                                                                                                                                                       |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`close_separation`](#voxy.separate.close_separation)()                            | Stop the Demucs workers (free their memory).                                                                                                                                                                                                                                     |
| [`separate`](#voxy.separate.separate)(audio, \*[, stem, sample_rate, run]) | `{stem: wav, "rest": wav}`: `audio` split into `stem` and everything else.                                                                                                                                                                                                       |
| [`speech_separator`](#voxy.separate.speech_separator)(\*[, stem, run])             | `separate` as a `(samples (frames, channels), sample_rate) -> (speech, music)` callable, the separator seam of reeleehq/mixing (`mixing.audio.separation.separate_speech(audio, separator=speech_separator())`), so mixing can run Demucs out of process, in voxy's environment. |
| [`worker_run`](#voxy.separate.worker_run)(\*[, model, threads, rootdir])     | The default `run`: the Demucs adapter in its own environment.                                                                                                                                                                                                                    |

### voxy.separate.DFLT_STEM *= 'vocals'*

The stem kept as “the voice”; Demucs also has drums, bass and other.

### voxy.separate.FLOAT_SUBTYPE *= 'FLOAT'*

float, so no level is lost.

* **Type:**
  What `speech_separator` asks the worker to write

### voxy.separate.MODEL_SOURCES *= {'htdemucs': ('drums', 'bass', 'other', 'vocals'), 'htdemucs_6s': ('drums', 'bass', 'other', 'vocals', 'guitar', 'piano'), 'htdemucs_ft': ('drums', 'bass', 'other', 'vocals')}*

The stems of the pretrained models whose sources are known before loading them.

### voxy.separate.SEPARATION_ENV *= EnvSpec(name='demucs', requirements=('demucs==4.0.1', 'torch>=2.11,<2.16', 'soundfile', 'numpy'), python='3.11', torch_backend='cpu', env_vars={'TORCH_HOME': '{models}'})*

torch is bounded to the releases Demucs 4.0.1 was checked with. torchaudio comes in
as a Demucs dependency, unpinned: Demucs imports it but never calls it here.

### voxy.separate.check_stem(stem, model)

Raise `ValueError` if `model` is known (`MODEL_SOURCES`) and lacks `stem`.

```pycon
>>> check_stem("vocals", "htdemucs_ft")
>>> check_stem("guitar", "htdemucs_ft")
Traceback (most recent call last):
  ...
ValueError: 'guitar' is not a stem of htdemucs_ft: ['drums', 'bass', 'other', 'vocals']
```

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.separate.close_separation()

Stop the Demucs workers (free their memory).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.separate.separate(audio, , stem='vocals', sample_rate=None, run=None, \*\*options)

`{stem: wav, "rest": wav}`: `audio` split into `stem` and everything else.

`rest` is the input minus the voice, so `stem + rest` rebuilds the input (as
the model hears it: resampled and with its channels). Both are 16-bit PCM WAV
with the model’s channels (Demucs: stereo; a mono input is duplicated, the first
two channels of a wider one are kept) at the model’s rate (Demucs: 44.1 kHz)
unless `sample_rate` says otherwise, whatever the input. If either stem would
clip, both are divided by one common factor (Demucs’s `prevent_clip`), so they
still add up to the input times that factor.

The whole file is decoded and held in memory, and Demucs takes about 24 s per
minute of audio on CPU for one model (`htdemucs_ft` runs only the sub-model
fine-tuned for `stem`); split hours-long recordings first.

* **Parameters:**
  * **audio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – A path to any format soundfile reads (WAV, FLAC, OGG, MP3 with a
    recent libsndfile), or WAV bytes.
  * **stem** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The stem to keep apart (`"vocals"`; Demucs also has drums, bass,
    other).
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The rate of the returned WAVs (default: the model’s).
  * **run** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(audio_path, out_dir, *, stem, sample_rate, \*\*options) ->
    {stem: path}`; default `worker_run()` (Demucs `htdemucs_ft` on CPU).
  * **\*\*options** – Passed to `run` (Demucs: `shifts`, `overlap`).
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]

### voxy.separate.speech_separator(, stem='vocals', run=None, \*\*options)

`separate` as a `(samples (frames, channels), sample_rate) -> (speech, music)`
callable, the separator seam of reeleehq/mixing
(`mixing.audio.separation.separate_speech(audio, separator=speech_separator())`),
so mixing can run Demucs out of process, in voxy’s environment.

Mono or stereo in; the stems come back at the input’s rate, channels and length,
at the input’s level (no clip rescaling, float WAV in between: `speech + music`
is the input). Keyword arguments are `separate`’s. For mixing’s `remove_talk`,
give it htdemucs’s context: `SeparationStrategy(..., context_s=3.0)` around this
callable (a bare callable gets mixing’s 1 s default).

```pycon
>>> def fake(audio, out_dir, *, stem, sample_rate, **options):
...     samples, sr = read_wav_channels(audio)
...     paths = {}
...     for name, part in ((stem, samples * 0.75), ("rest", samples * 0.25)):
...         paths[name] = f"{out_dir}/{name}.wav"
...         open(paths[name], "wb").write(write_wav_channels(part, sr))
...     return paths
>>> speech, music = speech_separator(run=fake)(np.ones((4, 2)), 8000)
>>> speech.shape, float(speech[0, 0]), float(music[0, 1])
((4, 2), 0.75, 0.25)
```

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, `ndarray`]]

### voxy.separate.worker_run(, model='htdemucs_ft', threads=None, rootdir=None)

The default `run`: the Demucs adapter in its own environment.

A stem the model lacks fails before the worker starts for the models in
`MODEL_SOURCES`, and inside the worker (a `WorkerError` carrying the same
message) for the others.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]
