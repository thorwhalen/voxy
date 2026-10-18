# voxy.wavutil

Mono PCM WAV in and out as float arrays, plus the few edits narration needs.

Every local backend and the reference-prep step exchange 16-bit PCM WAV files;
this module turns them into `float32` arrays in `[-1, 1]` and back, trims
silence, and joins pieces with pauses, using only numpy and the standard library.

```pycon
>>> import numpy as np
>>> data = write_wav(np.array([0.0, 0.5, -0.5], dtype=np.float32), 8000)
>>> samples, sr = read_wav(data)
>>> sr, np.round(samples, 3).tolist()
(8000, [0.0, 0.5, -0.5])
```

### Functions

| [`frame_db`](#voxy.wavutil.frame_db)(samples, sample_rate, \*[, frame_ms])   | RMS level (dBFS) of consecutive `frame_ms` frames.                                |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`read_wav`](#voxy.wavutil.read_wav)(source)                                 | `(samples, sample_rate)` of a PCM WAV (path or bytes), channels averaged.         |
| [`resample`](#voxy.wavutil.resample)(samples, from_rate, to_rate)            | Linear-interpolation resampling (enough to join pieces; use ffmpeg for masters).  |
| [`silence`](#voxy.wavutil.silence)(ms, sample_rate)                         | `ms` milliseconds of silence.                                                     |
| [`trim_silence`](#voxy.wavutil.trim_silence)(samples, sample_rate, \*[, ...])    | Drop leading and trailing frames quieter than `threshold_db` (keeping `keep_ms`). |
| [`write_wav`](#voxy.wavutil.write_wav)(samples, sample_rate)                  | Mono 16-bit PCM WAV bytes from floats in `[-1, 1]` (clipped).                     |

### voxy.wavutil.frame_db(samples, sample_rate, , frame_ms=10)

RMS level (dBFS) of consecutive `frame_ms` frames.

```pycon
>>> frame_db(np.zeros(160), 8000).round().tolist()
[-100.0, -100.0]
```

### voxy.wavutil.read_wav(source)

`(samples, sample_rate)` of a PCM WAV (path or bytes), channels averaged.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### voxy.wavutil.resample(samples, from_rate, to_rate)

Linear-interpolation resampling (enough to join pieces; use ffmpeg for masters).

```pycon
>>> len(resample(np.zeros(100), 8000, 16000))
200
```

* **Return type:**
  `ndarray`

### voxy.wavutil.silence(ms, sample_rate)

`ms` milliseconds of silence.

```pycon
>>> len(silence(250, 8000))
2000
```

* **Return type:**
  `ndarray`

### voxy.wavutil.trim_silence(samples, sample_rate, , threshold_db=-45.0, keep_ms=60)

Drop leading and trailing frames quieter than `threshold_db` (keeping `keep_ms`).

```pycon
>>> x = np.concatenate([np.zeros(800), 0.5 * np.ones(800), np.zeros(800)])
>>> len(trim_silence(x, 8000, keep_ms=0))
800
```

* **Return type:**
  `ndarray`

### voxy.wavutil.write_wav(samples, sample_rate)

Mono 16-bit PCM WAV bytes from floats in `[-1, 1]` (clipped).

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)
