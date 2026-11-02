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

### Module Attributes

| [`WAVE_FORMAT_PCM`](#voxy.wavutil.WAVE_FORMAT_PCM)        | integer PCM, IEEE float, and the extensible wrapper.                       |
|-------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`WAVE_FORMAT_FLOAT`](#voxy.wavutil.WAVE_FORMAT_FLOAT)      | integer PCM, IEEE float, and the extensible wrapper.                       |
| [`WAVE_FORMAT_EXTENSIBLE`](#voxy.wavutil.WAVE_FORMAT_EXTENSIBLE) | integer PCM, IEEE float, and the extensible wrapper.                       |
| [`TRAILING_CHUNK_IDS`](#voxy.wavutil.TRAILING_CHUNK_IDS)     | Chunks that can follow audio whose `data` size is a streaming placeholder. |
| [`DFLT_LOWPASS_TAPS`](#voxy.wavutil.DFLT_LOWPASS_TAPS)      | Taps of the anti-aliasing filter applied before downsampling (odd).        |
| [`DFLT_LOWPASS_SHARE`](#voxy.wavutil.DFLT_LOWPASS_SHARE)     | The filter's cutoff, as a share of the target rate's Nyquist frequency.    |

### Functions

| [`frame_db`](#voxy.wavutil.frame_db)(samples, sample_rate, \*[, frame_ms])   | RMS level (dBFS) of consecutive `frame_ms` frames.                                                                                                                                                                                  |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`lowpass`](#voxy.wavutil.lowpass)(samples, cutoff_hz, sample_rate, \*)     | `samples` with what lies above `cutoff_hz` removed (windowed-sinc FIR, Blackman window, zero phase: same length, no delay).                                                                                                         |
| [`pcm_to_float`](#voxy.wavutil.pcm_to_float)(raw[, width])                       | Little-endian PCM samples of `width` bytes as floats in about `[-1, 1]` (the most negative sample is a hair below -1).                                                                                                              |
| [`read_wav`](#voxy.wavutil.read_wav)(source)                                 | `(samples, sample_rate)` of a WAV (path or bytes), channels averaged; reads what [`read_wav_channels()`](#voxy.wavutil.read_wav_channels) reads.                                                                        |
| [`read_wav_channels`](#voxy.wavutil.read_wav_channels)(source)                        | `(samples (frames, channels), sample_rate)` of a WAV (path or bytes).                                                                                                                                                               |
| [`resample`](#voxy.wavutil.resample)(samples, from_rate, to_rate)            | Resampling for joining pieces (use ffmpeg or soxr for masters): low-passed below the target's Nyquist frequency first when downsampling (so a 24 kHz render put into an 8 kHz recording does not alias), then linear interpolation. |
| [`silence`](#voxy.wavutil.silence)(ms, sample_rate)                         | `ms` milliseconds of silence.                                                                                                                                                                                                       |
| [`trim_silence`](#voxy.wavutil.trim_silence)(samples, sample_rate, \*[, ...])    | Drop leading and trailing frames quieter than `threshold_db` (keeping `keep_ms`).                                                                                                                                                   |
| [`write_wav`](#voxy.wavutil.write_wav)(samples, sample_rate)                  | Mono 16-bit PCM WAV bytes from floats in `[-1, 1]` (clipped).                                                                                                                                                                       |
| [`write_wav_channels`](#voxy.wavutil.write_wav_channels)(samples, sample_rate)         | 32-bit float WAV bytes of `samples` `(frames, channels)` (or 1-D: mono), lossless, for handing audio to another process.                                                                                                            |

### voxy.wavutil.DFLT_LOWPASS_SHARE *= 0.95*

The filter’s cutoff, as a share of the target rate’s Nyquist frequency.

### voxy.wavutil.DFLT_LOWPASS_TAPS *= 255*

Taps of the anti-aliasing filter applied before downsampling (odd).

### voxy.wavutil.TRAILING_CHUNK_IDS *= re.compile(b'LIST|id3 |ID3 |fact|cue |smpl|bext|JUNK|PAD |iXML|_PMX|acid|inst|plst')*

Chunks that can follow audio whose `data` size is a streaming placeholder.

### voxy.wavutil.WAVE_FORMAT_EXTENSIBLE *= 65534*

integer PCM, IEEE float, and the extensible wrapper.

* **Type:**
  WAV `fmt` format tags

### voxy.wavutil.WAVE_FORMAT_FLOAT *= 3*

integer PCM, IEEE float, and the extensible wrapper.

* **Type:**
  WAV `fmt` format tags

### voxy.wavutil.WAVE_FORMAT_PCM *= 1*

integer PCM, IEEE float, and the extensible wrapper.

* **Type:**
  WAV `fmt` format tags

### voxy.wavutil.frame_db(samples, sample_rate, , frame_ms=10)

RMS level (dBFS) of consecutive `frame_ms` frames.

```pycon
>>> frame_db(np.zeros(160), 8000).round().tolist()
[-100.0, -100.0]
```

### voxy.wavutil.lowpass(samples, cutoff_hz, sample_rate, , taps=255)

`samples` with what lies above `cutoff_hz` removed (windowed-sinc FIR,
Blackman window, zero phase: same length, no delay).

```pycon
>>> sr = 48000; t = np.arange(sr) / sr
>>> hiss = lowpass(np.sin(2 * np.pi * 12000 * t), 4000, sr)
>>> bool(np.abs(hiss[1000:-1000]).max() < 0.01)
True
```

* **Return type:**
  `ndarray`

### voxy.wavutil.pcm_to_float(raw, width=2)

Little-endian PCM samples of `width` bytes as floats in about `[-1, 1]`
(the most negative sample is a hair below -1).

8-bit PCM is unsigned (silence is 128); a trailing partial sample (a stream cut
mid-sample) is dropped.

```pycon
>>> pcm_to_float(b"\x00\x00\xff\x7f\x01").tolist()
[0.0, 1.0]
>>> pcm_to_float(bytes([0, 128, 255]), 1).tolist()
[-1.0, 0.0, 0.9921875]
>>> pcm_to_float(bytes([0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x80]), 3).tolist()
[1.0, -1.0000001192092896]
```

* **Return type:**
  `ndarray`

### voxy.wavutil.read_wav(source)

`(samples, sample_rate)` of a WAV (path or bytes), channels averaged; reads
what [`read_wav_channels()`](#voxy.wavutil.read_wav_channels) reads.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### voxy.wavutil.read_wav_channels(source)

`(samples (frames, channels), sample_rate)` of a WAV (path or bytes).

Integer PCM (8, 16, 24, 32 bit) and 32-bit float, plain or extensible headers,
and streamed files whose header carries a placeholder size (the audio then ends
at a trailing chunk such as `LIST`, or at the end of the file).

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### voxy.wavutil.resample(samples, from_rate, to_rate)

Resampling for joining pieces (use ffmpeg or soxr for masters): low-passed below
the target’s Nyquist frequency first when downsampling (so a 24 kHz render put into
an 8 kHz recording does not alias), then linear interpolation.

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

### voxy.wavutil.write_wav_channels(samples, sample_rate)

32-bit float WAV bytes of `samples` `(frames, channels)` (or 1-D: mono),
lossless, for handing audio to another process.

```pycon
>>> import numpy as np
>>> x = np.array([[0.25, -0.5], [1.5, 0.0]], dtype=np.float32)
>>> back, sr = read_wav_channels(write_wav_channels(x, 8000))
>>> sr, back.tolist()
(8000, [[0.25, -0.5], [1.5, 0.0]])
```

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)
