"""
Mono PCM WAV in and out as float arrays, plus the few edits narration needs.

Every local backend and the reference-prep step exchange 16-bit PCM WAV files;
this module turns them into ``float32`` arrays in ``[-1, 1]`` and back, trims
silence, and joins pieces with pauses, using only numpy and the standard library.

>>> import numpy as np
>>> data = write_wav(np.array([0.0, 0.5, -0.5], dtype=np.float32), 8000)
>>> samples, sr = read_wav(data)
>>> sr, np.round(samples, 3).tolist()
(8000, [0.0, 0.5, -0.5])
"""

import io
import os
import wave

import numpy as np

PCM_SAMPLE_WIDTH = 2
PCM_SCALE = 32767.0
DFLT_SILENCE_DB = -45.0
DFLT_KEEP_MS = 60
FRAME_MS = 10


def read_wav(source: str | os.PathLike | bytes) -> tuple[np.ndarray, int]:
    """``(samples, sample_rate)`` of a PCM WAV (path or bytes), channels averaged."""
    handle = io.BytesIO(source) if isinstance(source, bytes) else open(source, "rb")
    with handle, wave.open(handle, "rb") as w:
        width, channels, sr = w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / PCM_SCALE
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483647.0
    elif width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sr


def write_wav(samples, sample_rate: int) -> bytes:
    """Mono 16-bit PCM WAV bytes from floats in ``[-1, 1]`` (clipped)."""
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1, 1) * PCM_SCALE).astype(
        "<i2"
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(PCM_SAMPLE_WIDTH)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buffer.getvalue()


def frame_db(samples: np.ndarray, sample_rate: int, *, frame_ms: int = FRAME_MS):
    """RMS level (dBFS) of consecutive ``frame_ms`` frames.

    >>> frame_db(np.zeros(160), 8000).round().tolist()
    [-100.0, -100.0]
    """
    size = max(1, int(sample_rate * frame_ms / 1000))
    n = len(samples) // size
    frames = np.asarray(samples[: n * size], dtype=np.float64).reshape(n, size)
    rms = np.sqrt((frames**2).mean(axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-5))


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = DFLT_SILENCE_DB,
    keep_ms: int = DFLT_KEEP_MS,
) -> np.ndarray:
    """Drop leading and trailing frames quieter than ``threshold_db`` (keeping ``keep_ms``).

    >>> x = np.concatenate([np.zeros(800), 0.5 * np.ones(800), np.zeros(800)])
    >>> len(trim_silence(x, 8000, keep_ms=0))
    800
    """
    levels = frame_db(samples, sample_rate)
    loud = np.flatnonzero(levels > threshold_db)
    if loud.size == 0:
        return samples[:0]
    size = max(1, int(sample_rate * FRAME_MS / 1000))
    keep = int(sample_rate * keep_ms / 1000)
    start = max(0, loud[0] * size - keep)
    end = min(len(samples), (loud[-1] + 1) * size + keep)
    return samples[start:end]


def silence(ms: float, sample_rate: int) -> np.ndarray:
    """``ms`` milliseconds of silence.

    >>> len(silence(250, 8000))
    2000
    """
    return np.zeros(int(sample_rate * ms / 1000), dtype=np.float32)


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear-interpolation resampling (enough to join pieces; use ffmpeg for masters).

    >>> len(resample(np.zeros(100), 8000, 16000))
    200
    """
    if from_rate == to_rate:
        return samples
    n = int(round(len(samples) * to_rate / from_rate))
    positions = np.linspace(0, len(samples) - 1, n)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
