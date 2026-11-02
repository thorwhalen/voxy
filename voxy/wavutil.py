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
#: WAV ``fmt`` format tags: integer PCM, IEEE float, and the extensible wrapper.
WAVE_FORMAT_PCM, WAVE_FORMAT_FLOAT, WAVE_FORMAT_EXTENSIBLE = 1, 3, 0xFFFE
DFLT_SILENCE_DB = -45.0
DFLT_KEEP_MS = 60
FRAME_MS = 10


def pcm_to_float(raw: bytes, width: int = PCM_SAMPLE_WIDTH) -> np.ndarray:
    """Little-endian PCM samples of ``width`` bytes as floats in ``[-1, 1]``.

    8-bit PCM is unsigned (silence is 128); a trailing partial sample (a stream cut
    mid-sample) is dropped.

    >>> pcm_to_float(b"\\x00\\x00\\xff\\x7f\\x01").tolist()
    [0.0, 1.0]
    >>> pcm_to_float(bytes([0, 128, 255]), 1).tolist()
    [-1.0, 0.0, 0.9921875]
    """
    raw = raw[: len(raw) - len(raw) % width]
    if width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / PCM_SCALE
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483647.0
    raise ValueError(f"Unsupported PCM sample width: {width} bytes")


def _wav_chunks(data: bytes):
    """``(chunk id, body)`` of a RIFF/WAVE payload; a data chunk whose size is a
    streaming placeholder (0 or past the end) runs to the end of the payload."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV payload")
    pos = 12
    while pos + 8 <= len(data):
        chunk, size = (
            data[pos : pos + 4],
            int.from_bytes(data[pos + 4 : pos + 8], "little"),
        )
        body = pos + 8
        if chunk == b"data" and not 0 < size <= len(data) - body:
            size = len(data) - body
        yield chunk, data[body : body + size]
        pos = body + size + size % 2


def read_wav(source: str | os.PathLike | bytes) -> tuple[np.ndarray, int]:
    """``(samples, sample_rate)`` of a WAV (path or bytes), channels averaged.

    Integer PCM (8, 16, 32 bit) and 32-bit float, plain or extensible headers, and
    streamed files whose header carries a placeholder size.
    """
    if not isinstance(source, bytes):
        with open(source, "rb") as f:
            source = f.read()
    fmt = None
    for chunk, body in _wav_chunks(source):
        if chunk == b"fmt ":
            tag, channels, rate = (
                int.from_bytes(body[i:j], "little") for i, j in ((0, 2), (2, 4), (4, 8))
            )
            width = int.from_bytes(body[14:16], "little") // 8
            if tag == WAVE_FORMAT_EXTENSIBLE and len(body) >= 26:
                tag = int.from_bytes(body[24:26], "little")  # the sub-format's tag
            fmt = (tag, channels, rate, width)
        elif chunk == b"data":
            if fmt is None:
                raise ValueError("WAV data chunk before its fmt chunk")
            tag, channels, rate, width = fmt
            if tag == WAVE_FORMAT_FLOAT and width == 4:
                samples = np.frombuffer(
                    body[: len(body) - len(body) % 4], "<f4"
                ).astype(np.float32)
            elif tag == WAVE_FORMAT_PCM:
                samples = pcm_to_float(body, width)
            else:
                raise ValueError(f"Unsupported WAV format: tag {tag}, {8 * width} bit")
            if channels > 1:
                samples = samples[: len(samples) - len(samples) % channels]
                samples = samples.reshape(-1, channels).mean(axis=1)
            return samples, rate
    raise ValueError("WAV payload has no data chunk")


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
