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
import re
import shutil
import subprocess
import wave

import numpy as np

PCM_SAMPLE_WIDTH = 2
PCM_SCALE = 32767.0
#: WAV ``fmt`` format tags: integer PCM, IEEE float, and the extensible wrapper.
WAVE_FORMAT_PCM, WAVE_FORMAT_FLOAT, WAVE_FORMAT_EXTENSIBLE = 1, 3, 0xFFFE
#: Chunks that can follow audio whose ``data`` size is a streaming placeholder.
TRAILING_CHUNK_IDS = re.compile(
    rb"LIST|id3 |ID3 |fact|cue |smpl|bext|JUNK|PAD |iXML|_PMX|acid|inst|plst"
)
DFLT_SILENCE_DB = -45.0
DFLT_KEEP_MS = 60
FRAME_MS = 10
#: Bitrate when a render is encoded to a lossy format (mp3, aac, opus) and none is given.
DFLT_LOSSY_BITRATE = "192k"
#: Formats whose encoder takes a bitrate (``-b:a``).
LOSSY_FORMATS = frozenset({"mp3", "aac", "opus", "ogg"})
#: Raw (headerless) formats voxy backends return, and ffmpeg's names for them.
RAW_FORMATS = {"pcm": "s16le", "ulaw": "mulaw", "alaw": "alaw"}
#: Audio formats ffmpeg_decode / ffmpeg_encode handle, with the muxer ffmpeg
#: calls each one when that is not its own name.
FFMPEG_MUXERS = {"mp3": "mp3", "flac": "flac", "ogg": "ogg", "opus": "opus",
                 "aac": "adts", **RAW_FORMATS}  # fmt: skip


def pcm_to_float(raw: bytes, width: int = PCM_SAMPLE_WIDTH) -> np.ndarray:
    """Little-endian PCM samples of ``width`` bytes as floats in about ``[-1, 1]``
    (the most negative sample is a hair below -1).

    8-bit PCM is unsigned (silence is 128); a trailing partial sample (a stream cut
    mid-sample) is dropped.

    >>> pcm_to_float(b"\\x00\\x00\\xff\\x7f\\x01").tolist()
    [0.0, 1.0]
    >>> pcm_to_float(bytes([0, 128, 255]), 1).tolist()
    [-1.0, 0.0, 0.9921875]
    >>> pcm_to_float(bytes([0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x80]), 3).tolist()
    [1.0, -1.0000001192092896]
    """
    raw = raw[: len(raw) - len(raw) % width]
    if width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / PCM_SCALE
    if width == 3:  # 24-bit: each sample in the top 3 bytes of an int32, shifted down
        padded = np.zeros((len(raw) // 3, 4), dtype=np.uint8)
        padded[:, 1:] = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        return (padded.view("<i4").ravel() >> 8).astype(np.float32) / 8388607.0
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483647.0
    raise ValueError(f"Unsupported PCM sample width: {width} bytes")


def _chunks_reach_end(data, pos: int) -> bool:
    """Does ``data`` from ``pos`` read as whole known trailing chunks ending exactly at
    its end? (Known ids only: a run of digital silence would otherwise read as
    zero-size chunks all the way to the end.)"""
    while pos < len(data):
        if pos + 8 > len(data) or not TRAILING_CHUNK_IDS.fullmatch(data[pos : pos + 4]):
            return False
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        pos += 8 + size + size % 2
    return pos in (len(data), len(data) + 1)  # a final pad byte may be missing


def _placeholder_data_end(data, body: int) -> int:
    """Where audio whose ``data`` size is unusable really ends: at the first known
    chunk id (word-aligned) from which the rest parses as chunks to the end of the
    payload, else at the end (a stream cut short)."""
    for match in TRAILING_CHUNK_IDS.finditer(data, body):
        pos = match.start()
        if pos % 2 == 0 and _chunks_reach_end(data, pos):
            return pos
    return len(data)


def _wav_chunks(data: bytes):
    """``(chunk id, body)`` of a RIFF/WAVE payload, the bodies as memoryviews (no
    copies). A ``data`` chunk whose size is a streaming placeholder (0 or past the
    end) runs to the next trailing chunk, or to the end of the payload."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV payload")
    view, pos = memoryview(data), 12
    while pos + 8 <= len(data):
        chunk, size = (
            data[pos : pos + 4],
            int.from_bytes(data[pos + 4 : pos + 8], "little"),
        )
        body = pos + 8
        if chunk == b"data" and not 0 < size <= len(data) - body:
            size = _placeholder_data_end(data, body) - body
        yield chunk, view[body : body + size]
        pos = body + size + size % 2


def read_wav_channels(source: str | os.PathLike | bytes) -> tuple[np.ndarray, int]:
    """``(samples (frames, channels), sample_rate)`` of a WAV (path or bytes).

    Integer PCM (8, 16, 24, 32 bit) and 32-bit float, plain or extensible headers,
    and streamed files whose header carries a placeholder size (the audio then ends
    at a trailing chunk such as ``LIST``, or at the end of the file).
    """
    if not isinstance(source, bytes):
        with open(source, "rb") as f:
            source = f.read()
    fmt = None
    for chunk, body in _wav_chunks(source):
        if chunk == b"fmt ":
            if len(body) < 16:
                raise ValueError(f"WAV fmt chunk too short ({len(body)} bytes)")
            tag, channels, rate = (
                int.from_bytes(body[i:j], "little") for i, j in ((0, 2), (2, 4), (4, 8))
            )
            width = int.from_bytes(body[14:16], "little") // 8
            if not (channels and width):
                raise ValueError(
                    f"WAV fmt chunk has {channels} channels of {width} bytes"
                )
            if tag == WAVE_FORMAT_EXTENSIBLE:
                if len(body) < 26:
                    raise ValueError(
                        "WAV extensible fmt chunk too short for its sub-format"
                    )
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
            samples = samples[: len(samples) - len(samples) % channels]
            return samples.reshape(-1, channels), rate
    raise ValueError("WAV payload has no data chunk")


def read_wav(source: str | os.PathLike | bytes) -> tuple[np.ndarray, int]:
    """``(samples, sample_rate)`` of a WAV (path or bytes), channels averaged; reads
    what :func:`read_wav_channels` reads."""
    samples, rate = read_wav_channels(source)
    return samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0], rate


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


def write_wav_channels(samples, sample_rate: int) -> bytes:
    """32-bit float WAV bytes of ``samples`` ``(frames, channels)`` (or 1-D: mono),
    lossless, for handing audio to another process.

    >>> import numpy as np
    >>> x = np.array([[0.25, -0.5], [1.5, 0.0]], dtype=np.float32)
    >>> back, sr = read_wav_channels(write_wav_channels(x, 8000))
    >>> sr, back.tolist()
    (8000, [[0.25, -0.5], [1.5, 0.0]])
    """
    samples = np.asarray(samples, dtype="<f4")
    if samples.ndim == 1:
        samples = samples[:, None]
    frames, channels = samples.shape
    data = samples.tobytes()
    block = channels * 4
    fmt = (
        WAVE_FORMAT_FLOAT.to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + int(sample_rate).to_bytes(4, "little")
        + (int(sample_rate) * block).to_bytes(4, "little")
        + block.to_bytes(2, "little")
        + (32).to_bytes(2, "little")
    )
    body = b"WAVE" + b"fmt " + len(fmt).to_bytes(4, "little") + fmt
    body += b"data" + len(data).to_bytes(4, "little") + data
    return b"RIFF" + len(body).to_bytes(4, "little") + body


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


#: Taps of the anti-aliasing filter applied before downsampling (odd).
DFLT_LOWPASS_TAPS = 255
#: The filter's cutoff, as a share of the target rate's Nyquist frequency.
DFLT_LOWPASS_SHARE = 0.95


def lowpass(
    samples: np.ndarray,
    cutoff_hz: float,
    sample_rate: int,
    *,
    taps: int = DFLT_LOWPASS_TAPS,
) -> np.ndarray:
    """``samples`` with what lies above ``cutoff_hz`` removed (windowed-sinc FIR,
    Blackman window, zero phase: same length, no delay).

    >>> sr = 48000; t = np.arange(sr) / sr
    >>> hiss = lowpass(np.sin(2 * np.pi * 12000 * t), 4000, sr)
    >>> bool(np.abs(hiss[1000:-1000]).max() < 0.01)
    True
    """
    taps = taps + 1 - taps % 2  # odd: a centred, delay-free kernel
    n = np.arange(taps) - taps // 2
    kernel = np.sinc(2 * cutoff_hz / sample_rate * n) * np.blackman(taps)
    kernel /= kernel.sum()
    x = np.asarray(samples, np.float64)
    full = np.convolve(x, kernel)  # "same" would return len(kernel) for a short input
    start = (len(kernel) - 1) // 2
    return full[start : start + len(x)].astype(np.float32)


#: Samples mirrored at each edge before band-limited interpolation, so the FFT does not
#: wrap the end of the signal onto its start (a click at both ends).
UPSAMPLE_EDGE = 1024


def _upsample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Band-limited interpolation (zero-padded spectrum): nothing above the source's
    Nyquist frequency is invented, unlike linear interpolation's images."""
    x = np.asarray(samples, np.float64)
    edge = min(UPSAMPLE_EDGE, len(x) - 1)
    padded = np.pad(x, edge, mode="reflect") if edge > 0 else x
    if len(padded) % 2:  # an even length keeps the source's Nyquist bin
        padded = np.append(padded, padded[-1])
    n_in, n_out = len(padded), int(round(len(padded) * to_rate / from_rate))
    spectrum = np.zeros(n_out // 2 + 1, np.complex128)
    source = np.fft.rfft(padded)
    spectrum[: len(source)] = source
    if n_in % 2 == 0:  # the source's Nyquist bin was the sum of two halves
        spectrum[len(source) - 1] /= 2
    y = np.fft.irfft(spectrum, n_out) * (n_out / n_in)
    start = int(round(edge * to_rate / from_rate))
    length = int(round(len(x) * to_rate / from_rate))
    return y[start : start + length].astype(np.float32)


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resampling for joining pieces (use ffmpeg or soxr for masters): band-limited
    interpolation when upsampling (a 24 kHz render put into a 48 kHz narration gets no
    images above 12 kHz); when downsampling, low-passed below the target's Nyquist
    frequency (so it does not alias), then linearly interpolated. The whole clip is
    held in memory (about 1 GB to take 10 minutes from 24 to 48 kHz): fine for
    rendered words and lines; use ffmpeg or soxr for long recordings.

    >>> len(resample(np.zeros(100), 8000, 16000)), len(resample(np.zeros(100), 16000, 8000))
    (200, 50)
    """
    if from_rate == to_rate:
        return samples
    if to_rate > from_rate and len(samples) > 1:
        return _upsample(samples, from_rate, to_rate)
    if to_rate < from_rate:
        samples = lowpass(samples, DFLT_LOWPASS_SHARE * to_rate / 2, from_rate)
    n = int(round(len(samples) * to_rate / from_rate))
    positions = np.linspace(0, len(samples) - 1, n)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


def check_audio_format(audio_format: str) -> None:
    """Raise ``ValueError`` (before any paid work) for a format ffmpeg_* can't do."""
    if audio_format != "wav" and audio_format not in FFMPEG_MUXERS:
        raise ValueError(
            f"Can't join audio in {audio_format!r}: use one of "
            f"{sorted({'wav', *FFMPEG_MUXERS})}, or ask the backend for WAV"
        )


def _ffmpeg_path(ffmpeg: str | None) -> str:
    path = ffmpeg or shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "This needs ffmpeg for compressed audio and it is not on PATH: install "
            "it (brew install ffmpeg / apt install ffmpeg), or ask the backend for "
            "WAV or PCM output instead"
        )
    return path


def _run_ffmpeg(cmd: list[str], data: bytes, what: str) -> bytes:
    done = subprocess.run(cmd, input=data, capture_output=True)
    if done.returncode:
        err = done.stderr.decode(errors="replace")[-300:]
        raise RuntimeError(f"ffmpeg could not {what}: {err}")
    return done.stdout


def ffmpeg_decode(
    audio: bytes,
    *,
    audio_format: str | None = None,
    sample_rate: int | None = None,
    ffmpeg: str | None = None,
) -> tuple[np.ndarray, int]:
    """``(samples, sample_rate)`` of audio in any format ffmpeg reads (mp3, ...),
    channels averaged, like :func:`read_wav`.

    A raw format (``RAW_FORMATS``: pcm, ulaw, alaw) has no header, so it needs
    ``audio_format`` and ``sample_rate``. Raises ``RuntimeError`` (naming the
    remedy) when ffmpeg is not installed.
    """
    cmd = [_ffmpeg_path(ffmpeg), "-hide_banner", "-loglevel", "error"]
    if audio_format in RAW_FORMATS:
        if not sample_rate:
            raise ValueError(f"Raw {audio_format!r} audio needs its sample_rate")
        cmd += ["-f", RAW_FORMATS[audio_format], "-ar", str(sample_rate), "-ac", "1"]
    cmd += ["-i", "pipe:0", "-f", "wav", "-c:a", "pcm_f32le", "pipe:1"]
    return read_wav(_run_ffmpeg(cmd, audio, "decode"))


def ffmpeg_encode(
    samples,
    sample_rate: int,
    audio_format: str,
    *,
    bitrate: str | None = DFLT_LOSSY_BITRATE,
    ffmpeg: str | None = None,
) -> bytes:
    """``samples`` (floats in ``[-1, 1]``) as ``audio_format`` bytes ('mp3', 'flac', ...).

    ``bitrate`` applies to the lossy formats (``LOSSY_FORMATS``); ``None`` leaves
    the encoder's default.
    """
    check_audio_format(audio_format)
    cmd = [_ffmpeg_path(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", "pipe:0"]
    if bitrate and audio_format in LOSSY_FORMATS:
        cmd += ["-b:a", bitrate]
    cmd += ["-f", FFMPEG_MUXERS[audio_format], "pipe:1"]
    return _run_ffmpeg(cmd, write_wav(samples, sample_rate), f"encode {audio_format}")
