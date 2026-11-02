"""Tests for voxy.wavutil's one WAV decoder and the encoder's input layouts (#71)."""

import io
import struct
import wave

import numpy as np
import pytest

from voxy.base import tensor_to_wav_bytes
from voxy.wavutil import pcm_to_float, read_wav, write_wav


def wav(
    frames: bytes,
    *,
    rate=8000,
    channels=1,
    width=2,
    tag=1,
    data_size=None,
    extensible=False,
):
    """A WAV payload built by hand (so headers the wave module won't write can be tested)."""
    block = channels * width
    if extensible:
        fmt = struct.pack(
            "<HHIIHH", 0xFFFE, channels, rate, rate * block, block, 8 * width
        )
        fmt += (
            struct.pack("<HHI", 22, 8 * width, 0)
            + struct.pack("<H", tag)
            + b"\x00" * 14
        )
    else:
        fmt = struct.pack(
            "<HHIIHH", tag, channels, rate, rate * block, block, 8 * width
        )
    size = len(frames) if data_size is None else data_size
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", size)
        + frames
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_round_trip_through_the_one_encoder_and_decoder():
    x = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    samples, sr = read_wav(write_wav(x, 16000))
    assert sr == 16000 and np.allclose(samples, x, atol=1 / 32767)


def test_a_streamed_wav_with_a_placeholder_size_decodes():
    frames = np.array([0, 16384, -16384], "<i2").tobytes()
    for placeholder in (0, 0xFFFFFFFF):
        samples, _ = read_wav(wav(frames, data_size=placeholder))
        assert np.allclose(samples, [0, 0.5, -0.5], atol=1e-4)


def test_unsigned_8_bit_32_bit_float_and_extensible_headers():
    assert read_wav(wav(bytes([0, 128, 255]), width=1))[0].tolist() == [
        -1.0,
        0.0,
        0.9921875,
    ]
    floats = np.array([0.25, -0.75], "<f4").tobytes()
    assert read_wav(wav(floats, width=4, tag=3))[0].tolist() == [0.25, -0.75]
    pcm = np.array([32767, 0], "<i2").tobytes()
    assert read_wav(wav(pcm, extensible=True))[0].tolist() == [1.0, 0.0]


def test_channels_are_averaged():
    frames = np.array([32767, 0, 0, 32767], "<i2").tobytes()  # two stereo frames
    assert np.allclose(read_wav(wav(frames, channels=2))[0], [0.5, 0.5])


def test_it_reads_what_the_wave_module_writes_and_refuses_what_it_cannot():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(22050)
        w.writeframes(np.array([100, -100], "<i2").tobytes())
    assert read_wav(buffer.getvalue())[1] == 22050
    with pytest.raises(ValueError, match="Not a WAV"):
        read_wav(b"ID3\x04" + b"\x00" * 40)
    with pytest.raises(ValueError, match="Unsupported WAV format"):
        read_wav(wav(b"\x00" * 8, width=4, tag=6))  # A-law


def test_pcm_decoding_drops_a_trailing_partial_sample():
    assert pcm_to_float(b"\x00\x00\xff").tolist() == [0.0]


def test_the_encoder_scales_unsigned_8_bit_around_128():
    silent = tensor_to_wav_bytes(np.full(4, 128, dtype=np.uint8), 8000)
    assert not read_wav(silent)[0].any()


def test_the_encoder_accepts_samples_by_channels():
    left_right = np.stack([np.full(100, 0.5), np.zeros(100)], axis=1)  # (100, 2)
    samples, _ = read_wav(tensor_to_wav_bytes(left_right, 8000))
    assert len(samples) == 100 and np.allclose(samples, 0.25, atol=1e-4)
    channels_first = left_right.T  # (2, 100)
    assert read_wav(tensor_to_wav_bytes(channels_first, 8000))[0].shape == (100,)


def test_a_trailing_chunk_after_placeholder_or_empty_data_is_not_audio():
    frames = np.array([0, 16384], "<i2").tobytes()
    info = b"LIST" + struct.pack("<I", 10) + b"INFOISFT\x02\x00"
    for placeholder in (0, 0xFFFFFFFF):
        samples, _ = read_wav(wav(frames + info, data_size=placeholder))
        assert np.allclose(samples, [0, 0.5], atol=1e-4)
    assert len(read_wav(wav(info, data_size=0))[0]) == 0  # really empty, then a LIST


def test_24_bit_pcm():
    frames = bytes([0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x00])
    assert read_wav(wav(frames, width=3))[0].tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    "payload, match",
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt \x04\x00\x00\x00\x01\x00\x01\x00", "too short"),
        (None, "channels"),
    ],
)
def test_degenerate_fmt_chunks_are_refused(payload, match):
    if payload is None:
        payload = wav(b"\x00\x00", channels=0)
    with pytest.raises(ValueError, match=match):
        read_wav(payload)


def test_digital_silence_after_a_stray_chunk_id_is_still_audio():
    """b"LIST" inside the samples, then zeros: zeros must not read as zero-size chunks."""
    frames = np.zeros(1000, "<i2")
    frames[100:102] = np.frombuffer(b"LIST", "<i2")
    samples, _ = read_wav(wav(frames.tobytes(), data_size=0))
    assert len(samples) == 1000


def test_downsampling_does_not_alias():
    from voxy.wavutil import resample

    sr = 48000
    t = np.arange(sr) / sr
    above = resample(np.sin(2 * np.pi * 10000 * t), sr, 8000)  # above 8 kHz's Nyquist
    kept = resample(np.sin(2 * np.pi * 1000 * t), sr, 8000)
    rms = lambda y: float(np.sqrt(np.mean(y[500:-500] ** 2)))
    assert rms(above) < 0.01 and abs(rms(kept) - 0.707) < 0.02


def test_upsampling_invents_nothing_above_the_source_nyquist():
    """#105: a 24 kHz render into a 48 kHz narration gets no images above 12 kHz."""
    from voxy.wavutil import resample

    sr, out = 24000, 48000
    t = np.arange(sr) / sr
    y = resample(np.sin(2 * np.pi * 3000 * t), sr, out)
    middle = y[2000:-2000] * np.hanning(len(y) - 4000)
    spectrum = np.abs(np.fft.rfft(middle))
    freqs = np.fft.rfftfreq(len(middle), 1 / out)
    assert spectrum[freqs > 12500].max() / spectrum.max() < 1e-4  # below -80 dB
    ideal = np.sin(2 * np.pi * 3000 * np.arange(len(y)) / out)
    assert np.abs(y[2000:-2000] - ideal[2000:-2000]).max() < 1e-3


@pytest.mark.parametrize("n, src, dst", [(1, 16000, 48000), (2, 16000, 48000), (999, 24000, 44100), (10, 22050, 44100)])
def test_upsampling_lengths_and_short_inputs(n, src, dst):
    from voxy.wavutil import resample

    y = resample(np.ones(n, np.float32), src, dst)
    assert len(y) == round(n * dst / src) and np.isfinite(y).all()


def test_lowpass_keeps_the_length_of_a_short_input():
    from voxy.wavutil import lowpass

    for n in (1, 50, 254, 255, 1000):
        assert len(lowpass(np.ones(n), 4000, 48000)) == n
