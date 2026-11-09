"""SSML pauses: parsing, planning per backend, and the facade's split-and-join."""

import shutil

import numpy as np
import pytest

from voxy import ElevenLabsSpeechModel, Speech, SpeechModel, text_to_speech
from voxy.pauses import (
    Section,
    break_seconds,
    join_sections,
    parse_sections,
    plan_sections,
)
from voxy.tests.test_elevenlabs_model import FakeClient
from voxy.wavutil import ffmpeg_decode, read_wav, write_wav

RATE = 8000


class Tone(SpeechModel):
    """Says each text as 0.1 s of tone (WAV), recording the calls."""

    name = "tone"
    dflt_voice = "v"

    def __init__(self, native_s: float = 0.0):
        super().__init__(device="cpu")
        self.calls, self.native_s = [], native_s

    def native_break_s(self, **_):
        return self.native_s

    def synthesize(self, text, voice=None, **kwargs):
        self.calls.append((text, kwargs))
        audio = write_wav(np.full(RATE // 10, 0.5, np.float32), RATE)
        return Speech(audio, format="wav", backend=self.name, voice=voice)


def test_time_units_strength_and_errors():
    assert break_seconds(' time="1.5s"') == 1.5
    assert break_seconds(" time='750ms'") == 0.75
    assert break_seconds(' strength="x-strong"') == 1.25
    with pytest.raises(ValueError, match="'3s' or '250ms'"):
        break_seconds(' time="3 seconds"')
    with pytest.raises(ValueError, match="strength"):
        break_seconds(' strength="huge"')


def test_sections_carry_pauses_and_marks():
    text = 'Ready?<mark name="hold"/> Breathe in.<break time="4s"/><break time="1s"/> Out.'
    assert parse_sections(text) == [
        Section("Ready?", 0.0, "hold"),
        Section("Breathe in.", 5.0),
        Section("Out."),
    ]
    assert parse_sections("No markup.") == [Section("No markup.")]
    assert parse_sections('<break time="2s"></break>') == [Section("", 2.0)]


def test_plan_keeps_native_breaks_and_drops_marks():
    text = 'A.<mark name="m"/> B.<break time="2s"/> C.<break time="8s"/> D.<break time="1s"/>'
    assert plan_sections(text, native_break_s=3.0) == [
        Section('A. B. <break time="2s" /> C.', 8.0),
        Section("D.", 1.0),  # a trailing break is silence, not markup
    ]
    assert [s.text for s in plan_sections(text)] == ["A. B.", "C.", "D."]


def test_facade_joins_sections_with_exact_silence():
    model = Tone()
    speech = text_to_speech('In.<break time="2s"/> Out.<break time="500ms"/>', model=model)
    samples, rate = read_wav(speech.audio)
    assert speech.format == "wav" and rate == RATE
    assert [c[0] for c in model.calls] == ["In.", "Out."]
    assert len(samples) == 2 * (RATE // 10) + int(2.5 * RATE)
    assert not samples[RATE // 10 : RATE // 10 + 2 * RATE].any()  # the 2 s is silent


def test_facade_single_call_when_backend_renders_the_breaks():
    model = Tone(native_s=3.0)
    text_to_speech('In.<break time="2s"/> Out.', model=model)
    assert model.calls == [('In. <break time="2s" /> Out.', {})]


def test_text_without_markup_is_untouched():
    model = Tone()
    text_to_speech("[softly] Hello <there>.", model=model)
    assert model.calls == [("[softly] Hello <there>.", {})]


def test_elevenlabs_v4_splits_and_asks_for_pcm():
    client = FakeClient()
    model = ElevenLabsSpeechModel(model_id="eleven_v4", client_factory=lambda k: client)
    model._explicit_api_key = "k"
    speech = text_to_speech('One.<break time="5s"/> Two.', "vid", model=model)
    tts = [c[2] for c in client.calls if c[0] == "convert"]
    texts, formats = [kw["text"] for kw in tts], {kw["output_format"] for kw in tts}
    assert texts == ["One.", "Two."] and formats == {"pcm_24000"}
    assert speech.format == "wav" and speech.text == 'One.<break time="5s"/> Two.'
    assert model.native_break_s(model_id="eleven_flash_v2_5") == 3.0


def test_compressed_sections_join_through_injected_codecs():
    """No ffmpeg needed: decode= / encode= are the seams."""
    sections = [Section("One.", 1.0), Section("Two.")]

    def synthesize(part):
        return Speech(b"fake-mp3", format="mp3", voice="v")

    decode = lambda speech: (np.full(RATE // 10, 0.5, np.float32), RATE)
    seen = {}

    def encode(samples, rate, fmt):
        seen.update(n=len(samples), rate=rate, fmt=fmt)
        return b"joined-mp3"

    out = join_sections(sections, synthesize, decode=decode, encode=encode)
    assert (out.audio, out.format, out.sample_rate) == (b"joined-mp3", "mp3", RATE)
    assert seen == {"n": RATE // 10 * 2 + RATE, "rate": RATE, "fmt": "mp3"}


def test_compressed_without_ffmpeg_is_explained(monkeypatch):
    monkeypatch.setattr("voxy.wavutil.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="needs ffmpeg"):
        join_sections([Section("One.")], lambda part: Speech(b"x", format="mp3"))


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _tone(seconds=0.5, rate=44100):
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


def _gap_s(samples, rate, *, floor=0.01):
    """Length of the silent stretch between the first and last sound."""
    loud = np.flatnonzero(np.abs(samples) > floor)
    quiet = np.diff(loud)
    return quiet.max() / rate


@needs_ffmpeg
@pytest.mark.parametrize("audio_format", ["mp3", "aac", "flac", "ogg"])
def test_compressed_sections_join_with_the_requested_gap(audio_format):
    from voxy.wavutil import ffmpeg_encode

    piece = ffmpeg_encode(_tone(), 44100, audio_format)
    out = join_sections(
        [Section("One.", 1.0), Section("Two.")],
        lambda part: Speech(piece, format=audio_format, voice="v"),
    )
    assert out.format == audio_format
    samples, rate = ffmpeg_decode(out.audio)
    assert rate == 44100
    assert abs(_gap_s(samples, rate) - 1.0) < 0.08  # encoder padding, not exact


@needs_ffmpeg
def test_raw_ulaw_sections_join():
    from voxy.wavutil import ffmpeg_encode

    piece = ffmpeg_encode(_tone(rate=8000), 8000, "ulaw")
    out = join_sections(
        [Section("One.", 0.5), Section("Two.")],
        lambda part: Speech(piece, format="ulaw", sample_rate=8000),
    )
    assert out.format == "ulaw" and out.sample_rate == 8000


def test_unsupported_format_is_refused_before_more_calls():
    calls = []

    def synthesize(part):
        calls.append(part)
        return Speech(b"x", format="weird")

    with pytest.raises(ValueError, match="Can't join audio in 'weird'"):
        join_sections([Section("A.", 1.0), Section("B.")], synthesize)
    assert calls == ["A."]


@needs_ffmpeg
def test_text_to_speech_mp3_output_format_keeps_mp3_and_rate():
    from voxy.wavutil import ffmpeg_encode

    mp3 = ffmpeg_encode(_tone(), 44100, "mp3")
    client = FakeClient(pcm=mp3)
    model = ElevenLabsSpeechModel(model_id="eleven_v4", client_factory=lambda k: client)
    model._explicit_api_key = "k"
    out = text_to_speech(
        'One.<break time="2s"/> Two.', "vid", model=model, output_format="mp3_44100_128"
    )
    assert out.format == "mp3" and out.sample_rate == 44100
    formats = [c[2]["output_format"] for c in client.calls if c[0] == "convert"]
    assert formats == ["mp3_44100_128"] * 2
