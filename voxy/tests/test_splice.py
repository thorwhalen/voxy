"""Tests for voxy.splice: replacing a spoken phrase, with every service faked."""

import numpy as np
import pytest

import voxy
from voxy import splice
from voxy.base import Speech
from voxy.splice import find_phrase, heard_check, replace_words, tts_render
from voxy.wavutil import read_wav, write_wav

SR = 8000


def tone(seconds, freq, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


GAP = np.zeros(SR // 2, dtype=np.float32)
#: "Hello Anna Smyth there": four tones, half a second apart.
SPEECH = np.concatenate(
    [tone(1, 200), GAP, tone(0.5, 300), GAP, tone(1, 400), GAP, tone(0.5, 500)]
)
WORDS = [
    {"text": " Hello", "start": 0.0, "end": 1.0},
    {"text": " Anna", "start": 1.5, "end": 2.0},
    {"text": " Smyth", "start": 2.5, "end": 3.5},
    {"text": " there.", "start": 4.0, "end": 4.5},
]


def fake_render(calls, seconds=0.8, amp=0.9):
    def render(text):
        calls.append(text)
        pad = np.zeros(SR // 10, dtype=np.float32)  # edge silence, to be trimmed
        return np.concatenate([pad, tone(seconds, 1000, amp), pad]), SR

    return render


def fake_transcriber(texts):
    """Returns ``texts`` as timed words, whatever the audio."""

    def transcriber(path):
        words = [
            {"text": f" {t}", "start": i, "end": i + 0.5} for i, t in enumerate(texts)
        ]
        return [
            {"start": 0, "end": len(texts), "text": " ".join(texts), "words": words}
        ]

    return transcriber


def test_the_phrase_is_replaced_and_the_rest_kept():
    calls = []
    fixed = replace_words(
        SPEECH,
        "anna smyth",
        "Anna Smith",
        words=WORDS,
        sample_rate=SR,
        render=fake_render(calls),
        verify=(),
    )
    assert calls == ["Anna Smith"]
    assert fixed.replaced == (1.375, 3.625)  # in the gaps, not in the words
    out, sr = read_wav(fixed.audio)
    assert sr == SR
    head, tail = int(1.2 * SR), int(0.6 * SR)
    assert np.allclose(out[:head], SPEECH[:head], atol=1e-3)  # "Hello" untouched
    assert np.allclose(out[-tail:], SPEECH[-tail:], atol=1e-3)  # "there" untouched
    a, b = int(fixed.start * SR), int(fixed.end * SR)
    spectrum = np.abs(np.fft.rfft(out[a:b]))
    assert abs(np.argmax(spectrum) * SR / (b - a) - 1000) < 20  # the render is there


def test_the_render_is_matched_to_the_loudness_it_replaces():
    fixed = replace_words(
        SPEECH,
        "Smyth",
        "Smith",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([], amp=0.9),
        verify=(),
    )
    out, _ = read_wav(fixed.audio)
    a, b = int(fixed.start * SR), int(fixed.end * SR)
    cut = SPEECH[int(fixed.replaced[0] * SR) : int(fixed.replaced[1] * SR)]
    rms = lambda x: float(np.sqrt(np.mean(x**2)))
    assert rms(out[a:b]) == pytest.approx(rms(cut), rel=0.15)


def test_words_come_from_the_transcriber_when_not_given():
    fixed = replace_words(
        write_wav(SPEECH, SR),
        "Smyth",
        "Smith",
        transcriber=lambda path: [{"words": WORDS}],
        render=fake_render([]),
        verify=(),
    )
    assert fixed.replaced == (2.375, 3.625)


def test_heard_check_reports_whether_the_new_words_are_transcribed():
    heard = heard_check(transcriber=fake_transcriber(["Anna", "Smith,", "there"]))
    missed = heard_check(transcriber=fake_transcriber(["Anna", "Smyth", "there"]))
    fixed = replace_words(
        SPEECH,
        "Anna Smyth",
        "Anna Smith",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=[heard, missed],
    )
    assert [c["passed"] for c in fixed.checks] == [True, False]
    assert fixed.checks[1]["heard"] == "anna smyth there"
    assert not fixed.heard


def test_heard_check_takes_the_phrase_as_asr_writes_it():
    asr = fake_transcriber(["with", "Dr.", "Patterson"])
    fixed = replace_words(
        SPEECH,
        "Smyth",
        "Doctor Patterson",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=[
            heard_check(transcriber=asr),
            heard_check(transcriber=asr, heard_as="Dr. Patterson"),
        ],
    )
    assert [c["passed"] for c in fixed.checks] == [False, True]


def test_the_default_check_uses_the_same_transcriber():
    fixed = replace_words(
        write_wav(SPEECH, SR),
        "Smyth",
        "Smith",
        words=WORDS,
        transcriber=fake_transcriber(["Anna", "Smith"]),
        render=fake_render([]),
    )
    assert fixed.heard and fixed.checks[0]["check"] == "heard"


def test_the_cut_snaps_past_an_early_asr_boundary():
    """Continuous speech: ASR ends the word 60 ms before its last sound fades."""
    x = np.concatenate(
        [tone(1, 300), tone(0.06, 300, 0.1), np.zeros(80), tone(0.5, 500)]
    )
    ws = [
        {"text": "Jones", "start": 0.0, "end": 1.0},
        {"text": "lived", "start": 1.0, "end": 1.5},
    ]
    start, end = splice.cut_points(x, SR, ws, 0, 0)
    assert 1.06 <= end <= 1.07  # in the 10 ms of silence, not at ASR's 1.0


def test_voice_check_compares_the_new_phrase_with_the_rest():
    from voxy.speakers import Windows

    def fake_embed(voice_of_new):
        def embed(path, *, spans, window, **kwargs):
            (start, _), *_ = spans
            inside = start > 0  # the new phrase's span; the rest starts at 0
            v = np.array(voice_of_new if inside else [1.0, 0.0])
            return Windows(np.array([[0.0, window]]), np.array([v / np.linalg.norm(v)]))

        return embed

    same = splice.voice_check(embed=fake_embed([1.0, 0.1]))
    other = splice.voice_check(embed=fake_embed([0.0, 1.0]))
    fixed = replace_words(
        SPEECH,
        "Smyth",
        "Smith",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=[same, other],
    )
    assert [c["passed"] for c in fixed.checks] == [True, False]
    assert fixed.checks[0]["similarity"] == pytest.approx(0.995, abs=1e-3)


def test_find_phrase_names_what_it_found():
    assert find_phrase(WORDS, "SMYTH") == (2, 2)
    with pytest.raises(ValueError, match="not found"):
        find_phrase(WORDS, "Smith")
    with pytest.raises(ValueError, match="found 1"):
        find_phrase(WORDS, "Smyth", occurrence=1)


def test_raw_samples_need_a_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        replace_words(SPEECH, "Smyth", "Smith", words=WORDS, render=fake_render([]))


def test_tts_render_asks_elevenlabs_for_pcm_and_decodes(monkeypatch):
    seen = []

    def fake_tts(text, voice=None, *, backend=None, **kwargs):
        seen.append((text, voice, backend, kwargs))
        return Speech(write_wav(tone(0.2, 300), SR), format="wav")

    monkeypatch.setattr(voxy.facade, "text_to_speech", fake_tts)
    samples, sr = tts_render("v1", backend="elevenlabs", voices={})("Smith")
    assert sr == SR and len(samples) == int(0.2 * SR)
    assert seen == [
        ("Smith", "v1", "elevenlabs", {"output_format": "pcm_24000", "voices": {}})
    ]
    tts_render("Daniel", backend="say", voices={})("Smith")
    assert seen[-1][3] == {"voices": {}}


def test_tts_render_refuses_a_compressed_render(monkeypatch):
    monkeypatch.setattr(
        voxy.facade,
        "text_to_speech",
        lambda *a, **k: Speech(b"ID3...", format="mp3"),
    )
    with pytest.raises(ValueError, match="output_format"):
        tts_render("v1", backend="say", voices={})("Smith")


def test_it_is_on_the_package():
    assert voxy.replace_words is splice.replace_words
