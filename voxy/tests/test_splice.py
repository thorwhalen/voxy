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


def rms(x):
    return float(np.sqrt(np.mean(x**2)))


def test_the_render_is_matched_to_the_loudness_of_the_speech_it_replaces():
    """The silence around the phrase in the cut does not lower the level matched."""
    long_gap = np.zeros(SR, dtype=np.float32)
    x = np.concatenate(
        [tone(0.5, 200), long_gap, tone(0.5, 300), long_gap, tone(0.5, 500)]
    )
    ws = [
        {"text": "Hello", "start": 0.0, "end": 0.5},
        {"text": "Smyth", "start": 1.5, "end": 2.0},
        {"text": "there", "start": 3.0, "end": 3.5},
    ]
    fixed = replace_words(
        x,
        "Smyth",
        "Smith",
        words=ws,
        sample_rate=SR,
        render=lambda text: (tone(0.6, 1000, 0.9), SR),
        verify=(),
    )
    cut = x[int(fixed.replaced[0] * SR) : int(fixed.replaced[1] * SR)]
    assert len(cut) >= 1.5 * int(0.5 * SR)  # a third of the cut is silence
    out, _ = read_wav(fixed.audio)
    middle = (fixed.start + fixed.end) / 2
    inserted = out[int((middle - 0.2) * SR) : int((middle + 0.2) * SR)]  # all tone
    phrase = x[int(1.5 * SR) : int(2.0 * SR)]
    assert rms(inserted) == pytest.approx(rms(phrase), rel=0.05)


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
    assert not fixed.passed


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
    assert fixed.passed and fixed.checks[0]["check"] == "heard"


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
    resolved = {"model": None, "use_library": False}
    assert seen == [
        ("Smith", "v1", "elevenlabs", {**resolved, "output_format": "pcm_24000"})
    ]
    tts_render("Daniel", backend="say", voices={})("Smith")
    assert seen[-1][3] == resolved


def test_tts_render_resolves_the_voice_once_with_the_library_settings(monkeypatch):
    calls = []

    def fake_resolve(voice, *, backend=None, voices=None, use_library=True):
        calls.append((voice, backend, use_library))
        return "say", "Daniel"

    monkeypatch.setattr(voxy.facade, "resolve_voice", fake_resolve)
    monkeypatch.setattr(
        voxy.facade,
        "text_to_speech",
        lambda *a, **k: Speech(write_wav(tone(0.2, 300), SR), format="wav"),
    )
    render = tts_render("ada", use_library=False, voices={})
    render("one")
    render("two")
    assert calls == [("ada", None, False)]


def test_tts_render_refuses_a_compressed_render(monkeypatch):
    monkeypatch.setattr(
        voxy.facade,
        "text_to_speech",
        lambda *a, **k: Speech(b"ID3...", format="mp3"),
    )
    with pytest.raises(ValueError, match="output_format"):
        tts_render("v1", backend="say", voices={})("Smith")


def test_the_last_word_is_cut_at_the_end_of_the_audio():
    """Whisper ends the final word at (or past) the end of the clip."""
    fixed = replace_words(
        SPEECH,
        "there",
        "here",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=(),
    )
    total = len(SPEECH) / SR
    assert fixed.replaced[1] == total
    assert fixed.end == pytest.approx(len(read_wav(fixed.audio)[0]) / SR)
    late = [*WORDS[:-1], {**WORDS[-1], "end": total + 0.2}]
    start, end = splice.cut_points(SPEECH, SR, late, 3, 3)
    assert end == total


def test_the_first_word_is_cut_at_the_start_of_the_audio():
    fixed = replace_words(
        SPEECH,
        "Hello",
        "Hi",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=(),
    )
    assert fixed.replaced[0] == 0.0 and fixed.start == 0.0


def test_a_silent_render_is_refused():
    with pytest.raises(ValueError, match="silent"):
        replace_words(
            SPEECH,
            "Smyth",
            "Smith",
            words=WORDS,
            sample_rate=SR,
            render=lambda text: (np.zeros(SR, dtype=np.float32), SR),
        )


def test_a_voice_or_a_render_is_required():
    with pytest.raises(ValueError, match="voice="):
        replace_words(SPEECH, "Smyth", "Smith", words=WORDS, sample_rate=SR)


def test_heard_check_fails_when_the_old_phrase_is_still_heard():
    asr = fake_transcriber(["Anna", "Smith", "Anna", "Smyth"])
    fixed = replace_words(
        SPEECH,
        "Anna Smyth",
        "Anna Smith",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=heard_check(transcriber=asr),
    )
    (check,) = fixed.checks
    assert check["new_heard"] and check["old_heard"] and not check["passed"]


def test_heard_check_expects_the_old_phrase_when_the_new_one_contains_it():
    fixed = replace_words(
        SPEECH,
        "Smyth",
        "Smyth Jones",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=heard_check(transcriber=fake_transcriber(["Anna", "Smyth", "Jones"])),
    )
    assert fixed.passed and not fixed.checks[0]["old_heard"]


def test_voice_check_passes_short_rest_spans_and_fails_without_rest():
    from types import SimpleNamespace

    from voxy.speakers import Windows

    seen = []

    def embed(path, *, spans, window, **kwargs):
        seen.append(list(spans))
        return Windows(np.array([[0.0, window]]), np.array([[1.0, 0.0]]))

    check = splice.voice_check(embed=embed, window=1.0)
    x = np.zeros(SR * 3, dtype=np.float32)
    result = check(x, SR, SimpleNamespace(start=0.5, end=2.8))
    assert seen[-1] == [(0.0, 0.5), (2.8, 3.0)]  # both shorter than the window
    assert result["passed"]
    whole = check(x, SR, SimpleNamespace(start=0.0, end=3.0))
    assert not whole["passed"] and "reason" in whole
    assert len(seen) == 2  # nothing embedded for the whole-file case


def test_find_phrase_refuses_an_empty_phrase():
    with pytest.raises(ValueError, match="no words"):
        find_phrase(WORDS, " … ")


def test_raw_samples_are_mono_floats():
    pcm = (SPEECH * 32768).astype(np.int16)
    fixed = replace_words(
        pcm,
        "Smyth",
        "Smith",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=(),
    )
    out, _ = read_wav(fixed.audio)
    head = int(1.2 * SR)
    assert np.allclose(out[:head], SPEECH[:head], atol=1e-3)
    with pytest.raises(ValueError, match="mono"):
        replace_words(
            np.stack([SPEECH, SPEECH], axis=1),
            "Smyth",
            "Smith",
            words=WORDS,
            sample_rate=SR,
            render=fake_render([]),
        )


def test_the_replacement_is_a_speech():
    fixed = replace_words(
        SPEECH,
        "Smyth",
        "Smith",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=(),
    )
    assert isinstance(fixed.speech, Speech) and fixed.speech.format == "wav"
    assert fixed.audio is fixed.speech.audio and fixed.sample_rate == SR
    assert fixed.passed is None


def test_it_is_on_the_package():
    assert voxy.replace_words is splice.replace_words


def test_voice_check_fails_rather_than_raises_on_a_silent_remainder():
    def silent_rest(path, *, spans, window, **kwargs):
        raise ValueError("No speech windows to find a voice in")

    check = splice.voice_check(embed=silent_rest)
    fixed = replace_words(
        SPEECH,
        "Smyth",
        "Smith",
        words=WORDS,
        sample_rate=SR,
        render=fake_render([]),
        verify=check,
    )
    assert (
        fixed.checks[0]["passed"] is False and "No speech" in fixed.checks[0]["reason"]
    )


def _noisy_speech(floor=0.01, seed=0):
    """Two "words" (tones) with a 0.4 s pause between them, over a noise floor."""
    rng = np.random.default_rng(seed)
    word = lambda s, f: tone(s, f, 0.3)
    x = np.concatenate([np.zeros(SR // 2), word(0.6, 300), np.zeros(int(0.4 * SR)), word(0.6, 500), np.zeros(SR // 2)])
    return (x + floor * rng.standard_normal(len(x))).astype(np.float32)


NOISY_WORDS = [{"text": "one", "start": 0.5, "end": 1.1}, {"text": "two", "start": 1.5, "end": 2.1}]


def _clean_render(text):
    pad = np.zeros(SR // 10, dtype=np.float32)
    return np.concatenate([pad, tone(0.5, 1000, 0.9), pad]), SR


def test_room_tone_fills_the_spliced_span_so_nothing_drops_to_silence():
    x = _noisy_speech()
    rms = lambda y: float(np.sqrt(np.mean(y**2)))
    for with_tone, expect_floor in ((True, True), (False, False)):
        fixed = replace_words(x, "two", "deux", words=NOISY_WORDS, sample_rate=SR,
                              render=_clean_render, verify=(), with_room_tone=with_tone)
        out, _ = read_wav(fixed.audio)
        quiet = out[int((fixed.start - 0.08) * SR) : int((fixed.start - 0.02) * SR)]  # the kept pause
        assert (rms(quiet) > 0.005) == expect_floor


def test_the_pauses_around_the_phrase_keep_their_length():
    x = _noisy_speech()
    fixed = replace_words(x, "two", "deux", words=NOISY_WORDS, sample_rate=SR,
                          render=_clean_render, verify=())
    out, _ = read_wav(fixed.audio)
    from voxy.wavutil import frame_db

    levels = frame_db(out, SR)
    loud = np.flatnonzero(levels > levels.max() - 20)
    gaps = np.diff(loud)
    pause = (gaps.max() - 1) * 0.01  # the longest run of quiet frames between the words
    assert abs(pause - 0.4) < 0.03  # the 0.4 s pause before "two" is still 0.4 s


def test_keep_pauses_off_gives_the_old_tighter_splice():
    x = _noisy_speech()
    kept = replace_words(x, "two", "deux", words=NOISY_WORDS, sample_rate=SR, render=_clean_render, verify=())
    tight = replace_words(x, "two", "deux", words=NOISY_WORDS, sample_rate=SR, render=_clean_render,
                          verify=(), keep_pauses=False)
    assert len(read_wav(tight.audio)[0]) < len(read_wav(kept.audio)[0])
