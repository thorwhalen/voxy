"""align_long (#129): a known text aligned to a recording of any length, in stretches."""

import io
import random
import wave

import voxy.align as va
from voxy.align import align_long, plan_stretches


def _silent_wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(va.ALIGN_RATE)
        w.writeframes(bytes(2 * int(seconds * va.ALIGN_RATE)))
    return buf.getvalue()


def _recording_crops(monkeypatch):
    """Record where each crop starts, so a fake aligner can answer on the crop's clock."""
    starts = []
    real = va._crop

    def crop(wav_path, start, end, out):
        starts.append((start, end))
        return real(wav_path, start, end, out)

    monkeypatch.setattr(va, "_crop", crop)
    return starts


def _episode(n_words=600, seed=0):
    """A synthetic hour-scale reading: word i at a jittered time, with a reference."""
    rng = random.Random(seed)
    words, ref, t = [], [], 0.0
    for i in range(n_words):
        t += rng.uniform(0.15, 0.6) + (rng.uniform(0.5, 2.0) if i % 12 == 11 else 0.0)
        dur = rng.uniform(0.2, 0.5)
        words.append(f"w{i}")
        ref.append((t, t + dur))
        t += dur
    return words, ref, t + 1.0


def test_stretches_cover_every_word_once_and_stay_short():
    words, ref, total = _episode()
    anchors = {j: ref[j] for j in range(len(words)) if j % 7}  # ASR missed one word in seven
    stretches = plan_stretches(len(words), anchors, total, max_stretch_s=25)
    assert stretches[0]["words"][0] == 0 and stretches[-1]["words"][1] == len(words)
    for a, b in zip(stretches, stretches[1:]):
        assert a["words"][1] == b["words"][0] and a["end"] == b["start"]
    assert all(s["end"] - s["start"] <= 25 and not s["too_long"] for s in stretches)
    # every word lies inside its stretch's audio
    for s in stretches:
        lo, hi = s["words"]
        assert all(s["start"] <= ref[j][0] and ref[j][1] <= s["end"] for j in range(lo, hi))


def test_a_long_unanchored_run_is_reported_not_cut_on_a_guess():
    anchors = {0: (0.0, 0.5), 1: (0.6, 1.0), 9: (70.0, 70.5)}
    stretches = plan_stretches(10, anchors, 80.0, max_stretch_s=25)
    assert [s["too_long"] for s in stretches] == [False, True]


def test_align_long_puts_each_word_near_its_reference(monkeypatch):
    words, ref, total = _episode(n_words=400, seed=1)
    rng = random.Random(2)
    # the rough pass: most words heard, times off by up to 0.3 s, a few misheard
    heard = [
        {"text": (w if rng.random() > 0.05 else "uh"), "start": s + rng.uniform(-0.3, 0.3), "end": e + rng.uniform(-0.3, 0.3)}
        for w, (s, e) in zip(words, ref)
        if rng.random() > 0.08
    ]
    index = {w: j for j, w in enumerate(words)}
    crops = _recording_crops(monkeypatch)

    def exact(audio, ws, *, start, end):  # a perfect aligner on the crop's clock, which checks the crop holds its words
        lo, hi = crops[-1]
        out = []
        for w in ws:
            s, e = ref[index[w]]
            assert lo <= s and e <= hi, (w, lo, hi)
            out.append({"start": s - lo, "end": e - lo, "score": 0.9, "chars": [{"char": w[0], "start": s - lo, "end": s - lo + 0.05}]})
        return out

    got = align_long(_silent_wav(total), " ".join(words), heard=lambda path: heard, run=exact, max_stretch_s=25)
    assert [w["word"] for w in got] == words
    close = sum(abs(w["start"] - s) <= 0.3 for w, (s, _) in zip(got, ref))
    assert close / len(words) >= 0.95
    assert all(w["reliable"] for w in got)
    assert all(abs(w["chars"][0]["start"] - w["start"]) < 1e-9 for w in got)  # chars moved too
    assert all(hi - lo < va.ALIGNER_MAX_S for lo, hi in crops)


def test_an_intro_an_outro_and_a_long_break_are_left_out_not_flagged(monkeypatch):
    # 50 words between 30 s and 80 s of a 120 s recording, and a 60 s break in the middle of them
    times = {j: (30 + 0.5 * j + (60 if j >= 25 else 0), 30.3 + 0.5 * j + (60 if j >= 25 else 0)) for j in range(50)}
    stretches = plan_stretches(50, times, 180.0, max_stretch_s=25)
    assert not any(s["too_long"] for s in stretches)
    assert stretches[0]["start"] >= 28 and stretches[-1]["end"] <= 117
    assert all(s["end"] - s["start"] <= 25 for s in stretches)


def test_punctuation_between_anchors_does_not_remove_the_cut():
    times = {0: (0.0, 0.4), 2: (30.0, 30.4)}  # word 1 is a dash: nothing hears it
    stretches = plan_stretches(3, times, 31.0, max_stretch_s=25, transparent=frozenset({1}))
    assert [s["too_long"] for s in stretches] == [False, False]


def test_a_crop_over_the_aligners_limit_is_refused_up_front():
    import pytest

    with pytest.raises(ValueError, match="must stay under"):
        align_long(b"", "a b", heard=lambda p: [], max_stretch_s=29.8)


def test_a_low_scoring_stretch_is_flagged():
    heard = [{"text": w, "start": 2.0 * i, "end": 2.0 * i + 1} for i, w in enumerate("a b c d".split())]
    poor = lambda audio, ws, *, start, end: [{"start": 0.0, "end": 0.1, "score": 0.1, "chars": []} for _ in ws]
    got = align_long(_silent_wav(9), "a b c d", heard=lambda p: heard, run=poor, max_stretch_s=4)
    assert not any(w["reliable"] for w in got)


def test_a_leading_dash_does_not_stop_the_lead_in_being_skipped():
    # word 0 is punctuation; word 1 is heard at 40 s after a 40 s intro
    times = {1: (40.0, 40.4), 2: (40.5, 40.9)}
    (s,) = plan_stretches(3, times, 45.0, max_stretch_s=25, transparent=frozenset({0}))
    assert s["start"] >= 38 and not s["too_long"]


def test_an_unheard_first_word_does_not_stop_the_lead_in_being_skipped():
    times = {1: (40.0, 40.4), 2: (40.5, 40.9)}  # word 0 was not heard
    (s,) = plan_stretches(3, times, 45.0, max_stretch_s=25)
    assert 30 <= s["start"] <= 40 and not s["too_long"]


def test_an_unheard_last_word_does_not_stop_the_outro_being_skipped():
    times = {0: (1.0, 1.4), 1: (1.5, 1.9)}  # word 2 was not heard; a 60 s outro
    (s,) = plan_stretches(3, times, 62.0, max_stretch_s=25)
    assert s["end"] <= 10 and not s["too_long"]
