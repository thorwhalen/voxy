"""Tests for voxy.speakers and the refprep speaker step, with made-up embeddings."""

import shutil

import numpy as np

from voxy import refprep
from voxy.speakers import (
    Windows,
    dominant_centroid,
    foreign_spans,
    mark_speaker,
    speaker_filter,
    voice_print,
)
from voxy.wavutil import write_wav

SR = 16000
TARGET, OTHER = np.eye(8)[0], np.eye(8)[1]
HOP, WIN = 0.1, 1.0


def _fake_embed(voice_at):
    """An ``embed`` whose window vector is the voice ``voice_at(center)`` returns."""

    def embed(path, *, spans=None, **_):
        n = 80
        times = np.array([[i * HOP, i * HOP + WIN] for i in range(n)])
        vectors = np.array([voice_at(s + WIN / 2) for s, _ in times])
        return Windows(times, vectors)

    return embed


def _interjected(center):
    """The target talks for 9 s, except someone else from 4.0 s to 4.5 s."""
    return OTHER if 4.0 <= center < 4.5 else TARGET


def _speechy_wav(path, seconds=9.0):
    rng = np.random.default_rng(0)
    path.write_bytes(write_wav(0.3 * rng.standard_normal(int(seconds * SR)), SR))
    return path


def _w(start, end, text):
    return {"start": start, "end": end, "text": text, "confidence": 0.9}


SEGMENTS = [
    {
        "start": 0.5,
        "end": 8.5,
        "text": "",
        "words": [
            _w(0.6, 1.5, " First"),
            _w(1.6, 3.6, " sentence."),
            _w(5.0, 6.0, " Second"),
            _w(6.1, 8.4, " sentence."),
        ],
    }
]


def test_dominant_centroid_finds_the_majority_voice():
    vectors = np.array([TARGET] * 30 + [OTHER] * 10)
    assert dominant_centroid(vectors) @ TARGET > 0.99


def test_an_untranscribed_interjection_becomes_a_gap_marker():
    windows = _fake_embed(_interjected)("x.wav")
    spans = foreign_spans(windows, windows.similarity(TARGET), pad=0)
    assert len(spans) == 1 and 3.9 <= spans[0][0] <= 4.1 and 4.4 <= spans[0][1] <= 4.6
    marked = mark_speaker(SEGMENTS, windows, windows.similarity(TARGET))
    assert all(w["keep"] for w in marked[0]["words"])
    (marker,) = marked[1]["words"]
    assert marker["keep"] is False and marker["text"] == ""


def test_a_transcribed_foreign_word_is_not_kept():
    windows = _fake_embed(lambda c: OTHER if 5.0 <= c <= 6.0 else TARGET)("x.wav")
    marked = mark_speaker(SEGMENTS, windows, windows.similarity(TARGET))
    keep = [(w["text"].strip(), w["keep"]) for w in marked[0]["words"]]
    assert keep == [
        ("First", True),
        ("sentence.", True),
        ("Second", False),
        ("sentence.", True),
    ]


#: One sentence with a pause in it, where the other voice says "uh-huh" (4.0-4.5 s).
PAUSED = [
    {
        "start": 0.5,
        "end": 7.0,
        "text": "",
        "words": [_w(0.6, 3.8, " This is"), _w(4.6, 7.0, " the answer.")],
    }
]


def test_without_the_speaker_step_the_clip_contains_the_interjection():
    (clip,) = refprep.plan_clips(PAUSED, min_seconds=1.5, max_seconds=12)
    assert clip["start"] < 4.0 and clip["end"] > 4.5


def test_with_it_no_clip_contains_the_interjection(tmp_path):
    wav = _speechy_wav(tmp_path / "a.wav")
    mark = speaker_filter(TARGET, embed=_fake_embed(_interjected))
    clips = refprep.plan_clips(mark(wav, PAUSED), min_seconds=1.5, max_seconds=12)
    assert [c["text"] for c in clips] == ["This is", "the answer."]
    for c in clips:
        assert c["end"] <= 4.0 or c["start"] >= 4.5


def test_dominant_filter_needs_no_voice_print(tmp_path):
    wav = _speechy_wav(tmp_path / "a.wav")
    marked = speaker_filter(embed=_fake_embed(_interjected))(wav, SEGMENTS)
    assert sum(not w["keep"] for s in marked for w in s["words"]) == 1


def test_voice_print_of_a_clean_recording(tmp_path):
    wav = _speechy_wav(tmp_path / "a.wav")
    assert voice_print(wav, embed=_fake_embed(lambda c: TARGET)) @ TARGET > 0.99


def test_prepare_reference_scores_the_speaker_match(tmp_path):
    src = _speechy_wav(tmp_path / "talk.wav")

    def conditioner(s, d, **_):
        shutil.copyfile(s, d)
        return d

    manifest = refprep.prepare_reference(
        [src],
        samples={},
        transcriber=lambda path, language=None: SEGMENTS,
        conditioner=conditioner,
        speaker=speaker_filter(TARGET, embed=_fake_embed(_interjected)),
        min_seconds=1.5,
        log=lambda *_: None,
    )
    assert sorted(c["text"] for c in manifest) == ["First sentence.", "Second sentence."]
    assert all(c["speaker_similarity"] > 0.99 for c in manifest)


#: An answer (target), a question (the other voice), a "Yes." (too short), an answer.
TURNS = [
    {"start": 0.0, "end": 2.0, "text": "", "words": [_w(0.1, 1.9, " Answer one.")]},
    {"start": 2.2, "end": 4.4, "text": "", "words": [_w(2.3, 4.3, " A question?")]},
    {"start": 4.5, "end": 4.8, "text": "", "words": [_w(4.5, 4.8, " Yes.")]},
    {"start": 5.0, "end": 7.8, "text": "", "words": [_w(5.1, 7.7, " Answer two.")]},
]


def test_a_turn_is_kept_or_dropped_as_a_whole(tmp_path):
    wav = _speechy_wav(tmp_path / "a.wav")
    other = lambda c: OTHER if 2.2 <= c <= 4.4 else TARGET  # noqa: E731
    marked = speaker_filter(TARGET, embed=_fake_embed(other))(wav, TURNS)
    keep = {w["text"].strip(): w["keep"] for s in marked[:4] for w in s["words"]}
    assert keep == {"Answer one.": True, "A question?": False, "Yes.": False, "Answer two.": True}
    clips = refprep.plan_clips(marked, min_seconds=1.5, max_seconds=12)
    assert [c["text"] for c in clips] == ["Answer one.", "Answer two."]


def test_a_listeners_yes_inside_a_turn_is_not_kept(tmp_path):
    wav = _speechy_wav(tmp_path / "a.wav")
    turn = [
        {
            "start": 0.0,
            "end": 7.8,
            "text": "",
            "words": [_w(0.1, 3.0, " One answer."), _w(3.4, 3.7, " Yes."), _w(4.2, 7.7, " Goes on.")],
        }
    ]
    yes = lambda c: OTHER if 3.4 <= c <= 3.7 else TARGET  # noqa: E731
    marked = speaker_filter(TARGET, embed=_fake_embed(yes))(wav, turn)
    assert [w["keep"] for w in marked[0]["words"]] == [True, False, True]
