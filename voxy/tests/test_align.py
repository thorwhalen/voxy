"""Tests for voxy.align, with the aligner faked."""

from pathlib import Path

import pytest

from voxy import align


def test_words_come_back_in_order_with_their_spans_and_none_for_unalignable():
    seen = []

    def run(audio, words):
        seen.append((Path(audio).read_bytes(), list(words)))
        return [None if w == "42" else {"start": i, "end": i + 0.4, "score": 0.8, "chars": [{"char": "x"}]}
                for i, w in enumerate(words)]

    spans = align.force_align(b"RIFF-audio", "Il a 42 ans", run=run)
    assert seen == [(b"RIFF-audio", ["Il", "a", "42", "ans"])]
    assert [s["word"] for s in spans] == ["Il", "a", "42", "ans"]
    assert spans[2]["start"] is None and spans[3]["start"] == 3


def test_a_list_of_words_and_a_path(tmp_path):
    wav = tmp_path / "line.wav"
    wav.write_bytes(b"RIFF")
    spans = align.force_align(wav, ["l'est"], run=lambda audio, words: [{"start": 0.1, "end": 0.3, "score": 1, "chars": []}])
    assert spans[0]["word"] == "l'est"


def test_a_short_answer_from_the_aligner_is_an_error():
    with pytest.raises(ValueError, match="2 words"):
        align.force_align(b"RIFF", "deux mots", run=lambda audio, words: [])


def test_the_default_run_uses_the_pooled_worker(monkeypatch):
    calls, made = [], []

    class FakeWorker:
        def call(self, op, **request):
            calls.append((op, request))
            return {"words": [None for _ in request["words"]]}

    monkeypatch.setattr(align._ALIGN_POOL, "_make", lambda spec, adapter, **key: made.append(key) or FakeWorker())
    align.close_alignment()
    try:
        align.force_align(b"RIFF", "un mot", run=align.worker_run())
    finally:
        align.close_alignment()
    (op, request), = calls
    assert op == "align" and Path(request["audio"]).is_absolute() and request["words"] == ["un", "mot"]
    assert made == [{"model": "MMS_FA", "threads": None, "rootdir": None}]


def test_which_said_ranks_candidates_with_margins_and_passes_the_crop():
    seen = []

    def run(audio, texts, *, start=None, end=None):
        seen.append((Path(audio).read_bytes(), list(texts), start, end))
        return [-10.0, -4.0, -6.5]

    ranked = align.which_said(b"RIFF-x", ["bitch", "witch", "which"], start=1.0, end=2.5, run=run)
    assert seen == [(b"RIFF-x", ["bitch", "witch", "which"], 1.0, 2.5)]
    assert [(c["text"], c["score"], c["margin"]) for c in ranked] == [
        ("witch", -4.0, 2.5), ("which", -6.5, 3.5), ("bitch", -10.0, None)
    ]


def test_the_default_scorer_calls_the_pooled_workers_score(monkeypatch):
    calls = []

    class FakeWorker:
        def call(self, op, **request):
            calls.append((op, request))
            return {"scores": [-1.0 for _ in request["texts"]]}

    monkeypatch.setattr(align._ALIGN_POOL, "_make", lambda spec, adapter, **key: FakeWorker())
    align.close_alignment()
    try:
        align.which_said(b"RIFF", ["a", "b"], end=0.5, run=align.worker_score())
    finally:
        align.close_alignment()
    (op, request), = calls
    assert op == "score" and request["texts"] == ["a", "b"] and request["end"] == 0.5
