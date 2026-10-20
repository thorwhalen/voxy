"""Tests for voxy.pronounce: CTC scoring of candidate readings, with a fake recogniser."""

import math

import numpy as np
import pytest

from voxy.pronounce import (
    Verdict,
    checks_from_risks,
    ctc_log_likelihood,
    failing,
    log_probs_from_path,
    score_readings,
    tokenize_phonemes,
    verify_words,
    word_spans,
)
from voxy.speechtext import flag_risks

VOCAB = ["<pad>", "ɛ", "s", "t", "ɛ̃", "a"]
READINGS = {"verb": "ɛ", "noun": "ɛst"}


def _ids(symbols):
    return [VOCAB.index(s) for s in symbols]


def test_ctc_matches_brute_force_on_a_tiny_case():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3, 3))
    lp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    def collapse(path):
        out, prev = [], None
        for p in path:
            if p != prev and p != 0:
                out.append(p)
            prev = p
        return out

    total = -math.inf
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if collapse([a, b, c]) == [1, 2]:
                    total = np.logaddexp(total, lp[0, a] + lp[1, b] + lp[2, c])
    assert ctc_log_likelihood(lp, [1, 2]) == pytest.approx(total)


def test_ctc_handles_repeated_symbols_and_impossible_targets():
    lp = log_probs_from_path(_ids(["s", "<pad>", "s"]), vocab_size=len(VOCAB))
    ss = tokenize_phonemes("ss", VOCAB)
    assert ctc_log_likelihood(lp, ss) > ctc_log_likelihood(
        lp, tokenize_phonemes("s", VOCAB)
    )
    # "ss" needs a blank between the two s: two frames cannot hold it.
    assert ctc_log_likelihood(lp[:2], ss) == -math.inf


def test_tokenize_prefers_the_longest_symbol_and_rejects_unknown_ones():
    assert tokenize_phonemes("ɛ̃ s", VOCAB) == _ids(["ɛ̃", "s"])
    assert tokenize_phonemes(["ɛ", "t"], VOCAB) == _ids(["ɛ", "t"])
    with pytest.raises(ValueError, match="not in the recogniser"):
        tokenize_phonemes("ʁ", VOCAB)


@pytest.mark.parametrize(
    "heard, best",
    [(["ɛ", "<pad>"], "verb"), (["ɛ", "s", "t", "<pad>"], "noun")],
)
def test_the_reading_heard_scores_best(heard, best):
    lp = log_probs_from_path(_ids(heard), vocab_size=len(VOCAB))
    scores = score_readings(lp, VOCAB, READINGS)
    assert max(scores, key=scores.get) == best


def test_word_spans_ignore_case_and_punctuation():
    words = [
        {"start": 0.0, "end": 0.2, "text": " Mon"},
        {"start": 0.9, "end": 1.1, "text": " Est"},
        {"start": 2.0, "end": 2.2, "text": " est,"},
    ]
    assert word_spans(words, "est") == [(0.9, 1.1), (2.0, 2.2)]


def test_verify_words_flags_the_take_that_said_the_noun():
    text = "Mon frere\nEst parti"
    risks = flag_risks(text, heteronyms={"est": ("verb", "noun")})
    checks = checks_from_risks(
        risks, readings={"Est": READINGS}, expected={"est": "verb"}
    )
    assert checks == [("Est", READINGS, "verb")]
    words = [{"start": 0.9, "end": 1.1, "text": "Est"}]
    calls = []

    def emitter(audio, start, end):  # this take said /ɛst/
        calls.append((audio, start, end))
        heard = _ids(["ɛ", "s", "t", "<pad>"])
        return log_probs_from_path(heard, vocab_size=len(VOCAB)), VOCAB

    verdicts = verify_words("take.wav", checks, words=words, emitter=emitter)
    assert calls == [("take.wav", pytest.approx(0.82), pytest.approx(1.18))]
    (verdict,) = verdicts
    assert (verdict.heard, verdict.decided, verdict.ok) == ("noun", True, False)
    assert failing(verdicts) == verdicts


def test_an_undecided_or_unexpected_verdict_is_not_a_failure():
    close = Verdict("est", 0, 1, {"verb": -1.0, "noun": -1.2}, expected="verb")
    report_only = Verdict("est", 0, 1, {"verb": -1.0, "noun": -5.0})
    assert (close.decided, close.ok) == (False, None)
    assert report_only.ok is None
    assert failing([close, report_only]) == []


def test_the_phoneme_adapter_imports_nothing_from_voxy():
    from voxy.workers import adapter_path

    source = adapter_path("phonemes").read_text()
    assert "voxy" not in source.split('"""', 2)[-1]
