"""Tests for voxy.pronounce: CTC scoring of candidate readings, with a fake recogniser."""

import math

import numpy as np
import pytest

from voxy.base import Speech
from voxy.pronounce import (
    Verdict,
    checks_by_line,
    checks_from_risks,
    load_readings,
    render_verified,
    speech_renderer,
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


# re-rendering ---------------------------------------------------------------


def _emitter_hearing(takes):
    """A fake recogniser that hears, in each take's audio, the reading listed for it."""

    def emit(audio, start, end):
        return log_probs_from_path(_ids(takes[audio]), vocab_size=len(VOCAB)), VOCAB

    return emit


def _words(audio):
    return [{"start": 0.0, "end": 0.3, "text": "Est"}]


def test_render_verified_stops_at_max_attempts_and_keeps_the_least_bad_take():
    heard = {}

    def render(text, line, attempt):
        audio = f"{line}-{attempt}"
        heard[audio] = "ɛ"  # every take says the verb
        return audio

    checks = {2: [("Est", READINGS, "noun")]}
    first, second = render_verified(
        ["plain line", "à l'Est"],
        checks,
        render=render,
        emitter=_emitter_hearing(heard),
        words_of=_words,
        max_attempts=3,
    )
    assert [t.audio for t in first.takes] == ["1-0"]  # no checks: rendered once
    assert first.verified and first.chosen.verdicts == ()
    assert [t.attempt for t in second.takes] == [0, 1, 2]
    assert not second.verified and second.chosen.attempt == 0


def test_a_line_whose_word_asr_never_found_is_not_verified_and_blank_lines_are_skipped():
    heard = {}

    def render(text, line, attempt):
        heard[f"{line}-{attempt}"] = "ɛst"
        return f"{line}-{attempt}"

    blank, line = render_verified(
        ["", "L'Est"],
        {2: [("Est", READINGS, "noun")]},
        render=render,
        emitter=_emitter_hearing(heard),
        words_of=lambda audio: [{"start": 0.0, "end": 0.3, "text": "L'Est"}],
    )
    assert blank.takes == () and blank.chosen is None and blank.verified
    assert len(line.takes) == 1 and line.unconfirmed == ["Est"] and not line.verified


def test_render_verified_validates_max_attempts():
    with pytest.raises(ValueError, match="max_attempts"):
        render_verified([], {}, render=None, emitter=None, max_attempts=0)


def test_checks_by_line_groups_risks_with_known_readings():
    risks = flag_risks("Il est là\nEst ou ouest\nrien", heteronyms={"est": ()})
    checks = checks_by_line(risks, readings={"est": READINGS}, expected={"est": "verb"})
    assert sorted(checks) == [1, 2]
    assert checks[2] == [("Est", READINGS, "verb")]


def test_speech_renderer_saves_each_take_and_passes_the_attempt(tmp_path):
    calls = []

    class FakeModel:
        def synthesize(self, text, voice=None, **kwargs):
            calls.append((text, voice, kwargs))
            return Speech(b"RIFF", format="wav")

    render = speech_renderer(FakeModel(), "v", outdir=tmp_path, take_kwarg="seed")
    path = render("hello", 3, 1)
    assert path == str(tmp_path / "0003-1.wav") and open(path, "rb").read() == b"RIFF"
    assert calls == [("hello", "v", {"seed": 1})]


def test_load_readings_from_tsv_and_json(tmp_path):
    tsv = tmp_path / "h.tsv"
    tsv.write_text("# word\tname\tipa\nest\tverb\tɛ\n\nest\tnoun\tɛst\n", "utf-8")
    assert load_readings(tsv) == {"est": READINGS}
    js = tmp_path / "h.json"
    js.write_text('{"est": {"verb": "ɛ", "noun": "ɛst"}}', "utf-8")
    assert load_readings(js) == {"est": READINGS}
    bad = tmp_path / "bad.tsv"
    bad.write_text("est\tverb\n", "utf-8")
    with pytest.raises(ValueError, match="bad.tsv:1"):
        load_readings(bad)


def test_checks_by_line_takes_a_detectors_expected_reading():
    from voxy.speechtext import Risk

    risks = [Risk(1, "est", "pos", expected="verb"), Risk(2, "Est", "pos", expected="noun")]
    checks = checks_by_line(risks, readings={"est": READINGS})
    assert checks == {1: [("est", READINGS, "verb")], 2: [("Est", READINGS, "noun")]}
    # an explicit `expected` still wins
    assert checks_by_line(risks[:1], readings={"est": READINGS}, expected={"est": "noun"}) == {
        1: [("est", READINGS, "noun")]
    }
