"""Tests for voxy.pronounce: CTC scoring of candidate readings, with a fake recogniser."""

import math

import numpy as np
import pytest

from voxy.base import Speech
from voxy.pronounce import (
    Verdict,
    audit,
    heteronym_readings,
    checks_by_line,
    checks_from_risks,
    crop_scorer,
    load_readings,
    phrase_scorer,
    reading_scorer,
    spot_scorer,
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

    verdicts = verify_words("take.wav", checks, words=words, emitter=emitter, scorer="crop")
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
        scorer="crop",
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
        scorer="crop",
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
    # text-wide, the two readings conflict: no single expected reading is checked
    assert checks_from_risks(risks, readings={"est": READINGS}) == [("est", READINGS, None)]
    # a resolved reading is not hidden by an earlier unresolved risk of the same word
    risks = [Risk(1, "est", "heteronym"), Risk(2, "est", "pos", expected="verb")]
    assert checks_from_risks(risks, readings={"est": READINGS}) == [("est", READINGS, "verb")]
    # an explicit `expected` still wins
    assert checks_by_line(risks[:1], readings={"est": READINGS}, expected={"est": "noun"}) == {
        1: [("est", READINGS, "noun")]
    }


def test_variants_of_one_reading_do_not_compete_and_count_as_that_reading():
    scores = {"verb": -1.0, "verb~liaison": -1.1, "noun": -3.0}
    verdict = Verdict("est", 0.0, 0.2, scores, expected="verb")
    assert verdict.decided and verdict.reading == "verb" and verdict.ok
    liaison = Verdict("est", 0.0, 0.2, {**scores, "verb~liaison": -0.9}, expected="verb")
    assert liaison.heard == "verb~liaison" and liaison.ok


def test_audit_checks_each_occurrence_against_its_own_expected_reading():
    text = "Il est là\nà l'est du jardin\nil est"
    words = [{"start": float(i), "end": i + 0.3, "text": w} for i, w in
             enumerate(["Il", "est", "là", "à", "l'est", "du", "jardin", "il", "est"])]
    # each window holds the word and its neighbours; the second "est" (a verb) came out as the noun
    first, second = ["a", "ɛ", "ɛ", "<pad>", "a"], ["a", "ɛ", "ɛ", "s", "s", "t", "t", "a"]

    def emitter(audio, start, end):
        return log_probs_from_path(_ids(first if start < 1 else second), vocab_size=len(VOCAB)), VOCAB

    findings = audit(text, "take.wav", readings={"est": READINGS}, expected={"est": "verb"},
                     scoring="spot", emitter=emitter, words_of=lambda audio: words)
    # "l'est" (line 2) is flagged too since #47, but {"est": ...} does not name it: report only
    assert [(f.risk.line, f.ok) for f in findings] == [(1, True), (2, None), (3, False)]
    findings = audit(text, "take.wav", readings={"est": READINGS}, expected={"est": "verb", "l’est": "noun"},
                     scoring="spot", emitter=emitter, words_of=lambda audio: words)
    assert [(f.risk.line, f.ok) for f in findings] == [(1, True), (2, True), (3, False)]


def test_audit_aligns_text_and_asr_words_and_leaves_missing_ones_unchecked():
    heard = ["a", "ɛ", "ɛ", "a"]
    findings = audit("est\nest", "take.wav", readings={"est": READINGS}, scoring="spot",
                     expected=lambda risk: "verb" if risk.line == 1 else "noun",
                     emitter=lambda audio, start, end: (log_probs_from_path(_ids(heard), vocab_size=len(VOCAB)), VOCAB),
                     words_of=lambda audio: [{"start": 0.0, "end": 0.2, "text": "est"}])
    assert [(f.risk.line, f.ok) for f in findings] == [(1, True), (2, None)]
    assert findings[1].verdict is None


def test_voxy_ships_a_french_heteronym_table():
    table = heteronym_readings("fr")
    assert {"est", "fils", "plus", "tous", "couvent"} <= set(table)
    with pytest.raises(ValueError, match="no heteronym table"):
        heteronym_readings("xx")


def test_the_pos_rules_name_readings_the_french_table_has():
    from voxy.pronounce import reading_name
    from voxy.speechtext import FRENCH_POS_HETERONYMS

    table = heteronym_readings("fr")
    for word, by_tag in FRENCH_POS_HETERONYMS.items():
        assert set(by_tag.values()) <= {reading_name(n) for n in table[word]}, word


def test_audit_scores_whole_phrases_so_a_neighbours_s_t_is_not_the_nouns():
    # "triste est le": the audio has s t (end of "triste") then ɛ: the verb, said right
    words = [{"start": 0.0, "end": 0.4, "text": "triste"}, {"start": 0.4, "end": 0.6, "text": "est"},
             {"start": 0.6, "end": 0.8, "text": "le"}]
    heard = ["s", "s", "t", "t", "ɛ", "ɛ", "<pad>", "a", "a"]
    ipa = {"triste": "st", "le": "a"}  # what the fake phonemizer writes out (VOCAB has no l, ə)
    findings = audit("triste est le", "take.wav", readings={"est": READINGS}, expected={"est": "verb"},
                     emitter=lambda audio, start, end: (log_probs_from_path(_ids(heard), vocab_size=len(VOCAB)), VOCAB),
                     phonemize=lambda texts, language: [ipa[t] for t in texts if t],
                     words_of=lambda audio: words)
    assert [f.ok for f in findings] == [True]
    with pytest.raises(ValueError, match="scoring"):
        audit("est", "take.wav", readings={"est": READINGS}, scoring="nope", words_of=lambda audio: words)


# "triste est le": the audio has s t (end of "triste") then ɛ, the verb said right
TRISTE_WORDS = [{"start": 0.0, "end": 0.4, "text": "triste"}, {"start": 0.4, "end": 0.6, "text": "est"},
                {"start": 0.6, "end": 0.8, "text": "le"}]
TRISTE_HEARD = ["s", "s", "t", "t", "ɛ", "ɛ", "<pad>", "a", "a"]


def _triste_phonemize(texts, language):
    ipa = {"triste": "st", "le": "a"}  # VOCAB has no l, ə
    return [ipa[t] for t in texts]


def _triste_emitter(audio, start, end):
    if start > 0.3:  # a crop of "est" alone still holds triste's s t
        return log_probs_from_path(_ids(["s", "t", "ɛ", "<pad>"]), vocab_size=len(VOCAB)), VOCAB
    return log_probs_from_path(_ids(TRISTE_HEARD), vocab_size=len(VOCAB)), VOCAB


TRISTE_PHRASE = phrase_scorer(emitter=_triste_emitter, phonemize=_triste_phonemize)


def test_verify_words_scores_whole_phrases_and_carries_the_scorers_margin():
    checks = [("est", READINGS, "verb")]
    kwargs = dict(words=TRISTE_WORDS, emitter=_triste_emitter)
    (phrase,) = verify_words("take.wav", checks, scorer=TRISTE_PHRASE, **kwargs)
    assert (phrase.reading, phrase.ok, phrase.margin) == ("verb", True, 2.0)
    (crop,) = verify_words("take.wav", checks, scorer="crop", **kwargs)
    assert (crop.reading, crop.margin) == ("noun", 0.5)  # what cropping got wrong
    (crop,) = verify_words("take.wav", checks, scorer=crop_scorer(emitter=_triste_emitter), **kwargs)
    assert crop.margin == 0.5
    (forced,) = verify_words("take.wav", checks, scorer=TRISTE_PHRASE, margin=9.0, **kwargs)
    assert forced.margin == 9.0 and forced.ok is None  # a margin given wins
    with pytest.raises(ValueError, match="scoring"):
        verify_words("take.wav", checks, scorer="nope", **kwargs)


def test_render_verified_passes_its_scorer_to_verify_words():
    renders = []

    def render(text, line, attempt):
        renders.append(attempt)
        return f"take-{attempt}"

    kwargs = dict(render=render, emitter=_triste_emitter, words_of=lambda audio: TRISTE_WORDS)
    checks = {1: [("est", READINGS, "verb")]}
    [line] = render_verified(["triste est le"], checks, scorer=TRISTE_PHRASE, **kwargs)
    assert line.verified and renders == [0]  # phrase scoring hears the verb: one take
    renders.clear()
    [line] = render_verified(["triste est le"], checks, scorer="crop", **kwargs)
    assert not line.verified and renders == [0, 1, 2]  # cropping "hears" the noun every take
    with pytest.raises(ValueError, match="scoring"):
        render_verified([], {}, render=render, emitter=None, scorer="nope")


def test_reading_scorer_rejects_an_unknown_scoring():
    with pytest.raises(ValueError, match="scoring"):
        reading_scorer("nope", emitter=_triste_emitter)


def test_each_scorer_carries_the_margin_it_is_calibrated_for():
    margins = {
        "phrase": phrase_scorer(emitter=_triste_emitter, phonemize=_triste_phonemize).margin,
        "spot": spot_scorer(emitter=_triste_emitter).margin,
        "crop": crop_scorer(emitter=_triste_emitter).margin,
    }
    assert margins == {"phrase": 2.0, "spot": 2.0, "crop": 0.5}
    assert reading_scorer("crop", emitter=_triste_emitter, margin=1.0).margin == 1.0
    with pytest.raises(TypeError):  # a scorer takes only its own options
        crop_scorer(emitter=_triste_emitter, phonemize=_triste_phonemize)


def test_a_phonemizer_that_drops_texts_is_an_error_not_a_short_zip():
    scorer = phrase_scorer(emitter=_triste_emitter, phonemize=lambda texts, language: ["st"])
    with pytest.raises(ValueError, match="1 IPA for 2 texts"):
        scorer("take.wav", TRISTE_WORDS, 1, READINGS)


def test_audit_takes_a_ready_scorer_over_its_scoring_options():
    findings = audit("triste est le", "take.wav", readings={"est": READINGS}, expected={"est": "verb"},
                     scorer=TRISTE_PHRASE, scoring="crop", words_of=lambda audio: TRISTE_WORDS)
    assert [(f.ok, f.verdict.margin) for f in findings] == [(True, 2.0)]
    (crop,) = audit("triste est le", "take.wav", readings={"est": READINGS}, expected={"est": "verb"},
                    scoring="crop", phonemize=_triste_phonemize, emitter=_triste_emitter,
                    words_of=lambda audio: TRISTE_WORDS)
    assert (crop.verdict.reading, crop.verdict.margin) == ("noun", 0.5)  # options it does not take are ignored


def test_audit_forwards_each_option_to_the_scorer_that_takes_it():
    words = [{"start": float(i), "end": i + 0.5, "text": w} for i, w in enumerate(["a", "b", "est", "c", "d"])]
    windows, languages = [], []

    def emitter(audio, start, end):
        windows.append((start, end))
        return log_probs_from_path(_ids(["a", "ɛ", "a"]), vocab_size=len(VOCAB)), VOCAB

    def phonemize(texts, language):
        languages.append(language)
        return ["a" for _ in texts]

    common = dict(readings={"est": READINGS}, emitter=emitter, words_of=lambda audio: words)
    audit("a b est c d", "t.wav", scoring="phrase", context=2, padding=0.0, language="en",
          phonemize=phonemize, **common)
    assert windows[-1] == (0.0, 4.5) and languages == ["en"]  # context and language reached it
    audit("a b est c d", "t.wav", scoring="crop", padding=0.25, **common)
    assert windows[-1] == (1.75, 2.75)  # padding reached the crop scorer
    lo, hi = (audit("a b est c d", "t.wav", scoring="spot", filler_penalty=f, **common)[0].verdict.scores
              for f in (0.5, 3.0))
    assert lo != hi  # filler_penalty reached the spot scorer


def test_an_explicit_margin_wins_over_the_scorers_in_audit_and_render_verified():
    words = [{"start": 0.0, "end": 0.3, "text": "est"}]
    emitter = _emitter_hearing({"take": ["ɛ"]})
    [finding] = audit("est", "take", readings={"est": READINGS}, scoring="crop", margin=7.0,
                      emitter=emitter, words_of=lambda audio: words)
    assert finding.verdict.margin == 7.0
    [line] = render_verified(["est"], {1: [("est", READINGS, "verb")]}, render=lambda *a: "take",
                             emitter=emitter, words_of=lambda audio: words, scorer="crop", margin=7.0)
    assert line.chosen.verdicts[0].margin == 7.0


# heteronyms elided onto a clitic (#47) ----------------------------------------


def test_flag_risks_flags_a_heteronym_after_a_noun_clitic_only():
    from voxy.speechtext import NOUN_CLITICS

    text = "Il est à l'Est, c'est loin, vent d’est"
    risks = flag_risks(text, heteronyms={"est": ()}, clitics=NOUN_CLITICS["fr"])
    assert [(r.clitic, r.word) for r in risks] == [("", "est"), ("l'", "Est"), ("d’", "est")]
    assert [r.word for r in flag_risks(text, heteronyms={"est": ()})] == ["est"]  # default: as before


def test_audit_keeps_c_est_l_est_and_est_apart_on_one_line():
    from voxy.speechtext import Risk

    text = "C'est à l'Est, il est là"
    # Whisper's split: the apostrophe goes with the word
    said = [" C", "'est", " à", " l", "'est", " il", " est", " là."]
    words = [{"start": float(i), "end": i + 0.3, "text": w} for i, w in enumerate(said)]

    def emitter(audio, start, end):  # the noun around "l'Est" (t = 4), the verb elsewhere
        noun, verb = ["a", "ɛ", "ɛ", "s", "s", "t", "t", "a"], ["a", "ɛ", "ɛ", "<pad>", "a"]
        heard = noun if 3 <= start + 0.08 < 5 else verb
        return log_probs_from_path(_ids(heard), vocab_size=len(VOCAB)), VOCAB

    def spacy_like(text):  # what pos_detector proposes: c'est's est comes first
        return [Risk(1, "est", "pos", expected="verb", clitic="C'"),
                Risk(1, "Est", "pos", expected="noun", clitic="l'"),
                Risk(1, "est", "pos", expected="verb")]

    findings = audit(text, "take.wav", readings={"est": READINGS}, detectors=[spacy_like],
                     scoring="spot", context=0, emitter=emitter, words_of=lambda audio: words)
    assert [(f.risk.clitic, f.verdict.expected, f.verdict.start, f.ok) for f in findings] == [
        ("l'", "noun", 4.0, True), ("", "verb", 6.0, True)]


def test_render_checks_skip_elided_risks_and_leave_that_words_reading_open():
    from voxy.speechtext import Risk

    risks = [Risk(1, "est", "pos", expected="verb"), Risk(1, "Est", "pos", expected="noun", clitic="l'"),
             Risk(2, "est", "pos", expected="verb"), Risk(3, "Est", "pos", expected="noun", clitic="l'")]
    # line 1: an ASR hearing "l'" | "Est" would judge that noun by "verb", so no reading;
    # line 2: no elided form, checked as before; line 3: only the elided form, not checked
    assert checks_by_line(risks, readings={"est": READINGS}) == {
        1: [("est", READINGS, None)], 2: [("est", READINGS, "verb")]}


def test_noun_clitics_are_elisions_and_a_line_initial_elided_capital_is_the_trigger():
    from voxy.speechtext import ELISIONS, NOUN_CLITICS

    for language, clitics in NOUN_CLITICS.items():
        assert set(clitics) <= set(ELISIONS[language]), language
    [risk] = flag_risks("Il dort,\nL'est du ciel", heteronyms={"est": ()}, clitics=("l'",))
    assert (risk.clitic, risk.word, risk.line_initial_capital) == ("L'", "est", True)


def test_phrase_scoring_writes_an_elided_clitic_instead_of_its_letter_name():
    from voxy.pronounce import phrase_readings, phrase_scorer, spoken_context

    fr = {"l'": "l", "d'": "d"}
    # Whisper's split inside the context, and a curly, capitalised clitic left at the end
    assert spoken_context([{"text": " l"}, {"text": "'aube"}, {"text": " L’"}], elisions=fr) == ("l'aube", "l")
    assert spoken_context([{"text": " read"}, {"text": " 'War"}], elisions=fr) == ("read 'War", "")
    vocab = ["<pad>", "ɛ", "s", "t", "a", "l", "d"]
    # "à l'Est d'": the clitics on both sides are written out, never phonemized
    words = [{"start": 0.0, "end": 0.2, "text": "à"}, {"start": 0.2, "end": 0.3, "text": "l'"},
             {"start": 0.3, "end": 0.6, "text": "Est"}, {"start": 0.6, "end": 0.7, "text": "d'"}]
    log_probs = log_probs_from_path([4, 5, 1, 2, 3, 6], vocab_size=len(vocab))
    asked = []

    def phonemize(texts, language):  # espeak's habit: a standalone l' is /ɛl/
        asked.extend(texts)
        return [t.replace("à", "a").replace("l'", "ɛl").replace(" ", "") for t in texts]

    emit = lambda audio, start, end: (log_probs, vocab)
    scores = phrase_scorer(emitter=emit, phonemize=phonemize, context=2)("take.wav", words, 2, READINGS)
    assert asked == ["à"] and scores == phrase_readings(log_probs, vocab, READINGS, before="al", after="d")
    asked.clear()  # no elision table: the old behaviour
    phrase_scorer(emitter=emit, phonemize=phonemize, context=2, elisions={})("take.wav", words, 2, READINGS)
    assert asked == ["à l'", "d'"]
