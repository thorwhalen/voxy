"""Display text vs speech text, and risk flags (no model needed)."""

import numpy as np

import voxy
from voxy.speechtext import Risk, flag_risks, prepare_speech_text

VERSE = "Mon frere\nEst parti\nLoin d'ici.\nEst-ce vrai ?"


def test_a_verse_initial_capital_that_is_not_a_sentence_start_is_lowercased():
    result = prepare_speech_text(VERSE)
    assert result.display == VERSE
    # "Est" follows "Mon frere" (no sentence end); "Loin" too; "Est-ce" follows a full stop
    assert result.speech.split("\n") == ["Mon frere", "est parti", "loin d'ici.", "Est-ce vrai ?"]
    assert [(c.line, c.original, c.replacement) for c in result.changes] == [
        (2, "Est", "est"),
        (3, "Loin", "loin"),
    ]


def test_deliberate_capitals_are_kept():
    text = "Il parle a Paris\nParis dort\nEst parti"
    assert prepare_speech_text(text).speech == "Il parle a Paris\nParis dort\nest parti"  # mid-line "Paris" is deliberate
    assert prepare_speech_text(text, keep={"Est"}).speech.endswith("Est parti")


def test_paragraph_and_sentence_starts_keep_their_capital():
    text = "Mon frere.\nEst parti\n\nLoin d'ici"
    assert prepare_speech_text(text).speech == "Mon frere.\nEst parti\n\nLoin d'ici"


def test_respellings_and_aliases_are_whole_word_and_recorded():
    result = prepare_speech_text(
        "Ils couvrent Zoe", lowercase_verse_initial=False,
        respellings={"couvrent": "couvre"}, aliases={"Zoe": "Zoh-ay"},
    )
    assert result.speech == "Ils couvre Zoh-ay"
    assert {c.reason for c in result.changes} == {"alias", "respelling"}


def test_transforms_run_last():
    assert prepare_speech_text("a", transforms=[str.upper]).speech == "A"


def test_flag_risks_marks_the_capital_trigger():
    risks = flag_risks(VERSE, heteronyms={"est": ("ɛ", "ɛst"), "frere": ()})
    by_line = {r.line: r for r in risks if r.word.lower() == "est"}
    assert by_line[2].line_initial_capital and by_line[2].readings == ("ɛ", "ɛst")
    assert 4 not in by_line  # "Est-ce" is a different word


def test_flag_risks_merges_extra_detectors():
    detector = lambda text: [Risk(1, "x", "custom")]  # noqa: E731
    assert [r.reason for r in flag_risks("a b", extra=[detector])] == ["custom"]


def test_local_clone_model_applies_text_prep_before_chunking():
    heard = []

    def generate(text, *, ref_audio, ref_text, **params):
        heard.append(text)
        return np.zeros(800, np.float32), 8000

    model = voxy.create_speech_model(
        "chatterbox", generate=generate, text_prep=lambda t: prepare_speech_text(t).speech
    )
    model.synthesize("Mon frere\nEst parti", voice={"ref_audio": "r.wav", "ref_text": "t"})
    assert any("est parti" in h for h in heard) and not any("Est parti" in h for h in heard)


def _fake_tagger(tags):
    """A tagger that tags each listed word with its POS, and every other word X."""

    def tag(text):
        import re

        return [
            (m.start(), m.group(0), tags.get(m.group(0).casefold(), "X"))
            for m in re.finditer(r"[^\W\d_][\w'-]*", text)
        ]

    return tag


def test_pos_detector_names_the_reading_the_part_of_speech_picks():
    from voxy.speechtext import pos_detector

    detect = pos_detector(tagger=_fake_tagger({"est": "AUX"}))
    risks = flag_risks("Mon frere\nest parti\nrien", extra=[detect])
    assert [(r.line, r.word, r.expected) for r in risks] == [(2, "est", "verb")]
    assert set(risks[0].readings) == {"verb", "noun"}


def test_pos_detector_keeps_a_heteronym_with_an_unmapped_tag_unresolved():
    from voxy.speechtext import pos_detector

    detect = pos_detector({"est": {"NOUN": "noun"}}, tagger=_fake_tagger({"est": "SYM"}))
    (risk,) = detect("vers est")
    assert risk.expected is None and "picks no known reading" in risk.reason


def test_spacy_tagger_says_how_to_install_spacy():
    import importlib.util

    import pytest

    from voxy.speechtext import spacy_tagger

    if importlib.util.find_spec("spacy"):
        pytest.skip("spaCy is installed here")
    with pytest.raises(ImportError, match=r"voxy\[pos\]"):
        spacy_tagger()("est")


def test_llm_detector_keeps_only_words_really_on_their_line():
    from voxy.speechtext import llm_detector

    prompts = []

    def chat(prompt):
        prompts.append(prompt)
        return (
            "Here you go:\n```json\n"
            '[{"line": 2, "word": "Est", "readings": ["verb", "noun"], "expected": "verb", "why": "verb after a subject"},'
            ' {"line": 1, "word": "ouest", "readings": ["a"]},'
            ' {"line": 9, "word": "est"},'
            ' {"line": 3, "word": "ici", "readings": ["x"], "expected": "not-a-reading"},'
            ' "junk"]\n```'
        )

    risks = flag_risks("Mon frere\nEst parti\nLoin d'ici.", extra=[llm_detector(chat=chat)])
    assert [(r.line, r.word, r.expected) for r in risks] == [(2, "Est", "verb"), (3, "ici", None)]
    assert "verb after a subject" in risks[0].reason
    assert "1: Mon frere\n2: Est parti" in prompts[0] and "French" in prompts[0]


def test_llm_detector_proposes_nothing_on_a_reply_that_is_not_json():
    from voxy.speechtext import llm_detector

    assert llm_detector(chat=lambda p: "I cannot [help] with that")("est") == []


def test_llm_detector_joins_a_streamed_reply_and_matches_abbreviations():
    from voxy.speechtext import llm_detector

    chunks = ['[{"line": 1, "word": "Dr.", ', '"readings": ["docteur"]}]']
    (risk,) = llm_detector(chat=lambda p: iter(chunks))("Le Dr. Martin")
    assert (risk.line, risk.word) == (1, "Dr.")


def test_llm_detector_live():
    """Opt-in: VOXY_LIVE_LLM=1 (a paid model call through aix)."""
    import os

    import pytest

    if os.environ.get("VOXY_LIVE_LLM") != "1":
        pytest.skip("set VOXY_LIVE_LLM=1 to call a real model")
    from voxy.speechtext import llm_detector

    risks = llm_detector()("Mon frere\nest parti vers l'est.")
    assert all(r.line in (1, 2) for r in risks)


def test_llm_detector_proposals_carry_the_clitic_of_an_elided_word():
    import json

    from voxy.speechtext import llm_detector

    def detect(text, *proposals):
        return llm_detector(chat=lambda prompt: json.dumps(list(proposals)))(text)

    def p(line, word, expected="noun"):
        return {"line": line, "word": word, "readings": ["verb", "noun"], "expected": expected}

    # the model writes it elided
    (risk,) = detect("Il est à l'Est", p(1, "l'Est"))
    assert (risk.clitic, risk.word, risk.expected) == ("l'", "Est", "noun")
    # the model writes it bare and the line only holds it elided
    (risk,) = detect("Loin, à l’Est", p(1, "Est"))
    assert (risk.clitic, risk.word) == ("l’", "Est")
    # the plain word on the line stays plain, even when the line has the elided form too
    (risk,) = detect("Il est à l'Est", p(1, "est", "verb"))
    assert (risk.clitic, risk.word) == ("", "est")
    # any elision counts (as pos_detector's); one the line does not hold adds nothing
    assert detect("Loin d'ici", p(1, "ici"))[0].clitic == "d'"
    assert llm_detector(chat=lambda _: json.dumps([p(1, "ici")]), clitics=())("Loin d'ici")[0].clitic == ""
    assert detect("Il est là", p(1, "l'Est")) == []
    assert [(r.clitic, r.word) for r in detect("Il est c'est", p(1, "c'est"))] == [("c'", "est")]


def test_llm_detector_places_a_bare_word_by_case_apostrophe_and_occurrence():
    import json

    from voxy.speechtext import llm_detector

    def detect(text, *words):
        proposals = [{"line": 1, "word": w, "readings": ["verb", "noun"]} for w in words]
        return [(r.word, r.clitic) for r in llm_detector(chat=lambda _: json.dumps(proposals))(text)]

    assert detect("Il est à l'Est", "Est") == [("Est", "l'")]  # case picks the elided noun
    assert detect("Il est à l'Est", "est", "Est") == [("est", ""), ("Est", "l'")]  # nth place
    assert detect("C'est à l'Est", "Est") == [("Est", "l'")]
    assert detect("l’Est", "l'Est") == [("Est", "l’")]  # either apostrophe; the line's spelling
    assert detect("Il est c'est", "est") == [("est", "")]  # the plain word first
    assert detect("C'est le vent de l'est", "est") == [("est", "C'")]  # a tie: the first place
    assert detect("C'est le vent de l'est", "l'est") == [("est", "l'")]  # elided: that place
    assert detect("l'Est et d'Est", "Est", "Est") == [("Est", "l'"), ("Est", "d'")]

