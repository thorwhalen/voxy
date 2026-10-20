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
