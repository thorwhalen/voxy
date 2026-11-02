"""Tests for voxy.lexique: deriving heteronym readings from Lexique rows."""

import json
from importlib.resources import files

import pytest

from voxy.lexique import (
    LEXIQUE_POS_NAMES,
    LEXIQUE_TO_IPA,
    REQUIRED_COLUMNS,
    heteronyms_from_rows,
    lexique_heteronyms,
    lexique_to_ipa,
)
from voxy.pronounce import DFLT_PHONEME_MODEL, heteronym_readings, load_readings
from voxy.speechtext import FRENCH_POS_HETERONYMS, FRENCH_READING_NAMES

#: The 37 symbols of the ``phon`` column of Lexique 3.83.
LEXIQUE_ALPHABET = "12589@EGNORSZabdefgijklmnopstuvwxyz§°"
#: Symbols of the phoneme model's vocabulary (the ``vocab.json`` of
#: facebook/wav2vec2-lv-60-espeak-cv-ft, checked by hand) that voxy's French table
#: does not use yet.
VOCAB_SYMBOLS_NOT_IN_FR_TABLE = {"ø", "œ", "œ̃", "ɲ", "ŋ", "ʃ", "b", "w", "x"}


def row(ortho, phon, cgram, films=1.0, books=1.0, *, lemma=None):
    return {
        "ortho": ortho,
        "phon": phon,
        "lemme": ortho if lemma is None else lemma,
        "cgram": cgram,
        "freqfilms2": str(films),
        "freqlivres": str(books),
    }


ROWS = [
    row("couvent", "kuv@", "NOM", 5, 8),
    row("couvent", "kuv", "VER", 0.1, 0.2, lemma="couver"),
    row("fils", "fis", "NOM", 300, 200),
    row("fils", "fil", "NOM", 3, 4, lemma="fil"),
    row("est", "e", "AUX", 900, 800, lemma="être"),
    row("est", "e", "VER", 50, 40, lemma="être"),
    row("est", "Est", "NOM", 20, 30),
    row("chat", "Sa", "NOM", 40, 30),
    row("rare", "RaR", "ADJ", 0.01, 0.0),
    row("rares", "RaR", "ADJ", 0.01, 0.0, lemma="rare"),
    row("pressent", "pRes", "VER:ind:pre", 0.5, 0.5, lemma="presser"),
    row("pressent", "pRes@", "VER:ind:pre", 0.3, 0.4, lemma="pressentir"),
]


def test_only_spellings_with_several_pronunciations():
    table = heteronyms_from_rows(ROWS)
    assert set(table) == {"couvent", "fils", "est", "pressent"}


def test_readings_are_named_by_part_of_speech_most_frequent_first():
    table = heteronyms_from_rows(ROWS)
    assert table["couvent"] == {"noun": "kuvɑ̃", "verb": "kuv"}
    assert list(table["est"]) == ["verb", "noun"]  # one reading per pronunciation


def test_same_part_of_speech_readings_are_named_by_lemma():
    table = heteronyms_from_rows(ROWS)
    assert table["fils"] == {"fils": "fis", "fil": "fil"}
    assert table["pressent"] == {"presser": "pʁes", "pressentir": "pʁesɑ̃"}


def test_same_part_of_speech_and_lemma_readings_are_told_apart_by_ipa():
    rows = [row("x", "fis", "NOM", 3), row("x", "fil", "NOM", 1)]
    assert heteronyms_from_rows(rows) == {"x": {"noun:fis": "fis", "noun:fil": "fil"}}


def test_rare_words_can_be_left_out():
    assert set(heteronyms_from_rows(ROWS, min_frequency=1)) == {
        "couvent",
        "fils",
        "est",
    }


def test_a_readings_weight_is_the_sum_of_its_rows():
    rows = [
        row("x", "a", "NOM", 3),
        row("x", "b", "ADJ", 2),
        row("x", "b", "ADV", 2),  # b: 4 per million in all, above a's 3
    ]
    assert list(heteronyms_from_rows(rows)["x"]) == ["adjective", "noun"]
    assert set(heteronyms_from_rows(rows, min_frequency=4)) == {"x"}
    assert heteronyms_from_rows(rows, min_frequency=4.5) == {}


# transcription noise ---------------------------------------------------------------


@pytest.mark.parametrize(
    "heavy, light",
    [("vjOl", "vjol"), ("pREs", "pRes"), ("R9", "R2"), ("R°la", "Rla")],
)
def test_pronunciations_equal_up_to_noise_are_one_reading_and_its_variant(heavy, light):
    rows = [row("x", heavy, "VER", 5), row("x", light, "VER", 1), row("x", "zz", "NOM")]
    readings = heteronyms_from_rows(rows)["x"]
    assert readings == {
        "verb": lexique_to_ipa(heavy),
        f"verb~{lexique_to_ipa(light)}": lexique_to_ipa(light),
        "noun": "zz",
    }


def test_a_pronunciation_with_a_verb_row_is_the_verb():
    rows = [  # Lexique also tags the verb form "vjOl" as a noun, more frequent
        row("violent", "vjOl", "NOM", 9),
        row("violent", "vjOl", "VER", 1, lemma="violer"),
        row("violent", "vjOl@", "ADJ", 30),
    ]
    assert heteronyms_from_rows(rows)["violent"] == {
        "adjective": "vjɔlɑ̃",
        "verb": "vjɔl",
    }


# names and symbols shared with the rest of voxy ------------------------------------


def test_reading_names_are_voxys_french_reading_names():
    assert set(LEXIQUE_POS_NAMES.values()) <= set(FRENCH_READING_NAMES.values())
    assert LEXIQUE_POS_NAMES["ADJ"] == "adjective"
    assert LEXIQUE_POS_NAMES["PRO"] == "pronoun"
    assert LEXIQUE_POS_NAMES["ART"] == "det"


def test_derived_names_match_the_hand_table_and_the_pos_detector():
    rows = [row("couvent", "kuv@", "NOM", 5), row("couvent", "kuv", "VER", 1)]
    rows += [row("content", "k§t@", "ADJ", 5), row("content", "k§t", "VER", 1)]
    derived, hand = heteronyms_from_rows(rows), heteronym_readings("fr")
    for word in derived:
        assert set(derived[word]) == set(hand[word])
        assert set(derived[word]) == set(FRENCH_POS_HETERONYMS[word].values())


def test_ipa_uses_the_phoneme_models_symbols():
    assert lexique_to_ipa("S1 9 2 °5 8 N G Z O g") == "ʃœ̃ œ ø əɛ̃ y ɲ ŋ ʒ ɔ ɡ"


def _lexique_ipa_symbols():
    passed_through = set(LEXIQUE_ALPHABET) - set(LEXIQUE_TO_IPA)
    return set(LEXIQUE_TO_IPA.values()) | passed_through


def test_every_ipa_symbol_is_one_the_phoneme_model_knows():
    fr = files("voxy").joinpath("data/heteronyms/fr.json").read_text(encoding="utf-8")
    fr_ipa = "".join(
        ipa
        for word, readings in json.loads(fr).items()
        if not word.startswith("_")
        for ipa in readings.values()
    )
    unknown = {
        s
        for s in _lexique_ipa_symbols()
        if s not in fr_ipa and s not in VOCAB_SYMBOLS_NOT_IN_FR_TABLE
    }
    assert not unknown


def test_every_ipa_symbol_is_in_the_cached_model_vocabulary():
    from voxy.envs import HF_SUBDIR, models_dir

    repo = "models--" + DFLT_PHONEME_MODEL.replace("/", "--")
    vocabs = sorted(models_dir(HF_SUBDIR, "hub", repo).glob("snapshots/*/vocab.json"))
    if not vocabs:
        pytest.skip("the phoneme model is not downloaded")
    vocab = json.loads(vocabs[-1].read_text(encoding="utf-8"))
    assert _lexique_ipa_symbols() <= set(vocab)


# files -----------------------------------------------------------------------------


def _write_tsv(path, rows, columns=REQUIRED_COLUMNS):
    lines = ["\t".join(columns)] + ["\t".join(r[c] for c in columns) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_reads_a_lexique_file_and_merges_with_the_shipped_table(tmp_path):
    derived = lexique_heteronyms(_write_tsv(tmp_path / "Lexique383.tsv", ROWS))
    assert derived == heteronyms_from_rows(ROWS)
    merged = {**derived, **heteronym_readings("fr")}  # the hand-made readings win
    assert merged["est"] == heteronym_readings("fr")["est"]


def test_keyword_arguments_reach_the_derivation(tmp_path):
    path = _write_tsv(tmp_path / "Lexique383.tsv", ROWS)
    assert lexique_heteronyms(path, min_frequency=1) == heteronyms_from_rows(
        ROWS, min_frequency=1
    )


@pytest.mark.parametrize("missing", ["freqfilms2", "lemme"])
def test_a_file_without_a_required_column_is_an_error(tmp_path, missing):
    columns = [c for c in REQUIRED_COLUMNS if c != missing]
    path = _write_tsv(tmp_path / "Lexique383.tsv", ROWS, columns)
    with pytest.raises(ValueError, match=missing):
        lexique_heteronyms(path)


def test_the_result_round_trips_through_load_readings(tmp_path):
    path = tmp_path / "fr-lexique.json"
    path.write_text(
        json.dumps(heteronyms_from_rows(ROWS), ensure_ascii=False), encoding="utf-8"
    )
    assert load_readings(path) == heteronyms_from_rows(ROWS)


@pytest.mark.parametrize("missing", ["ortho", "phon"])
def test_rows_without_a_spelling_or_pronunciation_are_skipped(missing):
    rows = [row("x", "a", "NOM"), row("x", "b", "NOM")]
    rows[1][missing] = ""
    assert heteronyms_from_rows(rows) == {}
