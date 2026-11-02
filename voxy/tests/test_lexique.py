"""Tests for voxy.lexique: deriving heteronym readings from Lexique rows."""

import pytest

from voxy.lexique import heteronyms_from_rows, lexique_heteronyms, lexique_to_ipa
from voxy.pronounce import heteronym_readings, load_readings


def row(ortho, phon, cgram, films=1.0, books=1.0):
    return {
        "ortho": ortho,
        "phon": phon,
        "cgram": cgram,
        "freqfilms2": str(films),
        "freqlivres": str(books),
    }


ROWS = [
    row("couvent", "kuv@", "NOM", 5, 8),
    row("couvent", "kuv", "VER", 0.1, 0.2),
    row("fils", "fis", "NOM", 300, 200),
    row("fils", "fil", "NOM", 3, 4),
    row("est", "e", "AUX", 900, 800),
    row("est", "e", "VER", 50, 40),
    row("est", "Est", "NOM", 20, 30),
    row("chat", "Sa", "NOM", 40, 30),
    row("rare", "RaR", "ADJ", 0.01, 0.0),
    row("rares", "RaR", "ADJ", 0.01, 0.0),
    row("pressent", "pRes", "VER:ind:pre", 0.5, 0.5),
    row("pressent", "pRes@", "VER:ind:pre", 0.3, 0.4),
]


def test_only_spellings_with_several_pronunciations():
    table = heteronyms_from_rows(ROWS)
    assert set(table) == {"couvent", "fils", "est", "pressent"}


def test_readings_are_named_by_part_of_speech_most_frequent_first():
    table = heteronyms_from_rows(ROWS)
    assert table["couvent"] == {"noun": "kuvɑ̃", "verb": "kuv"}
    assert list(table["est"]) == ["verb", "noun"]  # one reading per pronunciation


def test_same_part_of_speech_readings_are_told_apart_by_ipa():
    table = heteronyms_from_rows(ROWS)
    assert table["fils"] == {"noun:fis": "fis", "noun:fil": "fil"}
    assert table["pressent"] == {"verb:pʁes": "pʁes", "verb:pʁesɑ̃": "pʁesɑ̃"}


def test_rare_words_can_be_left_out():
    assert set(heteronyms_from_rows(ROWS, min_frequency=1)) == {
        "couvent",
        "fils",
        "est",
    }


def test_ipa_uses_the_phoneme_models_symbols():
    assert lexique_to_ipa("S1 9 2 °5 8 N G Z O") == "ʃœ̃ œ ø əɛ̃ ɥ ɲ ŋ ʒ ɔ"


def test_reads_a_lexique_file_and_merges_with_the_shipped_table(tmp_path):
    header = "ortho\tphon\tlemme\tcgram\tfreqfilms2\tfreqlivres\n"
    lines = [
        f"{r['ortho']}\t{r['phon']}\tx\t{r['cgram']}\t{r['freqfilms2']}\t{r['freqlivres']}"
        for r in ROWS
    ]
    path = tmp_path / "Lexique383.tsv"
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    derived = lexique_heteronyms(path)
    assert derived == heteronyms_from_rows(ROWS)
    merged = {**derived, **heteronym_readings("fr")}  # the hand-made readings win
    assert merged["est"] == heteronym_readings("fr")["est"]


def test_the_result_round_trips_through_load_readings(tmp_path):
    import json

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
