"""
French heteronyms from Lexique: every spelling with more than one pronunciation.

Lexique (New, Pallier et al., http://www.lexique.org) lists about 140 000 French
word forms with their pronunciation, part of speech and frequency. Its homographs
that are not homophones (*est* /ɛ/ or /ɛst/, *couvent* /kuv/ or /kuvɑ̃/,
*président*...) are the words a speech model can misread. voxy ships only a short
hand-made table (``voxy.pronounce.heteronym_readings``) and no Lexique data, so
derive the full table from your own copy:

    >>> from voxy.lexique import lexique_heteronyms                     # doctest: +SKIP
    >>> table = lexique_heteronyms("Lexique383.tsv", min_frequency=1)   # doctest: +SKIP
    >>> table["couvent"]                                                # doctest: +SKIP
    {'noun': 'kuvɑ̃', 'verb': 'kuv'}

The result has the shape ``voxy.pronounce`` takes (``{word: {reading name: IPA}}``,
IPA in the espeak symbols of its phoneme model), so it merges with the shipped
table: ``{**lexique_heteronyms(path), **heteronym_readings("fr")}``.

>>> rows = [
...     {"ortho": "est", "phon": "e", "cgram": "VER", "freqfilms2": "100", "freqlivres": "90"},
...     {"ortho": "est", "phon": "Est", "cgram": "NOM", "freqfilms2": "20", "freqlivres": "30"},
...     {"ortho": "chat", "phon": "Sa", "cgram": "NOM", "freqfilms2": "40", "freqlivres": "30"},
... ]
>>> heteronyms_from_rows(rows)
{'est': {'verb': 'e', 'noun': 'ɛst'}}
"""

import csv
import os
from collections.abc import Iterable, Mapping

#: Lexique's phonetic code -> the espeak IPA of voxy's phoneme model.
LEXIQUE_TO_IPA = {
    "E": "ɛ",
    "O": "ɔ",
    "°": "ə",
    "2": "ø",
    "9": "œ",
    "5": "ɛ̃",
    "1": "œ̃",
    "@": "ɑ̃",
    "§": "ɔ̃",
    "8": "ɥ",
    "R": "ʁ",
    "S": "ʃ",
    "Z": "ʒ",
    "N": "ɲ",
    "G": "ŋ",
}
#: Lexique's part of speech (``cgram``, before any ``:``) -> a reading name.
LEXIQUE_POS_NAMES = {
    "NOM": "noun",
    "VER": "verb",
    "AUX": "verb",
    "ADJ": "adj",
    "ADV": "adv",
    "PRE": "prep",
    "CON": "conj",
    "PRO": "pron",
    "ART": "art",
    "ONO": "interj",
}
#: Columns read from a Lexique 3 file.
ORTHO, PHON, CGRAM = "ortho", "phon", "cgram"
FREQUENCY_COLUMNS = ("freqfilms2", "freqlivres")


def lexique_to_ipa(phon: str, *, symbols: Mapping[str, str] = LEXIQUE_TO_IPA) -> str:
    """A Lexique pronunciation in IPA.

    >>> lexique_to_ipa("kuv@"), lexique_to_ipa("pRezid@"), lexique_to_ipa("Est")
    ('kuvɑ̃', 'pʁezidɑ̃', 'ɛst')
    """
    return "".join(symbols.get(c, c) for c in phon)


def _frequency(row: Mapping) -> float:
    """Occurrences per million (films and books, the larger)."""
    return max(float(row.get(c) or 0) for c in FREQUENCY_COLUMNS)


def _pos_name(cgram: str, names: Mapping[str, str]) -> str:
    tag = cgram.split(":")[0]
    return names.get(tag, tag.lower() or "word")


def heteronyms_from_rows(
    rows: Iterable[Mapping],
    *,
    min_frequency: float = 0.0,
    pos_names: Mapping[str, str] = LEXIQUE_POS_NAMES,
    symbols: Mapping[str, str] = LEXIQUE_TO_IPA,
) -> dict[str, dict[str, str]]:
    """``{word: {reading name: IPA}}`` for every spelling read in more than one way.

    A reading is named after the part of speech of its most frequent row; when two
    readings would share a name, both are named ``<pos>:<IPA>``. Words whose most
    frequent form is under ``min_frequency`` (per million) are left out.
    """
    readings: dict[str, dict[str, tuple[float, str]]] = {}
    for row in rows:
        word, phon = row[ORTHO], row[PHON]
        if not word or not phon:
            continue
        frequency, pos = _frequency(row), _pos_name(row[CGRAM], pos_names)
        best = readings.setdefault(word, {}).get(phon)
        if best is None or frequency > best[0]:
            readings[word][phon] = (frequency, pos)
    table = {}
    for word, by_phon in readings.items():
        if len(by_phon) < 2 or max(f for f, _ in by_phon.values()) < min_frequency:
            continue
        ordered = sorted(by_phon.items(), key=lambda kv: -kv[1][0])
        named = [
            (pos, lexique_to_ipa(phon, symbols=symbols)) for phon, (_, pos) in ordered
        ]
        poses = [pos for pos, _ in named]
        table[word] = {
            (pos if poses.count(pos) == 1 else f"{pos}:{ipa}"): ipa
            for pos, ipa in named
        }
    return table


def lexique_heteronyms(path: str | os.PathLike, **kwargs) -> dict[str, dict[str, str]]:
    """The heteronyms of a Lexique 3 file (tab-separated, e.g. ``Lexique383.tsv``).

    Keyword arguments go to ``heteronyms_from_rows`` (``min_frequency``, ...).
    """
    with open(path, encoding="utf-8", newline="") as f:
        return heteronyms_from_rows(csv.DictReader(f, delimiter="\t"), **kwargs)
