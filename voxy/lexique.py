"""
French heteronyms derived from Lexique, to merge under voxy's hand-made table.

Lexique (New, Pallier et al., http://www.lexique.org) lists about 140 000 French
word forms with their pronunciation, lemma, part of speech and frequency. Its
homographs that are not homophones (*est* /ɛ/ or /ɛst/, *couvent* /kuv/ or /kuvɑ̃/,
*président*...) are words a speech model can misread. voxy ships a short hand-made
table (``voxy.pronounce.heteronym_readings``) and no Lexique data, so derive more
from your own copy and let the hand-made readings win:

    >>> from voxy.lexique import lexique_heteronyms                     # doctest: +SKIP
    >>> from voxy.pronounce import heteronym_readings                   # doctest: +SKIP
    >>> derived = lexique_heteronyms("Lexique383.tsv", min_frequency=1) # doctest: +SKIP
    >>> table = {**derived, **heteronym_readings("fr")}                 # doctest: +SKIP
    >>> derived["couvent"]                                              # doctest: +SKIP
    {'noun': 'kuvɑ̃', 'verb': 'kuv'}

The derived table is not complete. Lexique stores one pronunciation per spelling,
part of speech and lemma, and no liaison or variant readings, so heteronyms that
differ only there are missing from it: *tous*, *os*, *six*, *dix*, *christ*,
*lacs*, *fait*. It extends the hand-made table; it does not replace it.

The result has the shape ``voxy.pronounce`` takes (``{word: {reading name: IPA}}``,
IPA in the symbols of its phoneme model). Pronunciations that differ only by
Lexique's transcription noise are one reading and its variants (``verb~e``); a
reading is named by its part of speech, with the names of
``voxy.speechtext.FRENCH_READING_NAMES``, or by its lemma when two readings share a
part of speech:

>>> rows = [
...     {"ortho": "est", "phon": "E", "lemme": "être", "cgram": "AUX", "freqfilms2": "100", "freqlivres": "90"},
...     {"ortho": "est", "phon": "e", "lemme": "être", "cgram": "VER", "freqfilms2": "9", "freqlivres": "9"},
...     {"ortho": "est", "phon": "Est", "lemme": "est", "cgram": "NOM", "freqfilms2": "20", "freqlivres": "30"},
...     {"ortho": "chat", "phon": "Sa", "lemme": "chat", "cgram": "NOM", "freqfilms2": "40", "freqlivres": "30"},
... ]
>>> heteronyms_from_rows(rows)
{'est': {'verb': 'ɛ', 'verb~e': 'e', 'noun': 'ɛst'}}
"""

import csv
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path

from voxy.pronounce import VARIANT_SEPARATOR
from voxy.speechtext import FRENCH_READING_NAMES

#: Lexique's phonetic code -> IPA in the vocabulary of voxy's phoneme model
#: (``facebook/wav2vec2-lv-60-espeak-cv-ft``); Lexique's other letters are that IPA.
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
    # /ɥ/ (*huit*, *lui*) is not in the model's vocabulary, which could never hear it;
    # /y/, the vowel it is the glide of, is the nearest symbol that is.
    "8": "y",
    "R": "ʁ",
    "S": "ʃ",
    "Z": "ʒ",
    "N": "ɲ",
    "G": "ŋ",
    "g": "ɡ",  # IPA's script g (U+0261), the model's symbol, not the ASCII letter
}
#: Lexique letters it writes inconsistently -> the letter they are compared as: two
#: pronunciations equal after this are one reading (open/closed vowels, schwa).
TRANSCRIPTION_NOISE = {"E": "e", "O": "o", "9": "2", "°": "", "8": "y"}
#: Lexique's part of speech (``cgram``, whole or before its ``:``) -> universal POS.
LEXIQUE_TO_UPOS = {
    "NOM": "NOUN",
    "VER": "VERB",
    "AUX": "AUX",
    "ADJ": "ADJ",
    "ADJ:num": "NUM",
    "ADJ:dem": "DET",
    "ADJ:ind": "DET",
    "ADJ:int": "DET",
    "ADJ:pos": "DET",
    "ADV": "ADV",
    "PRE": "ADP",
    "CON": "CCONJ",
    "PRO": "PRON",
    "ART": "DET",
    "ONO": "INTJ",
}
#: Lexique's part of speech -> a reading name (``voxy.speechtext.FRENCH_READING_NAMES``).
LEXIQUE_POS_NAMES = {
    tag: FRENCH_READING_NAMES[upos] for tag, upos in LEXIQUE_TO_UPOS.items()
}
#: A pronunciation with a verb row is named ``verb`` when its heaviest part of speech
#: is one of these: Lexique tags some verb forms as nouns or adjectives too (*violent*,
#: *insolent*), while an adverb (*plus*) is genuinely heavier than its verb rows.
VERB_NAME = FRENCH_READING_NAMES["VERB"]
VERB_OUTRANKS = frozenset({FRENCH_READING_NAMES["NOUN"], FRENCH_READING_NAMES["ADJ"]})
#: The name of a reading whose part of speech Lexique leaves empty.
UNTAGGED_NAME = "word"
#: Columns read from a Lexique 3 file.
ORTHO, PHON, LEMMA, CGRAM = "ortho", "phon", "lemme", "cgram"
FREQUENCY_COLUMNS = ("freqfilms2", "freqlivres")
REQUIRED_COLUMNS = (ORTHO, PHON, LEMMA, CGRAM, *FREQUENCY_COLUMNS)


def lexique_to_ipa(phon: str, *, symbols: Mapping[str, str] = LEXIQUE_TO_IPA) -> str:
    """A Lexique pronunciation in IPA.

    >>> lexique_to_ipa("kuv@"), lexique_to_ipa("pRezid@"), lexique_to_ipa("Est")
    ('kuvɑ̃', 'pʁezidɑ̃', 'ɛst')
    """
    return "".join(symbols.get(c, c) for c in phon)


def phon_key(phon: str, *, noise: Mapping[str, str] = TRANSCRIPTION_NOISE) -> str:
    """What a Lexique pronunciation is compared as, transcription noise removed.

    >>> phon_key("vjOl°"), phon_key("vjol")
    ('vjol', 'vjol')
    """
    return "".join(noise.get(c, c) for c in phon)


def _frequency(row: Mapping) -> float:
    """Occurrences per million (films and books, the larger)."""
    return max(float(row[c] or 0) for c in FREQUENCY_COLUMNS)


def _pos_name(cgram: str, names: Mapping[str, str]) -> str:
    tag = cgram.split(":")[0]
    return names.get(cgram) or names.get(tag) or tag.lower() or UNTAGGED_NAME


def _heaviest(weighted: Iterable[tuple[str, float]]) -> list[str]:
    """Keys by their summed weight, heaviest first (ties in first-seen order)."""
    totals: dict[str, float] = defaultdict(float)
    for key, weight in weighted:
        totals[key] += weight
    return sorted(totals, key=lambda k: -totals[k])


def _reading(rows: list[Mapping], pos_names: Mapping[str, str]) -> dict:
    """One reading of a spelling, from the rows of its pronunciations: its weight (the
    sum of the rows' frequencies), part of speech, lemma, and pronunciations (heaviest
    first)."""
    weighted = [(row, _frequency(row)) for row in rows]
    poses = _heaviest((_pos_name(r[CGRAM], pos_names), f) for r, f in weighted)
    return {
        "weight": sum(f for _, f in weighted),
        "pos": VERB_NAME
        if VERB_NAME in poses and poses[0] in VERB_OUTRANKS
        else poses[0],
        "lemma": _heaviest((r[LEMMA], f) for r, f in weighted)[0],
        "phons": _heaviest((r[PHON], f) for r, f in weighted),
    }


def _reading_names(readings: list[dict], *, symbols: Mapping[str, str]) -> list[str]:
    """A name per reading: its part of speech; readings sharing one are named by lemma,
    or ``<pos>:<IPA>`` when their lemmas coincide too."""
    poses = [r["pos"] for r in readings]
    names = []
    for r in readings:
        sharing = [s for s in readings if s["pos"] == r["pos"]]
        lemmas = [s["lemma"] for s in sharing]
        if len(sharing) == 1:
            names.append(r["pos"])
        elif r["lemma"] and lemmas.count(r["lemma"]) == 1 and r["lemma"] not in poses:
            names.append(r["lemma"])
        else:
            names.append(f"{r['pos']}:{lexique_to_ipa(r['phons'][0], symbols=symbols)}")
    return names


def heteronyms_from_rows(
    rows: Iterable[Mapping],
    *,
    min_frequency: float = 0.0,
    pos_names: Mapping[str, str] = LEXIQUE_POS_NAMES,
    symbols: Mapping[str, str] = LEXIQUE_TO_IPA,
    noise: Mapping[str, str] = TRANSCRIPTION_NOISE,
) -> dict[str, dict[str, str]]:
    """``{word: {reading name: IPA}}`` for the spellings Lexique reads in more than one way.

    Pronunciations equal under :func:`phon_key` are one reading, its heaviest
    pronunciation the reading and the others ``<name>~<IPA>`` variants. Readings come
    heaviest first (weight: the sum of their rows' frequencies) and are named by their
    heaviest part of speech (``verb`` if any row is a verb); readings that share one
    are named by lemma, or ``<pos>:<IPA>`` when their lemmas coincide too. Words whose
    heaviest reading is under ``min_frequency`` (per million) are left out.
    """
    by_word: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row[ORTHO] and row[PHON]:
            by_word[row[ORTHO]][phon_key(row[PHON], noise=noise)].append(row)
    table = {}
    for word, by_key in by_word.items():
        readings = sorted(
            (_reading(group, pos_names) for group in by_key.values()),
            key=lambda r: -r["weight"],
        )
        if len(readings) < 2 or readings[0]["weight"] < min_frequency:
            continue
        names = _reading_names(readings, symbols=symbols)
        table[word] = {
            (name if i == 0 else f"{name}{VARIANT_SEPARATOR}{ipa}"): ipa
            for name, r in zip(names, readings)
            for i, ipa in enumerate(
                lexique_to_ipa(p, symbols=symbols) for p in r["phons"]
            )
        }
    return table


def lexique_heteronyms(
    path: str | os.PathLike,
    *,
    min_frequency: float = 0.0,
    pos_names: Mapping[str, str] = LEXIQUE_POS_NAMES,
    symbols: Mapping[str, str] = LEXIQUE_TO_IPA,
    noise: Mapping[str, str] = TRANSCRIPTION_NOISE,
) -> dict[str, dict[str, str]]:
    """The heteronyms of a Lexique 3 file (tab-separated, e.g. ``Lexique383.tsv``), as
    :func:`heteronyms_from_rows` derives them.

    Raises ``ValueError`` if the file lacks one of :data:`REQUIRED_COLUMNS`.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(
                f"{os.fspath(path)!r} is not a Lexique 3 file: no column {missing}"
            )
        return heteronyms_from_rows(
            reader,
            min_frequency=min_frequency,
            pos_names=pos_names,
            symbols=symbols,
            noise=noise,
        )


#: Folder (under voxy's cache) of derived tables, one JSON file per file and arguments.
LEXIQUE_CACHE_KIND = "lexique"
#: Hex characters of a cache key (64 bits: no collision among a user's tables).
CACHE_KEY_CHARS = 16


def lexique_cache(*, cache_dir: str | os.PathLike | None = None) -> MutableMapping:
    """Derived tables as JSON files, ``<key>.json`` under ``cache_dir`` (default:
    ``voxy_cache_dir("lexique")``): derived from your Lexique copy, so never in a repo."""
    from dol import JsonFiles, mk_dirs_if_missing

    from voxy.stores import voxy_cache_dir

    folder = Path(cache_dir) if cache_dir else voxy_cache_dir(LEXIQUE_CACHE_KIND)
    return mk_dirs_if_missing(JsonFiles(str(folder) + os.sep))


def _cache_key(path: str | os.PathLike, arguments: Mapping) -> str:
    """The file's identity (real path, size, modification time) and the derivation's
    arguments, hashed: a changed file or changed arguments make a new key."""
    stat = os.stat(path)
    identity = {
        "path": os.path.realpath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "arguments": arguments,
    }
    text = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:CACHE_KEY_CHARS] + ".json"


def cached_lexique_heteronyms(
    path: str | os.PathLike,
    *,
    cache: MutableMapping | None = None,
    min_frequency: float = 0.0,
    pos_names: Mapping[str, str] = LEXIQUE_POS_NAMES,
    symbols: Mapping[str, str] = LEXIQUE_TO_IPA,
    noise: Mapping[str, str] = TRANSCRIPTION_NOISE,
) -> dict[str, dict[str, str]]:
    """:func:`lexique_heteronyms`, derived once and kept in ``cache`` (default
    :func:`lexique_cache`), rebuilt when the file or the arguments change.

    >>> import tempfile, pathlib
    >>> tsv = pathlib.Path(tempfile.mkdtemp()) / "Lexique383.tsv"
    >>> _ = tsv.write_text("ortho\\tphon\\tlemme\\tcgram\\tfreqfilms2\\tfreqlivres\\n"
    ...                    "est\\te\\têtre\\tVER\\t9\\t9\\nest\\tEst\\test\\tNOM\\t1\\t1\\n")
    >>> cache = {}
    >>> cached_lexique_heteronyms(tsv, cache=cache) == lexique_heteronyms(tsv), len(cache)
    (True, 1)
    """
    arguments = {
        "min_frequency": min_frequency,
        "pos_names": dict(pos_names),
        "symbols": dict(symbols),
        "noise": dict(noise),
    }
    cache = lexique_cache() if cache is None else cache
    key = _cache_key(path, arguments)
    if key not in cache:
        cache[key] = lexique_heteronyms(path, **arguments)
    return cache[key]
