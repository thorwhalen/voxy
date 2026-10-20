"""Display text vs speech text, and spotting words a model may misread.

A text a person reads and the text a model should be given are not always the same.
Measured case (French, a Qwen3-TTS clone): a verse line starting with a capitalised
verb, "Est parti" after a line "Mon frere", was read as the noun /est/ ("l'Est") in 4
of 4 takes; lowercased ("est parti") it was read correctly in 4 of 4. The capital, not
the line break, is the trigger: line-initial "Est" tokenises like the noun in "l'Est".
ASR checks cannot catch this (Whisper hears "est" either way), so the fix is upstream:

1. :func:`prepare_speech_text` turns display text into speech text: it lowercases a
   *verse-initial* capital that does not start a sentence (keeping deliberate capitals:
   words also capitalised mid-line, or named in ``keep``), then applies a per-language
   respelling table and proper-name aliases, and returns every change it made.
2. :func:`flag_risks` lists the words of a text a model may misread (heteronyms you
   supply, e.g. from Lexique; more detectors plug in through ``extra``), saying which
   ones sit where the capital trigger bites. :func:`pos_detector` is one such
   detector: it tags the text and, for heteronyms whose part of speech picks the
   reading (French "est": verb /ɛ/, noun /ɛst/), names the ``expected`` reading.

Engines differ in what they accept: for several local models text is the only lever
(they take no phonemes or SSML), so this is a text-level step; it plugs into
``LocalCloneSpeechModel(text_prep=...)``. Verifying the audio itself (phoneme
recognition of the flagged words) is :mod:`voxy.pronounce`.

>>> text = "Mon frere\\nEst parti\\nLoin d'ici."
>>> prepare_speech_text(text).speech
"Mon frere\\nest parti\\nloin d'ici."
>>> prepare_speech_text(text, keep={"Est", "Loin"}).speech == text
True
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field

#: What closes a sentence: the next line's capital then starts a new one.
SENTENCE_END = re.compile(r"[.!?…]['\"”’»)\]]*\s*$")
_WORD = re.compile(r"[^\W\d_][\w'’-]*", re.UNICODE)
_LEADING_WORD = re.compile(r"^(\s*[\"'“‘«(\[—–-]*\s*)([^\W\d_][\w'’-]*)", re.UNICODE)


@dataclass(frozen=True)
class Change:
    """One edit :func:`prepare_speech_text` made (``line`` counts from 1)."""

    line: int
    original: str
    replacement: str
    reason: str


@dataclass(frozen=True)
class SpeechText:
    """The text to show (``display``), the text to speak (``speech``) and what changed."""

    display: str
    speech: str
    changes: tuple[Change, ...] = ()

    def __str__(self) -> str:
        return self.speech


@dataclass(frozen=True)
class Risk:
    """A word a model may misread: where it is, why it is flagged, its known readings."""

    line: int
    word: str
    reason: str
    readings: tuple[str, ...] = ()
    line_initial_capital: bool = False
    #: The reading the context picks (a name among ``readings``), when a detector knows it.
    expected: str | None = None


def _starts_sentence(previous_line: str | None) -> bool:
    """Whether a line begins a sentence: it opens a paragraph, or the line before ended one."""
    return previous_line is None or bool(SENTENCE_END.search(previous_line))


def _mid_line_capitals(lines: list[str]) -> set[str]:
    """Words that appear capitalised somewhere other than the start of a line (deliberate)."""
    found = set()
    for line in lines:
        words = _WORD.findall(line)
        for word in words[1:]:
            if word[0].isupper():
                found.add(word)
    return found


def _replace_words(
    line: str, table: Mapping[str, str]
) -> tuple[str, list[tuple[str, str]]]:
    """Replace whole words of ``line`` found in ``table`` (case-insensitive); list the swaps."""
    lowered = {key.casefold(): value for key, value in table.items()}
    swaps: list[tuple[str, str]] = []

    def swap(match: re.Match) -> str:
        word = match.group(0)
        value = lowered.get(word.casefold())
        if value is None or value == word:
            return word
        swaps.append((word, value))
        return value

    return _WORD.sub(swap, line), swaps


def prepare_speech_text(
    text: str,
    *,
    lowercase_verse_initial: bool = True,
    keep: Collection[str] = (),
    respellings: Mapping[str, str] | None = None,
    aliases: Mapping[str, str] | None = None,
    transforms: Iterable[Callable[[str], str]] = (),
) -> SpeechText:
    """Display text in, speech text out, with a record of every change.

    Args:
        text: The text as a reader sees it (lines may be verse lines).
        lowercase_verse_initial: Lowercase a line-initial capital that does not start
            a sentence (the first line of a paragraph, or a line after one ending in
            ``. ! ? …``, keeps its capital).
        keep: Words whose capital is deliberate and stays (compared exactly). Words
            also capitalised elsewhere in a line are kept automatically.
        respellings: Word -> same-sound spelling, whole words, case-insensitive (e.g.
            a language's ambiguous forms written unambiguously).
        aliases: Proper name -> pronounceable spelling, whole words, case-insensitive.
        transforms: Extra ``text -> text`` steps applied last, over the whole text.

    Returns:
        A :class:`SpeechText`.
    """
    lines = text.split("\n")
    deliberate = set(keep) | _mid_line_capitals(lines)
    changes: list[Change] = []
    out: list[str] = []
    previous: str | None = None
    for number, line in enumerate(lines, 1):
        new = line
        if lowercase_verse_initial and line.strip():
            lead = _LEADING_WORD.match(line)
            if lead and lead.group(2)[0].isupper() and lead.group(2) not in deliberate:
                if not _starts_sentence(previous):
                    word = lead.group(2)
                    lowered = word[0].lower() + word[1:]
                    new = lead.group(1) + lowered + line[lead.end() :]
                    changes.append(
                        Change(
                            number,
                            word,
                            lowered,
                            "verse-initial capital, not a sentence start",
                        )
                    )
        for table, reason in ((aliases, "alias"), (respellings, "respelling")):
            if table:
                new, swaps = _replace_words(new, table)
                changes += [Change(number, a, b, reason) for a, b in swaps]
        out.append(new)
        previous = line if line.strip() else None  # a blank line opens a paragraph
    speech = "\n".join(out)
    for transform in transforms:
        speech = transform(speech)
    return SpeechText(display=text, speech=speech, changes=tuple(changes))


def flag_risks(
    text: str,
    *,
    heteronyms: Mapping[str, Iterable[str]] | Collection[str] = (),
    extra: Iterable[Callable[[str], Iterable[Risk]]] = (),
) -> list[Risk]:
    """Words of ``text`` a model may misread, in text order.

    Args:
        text: The text that will be spoken (run it through :func:`prepare_speech_text`
            first to see what is left).
        heteronyms: Words with more than one reading, as a collection or as
            ``{word: readings}`` (e.g. from Lexique). Matched case-insensitively.
        extra: More detectors, each ``text -> risks`` (a part-of-speech rule, an LLM
            proposer). Their risks are merged in by line.

    A heteronym that is capitalised at a line start where no sentence begins is flagged
    with ``line_initial_capital=True``: that is where the capital trigger bites.
    """
    table = (
        {k.casefold(): tuple(v) for k, v in heteronyms.items()}
        if isinstance(heteronyms, Mapping)
        else {w.casefold(): () for w in heteronyms}
    )
    risks: list[Risk] = []
    previous: str | None = None
    for number, line in enumerate(text.split("\n"), 1):
        lead = _LEADING_WORD.match(line)
        for match in _WORD.finditer(line):
            word = match.group(0)
            if word.casefold() not in table:
                continue
            first = bool(lead) and match.start() == lead.start(2)
            capital_trigger = (
                first and word[0].isupper() and not _starts_sentence(previous)
            )
            risks.append(
                Risk(
                    number,
                    word,
                    "a capital at the start of a verse line that does not start a sentence"
                    if capital_trigger
                    else "a word with more than one reading",
                    table[word.casefold()],
                    capital_trigger,
                )
            )
        previous = line if line.strip() else None
    for detector in extra:
        risks += list(detector(text))
    return sorted(risks, key=lambda r: r.line)


# Part-of-speech detector ------------------------------------------------------------

#: A tagger: ``text -> (char offset, token, universal POS tag)`` for every token.
Tagger = Callable[[str], Iterable[tuple[int, str, str]]]

#: The spaCy model :func:`spacy_tagger` loads by default (French).
DFLT_SPACY_MODEL = "fr_core_news_sm"

#: French heteronyms whose part of speech picks the reading: ``{word: {UPOS: reading}}``.
#: The reading names are the keys of the IPA table given to ``voxy.pronounce``.
FRENCH_POS_HETERONYMS: Mapping[str, Mapping[str, str]] = {
    "est": {
        "AUX": "verb",
        "VERB": "verb",
        "NOUN": "noun",
        "PROPN": "noun",
        "ADJ": "noun",
    },
}


def spacy_tagger(model: str = DFLT_SPACY_MODEL) -> Tagger:
    """A :data:`Tagger` over a spaCy pipeline, loaded on first use.

    Needs ``pip install 'voxy[pos]'`` and the model itself
    (``python -m spacy download fr_core_news_sm``).
    """
    nlp = None

    def tag(text: str) -> Iterable[tuple[int, str, str]]:
        nonlocal nlp
        if nlp is None:
            try:
                import spacy
            except ImportError as error:
                raise ImportError(
                    "the part-of-speech detector needs spaCy: pip install 'voxy[pos]'"
                ) from error
            try:
                nlp = spacy.load(model)
            except OSError as error:
                raise OSError(
                    f"spaCy model {model!r} is not installed: "
                    f"python -m spacy download {model}"
                ) from error
        return [(t.idx, t.text, t.pos_) for t in nlp(text)]

    return tag


def pos_detector(
    rules: Mapping[str, Mapping[str, str]] = FRENCH_POS_HETERONYMS,
    *,
    tagger: Tagger | None = None,
) -> Callable[[str], list[Risk]]:
    """A :func:`flag_risks` ``extra`` detector: heteronyms whose part of speech picks the reading.

    Run it on the *speech* text (:func:`prepare_speech_text`): a stray verse-initial
    capital fools the tagger the same way it fools the speech model.

    Args:
        rules: ``{word: {UPOS tag: reading name}}``, words matched case-insensitively
            (default :data:`FRENCH_POS_HETERONYMS`).
        tagger: ``text -> (offset, token, UPOS)``; default :func:`spacy_tagger`
            (French), loaded on first use.

    >>> fake = lambda text: [(0, "Il", "PRON"), (3, "est", "AUX"), (7, "parti", "VERB")]
    >>> [(r.word, r.expected) for r in pos_detector(tagger=fake)("Il est parti")]
    [('est', 'verb')]
    """
    table = {w.casefold(): dict(by_tag) for w, by_tag in rules.items()}
    tag = tagger

    def detect(text: str) -> list[Risk]:
        nonlocal tag
        if tag is None:
            tag = spacy_tagger()
        risks = []
        for offset, token, pos in tag(text):
            by_tag = table.get(token.casefold())
            if not by_tag:
                continue
            reading = by_tag.get(pos)
            risks.append(
                Risk(
                    text.count("\n", 0, offset) + 1,
                    token,
                    f"tagged {pos}: its part of speech picks the reading"
                    if reading
                    else f"tagged {pos}, which picks no known reading",
                    tuple(dict.fromkeys(by_tag.values())),
                    expected=reading,
                )
            )
        return risks

    return detect
