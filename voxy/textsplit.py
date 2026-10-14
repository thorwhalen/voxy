"""
Split long text (a chapter) into chunks a speech model can say in one breath.

Local cloning models drift, rush or hallucinate past roughly 20-40 seconds of
speech, so long text is spoken chunk by chunk and the audio joined with
controlled pauses. Paragraphs are kept apart (they get a longer pause); inside a
paragraph, whole sentences are packed into chunks of at most ``max_chars``, and
only a sentence longer than that is cut, at a clause break if it has one.

>>> split_text("One. Two!\\n\\nThree is a longer sentence, with a clause.", max_chars=20)
[['One. Two!'], ['Three is a longer', 'sentence,', 'with a clause.']]
"""

import re

DFLT_MAX_CHARS = 300
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?…][\"'”’)\]])\s+|(?<=[.!?…])\s+")
#: Words ending in a period that don't end a sentence ("Mr. Darcy").
ABBREVIATIONS = frozenset(
    "mr mrs ms dr st jr sr prof rev gen col capt lt sgt mt vs etc no vol ch fig".split()
)
_CLAUSE_BREAK = re.compile(r"(?<=[,;:—–])\s+")


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    """Greedily join ``pieces`` with spaces into strings of at most ``max_chars``."""
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Cut a too-long sentence at clause breaks, then at word boundaries."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces = []
    for clause in _CLAUSE_BREAK.split(sentence):
        pieces += [clause] if len(clause) <= max_chars else clause.split()
    return _pack(pieces, max_chars)


def split_sentences(paragraph: str) -> list[str]:
    """The sentences of ``paragraph`` (whitespace normalised).

    >>> split_sentences('He said "Go." Then   Mr. Darcy left.')
    ['He said "Go."', 'Then Mr. Darcy left.']
    """
    text = " ".join(paragraph.split())
    sentences: list[str] = []
    for piece in filter(None, _SENTENCE_END.split(text)):
        last_word = sentences[-1].rsplit(" ", 1)[-1] if sentences else ""
        if last_word.endswith(".") and last_word[:-1].lower() in ABBREVIATIONS:
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    return sentences


def split_text(text: str, *, max_chars: int = DFLT_MAX_CHARS) -> list[list[str]]:
    """Paragraphs of chunks: ``[[chunk, ...], ...]``, each chunk at most ``max_chars``."""
    paragraphs = [p for p in _PARAGRAPH_BREAK.split(text.strip()) if p.strip()]
    return [
        _pack(
            [part for s in split_sentences(p) for part in _split_long(s, max_chars)],
            max_chars,
        )
        for p in paragraphs
    ]
