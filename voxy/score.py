"""
Scores for singing: the notes of a melody (pitch, timing, word) in the form voxy's
singing backends take, read from MIDI, MusicXML or plain code.

A ``Note`` is one sung note in absolute seconds: its MIDI pitch, the word sung on it,
and whether the word starts on it (``starts_word``) or continues from the note
before (a melisma, or the later syllables of a word). Rests are the gaps between
notes. A singing backend turns a list of notes into its own input format
(``voxy.soulx.soulx_metadata`` for SoulX-Singer).

    >>> from voxy.score import read_score                               # doctest: +SKIP
    >>> notes = read_score("melody.mid", "Light a can-dle in the win-dow")  # doctest: +SKIP
    >>> notes = read_score("song.musicxml")         # the score's own lyrics  # doctest: +SKIP

Reading a score is audiate's job: ``read_score`` takes the sung line from
``audiate.sung_line`` (its module docstring states the reading rules: which line,
which verse, repeats, melismas, karaoke files) and makes ``Note``s of it. When
audiate is not sure (several candidate lines, a chord, a lyric on no note), it
raises and names the choices (``track=``, ``channel=``, ``part=``, ``verse=``);
for its other options (``voice=``, ``chords=``, ...) read the line with
``audiate.sung_line`` and pass the notes it returns. audiate is in the
``voxy[singing]`` extra and imported only when a score is read or lyrics attached.

Lyrics given by hand are one whitespace-separated token per sung note: ``can-dle``
spreads one word over two notes, a bare ``-`` (or ``_``, ``~``) holds the previous
word one more note (``audiate.sung.lyric_tokens``).

>>> notes = notes_from_pitches([(64, 1), (67, 1), (60, 2)], bpm=120)
>>> [(n.start, n.dur, n.pitch) for n in notes]
[(0.0, 0.5, 64), (0.5, 0.5, 67), (1.0, 1.0, 60)]
"""

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from voxy.base import optional_import

#: The module that reads scores (``sung_line``) and attaches lyrics (``attach_lyrics``).
SCORE_READER = "audiate.sung"
#: The word given to sung notes that have none (a score without lyrics on every note).
DFLT_FILLER_WORD = "la"
DFLT_BPM = 100


@dataclass(frozen=True)
class Note:
    """One sung note, in absolute seconds.

    Args:
        start: Onset (seconds from the start of the score).
        dur: Duration in seconds.
        pitch: MIDI note number (60 is middle C).
        word: The word sung on it ('' until lyrics are attached).
        starts_word: True if the word starts here; False if it continues the
            previous note's word (melisma, later syllable).
    """

    start: float
    dur: float
    pitch: int
    word: str = ""
    starts_word: bool = True

    @property
    def end(self) -> float:
        """Where the note stops (seconds)."""
        return self.start + self.dur


def notes_from_pitches(
    pitches: Iterable[tuple[int | None, float]],
    *,
    bpm: float = DFLT_BPM,
    start: float = 0.0,
) -> list[Note]:
    """Notes from ``(midi_pitch, beats)`` pairs, back to back; a pitch of 0 or None is a rest.

    >>> [(n.start, n.pitch) for n in notes_from_pitches([(60, 1), (0, 1), (62, 1)], bpm=60)]
    [(0.0, 60), (2.0, 62)]
    """
    beat = 60.0 / bpm
    notes, t = [], float(start)
    for pitch, beats in pitches:
        dur = beats * beat
        if pitch:
            notes.append(Note(round(t, 6), round(dur, 6), int(pitch)))
        t += dur
    return notes


def to_note(note) -> Note:
    """A ``Note`` from any note with ``start``, ``dur``, ``pitch`` (and optionally
    ``word``, ``starts_word``): an ``audiate.SungNote``, or a ``Note`` itself.

    >>> to_note(Note(0.0, 0.5, 60, "la"))
    Note(start=0.0, dur=0.5, pitch=60, word='la', starts_word=True)
    """
    if isinstance(note, Note):
        return note
    return Note(
        float(note.start),
        float(note.dur),
        int(note.pitch),
        getattr(note, "word", ""),
        getattr(note, "starts_word", True),
    )


def _is_notes(score) -> bool:
    return (
        isinstance(score, Sequence)
        and not isinstance(score, (str, bytes, bytearray))
        and all(hasattr(n, "pitch") and hasattr(n, "start") for n in score)
    )


def _with_lyrics(notes: Sequence[Note], lyrics: str) -> list[Note]:
    """``notes`` with the words of the lyric line ``lyrics``, through audiate's
    ``attach_lyrics`` (one token per note)."""
    reader = optional_import(SCORE_READER)
    sung = [reader.SungNote(n.start, n.dur, n.pitch) for n in notes]
    return [to_note(n) for n in reader.attach_lyrics(sung, lyrics)]


def read_score(
    score,
    lyrics: str | None = None,
    *,
    filler: str = DFLT_FILLER_WORD,
    track: int | None = None,
    channel: int | None = None,
    part: int | str | None = None,
    verse: int | str | None = None,
) -> list[Note]:
    """The notes of ``score``, with words: from ``lyrics`` if given, else the score's own.

    Args:
        score: A MIDI file (``.mid``, ``.midi``, ``.kar``), a MusicXML file (or
            anything music21 parses), its bytes, a music21 stream (all read with
            ``audiate.sung_line``), or a sequence of notes (``Note``, or
            ``audiate.SungNote`` as ``audiate.sung_line`` returns them).
        lyrics: A lyric line (one token per note, see ``audiate.sung.lyric_tokens``);
            replaces the score's own words, which are then not read.
        filler: The word for sung notes that still have none and continue no word
            (a note after a rest, in a score with lyrics on only some notes); a
            score with no words at all, and no ``lyrics``, is an error.
        track, channel: The MIDI line to sing (``track``: index in the file;
            ``channel``: 1 to 16).
        part, verse: The music21 part (an index, or a name or id) and the lyric
            verse to sing.

    Which line and verse are sung by default, and when the reading raises
    instead of guessing, is audiate's to say: see ``audiate.sung``.

    >>> [n.word for n in read_score([Note(0, 1, 60, "do"), Note(1, 1, 62, "re")])]
    ['do', 're']
    """
    if _is_notes(score):
        given = {"track": track, "channel": channel, "part": part, "verse": verse}
        if any(v is not None for v in given.values()):
            names = ", ".join(k for k, v in given.items() if v is not None)
            raise ValueError(f"{names}: pick a line of a score, not of notes")
        notes = sorted((to_note(n) for n in score), key=lambda n: n.start)
    else:
        if isinstance(score, (str, os.PathLike)) and not Path(score).exists():
            raise FileNotFoundError(f"No score at {score}")
        reader = optional_import(SCORE_READER)
        line = reader.sung_line(
            score,
            track=track,
            channel=channel,
            part=part,
            verse=verse,
            with_lyrics=lyrics is None,
        )
        notes = [to_note(n) for n in line]
    notes = [n for n in notes if n.dur > 0]
    if not notes:
        raise ValueError("The score has no notes")
    if lyrics is not None:
        return _with_lyrics(notes, lyrics)
    if not any(n.word for n in notes):
        raise ValueError("The score has no lyrics: pass lyrics='...'")
    return [n if n.word else replace(n, word=filler, starts_word=True) for n in notes]
