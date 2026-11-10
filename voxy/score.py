"""
Scores for singing: the notes of a melody (pitch, timing, word) from MIDI, MusicXML
or plain code, with lyrics attached.

A ``Note`` is one sung note in absolute seconds: its MIDI pitch, the word sung on it,
and whether the word starts on it (``starts_word``) or continues from the note
before (a melisma, or the later syllables of a word). Rests are the gaps between
notes. A singing backend turns a list of notes into its own input format
(``voxy.soulx.soulx_metadata`` for SoulX-Singer).

    >>> from voxy.score import read_score                               # doctest: +SKIP
    >>> notes = read_score("melody.mid", "Light a can-dle in the win-dow")  # doctest: +SKIP
    >>> notes = read_score("song.musicxml")         # the score's own lyrics  # doctest: +SKIP

Lyrics are a line of whitespace-separated tokens, one per sung note:
``can-dle`` spreads one word over two notes, a bare ``-`` (or ``_``) holds the
previous word one more note.

>>> notes = notes_from_pitches([(64, 1), (67, 1), (60, 2)], bpm=120)
>>> [(n.start, n.dur, n.pitch) for n in notes]
[(0.0, 0.5, 64), (0.5, 0.5, 67), (1.0, 1.0, 60)]
>>> [(n.word, n.starts_word) for n in attach_lyrics(notes, "hel-lo you")]
[('hello', True), ('hello', False), ('you', True)]

MIDI is read with ``mido`` (every tempo change honoured, lyric meta-events kept),
MusicXML (and anything else music21 parses) with ``music21``: both are in the
``voxy[singing]`` extra and imported only when a score of that kind is read.
"""

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from voxy.base import optional_import

MIDI_SUFFIXES = (".mid", ".midi")
#: Tokens of a lyric line that hold the previous word one more note.
HOLD_TOKENS = frozenset({"-", "_"})
#: The word given to sung notes that have none (a score without lyrics on every note).
DFLT_FILLER_WORD = "la"
DFLT_BPM = 100
#: MIDI tempo before any ``set_tempo`` event: 500 000 microseconds a beat (120 BPM).
DFLT_MIDI_TEMPO = 500_000


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


# --------------------------------------------------------------------------- readers


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


def notes_from_midi(path: str | os.PathLike, *, track: int | None = None) -> list[Note]:
    """The monophonic line of a MIDI file, with its lyric meta-events as words.

    Times follow the file's tempo map. Overlapping notes are cut at the next onset,
    so the line is strictly monophonic. ``track`` limits it to one track.
    """
    mido = optional_import("mido")
    mid = mido.MidiFile(str(path))
    tempo_map = _midi_tempo_map(mido.merge_tracks(mid.tracks))
    raw, lyrics = [], {}
    for trk in [mid.tracks[track]] if track is not None else mid.tracks:
        tick, active = 0, {}
        for msg in trk:
            tick += msg.time
            if msg.type == "lyrics" and msg.text.strip():
                lyrics[tick] = msg.text.strip()
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = tick
            elif msg.type in ("note_off", "note_on"):
                begin = active.pop((msg.channel, msg.note), None)
                if begin is not None and tick > begin:
                    raw.append((begin, tick, msg.note))
    if not raw:
        raise ValueError(f"No notes in {path}")
    raw.sort()

    def seconds(ticks: int) -> float:
        return _midi_ticks_to_seconds(ticks, tempo_map, mid.ticks_per_beat)

    notes = []
    for i, (begin, end, pitch) in enumerate(raw):
        if i + 1 < len(raw):
            end = min(end, raw[i + 1][0])
        if end > begin:
            onset = seconds(begin)
            notes.append(
                Note(onset, seconds(end) - onset, pitch, lyrics.get(begin, ""))
            )
    return notes


def _midi_tempo_map(merged) -> list[tuple[int, int]]:
    """``[(tick, microseconds_per_beat), ...]`` from a merged track, starting at tick 0."""
    changes, tick = {0: DFLT_MIDI_TEMPO}, 0
    for msg in merged:
        tick += msg.time
        if msg.type == "set_tempo":
            changes[tick] = msg.tempo
    return sorted(changes.items())


def _midi_ticks_to_seconds(ticks: int, tempo_map, ticks_per_beat: int) -> float:
    """Seconds at ``ticks``, through every tempo change before it.

    >>> _midi_ticks_to_seconds(960, [(0, 500_000), (480, 1_000_000)], 480)
    1.5
    """
    secs, prev_tick, prev_tempo = 0.0, 0, tempo_map[0][1]
    for tick, tempo in tempo_map[1:]:
        if tick >= ticks:
            break
        secs += (tick - prev_tick) * prev_tempo / ticks_per_beat / 1e6
        prev_tick, prev_tempo = tick, tempo
    return secs + (ticks - prev_tick) * prev_tempo / ticks_per_beat / 1e6


def notes_from_musicxml(score, *, part: int = 0) -> list[Note]:
    """The notes of one part of a MusicXML file (or any score music21 reads, or a
    music21 stream), with its lyrics as words.

    Tied notes become one. Syllables are joined into words with MusicXML's
    ``syllabic`` (begin, middle, end), so ``can-dle`` on two notes is ``candle`` on
    both, the second continuing the word. A note without a lyric inside a word, or
    right after a sung note (a melisma under a slur), continues that word.
    """
    music21 = optional_import("music21")
    if isinstance(score, (str, os.PathLike)):
        score = music21.converter.parse(str(score))
    stream = score.parts[part] if getattr(score, "parts", None) else score
    notes, word, syllables = [], [], []
    for item in stream.stripTies().flatten().secondsMap:
        element = item["element"]
        if not isinstance(element, music21.note.GeneralNote):
            continue
        if element.isRest:
            word = []
            continue
        start, dur = float(item["offsetSeconds"]), float(item["durationSeconds"])
        pitch = element.pitches[-1].midi  # the top note, if a chord slipped in
        lyric = element.lyrics[0] if element.lyrics else None
        if lyric is not None and lyric.text.strip():
            syllabic = lyric.syllabic or "single"
            if syllabic in ("single", "begin") or not word:
                syllables, word = [lyric.text.strip()], []
                notes.append(Note(start, dur, pitch))
            else:
                syllables.append(lyric.text.strip())
                notes.append(Note(start, dur, pitch, starts_word=False))
            word.append(len(notes) - 1)
            _name_word(notes, word, "".join(syllables))
            if syllabic in ("single", "end"):
                word = [len(notes) - 1]  # a melisma may still hold this word
        elif word:
            notes.append(Note(start, dur, pitch, starts_word=False))
            word.append(len(notes) - 1)
            _name_word(notes, word, "".join(syllables))
        else:
            notes.append(Note(start, dur, pitch))
    return notes


def _name_word(notes: list[Note], indices: Sequence[int], word: str) -> None:
    for i in indices:
        notes[i] = replace(notes[i], word=word)


# --------------------------------------------------------------------------- lyrics


def lyric_tokens(lyrics: str) -> list[tuple[str, bool]]:
    """``(word, starts_word)`` per sung note, from a lyric line.

    >>> lyric_tokens("Light a can-dle - in")
    [('Light', True), ('a', True), ('candle', True), ('candle', False), ('candle', False), ('in', True)]
    """
    tokens: list[tuple[str, bool]] = []
    for token in lyrics.split():
        if token in HOLD_TOKENS:
            if not tokens:
                raise ValueError(f"A hold token ({token!r}) can't come first")
            tokens.append((tokens[-1][0], False))
            continue
        syllables = [s for s in token.split("-") if s]
        word = "".join(syllables)
        tokens.append((word, True))
        tokens.extend((word, False) for _ in syllables[1:])
    return tokens


def attach_lyrics(notes: Sequence[Note], lyrics: str) -> list[Note]:
    """``notes`` with the words of ``lyrics`` (see ``lyric_tokens``), one token per note.

    >>> attach_lyrics(notes_from_pitches([(60, 1)]), "two words")
    Traceback (most recent call last):
      ...
    ValueError: 2 lyric syllables for 1 notes: write 'can-dle' for one word over two notes, '-' to hold a word
    """
    tokens = lyric_tokens(lyrics)
    if len(tokens) != len(notes):
        raise ValueError(
            f"{len(tokens)} lyric syllables for {len(notes)} notes: write 'can-dle' "
            "for one word over two notes, '-' to hold a word"
        )
    return [
        replace(n, word=word, starts_word=starts)
        for n, (word, starts) in zip(notes, tokens)
    ]


def read_score(
    score,
    lyrics: str | None = None,
    *,
    filler: str = DFLT_FILLER_WORD,
) -> list[Note]:
    """The notes of ``score``, with words: from ``lyrics`` if given, else the score's own.

    Args:
        score: A MIDI file (``.mid``, ``.midi``), a MusicXML file (``.musicxml``,
            ``.xml``, ``.mxl``, or anything music21 parses), a music21 stream, or
            a sequence of ``Note``.
        lyrics: A lyric line (one token per note, see ``lyric_tokens``); replaces
            the score's own words.
        filler: The word for sung notes that still have none (a MIDI file with a
            lyric event on only some notes); a score with no words at all, and no
            ``lyrics``, is an error.

    >>> [n.word for n in read_score(notes_from_pitches([(60, 1), (62, 1)]), "do re")]
    ['do', 're']
    """
    if isinstance(score, (str, os.PathLike)):
        path = Path(score)
        if not path.exists():
            raise FileNotFoundError(f"No score at {path}")
        if path.suffix.lower() in MIDI_SUFFIXES:
            notes = notes_from_midi(path)
        else:
            notes = notes_from_musicxml(path)
    elif isinstance(score, Sequence) and all(isinstance(n, Note) for n in score):
        notes = list(score)
    else:
        notes = notes_from_musicxml(score)
    if not notes:
        raise ValueError("The score has no notes")
    notes.sort(key=lambda n: n.start)
    if lyrics is not None:
        return attach_lyrics(notes, lyrics)
    if not any(n.word for n in notes):
        raise ValueError("The score has no lyrics: pass lyrics='...'")
    return [n if n.word else replace(n, word=filler, starts_word=True) for n in notes]
