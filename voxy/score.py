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
#: MIDI channel 10 (index 9): percussion, never a melody.
DRUM_CHANNEL = 9
#: A note may overlap the next by this much (legato) before the line counts as polyphonic.
DFLT_MAX_OVERLAP_S = 0.1
#: A gap this long between notes ends a word: a note after it without a lyric of
#: its own starts a new word instead of continuing (melisma).
WORD_BREAK_REST_S = 0.2


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


def notes_from_midi(
    path: str | os.PathLike,
    *,
    track: int | None = None,
    channel: int | None = None,
    max_overlap_s: float = DFLT_MAX_OVERLAP_S,
    with_lyrics: bool = True,
) -> list[Note]:
    """The melody of a MIDI file: the notes of one track and channel, with its lyric
    events joined into words.

    The line is the one track and channel that has notes, drums (channel 10) aside;
    with several, pass ``track=`` (its index in the file) and, for a file that mixes
    channels in one track, ``channel=`` (1 to 16); the error lists the candidates.
    Times follow the file's tempo map. A note overlapping the next by up to
    ``max_overlap_s`` (legato) is cut at the next onset; more, or two notes starting
    together, is polyphony, and an error.

    Words come from ``lyrics`` meta-events only (never ``text`` events, so a label
    is never sung), on the melody's track, else on the one other track that has
    them (type-1 files often keep them in track 0). Each is a syllable on the note
    sounding at its tick (from a little before its onset to its end); one that
    falls on no note, or a second one on a note, is an error. A trailing ``-``
    (``can-``) or a leading one (``-dle``) joins syllables into one word; a note
    without a syllable continues the word before it (a melisma), unless a rest
    separates them. ``with_lyrics=False`` reads no lyrics (words come from
    elsewhere). Karaoke files (``.kar``, lyrics in ``text`` events) are not
    supported yet: full score reading is planned in audiate
    (https://github.com/thorwhalen/audiate/issues/2).
    """
    mido = optional_import("mido")
    mid = mido.MidiFile(str(path))
    tempo_map = _midi_tempo_map(mido.merge_tracks(mid.tracks))
    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    syllables: dict[int, list[tuple[int, str]]] = {}
    for t, trk in enumerate(mid.tracks):
        tick, active = 0, {}
        for msg in trk:
            tick += msg.time
            if msg.type == "lyrics" and msg.text.strip():
                syllables.setdefault(t, []).append((tick, msg.text))
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = tick
            elif msg.type in ("note_off", "note_on"):
                begin = active.pop((msg.channel, msg.note), None)
                if begin is not None and tick > begin:
                    groups.setdefault((t, msg.channel), []).append(
                        (begin, tick, msg.note)
                    )
    (t, ch), raw = _midi_line(groups, track=track, channel=channel, path=path)

    def seconds(ticks: int) -> float:
        return _midi_ticks_to_seconds(ticks, tempo_map, mid.ticks_per_beat)

    raw.sort()
    notes, spans = [], []
    for i, (begin, end, pitch) in enumerate(raw):
        if i + 1 < len(raw):
            following = raw[i + 1][0]
            if following == begin or seconds(end) - seconds(following) > max_overlap_s:
                raise ValueError(
                    f"Track {t} channel {ch + 1} of {path} is polyphonic at "
                    f"{seconds(following):.2f} s: pick a one-voice line (track=, "
                    "channel=) or reduce it to its melody"
                )
            end = min(end, following)
        onset = seconds(begin)
        notes.append(Note(onset, seconds(end) - onset, pitch))
        spans.append((begin, end))
    lyric_at = {}
    if with_lyrics:
        events = _midi_lyric_events(syllables, t, path=path)
        lyric_at = _midi_syllables(
            events, spans, tolerance=mid.ticks_per_beat // 16, seconds=seconds
        )
    return _join_words(
        [
            (note, *_hyphenated(lyric_at[i]))
            if i in lyric_at
            else (note, None, False, False)
            for i, note in enumerate(notes)
        ]
    )


def _midi_lyric_events(syllables, track: int, *, path) -> list[tuple[int, str]]:
    """The lyric events of the melody's ``track``, else of the one other track with
    some; lyrics on several other tracks are an error (which one is sung?)."""
    if syllables.get(track):
        return syllables[track]
    others = sorted(t for t, events in syllables.items() if events)
    if len(others) > 1:
        raise ValueError(
            f"{path}: the melody (track {track}) has no lyrics, and tracks {others} "
            "have some: pass the words with lyrics="
        )
    return syllables[others[0]] if others else []


def _midi_line(groups, *, track, channel, path):
    """The one ``((track, channel), notes)`` the melody is, or an error naming the choices."""
    wanted = {
        key: notes
        for key, notes in groups.items()
        if (track is None or key[0] == track)
        and (key[1] == channel - 1 if channel is not None else key[1] != DRUM_CHANNEL)
    }
    if len(wanted) == 1:
        return next(iter(wanted.items()))
    if not wanted:
        raise ValueError(f"No notes in {path} (track={track}, channel={channel})")
    choices = ", ".join(
        f"track={t} channel={c + 1} ({len(n)} notes)"
        for (t, c), n in sorted(wanted.items())
    )
    raise ValueError(f"{path} has several lines; pick one: {choices}")


def _midi_syllables(events, spans, *, tolerance, seconds) -> dict[int, str]:
    """``{note index: syllable}``: each lyric event on the note sounding at its tick
    (``begin - tolerance <= tick < end``). An event on no note, or two on one, is
    an error rather than a word sung in the wrong place."""
    found = {}
    for tick, text in sorted(events):
        i = next(
            (k for k, (b, e) in enumerate(spans) if b - tolerance <= tick < e), None
        )
        if i is None:
            raise ValueError(
                f"The lyric {text.strip()!r} at {seconds(tick):.2f} s falls on no note "
                "of the melody: fix the file, or pass the words with lyrics="
            )
        if i in found:
            raise ValueError(
                f"Two lyrics ({found[i].strip()!r}, {text.strip()!r}) on the note at "
                f"{seconds(spans[i][0]):.2f} s: fix the file, or pass lyrics="
            )
        found[i] = text
    return found


def _hyphenated(syllable: str) -> tuple[str, bool, bool]:
    """``(text, continues_a_word, leaves_the_word_open)`` of a MIDI lyric syllable.

    >>> _hyphenated("can-"), _hyphenated("-dle"), _hyphenated("light ")
    (('can', False, True), ('dle', True, False), ('light', False, False))
    """
    text = syllable.strip()
    return text.strip("-").strip(), text.startswith("-"), text.endswith("-")


def _join_words(items, *, rest_s: float = WORD_BREAK_REST_S) -> list[Note]:
    """Notes named with whole words, from ``(note, syllable, continues, open_after)``
    items (``syllable`` None for a note without one; an item None for a rest).

    A syllable continues the current word when it says so or the one before left
    the word open; a note without a syllable continues the current word (melisma)
    unless a rest (an item None, or a gap of ``rest_s``) came between.
    """
    notes: list[Note] = []
    word: list[int] = []
    parts: list[str] = []
    open_word, previous = False, None
    for item in items:
        if item is None:
            word, open_word, previous = [], False, None
            continue
        note, syllable, continues, open_after = item
        after_rest = previous is not None and note.start - previous.end >= rest_s
        if syllable:
            if (continues or open_word) and word:
                parts.append(syllable)
                notes.append(replace(note, starts_word=False))
            else:
                parts, word = [syllable], []
                notes.append(replace(note, starts_word=True))
            word.append(len(notes) - 1)
            open_word = open_after
        elif word and not after_rest:
            notes.append(replace(note, starts_word=False))
            word.append(len(notes) - 1)
        else:
            notes.append(replace(note, word="", starts_word=True))
            word, open_word = [], False
        if word:
            _name_word(notes, word, "".join(parts))
        previous = note
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

    Only the part's first voice is read (timed by the part's own tempo marks), and
    only its first verse of lyrics; grace notes and zero-length notes are skipped;
    a chord sings its highest note. Tied notes become one. Syllables are
    joined into words with MusicXML's ``syllabic`` (begin, middle, end), so
    ``can-dle`` on two notes is ``candle`` on both, the second continuing the
    word. A note without a lyric continues the word before it (a melisma under a
    slur), unless a rest came between.
    """
    music21 = optional_import("music21")
    if isinstance(score, (str, os.PathLike)):
        score = music21.converter.parse(str(score))
    stream = score.parts[part] if getattr(score, "parts", None) else score
    stream = stream.stripTies()
    other_voices = {
        id(element)
        for measure in stream.getElementsByClass(music21.stream.Measure)
        for voice in list(measure.voices)[1:]
        for element in voice.recurse().notesAndRests
    }
    verse = min(
        (lyric.number for n in stream.recurse().notes for lyric in n.lyrics),
        default=1,
    )
    items = []
    for item in stream.flatten().secondsMap:
        element = item["element"]
        if not isinstance(element, music21.note.GeneralNote):
            continue
        if id(element) in other_voices:
            continue
        if element.isRest:
            items.append(None)
            continue
        start, dur = float(item["offsetSeconds"]), float(item["durationSeconds"])
        if element.duration.isGrace or dur <= 0:
            continue
        note = Note(start, dur, max(p.midi for p in element.pitches))
        lyric = next((ly for ly in element.lyrics if ly.number == verse), None)
        if lyric is None or not (lyric.text or "").strip():
            items.append((note, None, False, False))
            continue
        syllabic = lyric.syllabic or "single"
        items.append(
            (
                note,
                lyric.text.strip(),
                syllabic in ("middle", "end"),
                syllabic in ("begin", "middle"),
            )
        )
    return _join_words(items)


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
    track: int | None = None,
    channel: int | None = None,
    part: int = 0,
) -> list[Note]:
    """The notes of ``score``, with words: from ``lyrics`` if given, else the score's own.

    Args:
        score: A MIDI file (``.mid``, ``.midi``), a MusicXML file (``.musicxml``,
            ``.xml``, ``.mxl``, or anything music21 parses), a music21 stream, or
            a sequence of ``Note``.
        lyrics: A lyric line (one token per note, see ``lyric_tokens``); replaces
            the score's own words.
        filler: The word for sung notes that still have none and continue no word
            (a note after a rest, in a score with lyrics on only some notes); a
            score with no words at all, and no ``lyrics``, is an error.
        track, channel: The MIDI line to sing (see ``notes_from_midi``).
        part: The MusicXML part to sing.

    >>> [n.word for n in read_score(notes_from_pitches([(60, 1), (62, 1)]), "do re")]
    ['do', 're']
    """
    if isinstance(score, (str, os.PathLike)):
        path = Path(score)
        if not path.exists():
            raise FileNotFoundError(f"No score at {path}")
        if path.suffix.lower() in MIDI_SUFFIXES:
            notes = notes_from_midi(
                path, track=track, channel=channel, with_lyrics=lyrics is None
            )
        else:
            notes = notes_from_musicxml(path, part=part)
    elif isinstance(score, Sequence) and all(isinstance(n, Note) for n in score):
        notes = list(score)
    else:
        notes = notes_from_musicxml(score, part=part)
    notes = [n for n in notes if n.dur > 0]
    if not notes:
        raise ValueError("The score has no notes")
    notes.sort(key=lambda n: n.start)
    if lyrics is not None:
        return attach_lyrics(notes, lyrics)
    if not any(n.word for n in notes):
        raise ValueError("The score has no lyrics: pass lyrics='...'")
    return [n if n.word else replace(n, word=filler, starts_word=True) for n in notes]
