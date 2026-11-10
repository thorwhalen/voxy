# voxy.score

Scores for singing: the notes of a melody (pitch, timing, word) in the form voxy’s
singing backends take, read from MIDI, MusicXML or plain code.

A `Note` is one sung note in absolute seconds: its MIDI pitch, the word sung on it,
and whether the word starts on it (`starts_word`) or continues from the note
before (a melisma, or the later syllables of a word). Rests are the gaps between
notes. A singing backend turns a list of notes into its own input format
(`voxy.soulx.soulx_metadata` for SoulX-Singer).

```pycon
>>> from voxy.score import read_score
>>> notes = read_score("melody.mid", "Light a can-dle in the win-dow")
>>> notes = read_score("song.musicxml")         # the score's own lyrics
```

Reading a score is audiate’s job: `read_score` takes the sung line from
`audiate.sung_line` (its module docstring states the reading rules: which line,
which verse, repeats, melismas, karaoke files) and makes `Note``s of it. When
audiate is not sure (several candidate lines, a chord, a lyric on no note), it
raises and names the choices. ``read_score` (and `voxy.sing`) take them as
`track=`, `channel=`, `part=`, `verse=`, `chords=` and `score_voice=`
(audiate’s `voice=`: in voxy, `voice=` is the singer), and audiate’s errors
are reworded to match; for audiate’s other options read the line with
`audiate.sung_line` and pass the notes it returns. audiate is in the
`voxy[singing]` extra and imported only when a score is read or lyrics attached.

Lyrics given by hand are one whitespace-separated token per sung note: `can-dle`
spreads one word over two notes, a bare `-` (or `_`, `~`) holds the previous
word one more note (`audiate.sung.lyric_tokens`).

```pycon
>>> notes = notes_from_pitches([(64, 1), (67, 1), (60, 2)], bpm=120)
>>> [(n.start, n.dur, n.pitch) for n in notes]
[(0.0, 0.5, 64), (0.5, 0.5, 67), (1.0, 1.0, 60)]
```

### Module Attributes

| [`SCORE_READER`](#voxy.score.SCORE_READER)     | The module that reads scores (`sung_line`) and attaches lyrics (`attach_lyrics`).   |
|-------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`MIN_AUDIATE`](#voxy.score.MIN_AUDIATE)      | The first audiate release that has `SCORE_READER` (keep in step with pyproject).    |
| [`SUNG_LINE_NAMES`](#voxy.score.SUNG_LINE_NAMES)  | `read_score`'s name for each `audiate.sung_line` option it renames.                 |
| [`DFLT_FILLER_WORD`](#voxy.score.DFLT_FILLER_WORD) | The word given to sung notes that have none (a score without lyrics on every note). |

### Functions

| [`notes_from_pitches`](#voxy.score.notes_from_pitches)(pitches, \*[, bpm, start])   | Notes from `(midi_pitch, beats)` pairs, back to back; a pitch of 0 or None is a rest.                                                  |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| [`read_score`](#voxy.score.read_score)(score[, lyrics, filler, track, ...]) | The notes of `score`, with words: from `lyrics` if given, else the score's own.                                                        |
| [`to_note`](#voxy.score.to_note)(note)                                   | A `Note` from any note with `start`, `dur`, `pitch` (and optionally `word`, `starts_word`): an `audiate.SungNote`, or a `Note` itself. |

### Classes

| [`Note`](#voxy.score.Note)(start, dur, pitch[, word, starts_word])   | One sung note, in absolute seconds.   |
|-------------------------------------------------------------------------------------------------|---------------------------------------|

### voxy.score.DFLT_FILLER_WORD *= 'la'*

The word given to sung notes that have none (a score without lyrics on every note).

### voxy.score.MIN_AUDIATE *= '0.0.4'*

The first audiate release that has `SCORE_READER` (keep in step with pyproject).

### *class* voxy.score.Note(start, dur, pitch, word='', starts_word=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One sung note, in absolute seconds.

* **Parameters:**
  * **start** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Onset (seconds from the start of the score).
  * **dur** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Duration in seconds.
  * **pitch** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – MIDI note number (60 is middle C).
  * **word** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The word sung on it (’’ until lyrics are attached).
  * **starts_word** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – True if the word starts here; False if it continues the
    previous note’s word (melisma, later syllable).

#### *property* end *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Where the note stops (seconds).

### voxy.score.SCORE_READER *= 'audiate.sung'*

The module that reads scores (`sung_line`) and attaches lyrics (`attach_lyrics`).

### voxy.score.SUNG_LINE_NAMES *= {'score_voice': 'voice'}*

`read_score`’s name for each `audiate.sung_line` option it renames.

### voxy.score.notes_from_pitches(pitches, , bpm=100, start=0.0)

Notes from `(midi_pitch, beats)` pairs, back to back; a pitch of 0 or None is a rest.

```pycon
>>> [(n.start, n.pitch) for n in notes_from_pitches([(60, 1), (0, 1), (62, 1)], bpm=60)]
[(0.0, 60), (2.0, 62)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Note`](#voxy.score.Note)]

### voxy.score.read_score(score, lyrics=None, , filler='la', track=None, channel=None, part=None, verse=None, score_voice=None, chords=None)

The notes of `score`, with words: from `lyrics` if given, else the score’s own.

* **Parameters:**
  * **score** – A MIDI file (`.mid`, `.midi`, `.kar`), a MusicXML file (or
    anything music21 parses), its bytes, a music21 stream (all read with
    `audiate.sung_line`), or a sequence of notes (`Note`, or
    `audiate.SungNote` as `audiate.sung_line` returns them).
  * **lyrics** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A lyric line (one token per note, see `audiate.sung.lyric_tokens`);
    replaces the score’s own words, which are then not read.
  * **filler** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The word for sung notes that still have none and continue no word
    (a note after a rest, in a score with lyrics on only some notes); a
    score with no words at all, and no `lyrics`, is an error.
  * **track** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The MIDI line to sing (`track`: index in the file;
    `channel`: 1 to 16).
  * **channel** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The MIDI line to sing (`track`: index in the file;
    `channel`: 1 to 16).
  * **part** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The music21 part (an index, or a name or id) and the lyric
    verse to sing.
  * **verse** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The music21 part (an index, or a name or id) and the lyric
    verse to sing.
  * **score_voice** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The voice of that part (its MusicXML voice number; the
    `voice=` of `audiate.sung_line`).
  * **chords** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Two notes at once: an error (None, the default), or sing the
    `"top"` or `"bottom"` one.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Note`](#voxy.score.Note)]

Which line and verse are sung by default, and when the reading raises
instead of guessing, is audiate’s to say: see `audiate.sung`.

```pycon
>>> [n.word for n in read_score([Note(0, 1, 60, "do"), Note(1, 1, 62, "re")])]
['do', 're']
```

### voxy.score.to_note(note)

A `Note` from any note with `start`, `dur`, `pitch` (and optionally
`word`, `starts_word`): an `audiate.SungNote`, or a `Note` itself.

```pycon
>>> to_note(Note(0.0, 0.5, 60, "la"))
Note(start=0.0, dur=0.5, pitch=60, word='la', starts_word=True)
```

* **Return type:**
  [`Note`](#voxy.score.Note)
