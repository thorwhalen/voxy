# voxy.score

Scores for singing: the notes of a melody (pitch, timing, word) from MIDI, MusicXML
or plain code, with lyrics attached.

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

Lyrics are a line of whitespace-separated tokens, one per sung note:
`can-dle` spreads one word over two notes, a bare `-` (or `_`) holds the
previous word one more note.

```pycon
>>> notes = notes_from_pitches([(64, 1), (67, 1), (60, 2)], bpm=120)
>>> [(n.start, n.dur, n.pitch) for n in notes]
[(0.0, 0.5, 64), (0.5, 0.5, 67), (1.0, 1.0, 60)]
>>> [(n.word, n.starts_word) for n in attach_lyrics(notes, "hel-lo you")]
[('hello', True), ('hello', False), ('you', True)]
```

MIDI is read with `mido` (every tempo change honoured, lyric meta-events kept),
MusicXML (and anything else music21 parses) with `music21`: both are in the
`voxy[singing]` extra and imported only when a score of that kind is read.

### Module Attributes

| [`HOLD_TOKENS`](#voxy.score.HOLD_TOKENS)        | Tokens of a lyric line that hold the previous word one more note.                             |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| [`DFLT_FILLER_WORD`](#voxy.score.DFLT_FILLER_WORD)   | The word given to sung notes that have none (a score without lyrics on every note).           |
| [`DFLT_MIDI_TEMPO`](#voxy.score.DFLT_MIDI_TEMPO)    | 500 000 microseconds a beat (120 BPM).                                                        |
| [`DRUM_CHANNEL`](#voxy.score.DRUM_CHANNEL)       | percussion, never a melody.                                                                   |
| [`DFLT_MAX_OVERLAP_S`](#voxy.score.DFLT_MAX_OVERLAP_S) | A note may overlap the next by this much (legato) before the line counts as polyphonic.       |
| [`WORD_BREAK_REST_S`](#voxy.score.WORD_BREAK_REST_S)  | a note after it without a lyric of its own starts a new word instead of continuing (melisma). |

### Functions

| [`attach_lyrics`](#voxy.score.attach_lyrics)(notes, lyrics)                     | `notes` with the words of `lyrics` (see `lyric_tokens`), one token per note.                                          |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [`lyric_tokens`](#voxy.score.lyric_tokens)(lyrics)                             | `(word, starts_word)` per sung note, from a lyric line.                                                               |
| [`notes_from_midi`](#voxy.score.notes_from_midi)(path, \*[, track, channel, ...]) | The melody of a MIDI file: the notes of one track and channel, with its lyric events joined into words.               |
| [`notes_from_musicxml`](#voxy.score.notes_from_musicxml)(score, \*[, part])           | The notes of one part of a MusicXML file (or any score music21 reads, or a music21 stream), with its lyrics as words. |
| [`notes_from_pitches`](#voxy.score.notes_from_pitches)(pitches, \*[, bpm, start])    | Notes from `(midi_pitch, beats)` pairs, back to back; a pitch of 0 or None is a rest.                                 |
| [`read_score`](#voxy.score.read_score)(score[, lyrics, filler, track, ...])  | The notes of `score`, with words: from `lyrics` if given, else the score's own.                                       |

### Classes

| [`Note`](#voxy.score.Note)(start, dur, pitch[, word, starts_word])   | One sung note, in absolute seconds.   |
|-------------------------------------------------------------------------------------------------|---------------------------------------|

### voxy.score.DFLT_FILLER_WORD *= 'la'*

The word given to sung notes that have none (a score without lyrics on every note).

### voxy.score.DFLT_MAX_OVERLAP_S *= 0.1*

A note may overlap the next by this much (legato) before the line counts as polyphonic.

### voxy.score.DFLT_MIDI_TEMPO *= 500000*

500 000 microseconds a beat (120 BPM).

* **Type:**
  MIDI tempo before any `set_tempo` event

### voxy.score.DRUM_CHANNEL *= 9*

percussion, never a melody.

* **Type:**
  MIDI channel 10 (index 9)

### voxy.score.HOLD_TOKENS *= frozenset({'-', '_'})*

Tokens of a lyric line that hold the previous word one more note.

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

### voxy.score.WORD_BREAK_REST_S *= 0.2*

a note after it without a lyric of
its own starts a new word instead of continuing (melisma).

* **Type:**
  A gap this long between notes ends a word

### voxy.score.attach_lyrics(notes, lyrics)

`notes` with the words of `lyrics` (see `lyric_tokens`), one token per note.

```pycon
>>> attach_lyrics(notes_from_pitches([(60, 1)]), "two words")
Traceback (most recent call last):
  ...
ValueError: 2 lyric syllables for 1 notes: write 'can-dle' for one word over two notes, '-' to hold a word
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Note`](#voxy.score.Note)]

### voxy.score.lyric_tokens(lyrics)

`(word, starts_word)` per sung note, from a lyric line.

```pycon
>>> lyric_tokens("Light a can-dle - in")
[('Light', True), ('a', True), ('candle', True), ('candle', False), ('candle', False), ('in', True)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]]

### voxy.score.notes_from_midi(path, , track=None, channel=None, max_overlap_s=0.1, with_lyrics=True)

The melody of a MIDI file: the notes of one track and channel, with its lyric
events joined into words.

The line is the one track and channel that has notes, drums (channel 10) aside;
with several, pass `track=` (its index in the file) and, for a file that mixes
channels in one track, `channel=` (1 to 16); the error lists the candidates.
Times follow the file’s tempo map. A note overlapping the next by up to
`max_overlap_s` (legato) is cut at the next onset; more, or two notes starting
together, is polyphony, and an error.

Words come from `lyrics` meta-events only (never `text` events, so a label
is never sung), on the melody’s track, else on the one other track that has
them (type-1 files often keep them in track 0). Each is a syllable on the note
sounding at its tick (from a little before its onset to its end); one that
falls on no note, or a second one on a note, is an error. A trailing `-`
(`can-`) or a leading one (`-dle`) joins syllables into one word; a note
without a syllable continues the word before it (a melisma), unless a rest
separates them. `with_lyrics=False` reads no lyrics (words come from
elsewhere). Karaoke files (`.kar`, lyrics in `text` events) are not
supported yet: full score reading is planned in audiate
([https://github.com/thorwhalen/audiate/issues/2](https://github.com/thorwhalen/audiate/issues/2)).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Note`](#voxy.score.Note)]

### voxy.score.notes_from_musicxml(score, , part=0)

The notes of one part of a MusicXML file (or any score music21 reads, or a
music21 stream), with its lyrics as words.

Only the part’s first voice is read (timed by the part’s own tempo marks), and
only its first verse of lyrics; grace notes and zero-length notes are skipped;
a chord sings its highest note. Tied notes become one. Syllables are
joined into words with MusicXML’s `syllabic` (begin, middle, end), so
`can-dle` on two notes is `candle` on both, the second continuing the
word. A note without a lyric continues the word before it (a melisma under a
slur), unless a rest came between.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Note`](#voxy.score.Note)]

### voxy.score.notes_from_pitches(pitches, , bpm=100, start=0.0)

Notes from `(midi_pitch, beats)` pairs, back to back; a pitch of 0 or None is a rest.

```pycon
>>> [(n.start, n.pitch) for n in notes_from_pitches([(60, 1), (0, 1), (62, 1)], bpm=60)]
[(0.0, 60), (2.0, 62)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Note`](#voxy.score.Note)]

### voxy.score.read_score(score, lyrics=None, , filler='la', track=None, channel=None, part=0)

The notes of `score`, with words: from `lyrics` if given, else the score’s own.

* **Parameters:**
  * **score** – A MIDI file (`.mid`, `.midi`), a MusicXML file (`.musicxml`,
    `.xml`, `.mxl`, or anything music21 parses), a music21 stream, or
    a sequence of `Note`.
  * **lyrics** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A lyric line (one token per note, see `lyric_tokens`); replaces
    the score’s own words.
  * **filler** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The word for sung notes that still have none and continue no word
    (a note after a rest, in a score with lyrics on only some notes); a
    score with no words at all, and no `lyrics`, is an error.
  * **track** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The MIDI line to sing (see `notes_from_midi`).
  * **channel** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The MIDI line to sing (see `notes_from_midi`).
  * **part** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The MusicXML part to sing.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Note`](#voxy.score.Note)]

```pycon
>>> [n.word for n in read_score(notes_from_pitches([(60, 1), (62, 1)]), "do re")]
['do', 're']
```
