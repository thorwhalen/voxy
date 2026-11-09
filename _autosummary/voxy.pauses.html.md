# voxy.pauses

Pauses in text to speak: SSML breaks, honoured on every backend.

The standard way to ask a speech engine for silence is the W3C SSML 1.1 break
element, written into the text [1]:

> Breathe in.<break time=”4s”/> And out, slowly.<break time=”6s”/>

Engines honour it unevenly: ElevenLabs’ v2-family models up to 3 s, `eleven_v3`
and `eleven_v4` not at all [2], local models never. So `voxy.text_to_speech`
resolves breaks itself. A break the backend renders (`SpeechModel.native_break_s`)
stays in the text; at any other break the text is cut, each part is spoken on its
own, and the parts are joined with exactly that much silence. Text with no markup
takes the old path, untouched.

The structure underneath is a list of [`Section`](#voxy.pauses.Section) (`parse_sections`): the
text, cut at every `<break>` and every SSML `<mark name="..."/>`, each section
carrying the silence that follows it and the mark that ends it. A linear render
reads `plan_sections`, which cuts only where the backend needs a cut and drops
marks. A player that holds at a mark until the listener continues (SMIL 3.0’s
`end="<id>.activateEvent"` [3]) reads `parse_sections` and speaks section by
section: no other change is needed for that.

```pycon
>>> [(s.text, s.pause_s) for s in parse_sections('In.<break time="4s"/> Out.')]
[('In.', 4.0), ('Out.', 0.0)]
```

[1] W3C. Speech Synthesis Markup Language (SSML) Version 1.1, section 3.2.3.
: [https://www.w3.org/TR/speech-synthesis11/#edef_break](https://www.w3.org/TR/speech-synthesis11/#edef_break)

[2] ElevenLabs. Eleven v4 / Eleven v3 prompting guides (“SSML is not supported”).
: [https://elevenlabs.io/docs/overview/capabilities/text-to-speech/eleven-v4](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/eleven-v4)

[3] W3C. SMIL 3.0 Timing and Synchronization (event-based end values).
: [https://www.w3.org/TR/SMIL3/smil-timing.html](https://www.w3.org/TR/SMIL3/smil-timing.html)

### Module Attributes

| [`BREAK_STRENGTH_S`](#voxy.pauses.BREAK_STRENGTH_S)   | Seconds for each SSML `strength` (SSML leaves them to the engine; these are Azure Speech's documented values).   |
|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`DFLT_SILENCE_RATE`](#voxy.pauses.DFLT_SILENCE_RATE)  | The rate of a render that is nothing but silence (no section had text).                                          |

### Functions

| [`break_seconds`](#voxy.pauses.break_seconds)(attrs)                           | The silence a `<break>` asks for, from its attribute string.                                                                                        |
|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| [`decode_speech`](#voxy.pauses.decode_speech)(speech)                          | `(samples, rate)` of a `Speech`: WAV directly, anything else (mp3, ...) through ffmpeg.                                                             |
| [`encode_audio`](#voxy.pauses.encode_audio)(samples, rate, audio_format, \*)  | `samples` as `audio_format` bytes: WAV directly, anything else through ffmpeg (at `bitrate` if given, else the encoder default of `ffmpeg_encode`). |
| [`has_pause_markup`](#voxy.pauses.has_pause_markup)(text)                         | Whether `text` holds an SSML `<break>` or `<mark>`.                                                                                                 |
| [`join_sections`](#voxy.pauses.join_sections)(sections, synthesize, \*[, ...]) | One `Speech`: each section spoken by `synthesize`, then its silence.                                                                                |
| [`parse_sections`](#voxy.pauses.parse_sections)(text)                           | `text` cut at every SSML `<break>` and `<mark>`.                                                                                                    |
| [`plan_sections`](#voxy.pauses.plan_sections)(text, \*[, native_break_s])      | The sections to synthesize separately, for a backend that renders breaks of up to `native_break_s` seconds itself.                                  |
| [`render_break`](#voxy.pauses.render_break)(seconds)                          | The SSML break for `seconds`, in the form ElevenLabs documents.                                                                                     |

### Classes

| [`Section`](#voxy.pauses.Section)(text[, pause_s, mark])   | A stretch of text to speak, and how it ends.   |
|-----------------------------------------------------------------------------------|------------------------------------------------|

### voxy.pauses.BREAK_STRENGTH_S *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [float](https://docs.python.org/3/builtins/functions.html#float)]* *= {'medium': 0.75, 'none': 0.0, 'strong': 1.0, 'weak': 0.5, 'x-strong': 1.25, 'x-weak': 0.25}*

Seconds for each SSML `strength` (SSML leaves them to the engine; these are
Azure Speech’s documented values). A bare `<break/>` is `medium`.

### voxy.pauses.DFLT_SILENCE_RATE *= 24000*

The rate of a render that is nothing but silence (no section had text).

### *class* voxy.pauses.Section(text, pause_s=0.0, mark=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A stretch of text to speak, and how it ends.

`pause_s` is the silence after it; `mark` the name of the SSML mark that
ends it, if one does (a point a player may stop at).

### voxy.pauses.break_seconds(attrs)

The silence a `<break>` asks for, from its attribute string.

`time` (`"3s"`, `"250ms"`) wins over `strength`; neither means medium.

```pycon
>>> break_seconds(' time="250ms"'), break_seconds(" strength='strong'"), break_seconds("")
(0.25, 1.0, 0.75)
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### voxy.pauses.decode_speech(speech)

`(samples, rate)` of a `Speech`: WAV directly, anything else (mp3,
…) through ffmpeg.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### voxy.pauses.encode_audio(samples, rate, audio_format, , bitrate=None)

`samples` as `audio_format` bytes: WAV directly, anything else through
ffmpeg (at `bitrate` if given, else the encoder default of `ffmpeg_encode`).

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

### voxy.pauses.has_pause_markup(text)

Whether `text` holds an SSML `<break>` or `<mark>`.

```pycon
>>> has_pause_markup('Hi <break time="1s"/>'), has_pause_markup("Hi [softly]")
(True, False)
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### voxy.pauses.join_sections(sections, synthesize, \*, backend='', decode=<function decode_speech>, encode=<function encode_audio>)

One `Speech`: each section spoken by `synthesize`, then its silence.

`synthesize(text) -> Speech`. `decode(speech) -> (samples, rate)` reads a
section (default: WAV, or any format ffmpeg reads); sections at different
rates are resampled to the first one’s. The result is in the format the first
section came in: WAV, or `encode(samples, rate, format) -> bytes` for a
compressed one (default: ffmpeg), so a request for mp3 gets mp3 back. Lossy
encoders pad each section by tens of milliseconds, so a pause in a compressed
render can be up to ~50 ms off; ask for WAV or PCM where that matters.

### voxy.pauses.parse_sections(text)

`text` cut at every SSML `<break>` and `<mark>`.

Breaks in a row add up; a mark keeps the pause of breaks around it. A leading
break gives a first section with no text (silence first); a trailing one is
kept (silence last).

```pycon
>>> parse_sections('<break time="2s"/>One.<mark name="m1"/> Two.<break/><break/>')
[Section(text='', pause_s=2.0, mark=None), Section(text='One.', pause_s=0.0, mark='m1'), Section(text='Two.', pause_s=1.5, mark=None)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Section`](#voxy.pauses.Section)]

### voxy.pauses.plan_sections(text, , native_break_s=0.0)

The sections to synthesize separately, for a backend that renders breaks of
up to `native_break_s` seconds itself.

A break that short stays in the text (as [`render_break()`](#voxy.pauses.render_break)); a mark is
dropped (a linear render has nothing to do at it); every other break ends a
section.

```pycon
>>> t = 'A.<break time="1s"/> B.<break time="5s"/> C.'
>>> [(s.text, s.pause_s) for s in plan_sections(t, native_break_s=3)]
[('A. <break time="1s" /> B.', 5.0), ('C.', 0.0)]
>>> [s.text for s in plan_sections(t)]
['A.', 'B.', 'C.']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Section`](#voxy.pauses.Section)]

### voxy.pauses.render_break(seconds)

The SSML break for `seconds`, in the form ElevenLabs documents.

```pycon
>>> render_break(1.5)
'<break time="1.5s" />'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
