# voxy.singing

Singing: sing a score with lyrics in a voice (SVS), or re-voice a sung take (SVC).

```pycon
>>> import voxy
>>> voxy.sing("melody.mid", "Light a can-dle in the win-dow").save("sung.wav")
>>> voice = voxy.singing_voice("me.wav", "phrase.mid", "la la la la")
>>> voxy.sing("song.musicxml", voice=voice).save("song.wav")
>>> voxy.convert_singing("take.wav", voice=voice).save("take-in-my-voice.wav")
```

A singing *voice* is a sung clip, plus, for synthesis, the notes and words it sings
(`singing_voice`): a zero-shot singer copies the clip’s timbre, and SoulX-Singer
also aligns the clip with its notes and phonemes. The simplest way to get that
alignment is to make the clip from a known score: sing a short given phrase (its
MIDI or MusicXML, with its lyrics) and pass the recording with that score. Without
a voice, a synthetic one is used (`synthetic_singing_voice`: a formant
synthesiser singing “la”, which belongs to no one). Conversion needs only the clip.
Only clone a singing voice with its owner’s consent; each backend records its own
usage restrictions in `singing_backends()[name].consent`.

Scores are MIDI or MusicXML files, music21 streams, or lists of `voxy.score.Note`
(`voxy.score.read_score`); lyrics are one token per note (`can-dle` spreads a
word over two notes, `-` holds it one more).

Backends are `SingingBackend` records (`singing_backends()`), each running in
its own environment (`python -m voxy setup soulx`); `soulx` (SoulX-Singer,
Apache-2.0) is the built-in one. A model is the seam: `sing(..., model=...)`
takes any object with `sing(notes, voice, *, transpose, language, **params)` and
`convert(audio_path, voice, *, transpose, **params)` returning a `Speech`.

```pycon
>>> class Echo:
...     name = "echo"
...     def sing(self, notes, voice, **kw):
...         return Speech(b"RIFF", "wav", "echo", text=" ".join(n.word for n in notes))
>>> from voxy.score import notes_from_pitches
>>> sing(notes_from_pitches([(60, 1), (62, 1)]), "do re", voice=VoiceProfile(
...     {"prompt_audio": "v.wav", "prompt_notes": []}, 0, "echo", 24000), model=Echo()).text
'do re'
```

### Module Attributes

| [`DFLT_SINGING_SAMPLES`](#voxy.singing.DFLT_SINGING_SAMPLES)       | The samples store that keeps singing clips given as bytes.                  |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`FIT_VOICE`](#voxy.singing.FIT_VOICE)                  | `transpose` value meaning "move the melody into the voice's register".      |
| [`BUILTIN_SINGING_MODULES`](#voxy.singing.BUILTIN_SINGING_MODULES)    | Modules that register a built-in singing backend when imported.             |
| [`SYNTHETIC_PROMPT_PHRASE`](#voxy.singing.SYNTHETIC_PROMPT_PHRASE)    | `(midi, beats)`.                                                            |
| [`SYNTHETIC_PRESET_TRANSPOSE`](#voxy.singing.SYNTHETIC_PRESET_TRANSPOSE) | Semitones each synthetic voice preset sings the phrase at, from as written. |

### Functions

| [`close_singing_models`](#voxy.singing.close_singing_models)([models])                    | Stop the cached singing models' workers and free their memory (a model stays loaded, about 6 GB for SoulX, until this is called or the process ends).                                                                                     |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`convert_singing`](#voxy.singing.convert_singing)(audio, \*[, voice, backend, ...]) | Re-voice the sung `audio` in `voice`, keeping its melody, words and timing.                                                                                                                                                               |
| [`get_singing_model`](#voxy.singing.get_singing_model)([backend, models])              | The (cached) model of `backend`; `models` replaces the shared cache.                                                                                                                                                                      |
| [`register_singing_backend`](#voxy.singing.register_singing_backend)(backend, \*[, replace])  | Add a singing backend (`replace=False` keeps one already registered).                                                                                                                                                                     |
| [`resolve_singing_voice`](#voxy.singing.resolve_singing_voice)(voice, \*[, backend, ...])  | The singing voice `voice` names: a profile as is; a library name or alias (its `backend` profile); a path to a sung clip (for conversion); None, the synthetic voice.                                                                     |
| [`sing`](#voxy.singing.sing)(score[, lyrics, voice, backend, ...])        | Sing `score` with `lyrics` in `voice` and return the audio (`.save(path)`).                                                                                                                                                               |
| [`singing_backends`](#voxy.singing.singing_backends)()                                | Every singing backend, by name (the built-in ones registered first).                                                                                                                                                                      |
| [`singing_voice`](#voxy.singing.singing_voice)(audio[, score, lyrics, ...])        | A singing voice: a sung clip, and (for synthesis) the score it sings.                                                                                                                                                                     |
| [`synthetic_singing_voice`](#voxy.singing.synthetic_singing_voice)([preset, backend])        | A voice that belongs to no one: a formant synthesiser singing "la" on a short phrase (`voxy.formant_singer`), rendered once into voxy's cache (the file is named by a hash of everything it is rendered from, so a change re-renders it). |

### Classes

| [`SingingBackend`](#voxy.singing.SingingBackend)(name, env, factory[, init, ...])   | What voxy knows about one singing model.                                                                                             |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [`SingingModel`](#voxy.singing.SingingModel)(\*args, \*\*kwargs)                  | What `sing` and `convert_singing` call.                                                                                              |
| [`SingingWorker`](#voxy.singing.SingingWorker)(backend, \*[, init, threads, ...])  | The default model call of a singing backend: its adapter in its own environment, given the backend's fetched `source` as `repo_dir`. |

### voxy.singing.BUILTIN_SINGING_MODULES *= ('voxy.soulx',)*

Modules that register a built-in singing backend when imported.

### voxy.singing.DFLT_SINGING_SAMPLES *= 'singing-voices'*

The samples store that keeps singing clips given as bytes.

### voxy.singing.FIT_VOICE *= 'fit_voice'*

`transpose` value meaning “move the melody into the voice’s register”.

### voxy.singing.SYNTHETIC_PRESET_TRANSPOSE *= {'high': 0, 'low': -12}*

Semitones each synthetic voice preset sings the phrase at, from as written.

### voxy.singing.SYNTHETIC_PROMPT_PHRASE *= ((67, 1), (64, 1), (65, 1), (62, 1), (64, 1), (60, 1), (62, 1), (60, 2))*

`(midi, beats)`.

* **Type:**
  The phrase the synthetic voices sing (”la” on each note)

### *class* voxy.singing.SingingBackend(name, env, factory, init=dict(), params=dict(), source=None, licence='', consent='', notes='', sings=True, converts=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What voxy knows about one singing model.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Registry name (also its adapter’s, `workers/{name}_adapter.py`).
  * **env** ([`EnvSpec`](voxy.envs.html.md#voxy.envs.EnvSpec)) – The environment it runs in.
  * **factory** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`SingingModel`](#voxy.singing.SingingModel)]) – `(**kwargs) -> SingingModel`.
  * **init** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Default keyword arguments of the adapter’s `load`.
  * **params** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Documented parameters and their defaults (pass any to `sing`).
  * **source** ([`GitSource`](voxy.envs.html.md#voxy.envs.GitSource) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Code fetched at a pinned commit and handed to the adapter as
    `repo_dir` (for a model that pip can’t install), or None.
  * **licence** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Code and weights licences.
  * **consent** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The model’s own restrictions on cloning and impersonation.
  * **notes** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – One line on strengths and caveats.
  * **sings** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether it does SVS, SVC.
  * **converts** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether it does SVS, SVC.

### *class* voxy.singing.SingingModel(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

What `sing` and `convert_singing` call.

### *class* voxy.singing.SingingWorker(backend, , init=None, threads=None, rootdir=None, env_vars=None, auto_setup=True)

Bases: [`WorkerGenerate`](voxy.local_clone.html.md#voxy.local_clone.WorkerGenerate)

The default model call of a singing backend: its adapter in its own
environment, given the backend’s fetched `source` as `repo_dir`.

`worker(op, **request)` runs operation `op`, which writes a WAV to the
`out_path` it is given, and returns `(samples, sample_rate)`.

### voxy.singing.close_singing_models(models=None)

Stop the cached singing models’ workers and free their memory (a model stays
loaded, about 6 GB for SoulX, until this is called or the process ends).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.singing.convert_singing(audio, , voice=None, backend='soulx', output_path=None, transpose='fit_voice', model=None, models=None, voices=None, \*\*params)

Re-voice the sung `audio` in `voice`, keeping its melody, words and timing.

* **Parameters:**
  * **audio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – A sung clip without accompaniment (a path, or WAV bytes); separate a
    song first (`voxy.separate.separate`).
  * **voice** ([`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A singing voice, a library voice, a path to a clip of the target
    voice, or None (the synthetic voice).
  * **transpose** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'fit_voice'`]]) – ‘fit_voice’ (default: move the melody into the voice’s register)
    or semitones (0: keep it).
  * **arguments** (*Other*) – as `sing`.
* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

### voxy.singing.get_singing_model(backend='soulx', , models=None)

The (cached) model of `backend`; `models` replaces the shared cache.

* **Return type:**
  [`SingingModel`](#voxy.singing.SingingModel)

### voxy.singing.register_singing_backend(backend, , replace=True)

Add a singing backend (`replace=False` keeps one already registered).

* **Return type:**
  [`SingingBackend`](#voxy.singing.SingingBackend)

### voxy.singing.resolve_singing_voice(voice, , backend='soulx', voices=None)

The singing voice `voice` names: a profile as is; a library name or alias
(its `backend` profile); a path to a sung clip (for conversion); None, the
synthetic voice.

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

### voxy.singing.sing(score, lyrics=None, , voice=None, backend='soulx', output_path=None, transpose=0, language='English', track=None, channel=None, model=None, models=None, voices=None, \*\*params)

Sing `score` with `lyrics` in `voice` and return the audio (`.save(path)`).

* **Parameters:**
  * **score** – A MIDI or MusicXML file, a music21 stream, or 

    ```
    ``
    ```

    voxy.score.Note\`\`s.
  * **lyrics** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – One token per note (`can-dle`: a word over two notes; `-`: hold
    the word); default: the score’s own lyrics.
  * **voice** ([`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A singing voice (`singing_voice`), a library voice saved for
    `backend`, or None (the synthetic voice).
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The singing backend (`singing_backends()`).
  * **output_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Also save the audio there.
  * **transpose** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'fit_voice'`]]) – Semitones to move the melody (0: as written), or ‘fit_voice’
    (into the voice’s register).
  * **language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Of the lyrics (SoulX here: English).
  * **track** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Which line of a MIDI file to sing, when it has several
    (`voxy.score.notes_from_midi`).
  * **channel** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Which line of a MIDI file to sing, when it has several
    (`voxy.score.notes_from_midi`).
  * **model** ([`SingingModel`](#voxy.singing.SingingModel) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A ready model (its backend is then the backend).
  * **models** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Model cache to use instead of the shared one.
  * **voices** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice library store (default `voxy.voices_store()`).
  * **\*\*params** – The backend’s parameters (`singing_backends()[b].params`).
* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

### voxy.singing.singing_backends()

Every singing backend, by name (the built-in ones registered first).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`SingingBackend`](#voxy.singing.SingingBackend)]

### voxy.singing.singing_voice(audio, score=None, lyrics=None, , backend='soulx', sample_rate=24000, name='singing-voices', rootdir=None, \*\*metadata)

A singing voice: a sung clip, and (for synthesis) the score it sings.

* **Parameters:**
  * **audio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – The clip, 5 to 30 s of one voice singing, without accompaniment: a
    path, or WAV bytes (kept in the samples store of voice `name`, under
    the data root `rootdir`, so the voice outlives the session).
  * **score** – What the clip sings, timed as in the clip (a MIDI or MusicXML file,
    a music21 stream, or notes; see `voxy.score.read_score`). Needed by
    `sing`; `convert_singing` needs only the clip.
  * **lyrics** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Its words, if the score has none.
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The backend the voice is for (its `model_type`).
  * **\*\*metadata** – Kept with the voice (e.g. `consent=`, `description=`).
* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

The profile’s `segment` is plain data (`prompt_audio` path, `prompt_notes`),
so it can be saved in the voice library: `voxy.save_voice(name, voice,
consent=...)`.

### voxy.singing.synthetic_singing_voice(preset='high', , backend='soulx')

A voice that belongs to no one: a formant synthesiser singing “la” on a short
phrase (`voxy.formant_singer`), rendered once into voxy’s cache (the file is
named by a hash of everything it is rendered from, so a change re-renders it).

`preset` is ‘high’ or ‘low’ (an octave down, darker vowels).

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)
