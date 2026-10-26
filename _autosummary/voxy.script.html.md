# voxy.script

Render a cast script: text segments, each read by its own voice, one audio file per section.

For anything narrated by more than one voice (an audiobook with a narrator and a quoted
guide, a dialogue, a radio play). Three plain-data inputs, each a JSON-able mapping so a
person can edit them and the rest follows:

- the **script**: a list of sections `{"id", "title"?, "segments": [{"role", "text"}]}`;
- the **casting**: `{"roles": {role: voice name or None}, "takes": {section id: take},
  "default_role"?}`; a role mapped to `None` has no voice yet (its sections are listed,
  not rendered);
- the **voices**: voice name -> `{"backend"?, "init"?, "params"?, "tempo"?, "effect"?,
  "external"?}` plus how to load it (`"library": name`, `"reference": {"audio", "text"}`
  or `"voice": id-or-name` passed to the backend as is). `tempo` (pitch kept, rubberband)
  and `effect` (an ffmpeg audio filter) are finishing steps; `external` marks a voice
  synthesised elsewhere (`missing_takes` lists what it owes, `add_take` files the audio).

Files under `root`:

> segments/<key>.wav|json   one take of one segment; the key hashes (voice, settings,
> : spoken text, take number), so a take is never overwritten
>   and an edit makes a new one

> sections/<id>.mp3         a section’s takes joined: pauses, tempo, effect, loudness
> : (EBU R128) evened across voices, a gap where the voice changes

> index.json                per section: status, segments (key, role, voice), file, seconds

Render again after editing: only segments whose key changed are synthesised, and only
sections whose takes or finishing changed are rebuilt. A retake of a section is
`"takes": {"s1": 2}` in the casting (a `seed` in the voice’s params moves with the take).

```pycon
>>> script = [{"id": "s1", "segments": [{"role": "guide", "text": "Come."}, {"role": "author", "text": "I come."}]}]
>>> casting = {"roles": {"guide": "g", "author": "a"}}
>>> [(s["role"], s["voice"]) for s in plan_script(script, casting)[0]["segments"]]
[('guide', 'g'), ('author', 'a')]
```

### Functions

| [`add_take`](#voxy.script.add_take)(root, segment, audio, \*[, ...])       | File one take: `segments/<key>.wav` (`audio`: WAV bytes) and its `.json`.                           |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [`finish_section`](#voxy.script.finish_section)(wavs, out, finish)               | Join takes (each `(wav path, voice spec)`) into one MP3 at `out`.                                   |
| [`lengthen_pauses`](#voxy.script.lengthen_pauses)(samples, sr, \*, tiers[, ...])  | `samples` (mono float array) with each inner pause lengthened by its tier's extra seconds.          |
| [`load_voice_profile`](#voxy.script.load_voice_profile)(spec, \*[, root])            | The voice to hand `synthesize` from a voices entry (see the module docstring).                      |
| [`missing_takes`](#voxy.script.missing_takes)(script, root, \*[, casting, ...]) | Segments with no take yet (by default only those of external voices), in script order.              |
| [`plan_script`](#voxy.script.plan_script)(script, casting, \*[, text_prep])   | Each section with its segments resolved: `{role, voice, text, tts_text, take}`.                     |
| [`render_script`](#voxy.script.render_script)(script, root, \*[, casting, ...]) | Render what is missing or changed, section by section in script order; return (and save) the index. |
| [`segment_key`](#voxy.script.segment_key)(segment, voice)                     | A take's identity: a hash of the voice (loading and settings), spoken text and take number.         |
| [`status`](#voxy.script.status)(root)                                    | Progress from `index.json`: counts per status, total `seconds`, `segments_by_voice`.                |

### voxy.script.add_take(root, segment, audio, , generation_seconds=None)

File one take: `segments/<key>.wav` (`audio`: WAV bytes) and its `.json`.

`segment` is an entry of `missing_takes` (or the renderer’s own), with `key` and
`voice_spec`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.script.finish_section(wavs, out, finish)

Join takes (each `(wav path, voice spec)`) into one MP3 at `out`.

Per take: pauses, then tempo, effect and loudness; a silent gap follows each take but
the last. Needs `ffmpeg` (with rubberband for `tempo`).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.script.lengthen_pauses(samples, sr, , tiers, floor_db=-35.0)

`samples` (mono float array) with each inner pause lengthened by its tier’s extra seconds.

A pause is a run of 10 ms frames `floor_db` below the loud frames (95th percentile);
it is lengthened by repeating its own middle, so the room tone carries on. Leading and
trailing silence is left as is.

```pycon
>>> tone = np.sin(np.arange(8000) / 3.0).astype(np.float32)
>>> y = np.concatenate([tone, np.zeros(4000, np.float32), tone])  # 0.5 s pause at 8 kHz
>>> len(lengthen_pauses(y, 8000, tiers=[[0.4, 0.25]])) - len(y)
2000
```

### voxy.script.load_voice_profile(spec, , root=None)

The voice to hand `synthesize` from a voices entry (see the module docstring).

### voxy.script.missing_takes(script, root, , casting=None, voices=None, text_prep=None, external_only=True)

Segments with no take yet (by default only those of external voices), in script order.

Each is `{section, role, voice, text, tts_text, take, key, voice_spec}`: hand it to
whatever renders that voice, then file the audio with `add_take(root, segment, wav)`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.script.plan_script(script, casting, , text_prep=None)

Each section with its segments resolved: `{role, voice, text, tts_text, take}`.

`tts_text` is `text_prep(text)` (what is spoken; default: the text itself), so the
take’s key follows what the voice says, not the displayed text.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.script.render_script(script, root, \*, casting=None, voices=None, only=None, finish=None, text_prep=None, model_factory=None, voice_loader=<function load_voice_profile>, log=<built-in function print>)

Render what is missing or changed, section by section in script order; return (and save) the index.

* **Parameters:**
  * **script** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – Sections `{"id", "title"?, "segments": [{"role", "text"}]}`.
  * **root** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)) – The output folder; `casting` and `voices` default to its
    `casting.json` and `voices.json`.
  * **only** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Section ids to render (default all); the index still lists every section.
  * **finish** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Overrides of `DFLT_FINISH` (gap between voices, loudness, bitrate, pauses).
  * **text_prep** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `text -> spoken text` per segment (e.g. `voxy.speechtext` preparation).
  * **model_factory** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(backend, **init) -> SpeechModel` (default `voxy.create_speech_model`).
  * **voice_loader** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – `(voice spec, root=) -> voice` (default `load_voice_profile`).
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.script.segment_key(segment, voice)

A take’s identity: a hash of the voice (loading and settings), spoken text and take number.

`tempo` and `effect` are finishing steps, left out: changing them rebuilds the
section files without new takes.

```pycon
>>> a = segment_key({"tts_text": "Hi", "take": 1}, {"params": {"seed": 1}})
>>> a == segment_key({"tts_text": "Hi", "take": 1}, {"params": {"seed": 1}, "tempo": 0.8})
True
>>> a == segment_key({"tts_text": "Hi", "take": 2}, {"params": {"seed": 1}})
False
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.script.status(root)

Progress from `index.json`: counts per status, total `seconds`, `segments_by_voice`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
