# voxy.diagnosis

Diagnosis pages: render one text through a grid of (model x parameters), listen side by side.

```pycon
>>> from voxy import diagnosis
>>> variants = diagnosis.expand_grid([
...     {"backend": "f5tts", "params": {"nfe_step": [16, 32], "cfg_strength": 2}},
...     {"backend": "chatterbox", "params": {"exaggeration": [0.3, 0.6]}},
... ])
>>> results = diagnosis.render_grid(text, references, variants, "out/")
>>> diagnosis.build_page(results, "out/", title="First listen")
```

Or from a spec file: `python -m voxy diagnose spec.json --out DIR --publish`.

Every variant has a stable ID (`F5-7KQ2`) computed from its backend, model
options, parameters and reference clip, never from the text, so “use F5-7KQ2”
means the same configuration on every page and every chapter. The page shows,
per variant, the ID, an audio player, the parameters (those that differ from the
model’s defaults in bold), generation time and real-time factor, and lets each
listener rate it (1-5) and write a note. Ratings stay in the listener’s browser;
“Copy my notes” turns them into a text block to paste into a message.

For a page shared with other listeners: `notes_sink` saves every rating and
note as it is made (to an annoquest server, whose owner page then lists them per
listener), `appendix` folds the variants of some references into a collapsed
section, and `imported` adds clips rendered elsewhere (a cloud service).

### Module Attributes

| [`BACKEND_PREFIXES`](#voxy.diagnosis.BACKEND_PREFIXES)   | Short, readable ID prefixes (other backends use their first two letters).   |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`BACKEND_NAMES`](#voxy.diagnosis.BACKEND_NAMES)      | Display names for page sections and default variant labels.                 |

### Functions

| [`annoquest_request`](#voxy.diagnosis.annoquest_request)(results, \*, request_id, ...)   | The annoquest request a `notes_sink` page reports to: one item per variant (a 1-5 rating and a note), so the owner page reads the saved notes per listener.   |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`backend_name`](#voxy.diagnosis.backend_name)(backend)                             | The display name of a backend (its key when it has none).                                                                                                     |
| [`build_page`](#voxy.diagnosis.build_page)(results, out_dir, \*[, title, ...])    | Write `out_dir/index.html`: one card per variant, grouped by backend.                                                                                         |
| [`expand_grid`](#voxy.diagnosis.expand_grid)(grid, \*[, references])               | Variants from grid rows: list values are swept (cartesian product), scalars fixed.                                                                            |
| [`import_results`](#voxy.diagnosis.import_results)(entries, out_dir)                  | Results for clips rendered elsewhere (a cloud service, another machine).                                                                                      |
| [`page_id`](#voxy.diagnosis.page_id)(results, title)                           | A short ID for the page (its notes are stored under it).                                                                                                      |
| [`references_from_spec`](#voxy.diagnosis.references_from_spec)(spec, out_dir, \*[, log])    | `(voices, shown)`: reference voices for rendering, and their page entries.                                                                                    |
| [`render_grid`](#voxy.diagnosis.render_grid)(text, references, variants, ...)      | Render every variant to `out_dir/audio/<id>.<format>`; return (and save) the results.                                                                         |
| [`run_spec`](#voxy.diagnosis.run_spec)(spec, out_dir, \*[, log])                | Render a spec's grid and build its page; return the page path.                                                                                                |
| [`variant_labels`](#voxy.diagnosis.variant_labels)(results)                           | Readable names: the backend's name, then the values of the settings that vary between that backend's variants.                                                |

### Classes

| [`Variant`](#voxy.diagnosis.Variant)(backend[, params, init, reference, ...])   | One configuration to render: a backend, its model options and parameters.   |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|

### voxy.diagnosis.BACKEND_NAMES *= {'chatterbox': 'Chatterbox', 'elevenlabs': 'ElevenLabs', 'f5tts': 'F5-TTS', 'qwen3tts': 'Qwen3-TTS', 'voxcpm': 'VoxCPM2', 'xtts': 'XTTS-v2'}*

Display names for page sections and default variant labels.

### voxy.diagnosis.BACKEND_PREFIXES *= {'chatterbox': 'CB', 'elevenlabs': 'EL', 'f5tts': 'F5', 'qwen3tts': 'QW', 'voxcpm': 'VX', 'xtts': 'XT'}*

Short, readable ID prefixes (other backends use their first two letters).

### *class* voxy.diagnosis.Variant(backend, params=dict(), init=dict(), reference='', label='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One configuration to render: a backend, its model options and parameters.

#### *property* id *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Stable ID from backend, init, params and reference (not the text).

```pycon
>>> Variant("f5tts", {"nfe_step": 32}, reference="r1").id
'F5-...'
>>> Variant("f5tts", {"nfe_step": 32}).id == Variant("f5tts", {"nfe_step": 32}).id
True
```

### voxy.diagnosis.annoquest_request(results, , request_id, page_url, title, intro='', api_url='/api/annoquest', appendix=None, prompt='How much does this sound like the real voice, and how natural is it?')

The annoquest request a `notes_sink` page reports to: one item per variant
(a 1-5 rating and a note), so the owner page reads the saved notes per listener.
Register it on the server with `python -m annoquest_server put-request <file>`.

```pycon
>>> r = annoquest_request([{"id": "CB-AAAA", "backend": "chatterbox"}],
...     request_id="abcdefgh12", page_url="https://example.org/p/", title="T")
>>> [(i["id"], i["title"], i["response"]) for i in r["items"]]
[('CB-AAAA', 'Chatterbox', 'stars')]
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.diagnosis.backend_name(backend)

The display name of a backend (its key when it has none).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.diagnosis.build_page(results, out_dir, , title='Voice diagnosis', text='', references=None, intro='', appendix=None, backend_notes=None, notes_sink=None, noindex=False)

Write `out_dir/index.html`: one card per variant, grouped by backend.

* **Parameters:**
  * **references** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – name -> `{"audio": relative path, "text": ..., "label"?, "processing"?, ...}`
    shown so listeners can compare with the real voice.
  * **appendix** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – reference name -> a note: the variants made from those references
    (and the references themselves) go in a collapsed section at the end,
    under the note.
  * **backend_notes** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – backend -> a note shown under its section heading (default:
    the local backend’s notes and licence).
  * **notes_sink** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `{"kind": "annoquest", "url": "/api/annoquest", "request": id}`
    saves ratings, notes, the listener’s name and clip plays as they happen
    (the page says so). Without it they stay in the listener’s browser.
  * **noindex** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Ask search engines not to index the page (for a page shared by link).
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.diagnosis.expand_grid(grid, , references=('',))

Variants from grid rows: list values are swept (cartesian product), scalars fixed.

A row: `{"backend": ..., "params": {...}, "init": {...}, "label": ...}`.
`init` values may be lists too (e.g. `{"variant": ["original", "turbo"]}`).
Every variant is rendered once per reference name.

```pycon
>>> vs = expand_grid([{"backend": "f5tts", "params": {"nfe_step": [16, 32], "speed": 1}}])
>>> [(v.backend, v.params) for v in vs]
[('f5tts', {'nfe_step': 16, 'speed': 1}), ('f5tts', {'nfe_step': 32, 'speed': 1})]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Variant`](#voxy.diagnosis.Variant)]

### voxy.diagnosis.import_results(entries, out_dir)

Results for clips rendered elsewhere (a cloud service, another machine).

Each entry: `backend`, `audio_file`, and optionally `params`, `init`,
`reference`, `label`, `generation_seconds`. The audio is copied next to the
page and the entry gets the same ID scheme as rendered variants.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.diagnosis.page_id(results, title)

A short ID for the page (its notes are stored under it).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.diagnosis.references_from_spec(spec, out_dir, \*, log=<built-in function print>)

`(voices, shown)`: reference voices for rendering, and their page entries.

The spec names references directly (`"references": {name: {ref_audio, ref_text}}`)
or picks the best clips of a prepared voice (`"voice": name, "n_references": k`).
Reference audio is copied next to the page so listeners can play it.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.diagnosis.render_grid(text, references, variants, out_dir, \*, model_factory=None, audio_format='mp3', resume=True, log=<built-in function print>)

Render every variant to `out_dir/audio/<id>.<format>`; return (and save) the results.

Models are built one backend configuration at a time and closed before the
next, so only one model is in memory. With `resume`, variants already in
`results.json` (without error) are not rendered again.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – What every variant says.
  * **references** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Reference name -> voice (a `VoiceProfile` or `{"ref_audio", "ref_text"}`).
  * **variants** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Variant`](#voxy.diagnosis.Variant)]) – From `expand_grid` (their `reference` names key `references`).
  * **model_factory** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(backend, **init) -> SpeechModel` (default `create_speech_model`).
  * **audio_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ‘mp3’ (default, small) or ‘wav’.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.diagnosis.run_spec(spec, out_dir, \*, log=<built-in function print>)

Render a spec’s grid and build its page; return the page path.

Spec keys: `title`, `text` (or `text_file`), `references` or `voice`
(+ `n_references`), `grid` (rows for `expand_grid`), optional `intro`,
`audio_format`, `imported` (see `import_results`), `appendix`,
`backend_notes`, `noindex` and `notes_sink` (see `build_page`). With an
annoquest `notes_sink` and a `page_url` (where the page will be served), the
request to register on that server is written next to the page.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.diagnosis.variant_labels(results)

Readable names: the backend’s name, then the values of the settings that vary
between that backend’s variants. A result’s own `label` wins.

```pycon
>>> rs = [
...     {"id": "a", "backend": "chatterbox", "params": {"language": "fr", "exaggeration": 0.3}},
...     {"id": "b", "backend": "chatterbox", "params": {"language": "fr", "exaggeration": 0.5}},
...     {"id": "c", "backend": "voxcpm", "params": {"cfg_value": 2.0}},
...     {"id": "d", "backend": "elevenlabs", "label": "ElevenLabs v3"},
... ]
>>> variant_labels(rs)
{'a': 'Chatterbox 0.3', 'b': 'Chatterbox 0.5', 'c': 'VoxCPM2', 'd': 'ElevenLabs v3'}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
