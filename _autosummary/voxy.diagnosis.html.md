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

### Module Attributes

| [`BACKEND_PREFIXES`](#voxy.diagnosis.BACKEND_PREFIXES)   | Short, readable ID prefixes (other backends use their first two letters).   |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------|

### Functions

| [`build_page`](#voxy.diagnosis.build_page)(results, out_dir, \*[, title, ...])   | Write `out_dir/index.html`: one card per variant, grouped by backend.                 |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [`expand_grid`](#voxy.diagnosis.expand_grid)(grid, \*[, references])              | Variants from grid rows: list values are swept (cartesian product), scalars fixed.    |
| [`page_id`](#voxy.diagnosis.page_id)(results, title)                          | A short ID for the page (its notes are stored under it).                              |
| [`references_from_spec`](#voxy.diagnosis.references_from_spec)(spec, out_dir, \*[, log])   | `(voices, shown)`: reference voices for rendering, and their page entries.            |
| [`render_grid`](#voxy.diagnosis.render_grid)(text, references, variants, ...)     | Render every variant to `out_dir/audio/<id>.<format>`; return (and save) the results. |
| [`run_spec`](#voxy.diagnosis.run_spec)(spec, out_dir, \*[, log])               | Render a spec's grid and build its page; return the page path.                        |

### Classes

| [`Variant`](#voxy.diagnosis.Variant)(backend[, params, init, reference, ...])   | One configuration to render: a backend, its model options and parameters.   |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|

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

### voxy.diagnosis.build_page(results, out_dir, , title='Voice diagnosis', text='', references=None, intro='')

Write `out_dir/index.html`: one card per variant, grouped by backend.

`references`: name -> `{"audio": relative path, "text": ..., ...}` shown
at the top so listeners can compare with the real voice.

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
`audio_format`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
