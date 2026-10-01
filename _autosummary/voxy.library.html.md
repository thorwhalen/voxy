# voxy.library

A persistent voice library: clone a voice once, then use it by name.

A voice record (one per person, in `voxy.stores.voices_store`) holds what voxy
knows about a voice, plus one saved profile per backend:

> {“name”: “ada”, “aliases”: […], “consent”: “…”,
> : “profiles”: {“elevenlabs”: {“segment”: “<voice_id>”, “model_type”: “elevenlabs”, …}}}

Only profiles whose `segment` is plain data (an id, as for ElevenLabs) can be
saved; a local model’s in-memory segment (CSM) cannot.

```pycon
>>> import tempfile
>>> from voxy.base import VoiceProfile
>>> voices = voices_store(rootdir=tempfile.mkdtemp())
>>> _ = save_voice("ada", VoiceProfile("v1", 999, "elevenlabs", 24000), voices=voices,
...                aliases=["Ada"])
>>> load_voice("ada", voices=voices).segment
'v1'
>>> voices["ada"]["aliases"]
['Ada']
```

### Functions

| [`clone_from_samples`](#voxy.library.clone_from_samples)(name, \*[, model, ...])        | Clone the voice `name` from its stored samples and save the profile.                 |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [`design_from_description`](#voxy.library.design_from_description)(name, description, \*)    | Design a new voice from a text `description` and save it as `name`.                  |
| [`find_voice`](#voxy.library.find_voice)(name, \*[, voices])                    | The library key of the voice called `name` (case-insensitive, aliases too).          |
| [`load_voice`](#voxy.library.load_voice)(name, \*[, model_type, voices])        | The saved `model_type` profile of the voice called `name` (or an alias).             |
| [`profile_from_record`](#voxy.library.profile_from_record)(record[, model_type, prefer]) | The record's saved `model_type` profile.                                             |
| [`profile_to_dict`](#voxy.library.profile_to_dict)(profile)                          | A JSON-ready dict of `profile`; refuses segments that aren't plain data.             |
| [`save_voice`](#voxy.library.save_voice)(name, profile, \*[, voices])           | Save `profile` under `name` (merged into any existing record) and return the record. |

### voxy.library.clone_from_samples(name, , model=None, model_type='elevenlabs', samples=None, voices=None, record_fields=None, overwrite=False, \*\*clone_kwargs)

Clone the voice `name` from its stored samples and save the profile.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Voice name: the key in the samples and voices stores.
  * **model** ([`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A speech model; defaults to `create_speech_model(model_type)`.
  * **model_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Backend used when `model` is not given.
  * **samples** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `filename -> bytes`; defaults to `samples_store(name)`. Only
    its top-level audio files are uploaded.
  * **voices** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice records store; defaults to `voices_store()`.
  * **record_fields** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Extra fields for the voice record (aliases, consent…).
  * **overwrite** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Replace an existing profile of this backend for `name`
    (otherwise refused before anything is uploaded).
  * **\*\*clone_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Passed to `model.clone_voice` (e.g. `labels=`,
    `remove_background_noise=`). `name=` defaults to `name`.
* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

### voxy.library.design_from_description(name, description, , preview=None, model=None, model_type='elevenlabs', voices=None, record_fields=None, overwrite=False, \*\*design_kwargs)

Design a new voice from a text `description` and save it as `name`.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Library name for the voice.
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – What it sounds like (age, accent, tone, pace, character).
  * **preview** – The chosen preview (or its id) from the model’s
    `design_voice_previews`; if omitted, the first generated one.
  * **model** ([`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A speech model that can design voices (default: `model_type`’s).
  * **record_fields** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Extra fields for the voice record (aliases…).
  * **overwrite** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Replace an existing profile of this backend for `name`
    (otherwise refused before any paid call).
  * **\*\*design_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Passed to `model.design_voice` (`labels=`, `seed=`…).
* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

### voxy.library.find_voice(name, , voices=None)

The library key of the voice called `name` (case-insensitive, aliases too).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> lib = {"ada": {"aliases": ["Ada", "Addie"]}, "grace": {"aliases": ["Gracie"]}}
>>> find_voice("addie", voices=lib), find_voice("Grace", voices=lib), find_voice("x", voices=lib)
('ada', 'grace', None)
```

### voxy.library.load_voice(name, , model_type='elevenlabs', voices=None)

The saved `model_type` profile of the voice called `name` (or an alias).

`model_type=None` takes the voice’s first saved profile.

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

### voxy.library.profile_from_record(record, model_type=None, , prefer=None)

The record’s saved `model_type` profile.

With `model_type=None`: the `prefer` profile if the record has one, else
the record’s `default_backend`, else its first.

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

### voxy.library.profile_to_dict(profile)

A JSON-ready dict of `profile`; refuses segments that aren’t plain data.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.library.save_voice(name, profile, , voices=None, \*\*record_fields)

Save `profile` under `name` (merged into any existing record) and return the record.

`record_fields` (e.g. `aliases=`, `description=`, `consent=`) are set on
the record itself. Aliases must not already name another voice.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
