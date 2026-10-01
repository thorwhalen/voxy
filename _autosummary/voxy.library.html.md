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

| [`clone_from_samples`](#voxy.library.clone_from_samples)(name, \*[, model, ...])   | Clone the voice `name` from its stored samples and save the profile.                 |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| [`load_voice`](#voxy.library.load_voice)(name, \*[, model_type, voices])   | The saved `model_type` profile of the voice called `name`.                           |
| [`profile_to_dict`](#voxy.library.profile_to_dict)(profile)                     | A JSON-ready dict of `profile`; refuses segments that aren't plain data.             |
| [`save_voice`](#voxy.library.save_voice)(name, profile, \*[, voices])      | Save `profile` under `name` (merged into any existing record) and return the record. |

### voxy.library.clone_from_samples(name, , model=None, model_type='elevenlabs', samples=None, voices=None, record_fields=None, \*\*clone_kwargs)

Clone the voice `name` from its stored samples and save the profile.

* **Parameters:**
  * **name** (`str`) – Voice name: the key in the samples and voices stores.
  * **model** ([`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel) | `None`) – A speech model; defaults to `create_speech_model(model_type)`.
  * **model_type** (`str`) – Backend used when `model` is not given.
  * **samples** (`Mapping`[`str`, `bytes`] | `None`) – `filename -> bytes`; defaults to `samples_store(name)`. Only
    its top-level audio files are uploaded.
  * **voices** (`MutableMapping` | `None`) – Voice records store; defaults to `voices_store()`.
  * **record_fields** (`Mapping`[`str`, `Any`] | `None`) – Extra fields for the voice record (aliases, consent…).
  * **\*\*clone_kwargs** (`Any`) – Passed to `model.clone_voice` (e.g. `labels=`,
    `remove_background_noise=`). `name=` defaults to `name`.
* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

### voxy.library.load_voice(name, , model_type='elevenlabs', voices=None)

The saved `model_type` profile of the voice called `name`.

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

### voxy.library.profile_to_dict(profile)

A JSON-ready dict of `profile`; refuses segments that aren’t plain data.

* **Return type:**
  `dict`

### voxy.library.save_voice(name, profile, , voices=None, \*\*record_fields)

Save `profile` under `name` (merged into any existing record) and return the record.

`record_fields` (e.g. `aliases=`, `description=`, `consent=`) are set on
the record itself.

* **Return type:**
  `dict`
