# voxy.facade

Voice generation in one call, whatever the service behind it.

```pycon
>>> import voxy
>>> voxy.text_to_speech("Hello!", voice="ada").save("hi.mp3")
>>> voxy.text_to_speech("Hello!", voice="Daniel", backend="say")
>>> voxy.list_voices()               # the library: our named voices
>>> voxy.list_voices("elevenlabs")   # a backend's own voices
>>> voxy.voice_id("addie")            # 'xY12...' (aliases work)
```

How `voice` is understood, first match wins:

1. a `VoiceProfile`: used as is, on its own backend;
2. a name or alias in the voice library (`voxy.voices_store()`): that voice’s
   saved profile for `backend` (or, if `backend` is not given, its first one);
3. anything else: the backend’s own voice id or name (e.g. ‘nova’, ‘Daniel’);
4. `None`: the backend’s default voice, if it has one.

Pass `use_library=False` to reach a backend voice whose name is also a library
name or alias. With no `backend`, a library voice uses its profile for the
default backend if it has one, else its `default_backend`, else its first.

Backends are entries of `voxy.speech_model_factories`; add one with
`voxy.register_speech_model`. The default backend is `$VOXY_TTS_BACKEND`
(read at call time), else ‘elevenlabs’.

### Functions

| [`clear_speech_models`](#voxy.facade.clear_speech_models)([models])                    | Forget cached models (e.g. after changing keys or settings).                    |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| [`dflt_tts_backend`](#voxy.facade.dflt_tts_backend)()                               | `$VOXY_TTS_BACKEND`, else 'elevenlabs' (read on every call).                    |
| [`get_speech_model`](#voxy.facade.get_speech_model)([backend, models])              | The (cached) model for `backend`; `models` replaces the shared cache.           |
| [`list_voices`](#voxy.facade.list_voices)([backend, voices, model, models])    | Our named voices (`backend=None`), or the voices a backend offers.              |
| [`resolve_voice`](#voxy.facade.resolve_voice)(voice, \*[, backend, voices, ...]) | `(backend, voice)` to synthesize with (see the module docstring for the rules). |
| [`text_to_speech`](#voxy.facade.text_to_speech)(text[, voice, backend, ...])      | Speak `text` in `voice` and return the encoded audio (`.save(path)`).           |
| [`voice_id`](#voxy.facade.voice_id)(name, \*[, backend, voices])            | The provider's id for library voice `name` (for code that calls a provider).    |

### voxy.facade.clear_speech_models(models=None)

Forget cached models (e.g. after changing keys or settings).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.facade.dflt_tts_backend()

`$VOXY_TTS_BACKEND`, else ‘elevenlabs’ (read on every call).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.facade.get_speech_model(backend=None, , models=None)

The (cached) model for `backend`; `models` replaces the shared cache.

* **Return type:**
  [`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel)

### voxy.facade.list_voices(backend=None, , voices=None, model=None, models=None, \*\*kwargs)

Our named voices (`backend=None`), or the voices a backend offers.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`VoiceInfo`](voxy.base.html.md#voxy.base.VoiceInfo)]

```pycon
>>> lib = {"ada": {"name": "ada", "aliases": ["Addie"], "description": "d",
...                 "profiles": {"elevenlabs": {"segment": "v1"}}}}
>>> [(v.name, v.labels["backends"]) for v in list_voices(voices=lib)]
[('ada', ['elevenlabs'])]
```

### voxy.facade.resolve_voice(voice, , backend=None, voices=None, use_library=True)

`(backend, voice)` to synthesize with (see the module docstring for the rules).

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)]

```pycon
>>> lib = {"ada": {"name": "ada", "aliases": ["Addie"], "profiles": {
...     "elevenlabs": {"segment": "v1", "speaker_id": 1, "model_type": "elevenlabs",
...                    "sample_rate": 24000}}}}
>>> b, v = resolve_voice("addie", voices=lib)
>>> b, v.segment
('elevenlabs', 'v1')
>>> resolve_voice("Daniel", backend="say", voices=lib)
('say', 'Daniel')
```

### voxy.facade.text_to_speech(text, voice=None, , backend=None, output_path=None, voices=None, model=None, models=None, use_library=True, \*\*kwargs)

Speak `text` in `voice` and return the encoded audio (`.save(path)`).

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – What to say.
  * **voice** ([`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A library name or alias (‘ada’, ‘Addie’), a backend’s own voice
    (‘nova’, ‘Daniel’, an ElevenLabs id), a `VoiceProfile`, or None.
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Service to use (‘elevenlabs’, ‘say’, ‘aix’, ‘fal’, ‘csm’, or
    any registered). Inferred from library voices.
  * **output_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Also save the audio there.
  * **voices** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice library store (default `voxy.voices_store()`).
  * **model** ([`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A ready model to use (its backend is then the backend).
  * **models** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Model cache to use instead of the shared one.
  * **use_library** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Look `voice` up in the library first (False: always
    the backend’s own voice of that name).
  * **\*\*kwargs** – Backend-specific options (e.g. `output_format=` for
    ElevenLabs, `speed=` for aix, `quality=` for fal).
* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

### voxy.facade.voice_id(name, , backend='elevenlabs', voices=None)

The provider’s id for library voice `name` (for code that calls a provider).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> lib = {"ada": {"name": "ada", "aliases": ["Addie"], "profiles": {
...     "elevenlabs": {"segment": "v1", "speaker_id": 1, "model_type": "elevenlabs",
...                    "sample_rate": 24000}}}}
>>> voice_id("Addie", voices=lib)
'v1'
```
