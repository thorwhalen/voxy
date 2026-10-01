# voxy.aix_model

Speech through aix (LiteLLM): OpenAI’s TTS voices and the other providers LiteLLM routes.

The provider is picked by the `model` string (e.g. `"gpt-4o-mini-tts"`,
`"tts-1-hd"`); keys and defaults come from aix’s own config. aix is imported
lazily (`pip install 'voxy[aix]'`).

```pycon
>>> AixSpeechModel(tts=lambda text, **kw: None).name
'aix'
```

### Module Attributes

| [`OPENAI_TTS_VOICES`](#voxy.aix_model.OPENAI_TTS_VOICES)   | OpenAI's built-in TTS voices (which ones a model accepts depends on the model).   |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------|

### Classes

| [`AixSpeechModel`](#voxy.aix_model.AixSpeechModel)(\*[, model, response_format, tts])   | aix/LiteLLM text-to-speech as a voxy backend.   |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------|

### *class* voxy.aix_model.AixSpeechModel(, model=None, response_format='mp3', tts=None)

Bases: [`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel)

aix/LiteLLM text-to-speech as a voxy backend.

* **Parameters:**
  * **model** (`str` | `None`) – TTS model (None: aix’s configured default).
  * **response_format** (`str`) – ‘mp3’, ‘opus’, ‘aac’, ‘flac’, ‘wav’…
  * **tts** (`Callable` | `None`) – `(text, **kw) -> GeneratedAudio` (tests inject a fake).

#### list_voices()

OpenAI’s built-in voices (other LiteLLM providers have their own).

* **Return type:**
  `list`[[`VoiceInfo`](voxy.base.html.md#voxy.base.VoiceInfo)]

#### name *: str* *= 'aix'*

Registry name of the backend (also each profile’s `model_type`).

#### synthesize(text, voice=None, \*\*kwargs)

Speech from aix; extra kwargs (`speed=`, `api_key=`…) go to aix.

* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

### voxy.aix_model.OPENAI_TTS_VOICES *= ('alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer', 'verse')*

OpenAI’s built-in TTS voices (which ones a model accepts depends on the model).
