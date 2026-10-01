# voxy.fal_model

Speech through fal.ai, via falaw: many hosted TTS models, chosen by quality tier.

falaw picks a model for `quality` (‘fast’, ‘balanced’, ‘best’…) unless
`model_id` is given; `voice` means whatever that model calls a voice. falaw is
imported lazily (`pip install 'voxy[fal]'`).

```pycon
>>> FalSpeechModel(tts=lambda text, **kw: None, fetch=lambda url: b"").name
'fal'
```

### Classes

| [`FalSpeechModel`](#voxy.fal_model.FalSpeechModel)(\*[, quality, model_id, tts, ...])   | fal.ai text-to-speech (through falaw) as a voxy backend.   |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------|

### *class* voxy.fal_model.FalSpeechModel(, quality='balanced', model_id=None, tts=None, fetch=None)

Bases: [`SpeechModel`](voxy.base.md#voxy.base.SpeechModel)

fal.ai text-to-speech (through falaw) as a voxy backend.

* **Parameters:**
  * **quality** (`str`) – falaw quality tier used to pick a model.
  * **model_id** (`str` | `None`) – A specific fal model (overrides `quality`).
  * **tts** (`Callable` | `None`) – `(text, **kw) -> falaw.Result` (tests inject a fake).
  * **fetch** (`Callable`[[`str`], `bytes`] | `None`) – `url -> bytes` to download the result.

#### name *: str* *= 'fal'*

Registry name of the backend (also each profile’s `model_type`).

#### synthesize(text, voice=None, \*\*kwargs)

Speech from fal; other keyword arguments go to the model (falaw’s `extra`).

* **Return type:**
  [`Speech`](voxy.base.md#voxy.base.Speech)
