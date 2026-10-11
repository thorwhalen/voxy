# voxy.voicestudio_model

Local description-prompted speech through VoiceStudio (Parler-TTS and friends).

A voice here is a *text description* (“a warm, low female voice, close to the
microphone”), not a recording: `voice` may be that string or a `VoiceProfile`
made by `design_voice`. Nothing is uploaded; weights run locally (CPU works,
slowly) and are fetched from Hugging Face on first use (`model_id=`; the default
is several GB, so `pip install 'voxy[voicestudio]'` does not download them).

The model call is a seam: `generate(text, description) -> (waveform, sample_rate)`
is injectable, so tests (and other VoiceStudio architectures) never load weights.

```pycon
>>> model = VoiceStudioSpeechModel(generate=lambda text, description: ([0.0] * 8, 8000))
>>> model.synthesize("Hi", "a calm voice").format
'wav'
```

### Classes

| [`VoiceStudioSpeechModel`](#voxy.voicestudio_model.VoiceStudioSpeechModel)(\*[, model_id, ...])   | VoiceStudio (description-prompted TTS) as a voxy backend.   |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------|

### *class* voxy.voicestudio_model.VoiceStudioSpeechModel(, model_id='parler-tts/parler-tts-mini-v1', device='cpu', generate=None)

Bases: [`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel)

VoiceStudio (description-prompted TTS) as a voxy backend.

* **Parameters:**
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Hugging Face id of the base model (default Parler-TTS mini).
  * **device** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Torch device for the default `generate`.
  * **generate** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(text, description) -> (waveform, sample_rate)` (tests inject a fake).

#### design_voice(description, , speaker_id=0, \*\*\_)

A profile for a voice described in words (no model call, nothing stored remotely).

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

#### dflt_voice *: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)* *= 'A clear, natural voice speaking at a moderate pace with very clean audio.'*

the caller must).

* **Type:**
  Voice used when the caller names none (`None`

#### name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)* *= 'voicestudio'*

Registry name of the backend (also each profile’s `model_type`).

#### synthesize(text, voice=None, \*\*kwargs)

Speech in the voice described by `voice` (a description or a profile).

* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)
