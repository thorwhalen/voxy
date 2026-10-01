# voxy.base

Voxy: A flexible speech synthesis and voice cloning module.

This module provides a plugin architecture for working with different speech synthesis
models, with initial support for the CSM-1B model.

### Module Attributes

| [`speech_model_factories`](#voxy.base.speech_model_factories)   | Backend name -> factory (keys lowercase).                                     |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`backend_aliases`](#voxy.base.backend_aliases)          | Other names for a backend -> its registry name (one model, one profile type). |

### Functions

| [`audio_to_text`](#voxy.base.audio_to_text)(audio_input[, model_size, ...])   | Transcribe audio to text using Whisper.                                |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`canonical_backend`](#voxy.base.canonical_backend)(name)                         | The registry name of backend `name` (lowercased, aliases resolved).    |
| [`cleanup_audio`](#voxy.base.cleanup_audio)(audio, sample_rate[, ...])        | Clean up audio by normalizing volume and removing silence.             |
| [`create_speech_model`](#voxy.base.create_speech_model)([model_type])               | Create a speech model of the specified type.                           |
| [`register_speech_model`](#voxy.base.register_speech_model)(name, factory, \*[, ...]) | Register a backend (a class or `**kwargs -> SpeechModel` callable).    |
| [`tensor_to_wav_bytes`](#voxy.base.tensor_to_wav_bytes)(audio, sample_rate)         | Encode a tensor ([channels, samples] or [samples]) as mono 16-bit WAV. |

### Classes

| [`CSMSpeechModel`](#voxy.base.CSMSpeechModel)([model_path, device])          | Speech model implementation using Sesame's CSM-1B model.   |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`Speech`](#voxy.base.Speech)(audio, format[, backend, voice, ...])  | Synthesized speech: encoded audio plus what produced it.   |
| [`SpeechModel`](#voxy.base.SpeechModel)([device])                         | Base class for speech backends (local models or services). |
| [`VoiceInfo`](#voxy.base.VoiceInfo)(voice_id, name, backend[, ...])     | A voice a backend offers (stock, designed, or cloned).     |
| [`VoiceProfile`](#voxy.base.VoiceProfile)(segment, speaker_id, ...[, ...]) | Data class to store voice cloning information.             |

### *class* voxy.base.CSMSpeechModel(model_path=None, device='cpu')

Bases: [`SpeechModel`](#voxy.base.SpeechModel)

Speech model implementation using Sesame’s CSM-1B model.

#### clone_voice(audio_input, transcript=None, speaker_id=999, \*, cleanup_audio_fn=<function cleanup_audio>)

Create a voice profile from an audio sample and its transcript.

* **Parameters:**
  * **audio_input** (`str` | `bytes` | `BinaryIO` | `Tensor` | `ndarray`) – Audio in various formats
  * **transcript** (`str` | `None`) – Text transcription of the audio (if None, auto-transcribed)
  * **speaker_id** (`int`) – Unique ID for this voice
  * **cleanup_audio_fn** (`Callable` | `None`) – Function to clean up audio (None to skip)
* **Returns:**
  A packaged voice profile
* **Return type:**
  [`VoiceProfile`](#voxy.base.VoiceProfile)

#### generate_speech(text, voice_profile=None, output_path=None, max_length_ms=10000, temperature=0.7, topk=30)

Generate speech using a voice profile.

* **Parameters:**
  * **text** (`str` | `bytes` | `TextIOBase`) – Text to synthesize
  * **voice_profile** ([`VoiceProfile`](#voxy.base.VoiceProfile) | `None`) – Voice profile from clone_voice()
  * **output_path** (`str` | `None`) – Path to save the audio (optional)
  * **max_length_ms** (`int`) – Maximum audio length in milliseconds
  * **temperature** (`float`) – Sampling temperature (lower = more deterministic)
  * **topk** (`int`) – Top-k sampling parameter
* **Return type:**
  `Tensor`
* **Returns:**
  Generated audio tensor

#### name *: str* *= 'csm'*

Registry name of the backend (also each profile’s `model_type`).

#### *property* sample_rate *: int*

Sample rate of the generated audio (loads the model).

### *class* voxy.base.Speech(audio, format, backend='', voice=None, sample_rate=None, text=None)

Bases: `object`

Synthesized speech: encoded audio plus what produced it.

```pycon
>>> import tempfile, os
>>> speech = Speech(b"RIFF...", format="wav", backend="say", voice="Daniel")
>>> path = speech.save(os.path.join(tempfile.mkdtemp(), "hi.wav"))
>>> open(path, "rb").read()[:4]
b'RIFF'
```

#### save(path)

Write the audio to `path` (folders created) and return the path.

* **Return type:**
  `str`

### *class* voxy.base.SpeechModel(device='cpu')

Bases: `object`

Base class for speech backends (local models or services).

A backend implements whichever capabilities it has; the rest raise
`NotImplementedError` naming the backend:

- `synthesize(text, voice) -> Speech`: the facade’s one required method
  (the default renders `generate_speech` to WAV);
- `list_voices() -> list[VoiceInfo]`;
- `clone_voice(samples, ...) -> VoiceProfile`;
- `design_voice(description, ...) -> VoiceProfile`;
- `generate_speech(text, profile) -> torch.Tensor`.

#### clone_voice(audio_input, transcript=None, speaker_id=999, \*, cleanup_audio_fn=<function cleanup_audio>)

Create a voice profile from an audio sample and its transcript.

* **Parameters:**
  * **audio_input** (`str` | `bytes` | `BinaryIO` | `Tensor` | `ndarray`) – Audio in various formats
  * **transcript** (`str` | `None`) – Text transcription of the audio (if None, auto-transcribed)
  * **speaker_id** (`int`) – Unique ID for this voice
  * **cleanup_audio_fn** (`Callable` | `None`) – Function to clean up audio (None to skip)
* **Returns:**
  A packaged voice profile
* **Return type:**
  [`VoiceProfile`](#voxy.base.VoiceProfile)

#### design_voice(description, \*\*kwargs)

Create a new voice from a text description.

* **Return type:**
  [`VoiceProfile`](#voxy.base.VoiceProfile)

#### dflt_voice *: str | None* *= None*

the caller must).

* **Type:**
  Voice used when the caller names none (`None`

#### generate_speech(text, voice_profile=None, output_path=None, max_length_ms=10000, \*\*kwargs)

Generate speech using a voice profile.

* **Parameters:**
  * **text** (`str` | `bytes` | `TextIOBase`) – Text to synthesize
  * **voice_profile** ([`VoiceProfile`](#voxy.base.VoiceProfile) | `None`) – Voice profile from clone_voice()
  * **output_path** (`str` | `None`) – Path to save the audio (optional)
  * **max_length_ms** (`int`) – Maximum audio length in milliseconds
  * **\*\*kwargs** – Additional model-specific parameters
* **Return type:**
  `Tensor`
* **Returns:**
  Generated audio tensor

#### list_voices()

The voices this backend offers.

* **Return type:**
  `list`[[`VoiceInfo`](#voxy.base.VoiceInfo)]

#### name *: str* *= ''*

Registry name of the backend (also each profile’s `model_type`).

#### synthesize(text, voice=None, \*\*kwargs)

Render `text` in `voice` and return encoded audio.

This default (for tensor-producing models such as CSM) needs a
`VoiceProfile` or `None`; services override it.

* **Return type:**
  [`Speech`](#voxy.base.Speech)

### *class* voxy.base.VoiceInfo(voice_id, name, backend, description='', labels=None)

Bases: `object`

A voice a backend offers (stock, designed, or cloned).

### *class* voxy.base.VoiceProfile(segment, speaker_id, model_type, sample_rate, metadata=None)

Bases: `object`

Data class to store voice cloning information.

### voxy.base.audio_to_text(audio_input, model_size='base', , sample_rate=None)

Transcribe audio to text using Whisper.

* **Parameters:**
  * **audio_input** (`str` | `bytes` | `BinaryIO` | `Tensor` | `ndarray`) – Audio in various formats
  * **model_size** (`str`) – Whisper model size (‘tiny’, ‘base’, ‘small’, ‘medium’, ‘large’)
  * **sample_rate** (`int` | `None`) – Sample rate of `audio_input` when it is a raw tensor or
    numpy array. Required for correct transcription of raw audio that
    is not at `DFLT_ASSUMED_SAMPLE_RATE`; ignored when the input is a
    path, bytes or file-like object (those carry their own rate).
* **Return type:**
  `str`
* **Returns:**
  Transcribed text
* **Raises:**
  **ImportError** – If whisper is not installed

### voxy.base.backend_aliases *: dict[str, str]* *= {'csm-1b': 'csm'}*

Other names for a backend -> its registry name (one model, one profile type).

### voxy.base.canonical_backend(name)

The registry name of backend `name` (lowercased, aliases resolved).

* **Return type:**
  `str`

```pycon
>>> canonical_backend("ElevenLabs"), canonical_backend("CSM-1B")
('elevenlabs', 'csm')
```

### voxy.base.cleanup_audio(audio, sample_rate, normalize=True, remove_silence=True, silence_threshold=0.02, min_silence_duration=0.2)

Clean up audio by normalizing volume and removing silence.

* **Parameters:**
  * **audio** (`Tensor`) – Audio tensor [channels, samples] or [samples]
  * **sample_rate** (`int`) – Sample rate of the audio
  * **normalize** (`bool`) – Whether to normalize the audio volume
  * **remove_silence** (`bool`) – Whether to remove silence
  * **silence_threshold** (`float`) – Threshold for silence detection (0.0-1.0)
  * **min_silence_duration** (`float`) – Minimum silence duration in seconds
* **Return type:**
  `Tensor`
* **Returns:**
  Processed audio tensor

A 1-D input is given a channel dimension, and the loudest sample is
normalized to 1.0:

```pycon
>>> audio = torch.tensor([0.0, 0.5, 0.0, 0.0])
>>> processed = cleanup_audio(audio, sample_rate=8000)
>>> tuple(processed.shape)
(1, 4)
>>> round(float(processed.max()), 3)
1.0
```

Stereo input is mixed down to mono:

```pycon
>>> stereo = torch.tensor([[0.0, 0.5, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]])
>>> tuple(cleanup_audio(stereo, sample_rate=8000).shape)
(1, 4)
```

### voxy.base.create_speech_model(model_type='csm', \*\*kwargs)

Create a speech model of the specified type.

* **Parameters:**
  * **model_type** (`str`) – A key of `speech_model_factories` (‘csm’, ‘csm-1b’,
    ‘elevenlabs’, ‘aix’, ‘fal’, ‘say’, or any registered); case-insensitive,
    aliases in `backend_aliases` accepted.
  * **\*\*kwargs** – Additional model-specific parameters
* **Return type:**
  [`SpeechModel`](#voxy.base.SpeechModel)
* **Returns:**
  SpeechModel instance
* **Raises:**
  **ValueError** – If the model type is not supported

The returned model loads its (large) weights lazily, on first use:

```pycon
>>> model = create_speech_model("csm")
>>> type(model).__name__
'CSMSpeechModel'
>>> type(create_speech_model("elevenlabs", api_key="unused")).__name__
'ElevenLabsSpeechModel'
>>> create_speech_model("no-such-model")
Traceback (most recent call last):
  ...
ValueError: Unsupported model type: no-such-model (supported: csm, elevenlabs, aix, fal, say)
```

### voxy.base.register_speech_model(name, factory, , overwrite=False)

Register a backend (a class or `**kwargs -> SpeechModel` callable).

Returns `factory`, so it also works as a class decorator via `functools.partial`.

* **Return type:**
  `Callable`[`...`, [`SpeechModel`](#voxy.base.SpeechModel)]

```pycon
>>> class Echo(SpeechModel):
...     name = "echo"
>>> _ = register_speech_model("echo", Echo)
>>> type(create_speech_model("echo")).__name__
'Echo'
>>> register_speech_model("echo", Echo)
Traceback (most recent call last):
  ...
ValueError: A speech backend named 'echo' is already registered (pass overwrite=True)
>>> del speech_model_factories["echo"]
```

### voxy.base.speech_model_factories *: dict[str, Callable[[...], [SpeechModel](#voxy.base.SpeechModel)]]* *= {'aix': <function \_lazy_factory.<locals>.factory>, 'csm': <class 'voxy.base.CSMSpeechModel'>, 'elevenlabs': <function \_lazy_factory.<locals>.factory>, 'fal': <function \_lazy_factory.<locals>.factory>, 'say': <function \_lazy_factory.<locals>.factory>}*

Backend name -> factory (keys lowercase). Add one with `register_speech_model`.

### voxy.base.tensor_to_wav_bytes(audio, sample_rate)

Encode a tensor ([channels, samples] or [samples]) as mono 16-bit WAV.

Integer tensors are taken as PCM and scaled to [-1, 1] first.

* **Return type:**
  `bytes`

```pycon
>>> data = tensor_to_wav_bytes(torch.zeros(160), 16000)
>>> data[:4], len(data)
(b'RIFF', 364)
>>> tensor_to_wav_bytes(torch.tensor([[0, 32767]], dtype=torch.int16), 8000)[-2:]
b'\xff\x7f'
```
