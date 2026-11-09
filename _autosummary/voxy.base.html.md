# voxy.base

Voxy: A flexible speech synthesis and voice cloning module.

This module provides a plugin architecture for working with different speech synthesis
models, with initial support for the CSM-1B model.

torch, torchaudio and whisper are imported only where a local model needs them
(`voxy[local]`, `voxy[transcription]`), so a cloud backend runs without them.
Annotations are postponed and torch is imported only for type checking, so
`typing.get_type_hints` on these functions needs torch importable and in scope
(an accepted trade-off).

### Module Attributes

| [`OPTIONAL_MODULE_EXTRAS`](#voxy.base.OPTIONAL_MODULE_EXTRAS)                             | The extra that installs each optional module (for the error raised without it).   |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`optional_import`](#voxy.base.optional_import)(module, \*[, package, extra, ...]) | Import `module`, or raise a `ModuleNotFoundError` saying how to install it.       |
| [`speech_model_factories`](#voxy.base.speech_model_factories)                             | Backend name -> factory (keys lowercase).                                         |
| [`backend_aliases`](#voxy.base.backend_aliases)                                    | Other names for a backend -> its registry name (one model, one profile type).     |

### Functions

| [`audio_to_text`](#voxy.base.audio_to_text)(audio_input[, model_size, ...])   | Transcribe audio to text using Whisper.                                                  |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [`canonical_backend`](#voxy.base.canonical_backend)(name)                         | The registry name of backend `name` (lowercased, aliases resolved).                      |
| [`cleanup_audio`](#voxy.base.cleanup_audio)(audio, sample_rate[, ...])        | Clean up audio by normalizing volume and removing silence.                               |
| [`create_speech_model`](#voxy.base.create_speech_model)([model_type])               | Create a speech model of the specified type.                                             |
| [`dflt_voxy_device`](#voxy.base.dflt_voxy_device)()                              | `$DFLT_VOXY_DEVICE`, else 'cuda' if torch is installed and sees one, else 'cpu'.         |
| [`is_tensor`](#voxy.base.is_tensor)(obj)                                  | True if `obj` is a torch tensor (without importing torch to find out).                   |
| [`register_speech_model`](#voxy.base.register_speech_model)(name, factory, \*[, ...]) | Register a backend (a class or `**kwargs -> SpeechModel` callable).                      |
| [`tensor_to_wav_bytes`](#voxy.base.tensor_to_wav_bytes)(audio, sample_rate)         | Encode audio ([samples], [channels, samples] or [samples, channels]) as mono 16-bit WAV. |

### Classes

| [`CSMSpeechModel`](#voxy.base.CSMSpeechModel)([model_path, device])          | Speech model implementation using Sesame's CSM-1B model.   |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`Speech`](#voxy.base.Speech)(audio, format[, backend, voice, ...])  | Synthesized speech: encoded audio plus what produced it.   |
| [`SpeechModel`](#voxy.base.SpeechModel)([device])                         | Base class for speech backends (local models or services). |
| [`VoiceInfo`](#voxy.base.VoiceInfo)(voice_id, name, backend[, ...])     | A voice a backend offers (stock, designed, or cloned).     |
| [`VoiceProfile`](#voxy.base.VoiceProfile)(segment, speaker_id, ...[, ...]) | Data class to store voice cloning information.             |

### *class* voxy.base.CSMSpeechModel(model_path=None, device=None)

Bases: [`SpeechModel`](#voxy.base.SpeechModel)

Speech model implementation using Sesame’s CSM-1B model.

#### clone_voice(audio_input, transcript=None, speaker_id=999, \*, cleanup_audio_fn=<function cleanup_audio>)

Create a voice profile from an audio sample and its transcript.

* **Parameters:**
  * **audio_input** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO), `Tensor`, `ndarray`]) – Audio in various formats
  * **transcript** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Text transcription of the audio (if None, auto-transcribed)
  * **speaker_id** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Unique ID for this voice
  * **cleanup_audio_fn** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Function to clean up audio (None to skip)
* **Returns:**
  A packaged voice profile
* **Return type:**
  [`VoiceProfile`](#voxy.base.VoiceProfile)

#### generate_speech(text, voice_profile=None, output_path=None, max_length_ms=10000, temperature=0.7, topk=30)

Generate speech using a voice profile.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`TextIOBase`](https://docs.python.org/3/library/io.html#io.TextIOBase)) – Text to synthesize
  * **voice_profile** ([`VoiceProfile`](#voxy.base.VoiceProfile) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice profile from clone_voice()
  * **output_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to save the audio (optional)
  * **max_length_ms** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum audio length in milliseconds
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Sampling temperature (lower = more deterministic)
  * **topk** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Top-k sampling parameter
* **Return type:**
  `Tensor`
* **Returns:**
  Generated audio tensor

#### name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)* *= 'csm'*

Registry name of the backend (also each profile’s `model_type`).

#### *property* sample_rate *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Sample rate of the generated audio (loads the model).

### voxy.base.OPTIONAL_MODULE_EXTRAS *= {'huggingface_hub': 'local', 'torch': 'local', 'torchaudio': 'local', 'whisper': 'transcription'}*

The extra that installs each optional module (for the error raised without it).

### *class* voxy.base.Speech(audio, format, backend='', voice=None, sample_rate=None, text=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

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
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* voxy.base.SpeechModel(device=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

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
  * **audio_input** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO), `Tensor`, `ndarray`]) – Audio in various formats
  * **transcript** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Text transcription of the audio (if None, auto-transcribed)
  * **speaker_id** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Unique ID for this voice
  * **cleanup_audio_fn** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Function to clean up audio (None to skip)
* **Returns:**
  A packaged voice profile
* **Return type:**
  [`VoiceProfile`](#voxy.base.VoiceProfile)

#### design_voice(description, \*\*kwargs)

Create a new voice from a text description.

* **Return type:**
  [`VoiceProfile`](#voxy.base.VoiceProfile)

#### *property* device *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The inference device (resolved on first use).

#### dflt_voice *: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None)* *= None*

the caller must).

* **Type:**
  Voice used when the caller names none (`None`

#### generate_speech(text, voice_profile=None, output_path=None, max_length_ms=10000, \*\*kwargs)

Generate speech using a voice profile.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`TextIOBase`](https://docs.python.org/3/library/io.html#io.TextIOBase)) – Text to synthesize
  * **voice_profile** ([`VoiceProfile`](#voxy.base.VoiceProfile) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice profile from clone_voice()
  * **output_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to save the audio (optional)
  * **max_length_ms** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum audio length in milliseconds
  * **\*\*kwargs** – Additional model-specific parameters
* **Return type:**
  `Tensor`
* **Returns:**
  Generated audio tensor

#### list_voices()

The voices this backend offers.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`VoiceInfo`](#voxy.base.VoiceInfo)]

#### name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)* *= ''*

Registry name of the backend (also each profile’s `model_type`).

#### native_break_s(\*\*synth_kwargs)

The longest SSML `<break>` this backend renders itself, in seconds.

`0.0` (the default): it renders none, so `voxy.text_to_speech` cuts
the text at every break and joins the parts with real silence
(`voxy.pauses`). `synth_kwargs` are the call’s options, for a
backend whose answer depends on them (ElevenLabs: the `model_id`).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

#### synthesize(text, voice=None, \*\*kwargs)

Render `text` in `voice` and return encoded audio.

This default (for tensor-producing models such as CSM) needs a
`VoiceProfile` or `None`; services override it.

* **Return type:**
  [`Speech`](#voxy.base.Speech)

### *class* voxy.base.VoiceInfo(voice_id, name, backend, description='', labels=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A voice a backend offers (stock, designed, or cloned).

### *class* voxy.base.VoiceProfile(segment, speaker_id, model_type, sample_rate, metadata=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Data class to store voice cloning information.

### voxy.base.audio_to_text(audio_input, model_size='base', , sample_rate=None)

Transcribe audio to text using Whisper.

* **Parameters:**
  * **audio_input** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO), `Tensor`, `ndarray`]) – Audio in various formats
  * **model_size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Whisper model size (‘tiny’, ‘base’, ‘small’, ‘medium’, ‘large’)
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Sample rate of `audio_input` when it is a raw tensor or
    numpy array. Required for correct transcription of raw audio that
    is not at `DFLT_ASSUMED_SAMPLE_RATE`; ignored when the input is a
    path, bytes or file-like object (those carry their own rate).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Transcribed text
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If whisper is not installed

### voxy.base.backend_aliases *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [str](https://docs.python.org/3/builtins/stdtypes.html#str)]* *= {'csm-1b': 'csm', 'f5-tts': 'f5tts', 'qwen3-tts': 'qwen3tts', 'voxcpm2': 'voxcpm', 'xtts-v2': 'xtts', 'xtts_v2': 'xtts'}*

Other names for a backend -> its registry name (one model, one profile type).

### voxy.base.canonical_backend(name)

The registry name of backend `name` (lowercased, aliases resolved).

```pycon
>>> canonical_backend("ElevenLabs"), canonical_backend("CSM-1B")
('elevenlabs', 'csm')
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.base.cleanup_audio(audio, sample_rate, normalize=True, remove_silence=True, silence_threshold=0.02, min_silence_duration=0.2)

Clean up audio by normalizing volume and removing silence.

* **Parameters:**
  * **audio** (`Tensor`) – Audio tensor [channels, samples] or [samples]
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Sample rate of the audio
  * **normalize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to normalize the audio volume
  * **remove_silence** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to remove silence
  * **silence_threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Threshold for silence detection (0.0-1.0)
  * **min_silence_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum silence duration in seconds
* **Return type:**
  `Tensor`
* **Returns:**
  Processed audio tensor

A 1-D input is given a channel dimension, and the loudest sample is
normalized to 1.0:

```pycon
>>> import torch
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
  * **model_type** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A key of `speech_model_factories` (‘csm’, ‘csm-1b’,
    ‘elevenlabs’, ‘aix’, ‘fal’, ‘say’, ‘voicestudio’, or any registered); case-insensitive,
    aliases in `backend_aliases` accepted.
  * **\*\*kwargs** – Additional model-specific parameters
* **Return type:**
  [`SpeechModel`](#voxy.base.SpeechModel)
* **Returns:**
  SpeechModel instance
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If the model type is not supported

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
ValueError: Unsupported model type: no-such-model (supported: csm, elevenlabs, ...)
```

### voxy.base.dflt_voxy_device()

`$DFLT_VOXY_DEVICE`, else ‘cuda’ if torch is installed and sees one, else ‘cpu’.

MPS is skipped even when available: CSM fails on it (“Output channels > 65536
not supported at the MPS device”).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.base.is_tensor(obj)

True if `obj` is a torch tensor (without importing torch to find out).

```pycon
>>> is_tensor([0.0])
False
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### voxy.base.optional_import(module, , package='voxy', extra=None, extras={'huggingface_hub': 'local', 'torch': 'local', 'torchaudio': 'local', 'whisper': 'transcription'})

`optional_import(module)`: the module, or an ImportError naming the voxy extra.

* **Return type:**
  [`ModuleType`](https://docs.python.org/3/library/types.html#types.ModuleType)

### voxy.base.register_speech_model(name, factory, , overwrite=False)

Register a backend (a class or `**kwargs -> SpeechModel` callable).

Returns `factory`, so it also works as a class decorator via `functools.partial`.

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

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`SpeechModel`](#voxy.base.SpeechModel)]

### voxy.base.speech_model_factories *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[..., [SpeechModel](#voxy.base.SpeechModel)]]* *= {'aix': <function \_lazy_factory.<locals>.factory>, 'chatterbox': <function \_lazy_factory.<locals>.factory>, 'csm': <class 'voxy.base.CSMSpeechModel'>, 'elevenlabs': <function \_lazy_factory.<locals>.factory>, 'f5tts': <function \_lazy_factory.<locals>.factory>, 'fal': <function \_lazy_factory.<locals>.factory>, 'kokoro': <function \_lazy_factory.<locals>.factory>, 'qwen3tts': <function \_lazy_factory.<locals>.factory>, 'say': <function \_lazy_factory.<locals>.factory>, 'voicestudio': <function \_lazy_factory.<locals>.factory>, 'voxcpm': <function \_lazy_factory.<locals>.factory>, 'xtts': <function \_lazy_factory.<locals>.factory>}*

Backend name -> factory (keys lowercase). Add one with `register_speech_model`.

### voxy.base.tensor_to_wav_bytes(audio, sample_rate)

Encode audio ([samples], [channels, samples] or [samples, channels]) as mono
16-bit WAV.

`audio` is a numpy array or a torch tensor (torch is not needed for arrays).
Integer audio is taken as PCM and scaled to [-1, 1] first (8-bit PCM is
unsigned). A 2-D input is `[channels, samples]` unless it has more rows than
columns, which only `[samples, channels]` (soundfile’s layout) can have (a
square input is read channels-first); channels are averaged, and
`voxy.wavutil.write_wav` does the encoding.

```pycon
>>> data = tensor_to_wav_bytes(np.zeros(160), 16000)
>>> data[:4], len(data)
(b'RIFF', 364)
>>> tensor_to_wav_bytes(np.array([[0, 32767]], dtype=np.int16), 8000)[-2:]
b'\xff\x7f'
>>> tensor_to_wav_bytes(np.full((5, 2), 128, dtype=np.uint8), 8000)[-4:]  # silence
b'\x00\x00\x00\x00'
>>> tensor_to_wav_bytes(np.zeros((1, 2, 3)), 8000)
Traceback (most recent call last):
  ...
ValueError: Audio must be 1-D or 2-D (channels and samples), not 3-D
```

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)
