# voxy.base

Voxy: A flexible speech synthesis and voice cloning module.

This module provides a plugin architecture for working with different speech synthesis
models, with initial support for the CSM-1B model.

### Module Attributes

| [`speech_model_factories`](#voxy.base.speech_model_factories)   | Backend name -> factory.   |
|---------------------------------------------------------------------------|----------------------------|

### Functions

| [`audio_to_text`](#voxy.base.audio_to_text)(audio_input[, model_size, ...])   | Transcribe audio to text using Whisper.                    |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`cleanup_audio`](#voxy.base.cleanup_audio)(audio, sample_rate[, ...])        | Clean up audio by normalizing volume and removing silence. |
| [`create_speech_model`](#voxy.base.create_speech_model)([model_type])               | Create a speech model of the specified type.               |

### Classes

| [`CSMSpeechModel`](#voxy.base.CSMSpeechModel)([model_path, device])          | Speech model implementation using Sesame's CSM-1B model.   |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [`SpeechModel`](#voxy.base.SpeechModel)([device])                         | Base class for speech synthesis models.                    |
| [`VoiceProfile`](#voxy.base.VoiceProfile)(segment, speaker_id, ...[, ...]) | Data class to store voice cloning information.             |

### *class* voxy.base.CSMSpeechModel(model_path=None, device='cpu')

Bases: [`SpeechModel`](#voxy.base.SpeechModel)

Speech model implementation using Sesame’s CSM-1B model.

#### clone_voice(audio_input, transcript=None, speaker_id=999, \*, cleanup_audio_fn=<function cleanup_audio>)

Create a voice profile from an audio sample and its transcript.

* **Parameters:**
  * **audio_input** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO) | `Tensor` | `ndarray`) – Audio in various formats
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

### *class* voxy.base.SpeechModel(device='cpu')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base class for speech synthesis models.

#### clone_voice(audio_input, transcript=None, speaker_id=999, \*, cleanup_audio_fn=<function cleanup_audio>)

Create a voice profile from an audio sample and its transcript.

* **Parameters:**
  * **audio_input** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO) | `Tensor` | `ndarray`) – Audio in various formats
  * **transcript** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Text transcription of the audio (if None, auto-transcribed)
  * **speaker_id** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Unique ID for this voice
  * **cleanup_audio_fn** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Function to clean up audio (None to skip)
* **Returns:**
  A packaged voice profile
* **Return type:**
  [`VoiceProfile`](#voxy.base.VoiceProfile)

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

### *class* voxy.base.VoiceProfile(segment, speaker_id, model_type, sample_rate, metadata=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Data class to store voice cloning information.

### voxy.base.audio_to_text(audio_input, model_size='base', , sample_rate=None)

Transcribe audio to text using Whisper.

* **Parameters:**
  * **audio_input** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO) | `Tensor` | `ndarray`) – Audio in various formats
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
    ‘elevenlabs’); case-insensitive.
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
ValueError: Unsupported model type: no-such-model (supported: csm, csm-1b, elevenlabs)
```

### voxy.base.speech_model_factories *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Callable](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[...], [SpeechModel](#voxy.base.SpeechModel)]]* *= {'csm': <class 'voxy.base.CSMSpeechModel'>, 'csm-1b': <class 'voxy.base.CSMSpeechModel'>, 'elevenlabs': <function \_elevenlabs_speech_model>}*

Backend name -> factory. Add a backend by adding an entry here.
