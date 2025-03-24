# voxy

Facade for voice cloning and speech synthesis

To install:	```pip install voxy```

Voxy is a flexible Python module for speech synthesis and voice cloning, with initial support for the Sesame CSM-1B model. It provides a plugin architecture that can be extended to support other models in the future.

## Features

- Voice cloning from audio samples
- High-quality speech synthesis
- Flexible input formats (file paths, bytes, streams, tensors)
- Audio cleanup utilities
- Automatic audio transcription (using Whisper)
- Plugin architecture for different speech models

## Installation

### Prerequisites

- Python 3.10+
- PyTorch and TorchAudio
- CUDA-compatible GPU (recommended)
- FFmpeg for audio processing

### Install the CSM Model

The intention is to make `voxy` into a plugin-enabled facade, where you can chose your 
own engine (for voice cloning, voice synthesis, etc.). 
But for now, we just support, what seems to be the best open-source model out there
(at the time of writing this): 
[Sesame AI Lab's](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) 
CSM model. It's just that, well, they did an amazing job at the model, but a terrible one
(so far) for the python interface -- which is what inspired me to develop `voxy` 
in the first place.

Follow the instructions in the [CSM repository](https://github.com/SesameAILabs/csm) 
to install the CSM model and its dependencies.

## Quick Start

### Basic Usage

```python
from voxy import create_speech_model

# Create a speech model
model = create_speech_model(model_type="csm")

# Generate speech with default voice
audio = model.generate_speech(
    text="Hello, this is a test of the CSM speech model.", 
    output_path="output.wav"
)
```

### Voice Cloning

```python
from voxy import create_speech_model

# Create a speech model
model = create_speech_model(model_type="csm")

# Clone a voice from an audio file
voice_profile = model.clone_voice(
    audio_input="sample_voice.wav",
    transcript="This is a sample of my voice for cloning purposes."
)

# Generate speech with the cloned voice
audio = model.generate_speech(
    text="This is my cloned voice speaking. Isn't it amazing?",
    voice_profile=voice_profile,
    output_path="cloned_voice.wav"
)
```

### Automatic Transcription

```python
from voxy import create_speech_model

# Create a speech model
model = create_speech_model(model_type="csm")

# Clone a voice with automatic transcription
voice_profile = model.clone_voice(
    audio_input="sample_voice.wav",
    # No transcript provided, will use automatic transcription
)

# Generate speech with the cloned voice
audio = model.generate_speech(
    text="This voice was cloned using automatic transcription.",
    voice_profile=voice_profile,
    output_path="auto_transcribed_voice.wav"
)
```

### Flexible Input Formats

The module supports various input formats:

```python
# From file path
voice_profile1 = model.clone_voice(
    audio_input="sample_voice.wav",
    transcript="Text transcript."
)

# From bytes
with open("sample_voice.wav", "rb") as f:
    audio_bytes = f.read()
voice_profile2 = model.clone_voice(
    audio_input=audio_bytes,
    transcript="Text transcript."
)

# From file object
with open("sample_voice.wav", "rb") as f:
    voice_profile3 = model.clone_voice(
        audio_input=f,
        transcript="Text transcript."
    )

# From tensor
import torch
import torchaudio
audio_tensor, sample_rate = torchaudio.load("sample_voice.wav")
voice_profile4 = model.clone_voice(
    audio_input=audio_tensor,
    transcript="Text transcript."
)
```

## Configuration

You can configure the default device by setting the `DFLT_VOXY_DEVICE` environment variable:

```bash
# Use CUDA
export DFLT_VOXY_DEVICE=cuda

# Use CPU
export DFLT_VOXY_DEVICE=cpu

# Use MPS (Apple Silicon)
export DFLT_VOXY_DEVICE=mps
```

## Advanced Usage

### Audio Cleanup

The module includes an audio cleanup function that normalizes volume and removes silence:

```python
from voxy import cleanup_audio
import torchaudio

# Load audio
audio, sample_rate = torchaudio.load("noisy_audio.wav")

# Clean up audio
cleaned_audio = cleanup_audio(
    audio=audio,
    sample_rate=sample_rate,
    normalize=True,
    remove_silence=True,
    silence_threshold=0.02,
    min_silence_duration=0.2
)

# Save cleaned audio
torchaudio.save("cleaned_audio.wav", cleaned_audio, sample_rate)
```

### Disabling Audio Cleanup

You can disable audio cleanup when cloning a voice:

```python
voice_profile = model.clone_voice(
    audio_input="sample_voice.wav",
    transcript="This is a sample of my voice.",
    cleanup_audio_fn=None  # Disable audio cleanup
)
```

### Custom Audio Cleanup

You can also provide your own audio cleanup function:

```python
def my_custom_cleanup(audio, sample_rate, **kwargs):
    # Custom cleanup logic
    return processed_audio

voice_profile = model.