# voxy

Facade for voice generation: speech synthesis in any voice, on any service, plus cloning and designing voices.

To install:	```pip install voxy``` (add `'voxy[elevenlabs]'`, `'voxy[aix]'` or `'voxy[fal]'` for those services)

```python
import voxy

voxy.text_to_speech("Hello!", voice="cora").save("hi.mp3")          # a named voice from the library
voxy.text_to_speech("A quick draft.", voice="Daniel", backend="say")  # free local voice (macOS)
voxy.list_voices()                # our named voices
voxy.list_voices("elevenlabs")    # a service's voices
```

Backends: `elevenlabs` (cloning, voice design, TTS), `say` (macOS, offline), `aix` (OpenAI voices and other LiteLLM providers), `fal` (fal.ai models via falaw), `csm` (local Sesame CSM-1B). Register more with `voxy.register_speech_model(name, factory)`.

Command line: `python -m voxy voices [--backend X]` and `python -m voxy speak "text" --voice cora -o out.mp3`.

### Designing a voice from a description

```python
model = voxy.get_speech_model("elevenlabs")
previews = model.design_voice_previews("a warm, slow, elderly storyteller", text="Once upon a time...")
for i, p in enumerate(previews):
    p.save(f"preview_{i}.mp3")          # listen, then pick one
voxy.design_from_description("storyteller", "a warm, slow, elderly storyteller", preview=previews[1])
```

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

### Install at least one model

#### Example: the CSM Model

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

Try [the csm spaces](https://huggingface.co/spaces/sesame/csm-1b) where you can try various functionalities of csm.

## Quick Start

### Basic Usage

```python
from voxy import create_speech_model

# Create a speech model
model = create_speech_model(model_type="csm")

# Generate speech with default voice
audio = model.generate_speech(
    text="Hello, this is a test of the CSM speech model.", output_path="output.wav"
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
    transcript="This is a sample of my voice for cloning purposes.",
)

# Generate speech with the cloned voice
audio = model.generate_speech(
    text="This is my cloned voice speaking. Isn't it amazing?",
    voice_profile=voice_profile,
    output_path="cloned_voice.wav",
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
    output_path="auto_transcribed_voice.wav",
)
```

### Flexible Input Formats

The module supports various input formats:

```python
# From file path
voice_profile1 = model.clone_voice(
    audio_input="sample_voice.wav", transcript="Text transcript."
)

# From bytes
with open("sample_voice.wav", "rb") as f:
    audio_bytes = f.read()
voice_profile2 = model.clone_voice(
    audio_input=audio_bytes, transcript="Text transcript."
)

# From file object
with open("sample_voice.wav", "rb") as f:
    voice_profile3 = model.clone_voice(audio_input=f, transcript="Text transcript.")

# From tensor
import torch
import torchaudio

audio_tensor, sample_rate = torchaudio.load("sample_voice.wav")
voice_profile4 = model.clone_voice(
    audio_input=audio_tensor, transcript="Text transcript."
)
```

### ElevenLabs backend

Install with `pip install 'voxy[elevenlabs]'` and set `ELEVEN_API_KEY` or `ELEVENLABS_API_KEY` (checked in that order), or pass `api_key=`.

```python
from voxy import create_speech_model

model = create_speech_model("elevenlabs")

# Instant voice clone: one sample or several (paths, bytes, file-likes, tensors), at most 25 files.
# ElevenLabs recommends 1-2 minutes of clean single-speaker audio in total (at most ~3).
profile = model.clone_voice(["clip1.wav", "clip2.m4a"], name="my-voice")
print(profile.segment)  # the ElevenLabs voice_id

audio = model.generate_speech(
    "Hello from my cloned voice.", profile, output_path="hello.wav"
)

# Later sessions: rebuild the profile from the id, no re-upload.
profile = model.voice_profile("<voice_id>")
```

`generate_speech` returns a tensor for `pcm_*`/`wav_*` output formats (default `pcm_24000`); `synthesize_bytes` returns the raw payload in any format (e.g. `mp3_44100_128`). `delete_voice` removes a clone from the account. ElevenLabs requires that you have the right, and the voice owner's consent, to clone a voice.

Live tests are opt-in: `VOXY_LIVE_ELEVENLABS=1` runs a short synthesis; adding `VOXY_LIVE_ELEVENLABS_SAMPLE=<audio you may clone>` also clones, speaks and deletes.

### The voice library: clone once, reuse by name

voxy keeps voice data in stores (`MutableMapping`s) under `$VOXY_DATA_DIR` (default `~/.local/share/voxy`): `voices/{name}.json` records (with each backend's saved profile, e.g. the ElevenLabs `voice_id`), `samples/{name}/` (audio a clone is made from), `sources/{name}/` (raw media). Every library function takes its stores as arguments (`voices=`, `samples=`), so any `MutableMapping` (another folder, S3) can stand in.

```python
from voxy import samples_store, clone_from_samples, load_voice, create_speech_model

samples = samples_store("ada")
samples["clip1.wav"] = open("clip1.wav", "rb").read()
clone_from_samples("ada")  # uploads the samples, saves the profile

model = create_speech_model("elevenlabs")
model.generate_speech("Hi!", load_voice("ada"), output_path="hi.wav")
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
    min_silence_duration=0.2,
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
    cleanup_audio_fn=None,  # Disable audio cleanup
)
```

### Custom Audio Cleanup

You can also provide your own audio cleanup function:

```python
def my_custom_cleanup(audio, sample_rate, **kwargs):
    # Custom cleanup logic
    return processed_audio

voice_profile = model.
