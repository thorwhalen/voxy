# voxy

Facade for voice generation: speech synthesis in any voice, on any service, plus cloning and designing voices.

To install:	```pip install voxy``` (add `'voxy[elevenlabs]'`, `'voxy[aix]'` or `'voxy[fal]'` for those services) The part-of-speech heteronym detector (`voxy.speechtext.pos_detector`) needs `'voxy[pos]'` plus `python -m spacy download fr_core_news_sm`. The LLM proposer (`voxy.speechtext.llm_detector`) uses `'voxy[aix]'` unless you pass `chat=`.

```python
import voxy

voxy.text_to_speech("Hello!", voice="ada").save(
    "hi.mp3"
)  # a named voice from the library
voxy.text_to_speech(
    "A quick draft.", voice="Daniel", backend="say"
)  # free local voice (macOS)
voxy.list_voices()  # our named voices
voxy.list_voices("elevenlabs")  # a service's voices
```

Backends: `elevenlabs` (cloning, voice design, TTS), `say` (macOS, offline), `aix` (OpenAI voices and other LiteLLM providers), `fal` (fal.ai models via falaw), `voicestudio` (local text-described voices, `voxy[voicestudio]`), `csm` (local Sesame CSM-1B). Register more with `voxy.register_speech_model(name, factory)`.

Command line: `python -m voxy voices [--backend X]` and `python -m voxy speak "text" --voice ada -o out.mp3`.

### Designing a voice from a description

```python
model = voxy.get_speech_model("elevenlabs")
line = (
    "Once upon a time, in a village at the edge of a very old forest, there lived a "
    "baker who could hear the bread singing as it rose."
)  # 100-1000 characters
previews = model.design_voice_previews("a warm, slow, elderly storyteller", text=line)
for i, p in enumerate(previews):
    p.save(f"preview_{i}.mp3")  # listen, then pick one
voxy.design_from_description(
    "storyteller", "a warm, slow, elderly storyteller", preview=previews[1]
)
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

### Local voice cloning (no service, CPU works)

Five local models clone a voice from one clean clip and its transcript: `voxcpm` (VoxCPM2), `qwen3tts` (Qwen3-TTS), `chatterbox`, `f5tts` (F5-TTS) and `xtts` (XTTS-v2). Each runs in its own Python environment, built on first use with [uv](https://docs.astral.sh/uv/) under `$VOXY_DATA_DIR/envs/` (weights under `$VOXY_DATA_DIR/models/`), so their clashing torch/transformers pins never touch yours. Requirements: `uv` and `ffmpeg` on the PATH.

```python
import voxy
from voxy.local_clone import reference_profile

model = voxy.create_speech_model("voxcpm")  # or qwen3tts, chatterbox, f5tts, xtts
voice = reference_profile("ref.wav", "Exactly what the clip says.")
model.synthesize(chapter_text, voice, cfg_value=2.0).save("chapter1.wav")
model.close()  # free the model's memory
```

Long text is split into paragraphs and sentence-packed chunks, each spoken separately and joined with controlled pauses (`gap_ms=`, `paragraph_gap_ms=`). Model parameters are keyword arguments of `synthesize` (see `voxy.local_clone.LOCAL_CLONE_BACKENDS[name].params`); model options (a variant, a fine-tuned checkpoint) are keyword arguments of `create_speech_model`. Threads are capped at `$VOXY_NUM_THREADS` (default 8). Licences differ: F5-TTS and XTTS-v2 weights are non-commercial (XTTS needs `accept_licence=True`); VoxCPM2, Qwen3-TTS and Chatterbox are Apache-2.0/MIT.

Languages: VoxCPM2 (30 languages), Qwen3-TTS (10, pass `language="French"` etc.), XTTS-v2 (17, `language="fr"`) and Chatterbox with `variant="multilingual"` (23, `language="fr"`) speak beyond English; Chatterbox's `original` and `turbo` variants and the F5-TTS base model are English (F5 also Chinese).

**Which backend?** [docs/benchmarks.md](docs/benchmarks.md) compares the cloning backends on published benchmarks (French and English accuracy, voice likeness rated by people, arenas, licence, price, watermark), with tables, charts and a link to every source. The data is `voxy/data/benchmarks.json`; `python -m voxy benchmarks docs/benchmarks.md` regenerates the tables and charts after you edit it.

```python
model = voxy.create_speech_model("chatterbox", variant="multilingual")
speech = model.synthesize("Bonjour à tous.", voice, language="fr")
```

```bash
python -m voxy setup voxcpm qwen3tts      # build environments ahead of time
python -m voxy prep ada tape1.m4a tape2.mp3 --language en
python -m voxy prep ada interview.m4a --speaker dominant   # drop the interviewer's voice
python -m voxy diagnose spec.json --out renders/first-listen --publish
```

**Reference preparation** (`voxy.refprep.prepare_reference`): converts any recording to mono WAV, denoises (`denoise=` `afftdn`, `anlmdn`, or `deepfilternet`, which lowers real room noise far more but builds its own torch environment on first use) and normalises loudness (ffmpeg), transcribes locally with Whisper (scribed's faster-whisper backend) with word timings, cuts clips of 3-12 s at sentence boundaries, and scores each clip (signal-to-noise, clipping, level, ASR confidence, speaking rate). With `speaker=voxy.speakers.speaker_filter()` (`--speaker`) it keeps one voice: ECAPA speaker embeddings (SpeechBrain, in its own environment) decide each transcribed turn as a whole against the target's voice print (the recording's dominant voice by default, or `speakers.voice_print(clean_recordings)`), and interviewer questions, interjections such as "Yes." and untranscribed "uh-huh"s never end up inside a clip. Clips and `manifest.jsonl` go to the voice's samples store; `best_references(manifest)` picks reference clips and `voxy.finetune.export_dataset` writes training sets (LJSpeech or F5-TTS layout).

**Diagnosis pages** (`voxy.diagnosis`): render one text through a grid of models and parameters, then build one page with a player per variant, its parameters, generation time and real-time factor. Each variant has a stable ID (e.g. `VX-7KQ2`) derived from its configuration, not the text, so listeners can say "use VX-7KQ2" across pages and chapters; each listener can rate and annotate variants and copy their notes. A spec:

```json
{"title": "First listen", "text_file": "excerpt.txt", "voice": "ada", "n_references": 1,
 "grid": [{"backend": "voxcpm", "params": {"cfg_value": [1.5, 2.0, 2.5]}},
          {"backend": "chatterbox", "params": {"exaggeration": [0.3, 0.6]}}]}
```

Cards are titled with readable names (the backend, then the settings that vary, e.g. "Chatterbox 0.3"). For a page you share with other listeners, optional spec keys help: `references` (named clips, e.g. a raw and a denoised one, each with a `label`), `appendix` (reference name → a note: those variants go in a collapsed section at the end), `imported` (clips rendered elsewhere, such as a cloud service, with their `backend`, `label` and `audio_file`), `backend_notes`, `noindex`, and `notes_sink` + `page_url`, which save every rating, note, name and clip play as they happen to an [annoquest](https://github.com/i2mint/annoquest) server (`{"kind": "annoquest", "url": "/api/annoquest", "request": "<id>"}`). The page says so, and `annoquest-request.json` is written next to it for registering on that server; add `?reader=<name>` to the link you send each listener to tell them apart.

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
