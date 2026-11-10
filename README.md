# voxy

Facade for voice generation: speech synthesis in any voice, on any service, plus cloning and designing voices.

To install:	```pip install voxy``` (add `'voxy[elevenlabs]'`, `'voxy[aix]'` or `'voxy[fal]'` for those services). torch is not installed by default: the local CSM model and tensor helpers (`cleanup_audio`, `generate_speech`) need `'voxy[local]'`, and Whisper transcription `'voxy[transcription]'`; the local backends that run in their own environments (kokoro, f5tts, ...) need neither. The part-of-speech heteronym detector (`voxy.speechtext.pos_detector`) needs `'voxy[pos]'` plus `python -m spacy download fr_core_news_sm`. The LLM proposer (`voxy.speechtext.llm_detector`) uses `'voxy[aix]'` unless you pass `chat=`.

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

Backends: `elevenlabs` (cloning, voice design, TTS), `say` (macOS, offline), `aix` (OpenAI voices and other LiteLLM providers), `fal` (fal.ai models via falaw), `kokoro` (local stock voices, CPU, no key; see below), `voicestudio` (local text-described voices, `voxy[voicestudio]`), `csm` (local Sesame CSM-1B). Register more with `voxy.register_speech_model(name, factory)`. Singing (`voxy.sing`, `voxy.convert_singing`): `soulx` (SoulX-Singer, local), see below.

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

Change the voice of a recording (speech to speech, ElevenLabs): the words, timing and intonation stay the recording's, the voice becomes the target's. A correction can be spoken by anyone and come out in the narrator's voice:

```python
voxy.speech_to_speech("my_take.wav", voice="ada").save("ada_take.mp3")
```

### Pauses: SSML breaks, on any backend

Write silences as standard SSML breaks in the text. voxy keeps a break the backend renders itself (ElevenLabs' v2 models, up to 3 s) and, for any other one, speaks the parts separately and joins them with exactly that much silence (as WAV; or, when you ask for a compressed `output_format` such as `mp3_44100_128`, in that format through ffmpeg, which is how to get 44.1 kHz on ElevenLabs tiers below Pro). `eleven_v3` and `eleven_v4` render no SSML, so every break is cut there:

```python
model = voxy.ElevenLabsSpeechModel(
    model_id="eleven_v4"
)  # joins at pcm_24000 (44.1 kHz PCM needs Pro)
voxy.text_to_speech(
    '[calmly] Breathe in.<break time="4s"/> And out, slowly.<break time="6s"/>',
    "my-voice-id",
    model=model,
).save("breath.wav")
```

`time` takes `s` or `ms`; `<break strength="strong"/>` works too. `voxy.parse_sections(text)` gives the text as `Section(text, pause_s, mark)`, cut at every break and at every `<mark name="..."/>`: what a player that waits at a mark for the listener would read. Why SSML and not a custom tag, and the limits per engine: `misc/docs/research/ssml-pauses-and-hold-points.md`.

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

### A local stock voice (no service, no clip, no torch)

`kokoro` speaks in one of 54 stock voices (`af_heart`, `am_michael`, `bf_emma`, ...) with Kokoro-82M (Apache-2.0) through kokoro-onnx, faster than realtime on a CPU. It runs in its own small environment, which `python -m voxy setup kokoro` builds (or the first call does); the model files (about 354 MB) are fetched from the Hugging Face hub on first use into `$VOXY_DATA_DIR/models/hf`, like every other local backend's weights. Listing its voices needs neither. A voice's first letter is its language (`a` American, `b` British, `f` French, `e` Spanish, `i` Italian, `p` Portuguese, `h` Hindi, `j` Japanese, `z` Chinese). It runs on Linux and macOS (an environment built with the older kokoro-onnx pins failed on macOS, thorwhalen/voxy#61; delete `$VOXY_DATA_DIR/envs/kokoro` and run `python -m voxy setup kokoro` if you still see that), with no cloud key.

```python
voxy.text_to_speech("Hello.", "bf_emma", backend="kokoro", speed=1.1).save("hi.wav")
voxy.list_voices("kokoro")
```

### Separating the voice from music

`voxy.separate.separate("song.mp3")` returns `{"vocals": wav_bytes, "rest": wav_bytes}`: Demucs (`htdemucs_ft` by default, MIT) in its own environment, built on first use (or `python -m voxy setup demucs`), weights under `$VOXY_DATA_DIR/models/demucs`. Use it before cloning, transcribing or editing a voice recorded over music. `rest` is the input minus the voice, so the two add back up to it. Both are 16-bit stereo WAV at 44.1 kHz whatever the input (`sample_rate=` to change the rate); loud input is scaled down, both stems by the same factor, rather than clipped. The whole file is held in memory, and it takes about 24 s per minute of audio on CPU.

### Correcting words in a recording

`voxy.replace_words` replaces a spoken phrase with a new one in the speaker's voice: it finds the phrase in the recording's ASR words, renders the new phrase (TTS in `voice=` by default, or any `render=`), cuts at the quietest moment near each end, matches the loudness of the speech it replaces (pauses excluded), crossfades, and checks that the new words are heard there and the old ones are gone. Pass `voice=` (the speaker's voice) or `render=`.

```python
fixed = voxy.replace_words("narration.wav", "Anna Jones", "Anna Smith", voice="ada")
fixed.passed, fixed.checks  # True, [{'check': 'heard', 'passed': True, ...}]
fixed.save("narration-fixed.wav")
```

Replace the whole phrase around a word (first name and surname), not the word alone: the neighbouring sounds anticipate it. `old` is the phrase as ASR heard it.

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

Long text is split into paragraphs and sentence-packed chunks, each spoken separately and joined with controlled pauses (`gap_ms=`, `paragraph_gap_ms=`). Model parameters are keyword arguments of `synthesize` (see `voxy.local_clone.local_backends()[name].params`); model options (a variant, a fine-tuned checkpoint) are keyword arguments of `create_speech_model`. Threads are capped at `$VOXY_NUM_THREADS` (default 8). Licences differ: F5-TTS and XTTS-v2 weights are non-commercial (XTTS needs `accept_licence=True`); VoxCPM2, Qwen3-TTS and Chatterbox are Apache-2.0/MIT.

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

**Reference preparation** (`voxy.refprep.prepare_reference`): converts any recording to mono WAV, denoises (`denoise=` `afftdn`, `anlmdn`, or `deepfilternet`, which lowers real room noise far more but builds its own torch environment on first use) and normalises loudness (ffmpeg), transcribes locally with Whisper (scribed's faster-whisper backend) with word timings, cuts clips of 3-12 s at sentence boundaries, and scores each clip (signal-to-noise, clipping, level, ASR confidence, speaking rate). With `speaker=voxy.speakers.speaker_filter()` (`--speaker`) it keeps one voice: ECAPA speaker embeddings (SpeechBrain, in its own environment) decide each transcribed turn as a whole against the target's voice print (the recording's dominant voice by default, or `speakers.voice_print(clean_recordings)`), and interviewer questions, interjections such as "Yes." and untranscribed "uh-huh"s never end up inside a clip. Clips and `manifest.jsonl` go to the voice's samples store, which a run only ever adds to: running again (more recordings, or the same file re-cut) writes under fresh keys and keeps every earlier clip, its bytes and its manifest entry, so saved reference profiles stay valid; the manifest records each source as soon as it is done, so a run that fails part-way loses only the failing source's work, whose keys (a conditioned file, maybe some clips) no entry names. `refprep.forget(samples, source, run=None, dry_run=False)` removes a source's clips (`dry_run=True` lists what it would delete), and `refprep.orphans(samples)` lists clip and conditioned keys no manifest entry claims (a crashed run's, a zero-clip source's), which `refprep.forget_orphans(samples, dry_run=False)` deletes (it is a dry run by default); those two are the only functions that delete samples. A source is identified by its real path in the case the filesystem stores it, so `Talk.wav` and `talk.wav` typed for one file on macOS or Windows are one source. `best_references(manifest)` picks reference clips and `voxy.finetune.export_dataset` writes training sets (LJSpeech or F5-TTS layout), both from each source's latest run (`refprep.latest_runs`; `latest_only=False` for every run).

**Diagnosis pages** (`voxy.diagnosis`): render one text through a grid of models and parameters, then build one page with a player per variant, its parameters, generation time and real-time factor. Each variant has a stable ID (e.g. `VX-7KQ2`) derived from its configuration, not the text, so listeners can say "use VX-7KQ2" across pages and chapters; each listener can rate and annotate variants and copy their notes. A spec:

```json
{"title": "First listen", "text_file": "excerpt.txt", "voice": "ada", "n_references": 1,
 "grid": [{"backend": "voxcpm", "params": {"cfg_value": [1.5, 2.0, 2.5]}},
          {"backend": "chatterbox", "params": {"exaggeration": [0.3, 0.6]}}]}
```

Cards are titled with readable names (the backend, then the settings that vary, e.g. "Chatterbox 0.3"). For a page you share with other listeners, optional spec keys help: `references` (named clips, e.g. a raw and a denoised one, each with a `label`), `appendix` (reference name → a note: those variants go in a collapsed section at the end), `imported` (clips rendered elsewhere, such as a cloud service, with their `backend`, `label` and `audio_file`), `backend_notes`, `noindex`, and `notes_sink` + `page_url`, which save every rating, note, name and clip play as they happen to an [annoquest](https://github.com/i2mint/annoquest) server (`{"kind": "annoquest", "url": "/api/annoquest", "request": "<id>"}`). The page says so, and `annoquest-request.json` is written next to it for registering on that server; add `?reader=<name>` to the link you send each listener to tell them apart.

### Singing: a score with lyrics, or a sung take re-voiced

`voxy.sing` sings a melody (MIDI, MusicXML, a music21 stream, or `voxy.score.Note`s) with lyrics in a voice; `voxy.convert_singing` re-voices a sung take, keeping its melody, words and timing. The backend is `soulx` ([SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer), Soul AI Lab, code and weights Apache-2.0), in its own environment like the local cloners: `python -m voxy setup soulx`, or built on first use (Python 3.11, torch 2.2; needs `uv` and `git`). SoulX-Singer is not pip-installable, so voxy fetches its repository at a pinned commit into `$VOXY_DATA_DIR/models/soulx/source/`; the weights (two 2.8 GB checkpoints plus RMVPE) come from the Hugging Face Hub at pinned revisions; `openai/whisper-base` (loaded by upstream for conversion) and g2p_en's NLTK data are fetched unpinned, at their latest versions, on first use. Reading scores needs `'voxy[singing]'` (mido for MIDI, music21 for MusicXML).

```python
import voxy

# lyrics: one token per note; "can-dle" spreads a word over two notes, "-" holds it one more
voxy.sing("melody.mid", "Light a can-dle in the win-dow").save("sung.wav")
voxy.sing("song.musicxml", transpose=-2).save("song.wav")  # the score's own lyrics, 2 semitones down
voxy.sing("arrangement.mid", track=2)  # a MIDI file with several lines: say which one sings

# a voice: a sung clip and the score it sings, timed as in the clip
voice = voxy.singing_voice("phrase-take.wav", "phrase.mid", "la la la la la la la la")
voxy.sing("melody.mid", "Light a can-dle in the win-dow", voice=voice, seed=0)
voxy.convert_singing("take.wav", voice=voice).save("take-revoiced.wav")
```

**Making a voice.** Synthesis copies the timbre of a sung clip and aligns that clip with its notes and phonemes, so a voice is a clip plus its score. The simplest way to get both: record the voice singing a short phrase written beforehand (a MIDI or MusicXML file with its lyrics, sung in time, 5 to 30 s, no accompaniment), and pass the recording with that score. Conversion needs only the clip. Without `voice=`, a synthetic voice is used (`voxy.synthetic_singing_voice("high" | "low")`: a formant synthesiser singing "la", which belongs to no one); words sung in it are less clear than with a real voice. A voice is plain data, so `voxy.save_voice(name, voice, consent="...")` keeps it in the library and `voice=name` finds it again.

**Scores.** A MIDI file's melody is its one track and channel with notes (drums aside); with several, the error lists them and `track=` (and `channel=`, 1 to 16) picks one; a line with chords is refused. Lyric events are syllables: `can-` / `-dle` join into one word, karaoke `/` and `\` marks are dropped, and a note without a syllable holds the word before it unless a rest came between. MusicXML is read through music21: the part's first voice (`part=` picks the part), syllabic lyrics, ties merged, grace notes skipped, the top note of a chord. Each sung note gets one word without spaces.

**Options.** `transpose=` is semitones (0, the default for `sing`: as written) or `"fit_voice"` (into the voice's register, the default for `convert_singing`). Model parameters are keyword arguments: `n_steps=32` (16 is twice as fast but blurs words), `cfg=3.0`, `seed=`, `peak=0.95` (output above it is scaled down), `crossfade_s=0.02` (fades where segments meet: a song is sung in segments of at most 30 s, cut at rests where possible). `"fit_voice"` is computed once for the whole song (median note of the voice's clip minus median note of the score). English lyrics only for now (upstream also sings Mandarin and Cantonese). On Apple Silicon it runs on MPS with the vocoder on the CPU (torch 2.2's MPS backend fails on its complex multiply): about 1.4x real time on an M1 Max once loaded (a first call, which loads the model, took 48 s for 6 s of singing, and conversion 35 s) and up to 6 GB of memory per model; on the CPU alone about 8x real time. A model stays loaded (about 6 GB) until `voxy.close_singing_models()` or the end of the process; switching between `sing` and `convert_singing` swaps one model for the other rather than holding both.

**Consent.** SoulX-Singer's usage disclaimer (not a licence term) forbids impersonating anyone without authorisation and making deceptive audio, and asks to respect intellectual property, privacy and consent. Clone only your own voice, a consenting singer's, or a synthetic one. Each singing backend records its restrictions in `voxy.singing_backends()[name].consent`.

### Heteronyms: what to send, and checking what was said

Speech models misread words with two pronunciations (French *est* as the verb /ɛ/ or the noun /ɛst/, *plus*, *tous*, *fils*, the *-ent* verbs), and a verse line's capital makes it worse. An ASR check never sees it: Whisper writes "est" either way. Prepare the text, then audit the audio:

```python
from voxy.speechtext import prepare_speech_text
from voxy.pronounce import audit

text = prepare_speech_text(
    verse, keep={"Vous"}
).speech  # verse capitals lowercased, deliberate ones kept
# ... render `text` to take.wav with any backend ...
findings = audit(
    text, "take.wav", expected={"est": "verb"}
)  # voxy's French heteronym table by default
misread = [f for f in findings if f.ok is False]  # re-render these
```

voxy ships a short hand-made French table (`heteronym_readings("fr")`). For more, derive one from your copy of [Lexique](http://www.lexique.org) (voxy ships no Lexique data): `voxy.lexique.cached_lexique_heteronyms("Lexique383.tsv", min_frequency=1)` (derived once, then kept in voxy's cache) returns the spellings Lexique gives more than one pronunciation, in the same `{word: {reading: IPA}}` shape, readings named by part of speech (or by lemma when two share one) and pronunciations that differ only by Lexique's open/closed vowels or schwa folded into `name~variant`. Merge it under the hand-made table, `{**derived, **heteronym_readings("fr")}`: Lexique keeps one pronunciation per spelling, part of speech and lemma, and no liaison or variant readings, so *tous*, *os*, *six*, *dix*, *christ*, *lacs* and *fait* are not in it.

`audit` finds each flagged word in the render and scores its readings, written in IPA, as the whole phrase around it (neighbours written out by a bundled espeak-ng) with a wav2vec2 phoneme recogniser; both run in their own environment, built on first use. Expected readings come from `expected` (a mapping, or a function of each risk) or from a detector such as `voxy.speechtext.pos_detector()`. A heteronym elided onto a clitic that can precede its noun reading (French *l'Est*, *vent d'est*) is flagged and checked too; name it in the mapping by its clitic (`{"est": "verb", "l'est": "noun"}`), since `"est"` alone does not cover it. A mapping is checked whole up front: a key that is not a word of the readings table (a typo), or a reading that word does not have, raises a `ValueError` before any ASR, and `audit_verifier` raises it when built, before the first render; a detector's proposal naming no reading is dropped with a warning. While rendering, `render_verified` scores the same way and re-renders a line whose word came out wrong; its one `verify=` seam, `(text, audio, line) -> findings`, defaults to `audit` on the line, occurrence by occurrence, elided forms included; pass `verify=audit_verifier(lines, readings=..., expected=..., detectors=...)` for `audit`'s own options. The older `checks=` (`checks_by_line`, `verify_words`) is a legacy adapter that matches words by spelling. `audit` takes `scoring=` (`"phrase"`, `"spot"` or `"crop"`), and all three take a `scorer=` (a name or a scorer from `phrase_scorer`/`spot_scorer`/`crop_scorer`, each carrying the margin it is calibrated for); cropping the word at its ASR times credits a neighbour's /s/ or /t/ to the longer reading, so keep the default.

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
