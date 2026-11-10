# voxy.workers

Backend adapters run by `voxy.worker` inside each backend’s own environment.

Each `*_adapter.py` wraps one model library with `load(**init)` and operations
such as `synthesize(state, *, text, ref_audio, ref_text, out_path, **params)`;
`ps.isolated.Worker` serves it. These files are executed by other interpreters,
so they import nothing from voxy; this folder is on their `sys.path`, so they
share helpers by plain import (`from _worker_threads import cap_torch_threads`).

### Functions

| [`adapter_path`](#voxy.workers.adapter_path)(name)   | The adapter file for `name` (e.g. 'f5tts' -> `f5tts_adapter.py`).   |
|-----------------------------------------------------------------------|---------------------------------------------------------------------|

### voxy.workers.adapter_path(name)

The adapter file for `name` (e.g. ‘f5tts’ -> `f5tts_adapter.py`).

```pycon
>>> adapter_path("f5tts").name
'f5tts_adapter.py'
```

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### Modules

| [`align_adapter`](voxy.workers.align_adapter.html.md#module-voxy.workers.align_adapter)           | Forced alignment with torchaudio's `MMS_FA` bundle (Meta's multilingual CTC aligner): character spans of words already known to be in the audio.   |
|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| [`asr_adapter`](voxy.workers.asr_adapter.html.md#module-voxy.workers.asr_adapter)               | Speech recognition with timed words (scribed's faster-whisper backend), for reference prep.                                                        |
| [`chatterbox_adapter`](voxy.workers.chatterbox_adapter.html.md#module-voxy.workers.chatterbox_adapter) | Chatterbox (Resemble AI): 'original' (emotion exaggeration, CFG), 'turbo' (faster), or 'multilingual' (23 languages; pass `language`, e.g. 'fr').  |
| [`demucs_adapter`](voxy.workers.demucs_adapter.html.md#module-voxy.workers.demucs_adapter)         | Demucs (Meta, MIT): split a recording into a voice stem and the rest (music, noise).                                                               |
| [`f5tts_adapter`](voxy.workers.f5tts_adapter.html.md#module-voxy.workers.f5tts_adapter)           | F5-TTS (flow matching, zero-shot cloning from a reference clip and its text).                                                                      |
| [`kokoro_adapter`](voxy.workers.kokoro_adapter.html.md#module-voxy.workers.kokoro_adapter)         | Kokoro-82M through kokoro-onnx: stock voices on CPU, no torch, no reference clip.                                                                  |
| [`phonemes_adapter`](voxy.workers.phonemes_adapter.html.md#module-voxy.workers.phonemes_adapter)     | Phoneme recognition (wav2vec2 espeak CTC): frame-wise log-probabilities of a crop, and espeak phonemization.                                       |
| [`qwen3tts_adapter`](voxy.workers.qwen3tts_adapter.html.md#module-voxy.workers.qwen3tts_adapter)     | Qwen3-TTS (Alibaba Qwen): Base voice cloning from a clip and its text (ICL mode), and VoiceDesign (a voice from a description).                    |
| [`speaker_adapter`](voxy.workers.speaker_adapter.html.md#module-voxy.workers.speaker_adapter)       | Speaker embeddings (SpeechBrain ECAPA-TDNN) of sliding windows, for speaker filtering.                                                             |
| [`voxcpm_adapter`](voxy.workers.voxcpm_adapter.html.md#module-voxy.workers.voxcpm_adapter)         | VoxCPM2 (OpenBMB): tokenizer-free diffusion-autoregressive cloning, 48 kHz output.                                                                 |
| [`xtts_adapter`](voxy.workers.xtts_adapter.html.md#module-voxy.workers.xtts_adapter)             | XTTS-v2 (Coqui, maintained by idiap as coqui-tts): GPT-style cloning from a short clip.                                                            |
