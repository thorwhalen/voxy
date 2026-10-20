# voxy.workers

Backend adapters run by `voxy.worker` inside each backend’s own environment.

`serve.py` is the entry point; each `*_adapter.py` wraps one model library
with `load(**init)` and `synthesize(state, *, text, ref_audio, ref_text,
out_path, \*\*params)`. These files are executed by other interpreters, so they
import nothing from voxy.

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

| [`asr_adapter`](voxy.workers.asr_adapter.html.md#module-voxy.workers.asr_adapter)               | Speech recognition with timed words (scribed's faster-whisper backend), for reference prep.                                                       |
|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [`chatterbox_adapter`](voxy.workers.chatterbox_adapter.html.md#module-voxy.workers.chatterbox_adapter) | Chatterbox (Resemble AI): 'original' (emotion exaggeration, CFG), 'turbo' (faster), or 'multilingual' (23 languages; pass `language`, e.g. 'fr'). |
| [`f5tts_adapter`](voxy.workers.f5tts_adapter.html.md#module-voxy.workers.f5tts_adapter)           | F5-TTS (flow matching, zero-shot cloning from a reference clip and its text).                                                                     |
| [`phonemes_adapter`](voxy.workers.phonemes_adapter.html.md#module-voxy.workers.phonemes_adapter)     | Phoneme recognition (wav2vec2 espeak CTC): frame-wise log-probabilities of a crop.                                                                |
| [`qwen3tts_adapter`](voxy.workers.qwen3tts_adapter.html.md#module-voxy.workers.qwen3tts_adapter)     | Qwen3-TTS Base (Alibaba Qwen): voice cloning from a clip and its text (ICL mode).                                                                 |
| [`serve`](voxy.workers.serve.html.md#module-voxy.workers.serve)                           | Worker entry point: run a backend adapter inside its own environment.                                                                             |
| [`voxcpm_adapter`](voxy.workers.voxcpm_adapter.html.md#module-voxy.workers.voxcpm_adapter)         | VoxCPM2 (OpenBMB): tokenizer-free diffusion-autoregressive cloning, 48 kHz output.                                                                |
| [`xtts_adapter`](voxy.workers.xtts_adapter.html.md#module-voxy.workers.xtts_adapter)             | XTTS-v2 (Coqui, maintained by idiap as coqui-tts): GPT-style cloning from a short clip.                                                           |
