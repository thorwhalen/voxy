# voxy.workers.qwen3tts_adapter

Qwen3-TTS (Alibaba Qwen): Base voice cloning from a clip and its text (ICL mode), and VoiceDesign (a voice from a description).

The VoiceDesign model is loaded on the first `design` call, apart from Base.

### Functions

| [`design`](#voxy.workers.qwen3tts_adapter.design)(state, \*, text, instruct, out_path[, ...])   | Speak `text` in a voice described by `instruct` (a new voice per seed).   |
|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| `load`(\*[, model_id, design_model_id, device, ...])                                                  |                                                                           |
| `synthesize`(state, \*, text, ref_audio, ...[, ...])                                                  |                                                                           |

### voxy.workers.qwen3tts_adapter.design(state, , text, instruct, out_path, seed=None, language='English', non_streaming_mode=True, \*\*params)

Speak `text` in a voice described by `instruct` (a new voice per seed).
