# voxy.workers.speaker_adapter

Speaker embeddings (SpeechBrain ECAPA-TDNN) of sliding windows, for speaker filtering.

### Functions

| [`embed`](#voxy.workers.speaker_adapter.embed)(state, \*, audio, out_path[, spans, ...])   | Embed windows of `window` s every `hop` s inside `spans` (default: all).   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `load`(\*[, model, device])                                                                        |                                                                            |

### voxy.workers.speaker_adapter.embed(state, , audio, out_path, spans=None, window=1.0, hop=0.1, batch=64)

Embed windows of `window` s every `hop` s inside `spans` (default: all).

Saves `times` (n, 2) and unit `vectors` (n, d) to the `.npz` `out_path`.
A span shorter than `window` gets one window, its own length.
