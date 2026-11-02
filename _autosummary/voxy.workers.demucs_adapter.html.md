# voxy.workers.demucs_adapter

Demucs (Meta, MIT): split a recording into a voice stem and the rest (music, noise).

Audio is read and written with soundfile; channels are matched and the rate converted
with Demucs’s own `convert_audio` (`julius.resample_frac`), so torchaudio’s I/O
backends play no part. `TORCH_HOME` (set by voxy) keeps the weights under voxy’s
models folder.

`rest` is the input minus the voice, so the two stems add back up to the input; a
model’s other stems never enter it. When the model is a bag in which one sub-model
alone makes the wanted stem (`htdemucs_ft`: one fine-tuned model per stem), only that
sub-model runs.

### Module Attributes

| [`CLIP_MODE`](#voxy.workers.demucs_adapter.CLIP_MODE)   | divide every stem by one common factor.   |
|--------------------------------------------------------------|-------------------------------------------|

### Functions

| `load`(\*[, model, device])                                                                       |                                                                               |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`separate`](#voxy.workers.demucs_adapter.separate)(state, \*, audio, out_dir[, stem, ...]) | Write `<stem>.wav` and `rest.wav` (the input minus the voice) into `out_dir`. |
| [`stem_model`](#voxy.workers.demucs_adapter.stem_model)(model, index)                         | The one sub-model of a bag that alone makes source `index`, else `model`.     |

### voxy.workers.demucs_adapter.CLIP_MODE *= 'rescale'*

divide every stem by one common factor.

* **Type:**
  Demucs’s guard against clipping

### voxy.workers.demucs_adapter.separate(state, , audio, out_dir, stem='vocals', sample_rate=None, shifts=1, overlap=0.25)

Write `<stem>.wav` and `rest.wav` (the input minus the voice) into `out_dir`.

Both are 16-bit PCM WAV with the model’s channels, at `sample_rate` (default: the
model’s rate). If either stem would clip, both are scaled by one common factor, so
`voice + rest` is the input times that factor.

### voxy.workers.demucs_adapter.stem_model(model, index)

The one sub-model of a bag that alone makes source `index`, else `model`.

A bag weighs each sub-model’s estimate of each source; when a single sub-model
has a non-zero weight for a source, the bag’s estimate of it is that sub-model’s.
