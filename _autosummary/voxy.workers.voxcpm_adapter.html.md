# voxy.workers.voxcpm_adapter

VoxCPM2 (OpenBMB): tokenizer-free diffusion-autoregressive cloning, 48 kHz output.

### Functions

| `load`(\*[, model_id, device, optimize, ...])                                                       |                                                                                                               |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| [`synthesize`](#voxy.workers.voxcpm_adapter.synthesize)(state, \*, text, ref_audio, ...[, ...]) | `prompt_mode`: 'continuation' (clip + its text: closest match), 'reference' (clip only, isolated), or 'both'. |

### voxy.workers.voxcpm_adapter.synthesize(state, , text, ref_audio, ref_text, out_path, seed=None, prompt_mode='continuation', \*\*params)

`prompt_mode`: ‘continuation’ (clip + its text: closest match), ‘reference’
(clip only, isolated), or ‘both’.
