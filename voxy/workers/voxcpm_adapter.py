"""VoxCPM2 (OpenBMB): tokenizer-free diffusion-autoregressive cloning, 48 kHz output."""


def load(
    *, model_id="openbmb/VoxCPM2", device="cpu", optimize=False, lora_weights_path=None
):
    from voxcpm import VoxCPM

    return VoxCPM.from_pretrained(
        model_id,
        load_denoiser=False,
        optimize=optimize,
        device=device,
        lora_weights_path=lora_weights_path,
    )


def synthesize(
    state,
    *,
    text,
    ref_audio,
    ref_text,
    out_path,
    seed=None,
    prompt_mode="continuation",
    **params,
):
    """``prompt_mode``: 'continuation' (clip + its text: closest match), 'reference'
    (clip only, isolated), or 'both'."""
    import soundfile
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    if isinstance(ref_audio, list):
        ref_audio = ref_audio[0]
    prompt = {}
    if prompt_mode in ("continuation", "both"):
        prompt.update(prompt_wav_path=ref_audio, prompt_text=ref_text)
    if prompt_mode in ("reference", "both"):
        prompt.update(reference_wav_path=ref_audio)
    if not prompt:
        raise ValueError(f"Unknown prompt_mode {prompt_mode!r}")
    wav = state.generate(text=text, **prompt, **params)
    sample_rate = int(state.tts_model.sample_rate)
    soundfile.write(out_path, wav, sample_rate, subtype="PCM_16")
    return {"sample_rate": sample_rate, "seed": seed}
