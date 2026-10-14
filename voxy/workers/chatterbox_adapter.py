"""Chatterbox (Resemble AI): 'original' (emotion exaggeration, CFG) or 'turbo' (faster)."""


def load(*, variant="original", device="cpu"):
    if variant == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
    elif variant == "original":
        from chatterbox.tts import ChatterboxTTS as Model
    else:
        raise ValueError(f"Unknown Chatterbox variant {variant!r} (original, turbo)")
    return Model.from_pretrained(device=device)


def synthesize(state, *, text, ref_audio, ref_text, out_path, seed=None, **params):
    import soundfile
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    wav = state.generate(text, audio_prompt_path=ref_audio, **params)
    soundfile.write(out_path, wav.squeeze(0).cpu().numpy(), state.sr, subtype="PCM_16")
    return {"sample_rate": int(state.sr), "seed": seed}
