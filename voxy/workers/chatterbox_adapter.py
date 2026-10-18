"""Chatterbox (Resemble AI): 'original' (emotion exaggeration, CFG), 'turbo' (faster), or
'multilingual' (23 languages; pass ``language``, e.g. 'fr')."""

VARIANTS = ("original", "turbo", "multilingual")


def load(*, variant="original", device="cpu"):
    if variant == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
    elif variant == "original":
        from chatterbox.tts import ChatterboxTTS as Model
    elif variant == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
    else:
        raise ValueError(f"Unknown Chatterbox variant {variant!r} ({', '.join(VARIANTS)})")
    return Model.from_pretrained(device=device)


def synthesize(state, *, text, ref_audio, ref_text, out_path, seed=None, language="en", **params):
    import soundfile
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    if type(state).__name__ == "ChatterboxMultilingualTTS":
        params["language_id"] = language
    elif language not in ("en", "english", "English"):
        raise ValueError(f"This Chatterbox variant speaks English only; use variant='multilingual' for {language!r}")
    wav = state.generate(text, audio_prompt_path=ref_audio, **params)
    soundfile.write(out_path, wav.squeeze(0).cpu().numpy(), state.sr, subtype="PCM_16")
    return {"sample_rate": int(state.sr), "seed": seed}
