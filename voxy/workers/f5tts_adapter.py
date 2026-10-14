"""F5-TTS (flow matching, zero-shot cloning from a reference clip and its text)."""


def load(*, model="F5TTS_v1_Base", ckpt_file="", vocab_file="", device="cpu", **kwargs):
    from f5_tts.api import F5TTS

    return F5TTS(
        model=model, ckpt_file=ckpt_file, vocab_file=vocab_file, device=device, **kwargs
    )


def synthesize(state, *, text, ref_audio, ref_text, out_path, seed=None, **params):
    import soundfile

    wav, sample_rate, _ = state.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=text,
        seed=seed,
        show_info=lambda *a, **k: None,
        **params,
    )
    soundfile.write(out_path, wav, sample_rate, subtype="PCM_16")
    return {"sample_rate": int(sample_rate), "seed": state.seed}
