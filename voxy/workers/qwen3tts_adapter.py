"""Qwen3-TTS Base (Alibaba Qwen): voice cloning from a clip and its text (ICL mode)."""

DFLT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def load(*, model_id=DFLT_MODEL_ID, device="cpu", dtype="float32"):
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
    )
    return {"model": model, "prompts": {}}


def synthesize(
    state,
    *,
    text,
    ref_audio,
    ref_text,
    out_path,
    seed=None,
    language="English",
    non_streaming_mode=True,
    x_vector_only_mode=False,
    **params,
):
    import soundfile
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    if isinstance(ref_audio, list):
        ref_audio = ref_audio[0]
    key = (ref_audio, ref_text, x_vector_only_mode)
    if key not in state["prompts"]:  # the voice prompt is computed once per reference
        state["prompts"][key] = state["model"].create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
    wavs, sample_rate = state["model"].generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=state["prompts"][key],
        non_streaming_mode=non_streaming_mode,
        **params,
    )
    soundfile.write(out_path, wavs[0], sample_rate, subtype="PCM_16")
    return {"sample_rate": int(sample_rate), "seed": seed}
