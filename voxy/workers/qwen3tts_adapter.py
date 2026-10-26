"""Qwen3-TTS (Alibaba Qwen): Base voice cloning from a clip and its text (ICL mode), and VoiceDesign (a voice from a description).

The VoiceDesign model is loaded on the first ``design`` call, apart from Base.
"""

DFLT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DFLT_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def _from_pretrained(model_id, device, dtype):
    import torch
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
    )


def load(
    *,
    model_id=DFLT_MODEL_ID,
    design_model_id=DFLT_DESIGN_MODEL_ID,
    device="cpu",
    dtype="float32",
):
    return {
        "model": _from_pretrained(model_id, device, dtype),
        "prompts": {},
        "design": {"model_id": design_model_id, "device": device, "dtype": dtype},
    }


def design(
    state,
    *,
    text,
    instruct,
    out_path,
    seed=None,
    language="English",
    non_streaming_mode=True,
    **params,
):
    """Speak ``text`` in a voice described by ``instruct`` (a new voice per seed)."""
    import soundfile
    import torch

    config = state["design"]
    if "model" not in config:  # 4+ GB: only when a voice is actually designed
        config["model"] = _from_pretrained(
            config["model_id"], config["device"], config["dtype"]
        )
    if seed is not None:
        torch.manual_seed(seed)
    wavs, sample_rate = config["model"].generate_voice_design(
        text=text,
        instruct=instruct,
        language=language,
        non_streaming_mode=non_streaming_mode,
        **params,
    )
    soundfile.write(out_path, wavs[0], sample_rate, subtype="PCM_16")
    return {"sample_rate": int(sample_rate), "seed": seed}


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
