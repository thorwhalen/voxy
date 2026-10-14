"""XTTS-v2 (Coqui, maintained by idiap as coqui-tts): GPT-style cloning from a short clip."""

DFLT_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def load(
    *, model_name=DFLT_MODEL_NAME, model_path=None, config_path=None, device="cpu"
):
    from TTS.api import TTS

    if model_path:  # a fine-tuned checkpoint
        return TTS(model_path=model_path, config_path=config_path).to(device)
    return TTS(model_name).to(device)


def synthesize(
    state, *, text, ref_audio, ref_text, out_path, seed=None, language="en", **params
):
    import numpy
    import soundfile
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    refs = ref_audio if isinstance(ref_audio, list) else [ref_audio]
    wav = state.tts(text=text, speaker_wav=refs, language=language, **params)
    sample_rate = int(state.synthesizer.output_sample_rate)
    soundfile.write(out_path, numpy.asarray(wav), sample_rate, subtype="PCM_16")
    return {"sample_rate": sample_rate, "seed": seed}
