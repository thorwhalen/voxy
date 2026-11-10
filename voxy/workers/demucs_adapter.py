"""Demucs (Meta, MIT): split a recording into a voice stem and the rest (music, noise).

Audio is read and written with soundfile; channels are matched and the rate converted
with Demucs's own ``convert_audio`` (``julius.resample_frac``), so torchaudio's I/O
backends play no part. ``TORCH_HOME`` (set by voxy) keeps the weights under voxy's
models folder.

``rest`` is the input minus the voice, so the two stems add back up to the input; a
model's other stems never enter it. When the model is a bag in which one sub-model
alone makes the wanted stem (``htdemucs_ft``: one fine-tuned model per stem), only that
sub-model runs.
"""

REST_STEM = "rest"
WAV_SUBTYPE = "PCM_16"
#: Demucs's guard against clipping: divide every stem by one common factor.
CLIP_MODE = "rescale"


def load(*, model="htdemucs_ft", device="cpu"):
    from _torch_threads import cap_torch_threads

    cap_torch_threads()

    from demucs.pretrained import get_model

    separator = get_model(model)
    separator.to(device).eval()
    return {"model": separator, "model_name": model, "device": device}


def stem_model(model, index):
    """The one sub-model of a bag that alone makes source ``index``, else ``model``.

    A bag weighs each sub-model's estimate of each source; when a single sub-model
    has a non-zero weight for a source, the bag's estimate of it is that sub-model's.
    """
    weights = getattr(model, "weights", None)
    if weights is None:
        return model
    makers = [sub for sub, w in zip(model.models, weights) if w[index]]
    return makers[0] if len(makers) == 1 else model


def separate(
    state,
    *,
    audio,
    out_dir,
    stem="vocals",
    sample_rate=None,
    shifts=1,
    overlap=0.25,
    clip_mode=CLIP_MODE,
    subtype=WAV_SUBTYPE,
):
    """Write ``<stem>.wav`` and ``rest.wav`` (the input minus the voice) into ``out_dir``.

    Both are WAV (``subtype``, default 16-bit PCM) with the model's channels, at
    ``sample_rate`` (default: the model's rate). With ``clip_mode="rescale"`` (the
    default), if either stem would clip both are scaled by one common factor, so
    ``voice + rest`` is the input times that factor; ``clip_mode=None`` with
    ``subtype="FLOAT"`` keeps the level exactly (values may exceed 1).
    """
    import os

    import julius
    import numpy as np
    import soundfile
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio, prevent_clip

    model = state["model"]
    if stem not in model.sources:
        raise ValueError(
            f"{stem!r} is not a stem of {state['model_name']}: {list(model.sources)}"
        )
    index = model.sources.index(stem)
    samples, in_rate = soundfile.read(audio, dtype="float32", always_2d=True)
    mix = convert_audio(
        torch.from_numpy(samples.T.copy()),  # (channels, frames)
        in_rate,
        model.samplerate,
        model.audio_channels,
    )
    # Demucs's own normalisation (demucs.separate): zero mean, unit std of the mono ref
    ref = mix.mean(0)
    offset, scale = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        sources = apply_model(
            stem_model(model, index),
            ((mix - offset) / scale)[None],
            device=state["device"],
            shifts=shifts,
            overlap=overlap,
        )[0]
    voice = sources[index] * scale + offset
    stems = torch.stack([voice, mix - voice])
    out_rate = sample_rate or model.samplerate
    if out_rate != model.samplerate:
        stems = julius.resample_frac(stems, model.samplerate, out_rate)
    if clip_mode:
        stems = prevent_clip(stems, mode=clip_mode)
    paths = {}
    for name, track in zip((stem, REST_STEM), stems):
        path = os.path.join(out_dir, f"{name}.wav")
        soundfile.write(
            path, np.ascontiguousarray(track.numpy().T), out_rate, subtype=subtype
        )
        paths[name] = path
    return {"paths": paths, "sample_rate": out_rate, "sources": list(model.sources)}
