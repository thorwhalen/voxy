"""Demucs (Meta, MIT): split a recording into a voice stem and the rest (music, noise).

Audio is read and written with soundfile, and resampled with torchaudio's
functional resampler, so the result does not depend on torchaudio's I/O backends.
``TORCH_HOME`` (set by voxy) keeps the weights under voxy's models folder.
"""


def load(*, model="htdemucs_ft", device="cpu"):
    from demucs.pretrained import get_model

    separator = get_model(model)
    separator.to(device).eval()
    return {"model": separator, "device": device}


def separate(state, *, audio, out_dir, stem="vocals", shifts=1, overlap=0.25):
    """Write ``<stem>.wav`` and ``rest.wav`` (everything else) into ``out_dir``."""
    import os

    import numpy as np
    import soundfile
    import torch
    import torchaudio.functional as F
    from demucs.apply import apply_model

    model = state["model"]
    samples, sample_rate = soundfile.read(audio, dtype="float32", always_2d=True)
    wav = torch.from_numpy(samples.T.copy())  # (channels, frames)
    if wav.shape[0] == 1:
        wav = wav.repeat(model.audio_channels, 1)
    wav = F.resample(wav, sample_rate, model.samplerate)
    ref = wav.mean(0)
    scale = ref.std() + 1e-8
    with torch.no_grad():
        sources = apply_model(
            model,
            ((wav - ref.mean()) / scale)[None],
            device=state["device"],
            shifts=shifts,
            overlap=overlap,
        )[0]
    sources = sources * scale + ref.mean()
    index = model.sources.index(stem)
    voice = sources[index]
    rest = sources.sum(0) - voice
    paths = {}
    for name, track in ((stem, voice), ("rest", rest)):
        path = os.path.join(out_dir, f"{name}.wav")
        soundfile.write(
            path,
            np.ascontiguousarray(track.numpy().T),
            model.samplerate,
            subtype="PCM_16",
        )
        paths[name] = path
    return {
        "paths": paths,
        "sample_rate": model.samplerate,
        "sources": list(model.sources),
    }
