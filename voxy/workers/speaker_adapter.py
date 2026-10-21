"""Speaker embeddings (SpeechBrain ECAPA-TDNN) of sliding windows, for speaker filtering."""

import os

DFLT_SAMPLE_RATE = 16000
DFLT_BATCH = 64


def load(*, model="speechbrain/spkrec-ecapa-voxceleb", device="cpu"):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    savedir = os.path.join(os.environ.get("VOXY_SPEAKER_MODELS", "."), "ecapa")
    encoder = EncoderClassifier.from_hparams(
        source=model, savedir=savedir, run_opts={"device": device}
    )
    return {"torch": torch, "encoder": encoder}


def embed(state, *, audio, out_path, spans=None, window=1.0, hop=0.1, batch=DFLT_BATCH):
    """Embed windows of ``window`` s every ``hop`` s inside ``spans`` (default: all).

    Saves ``times`` (n, 2) and unit ``vectors`` (n, d) to the ``.npz`` ``out_path``.
    A span shorter than ``window`` gets one window, its own length.
    """
    import librosa
    import numpy as np

    torch = state["torch"]
    samples, sr = librosa.load(audio, sr=DFLT_SAMPLE_RATE, mono=True)
    total = len(samples) / sr
    spans = spans or [[0.0, total]]
    times = []
    for start, end in spans:
        start, end = max(0.0, start), min(total, end)
        if end - start <= window:
            if end > start:
                times.append((start, end))
            continue
        n = int((end - start - window) // hop) + 1
        times += [(start + i * hop, start + i * hop + window) for i in range(n)]
    vectors = []
    for i in range(0, len(times), batch):
        chunk = times[i : i + batch]
        pieces = [samples[int(s * sr) : int(e * sr)] for s, e in chunk]
        width = max(len(p) for p in pieces)
        padded = np.stack([np.pad(p, (0, width - len(p))) for p in pieces])
        lengths = torch.tensor([len(p) / width for p in pieces])
        with torch.no_grad():
            emb = state["encoder"].encode_batch(torch.from_numpy(padded), lengths)
        vectors.append(torch.nn.functional.normalize(emb.squeeze(1), dim=-1).numpy())
    dim = vectors[0].shape[1] if vectors else 0
    np.savez(
        out_path,
        times=np.array(times, dtype=float).reshape(-1, 2),
        vectors=np.concatenate(vectors) if vectors else np.zeros((0, dim)),
    )
    return {"out_path": out_path, "windows": len(times)}
