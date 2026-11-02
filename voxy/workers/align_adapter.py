"""Forced alignment with torchaudio's ``MMS_FA`` bundle (Meta's multilingual CTC
aligner): character spans of words already known to be in the audio.

Run by ``voxy.worker`` in its own environment; audio is read with soundfile. The
aligner's weights come from Meta's MMS release, under a non-commercial licence
(CC-BY-NC-4.0, as far as the MMS release states; check before commercial use).
"""

import unicodedata


def load(*, model="MMS_FA", device="cpu"):
    import torchaudio

    bundle = getattr(torchaudio.pipelines, model)
    aligner = bundle.get_model().to(device).eval()
    labels = bundle.get_labels(star=None)
    return {
        "bundle": bundle,
        "model": aligner,
        "device": device,
        "index": {c: i for i, c in enumerate(labels)},
    }


def _letters(word, index):
    """``word`` lowercased, accents dropped, keeping the aligner's letters only."""
    plain = unicodedata.normalize("NFKD", word.lower())
    return "".join(c for c in plain if c in index and c != "-")


def align(state, *, audio, words):
    """``{"words": [{"start", "end", "score", "chars": [...]} or None, ...]}`` for
    ``words`` (in order) in ``audio``; a word with no alignable letter gets None."""
    import soundfile
    import torch
    import torchaudio.functional as F

    bundle, index = state["bundle"], state["index"]
    samples, rate = soundfile.read(audio, dtype="float32", always_2d=True)
    wav = torch.from_numpy(samples.mean(axis=1).copy())[None]
    wav = F.resample(wav, rate, bundle.sample_rate)
    letters = [_letters(w, index) for w in words]
    tokens = [index[c] for ls in letters for c in ls]
    if not tokens:
        return {"words": [None] * len(words)}
    with torch.inference_mode():
        emission, _ = state["model"](wav.to(state["device"]))
    path, scores = F.forced_align(
        emission.cpu(), torch.tensor([tokens], dtype=torch.int32), blank=0
    )
    spans = F.merge_tokens(path[0], scores[0].exp())
    seconds = wav.shape[1] / emission.shape[1] / bundle.sample_rate
    out, k = [], 0
    for word_letters in letters:
        if not word_letters:
            out.append(None)
            continue
        chars = [
            {
                "char": c,
                "start": round(s.start * seconds, 4),
                "end": round(s.end * seconds, 4),
                "score": round(float(s.score), 4),
            }
            for c, s in zip(word_letters, spans[k : k + len(word_letters)])
        ]
        k += len(word_letters)
        out.append(
            {
                "start": chars[0]["start"],
                "end": chars[-1]["end"],
                "score": round(sum(c["score"] for c in chars) / len(chars), 4),
                "chars": chars,
            }
        )
    return {"words": out}
