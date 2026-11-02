"""Forced alignment with torchaudio's ``MMS_FA`` bundle (Meta's multilingual CTC
aligner): character spans of words already known to be in the audio.

Run by ``voxy.worker`` in its own environment; audio is read with soundfile. The
aligner's weights come from Meta's MMS release, under a non-commercial licence
(CC-BY-NC-4.0, as far as the MMS release states; check before commercial use).
"""

import unicodedata

#: Letters NFKD leaves whole, spelled with the aligner's letters (French œ, ...).
LIGATURES = {"œ": "oe", "æ": "ae", "ß": "ss", "ø": "o", "ł": "l", "’": "'", "ʼ": "'"}
#: The aligner's token for a word it has no letters for (a number, a symbol).
STAR = "*"
#: The shortest stretch the aligner's convolutions take.
MIN_CROP_SECONDS = 0.1


def load(*, model="MMS_FA", device="cpu"):
    import torchaudio

    bundle = getattr(torchaudio.pipelines, model)
    aligner = bundle.get_model().to(device).eval()
    labels = bundle.get_labels(star=STAR)
    return {
        "bundle": bundle,
        "model": aligner,
        "device": device,
        "index": {c: i for i, c in enumerate(labels)},
    }


def _letters(word, index):
    """``word`` lowercased, ligatures spelled out, accents dropped, keeping the aligner's
    letters only; the star token for a word with none (a number, a symbol)."""
    lowered = "".join(LIGATURES.get(c, c) for c in word.lower())
    plain = unicodedata.normalize("NFKD", lowered)
    letters = "".join(c for c in plain if c in index and c not in "-" + STAR)
    return letters or STAR


def align(state, *, audio, words, start=None, end=None, max_seconds=30.0):
    """``{"words": [{"start", "end", "score", "chars": [...]}, ...]}`` for ``words``
    (in order) in ``audio``, or in its ``start``-``end`` stretch (s; times stay on the
    recording's clock); a word with no letter (a number) is placed as the star
    token. A stretch longer than ``max_seconds`` is refused (memory grows fast: 150 s
    took about 6 GB): align it in stretches."""
    import soundfile
    import torch
    import torchaudio.functional as F

    bundle, index = state["bundle"], state["index"]
    samples, rate = soundfile.read(audio, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    total = len(mono) / rate
    lo, hi = max(0.0, start or 0.0), min(total, total if end is None else end)
    if hi - lo < MIN_CROP_SECONDS:
        raise ValueError(f"stretch {lo:.2f}-{hi:.2f} s of {total:.2f} s is too short")
    if hi - lo > max_seconds:
        raise ValueError(
            f"{hi - lo:.0f} s of audio is over max_seconds={max_seconds}: "
            "align it in stretches (a line, a word and its neighbours)"
        )
    wav = torch.from_numpy(mono[int(lo * rate) : int(hi * rate)].copy())[None]
    wav = F.resample(wav, rate, bundle.sample_rate)
    letters = [_letters(w, index) for w in words]
    tokens = [index[c] for ls in letters for c in ls]
    if not tokens:
        return {"words": []}
    with torch.inference_mode():
        emission, _ = state["model"](wav.to(state["device"]))
    try:
        path, scores = F.forced_align(
            emission.cpu(), torch.tensor([tokens], dtype=torch.int32), blank=0
        )
    except RuntimeError as e:  # more letters than the audio has frames
        raise ValueError(
            f"{len(tokens)} letters do not fit in {emission.shape[1]} frames of audio: "
            "is the text longer than what is spoken?"
        ) from e
    spans = F.merge_tokens(path[0], scores[0].exp())
    if len(spans) != len(tokens):  # one span per token; anything else shifts words
        raise RuntimeError(f"{len(spans)} spans for {len(tokens)} tokens")
    seconds = wav.shape[1] / emission.shape[1] / bundle.sample_rate
    out, k = [], 0
    for word_letters in letters:
        chars = [
            {
                "char": c,
                "start": round(lo + s.start * seconds, 4),
                "end": round(lo + s.end * seconds, 4),
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


def score(state, *, audio, texts, start=None, end=None, max_seconds=30.0):
    """``{"scores": [...]}``: for each of ``texts``, the log-likelihood the aligner
    gives it over ``audio`` (cropped to ``start``-``end`` seconds when given), i.e.
    minus the CTC loss; ``-inf`` for a text too long for the crop. Higher is more
    likely; compare texts on the same crop only.

    Every candidate must be spelled in letters: a number or a symbol would become the
    "any sound" token, which fits any audio and would always win.
    """
    import math

    import soundfile
    import torch
    import torchaudio.functional as F

    bundle, index = state["bundle"], state["index"]
    letters = []
    for text in texts:
        spelled = "".join(_letters(w, index) for w in text.split())
        if not spelled or STAR in spelled:
            raise ValueError(
                f"candidate {text!r} needs letters only (spell numbers out; no empty text)"
            )
        letters.append(spelled)
    samples, rate = soundfile.read(audio, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    total = len(mono) / rate
    lo, hi = max(0.0, start or 0.0), min(total, total if end is None else end)
    if hi - lo < MIN_CROP_SECONDS:
        raise ValueError(f"crop {lo:.2f}-{hi:.2f} s of {total:.2f} s is too short")
    if hi - lo > max_seconds:
        raise ValueError(f"crop of {hi - lo:.0f} s is over max_seconds={max_seconds}")
    crop = mono[int(lo * rate) : int(hi * rate)]
    wav = F.resample(torch.from_numpy(crop.copy())[None], rate, bundle.sample_rate)
    with torch.inference_mode():
        emission, _ = state["model"](wav.to(state["device"]))
    log_probs = emission[0].cpu().log_softmax(-1)[:, None]
    scores = []
    for spelled in letters:
        tokens = torch.tensor([[index[c] for c in spelled]], dtype=torch.long)
        loss = float(
            torch.nn.functional.ctc_loss(
                log_probs,
                tokens,
                [log_probs.shape[0]],
                [tokens.shape[1]],
                blank=0,
                reduction="sum",
                zero_infinity=False,
            )
        )
        scores.append(-math.inf if math.isinf(loss) else round(-loss, 4))
    return {"scores": scores}
