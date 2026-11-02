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


def align(state, *, audio, words, max_seconds=30.0):
    """``{"words": [{"start", "end", "score", "chars": [...]}, ...]}`` for ``words``
    (in order) in ``audio``; a word with no letter (a number) is placed as the star
    token. Audio longer than ``max_seconds`` is refused (memory grows fast: 150 s
    took about 6 GB): align it in stretches."""
    import soundfile
    import torch
    import torchaudio.functional as F

    bundle, index = state["bundle"], state["index"]
    samples, rate = soundfile.read(audio, dtype="float32", always_2d=True)
    wav = torch.from_numpy(samples.mean(axis=1).copy())[None]
    if len(samples) / rate > max_seconds:
        raise ValueError(
            f"{len(samples) / rate:.0f} s of audio is over max_seconds={max_seconds}: "
            "align it in stretches (a line, a word and its neighbours)"
        )
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


def score(state, *, audio, texts, start=None, end=None):
    """``{"scores": [...]}``: for each of ``texts``, the log-likelihood the aligner
    gives it over ``audio`` (cropped to ``start``-``end`` seconds when given), i.e.
    minus the CTC loss. Higher is more likely; compare texts on the same crop only."""
    import soundfile
    import torch
    import torchaudio.functional as F

    bundle, index = state["bundle"], state["index"]
    samples, rate = soundfile.read(audio, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    lo = int((start or 0) * rate)
    hi = len(mono) if end is None else int(end * rate)
    wav = F.resample(
        torch.from_numpy(mono[lo:hi].copy())[None], rate, bundle.sample_rate
    )
    with torch.inference_mode():
        emission, _ = state["model"](wav.to(state["device"]))
    log_probs = emission[0].cpu().log_softmax(-1)[:, None]
    scores = []
    for text in texts:
        letters = "".join(_letters(w, index) for w in text.split())
        tokens = torch.tensor([[index[c] for c in letters]], dtype=torch.long)
        loss = torch.nn.functional.ctc_loss(
            log_probs,
            tokens,
            [log_probs.shape[0]],
            [tokens.shape[1]],
            blank=0,
            reduction="sum",
            zero_infinity=True,
        )
        scores.append(round(-float(loss), 4))
    return {"scores": scores}
