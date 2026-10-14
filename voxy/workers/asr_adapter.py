"""Speech recognition with timed words (scribed's faster-whisper backend), for reference prep."""

import os


def load(*, model="large-v3-turbo", compute_type="int8", device="cpu"):
    os.environ.setdefault("SCRIBED_FASTER_WHISPER_COMPUTE", compute_type)
    os.environ.setdefault("SCRIBED_FASTER_WHISPER_DEVICE", device)
    import scribed

    return {"scribed": scribed, "model": model}


def transcribe(state, *, audio, language=None, **params):
    options = {"word_timestamps": True, "model": state["model"], **params}
    if language:
        options["language"] = language
    transcript = state["scribed"].transcribe(audio, backend="faster-whisper", **options)
    return {"segments": [_segment(s) for s in transcript.segments]}


def _segment(s):
    return {
        "start": s.start,
        "end": s.end,
        "text": s.text.strip(),
        "confidence": s.confidence,
        "words": [
            {"start": w.start, "end": w.end, "text": w.text, "confidence": w.confidence}
            for w in (s.words or ())
        ],
    }
