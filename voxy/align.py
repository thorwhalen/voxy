"""
Where each word of a known text is in a recording, to the character: forced alignment.

ASR gives word times for what it heard; when the text is known (a script, lyrics, the
words around a spot to edit), forced alignment places exactly those words, with each
character's span. That is what a splice needs at a word's onset, and what a check
needs when ASR heard a different word than the one spoken.

    >>> from voxy.align import force_align                              # doctest: +SKIP
    >>> spans = force_align("line.wav", "Mon frère est parti")          # doctest: +SKIP
    >>> spans[2]["word"], spans[2]["start"], spans[2]["chars"][0]       # doctest: +SKIP

The default aligner is torchaudio's ``MMS_FA`` (multilingual, Latin letters; accents are
dropped for matching) in its own environment, built on first use (or ``python -m voxy
setup align``). Its weights are non-commercial (see ``workers/align_adapter.py``).
Align short stretches (a line, a word and its neighbours, under about 30 s): one pass
over a long recording drifts, sung audio most.

``run`` is the seam: ``(audio_path, words) -> [span or None, ...]``.

>>> fake = lambda audio, words: [{"start": i, "end": i + 0.5, "score": 0.9, "chars": []}
...                              for i, _ in enumerate(words)]
>>> [s["word"] for s in force_align(b"RIFF....", "Bonjour  le monde", run=fake)]
['Bonjour', 'le', 'monde']
"""

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from voxy.envs import EnvSpec
from voxy.worker import WorkerPool

DFLT_ALIGNER = "MMS_FA"
#: torch and torchaudio bounded to releases that still ship ``forced_align``.
ALIGN_ENV = EnvSpec(
    "align",
    ("torch>=2.6,<2.12", "torchaudio>=2.6,<2.12", "soundfile", "numpy"),
    env_vars={"TORCH_HOME": "{models}"},
)

#: The aligner workers, one per (model, threads, data root), made on demand.
_ALIGN_POOL = WorkerPool(ALIGN_ENV, "align")

Run = Callable[[str, Sequence[str]], Sequence["dict | None"]]


def close_alignment() -> None:
    """Stop the aligner workers (free their memory)."""
    _ALIGN_POOL.close()


def worker_run(
    *,
    model: str = DFLT_ALIGNER,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> Run:
    """The default ``run``: the aligner adapter in its own environment."""

    def run(audio: str, words: Sequence[str]):
        worker = _ALIGN_POOL.get(model, threads, rootdir)
        # the worker runs in its own working directory: an absolute path
        reply = worker.call(
            "align", audio=str(Path(audio).resolve()), words=list(words)
        )
        return reply["words"]

    return run


def force_align(
    audio: str | os.PathLike | bytes,
    text: str | Sequence[str],
    *,
    run: Run | None = None,
) -> list[dict]:
    """Each word of ``text``, in order, with its ``start``, ``end`` (s), ``score`` (mean
    character confidence) and ``chars`` (``char``, ``start``, ``end``, ``score``).

    Args:
        audio: A WAV (or any format soundfile reads) path, or WAV bytes.
        text: The words spoken, as a string (split on whitespace) or a list.
        run: ``(audio_path, words) -> [span or None, ...]``; default ``worker_run()``.

    A word with no letter the aligner knows (a number, a symbol) is placed as one
    "any sound" token. Low scores (under about 0.5) mean the text is not what is said
    there. Audio over 30 s is refused by the default aligner: align in stretches. A
    run that returns None for a word gives ``start``/``end`` None.
    """
    words = text.split() if isinstance(text, str) else list(text)
    run = run or worker_run()
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(audio, bytes):
            path = Path(tmp) / "audio.wav"
            path.write_bytes(audio)
            audio = path
        spans = run(str(audio), words)
    if len(spans) != len(words):
        raise ValueError(
            f"the aligner returned {len(spans)} spans for {len(words)} words"
        )
    empty = {"start": None, "end": None, "score": None, "chars": []}
    return [{"word": w, **(span or empty)} for w, span in zip(words, spans)]


ScoreRun = Callable[..., Sequence[float]]
#: A lead (in log-likelihood) under this is a tie: tight crops can favour a slightly
#: shorter text by fractions of a point.
DFLT_DECISIVE_MARGIN = 1.0


def worker_score(
    *,
    model: str = DFLT_ALIGNER,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> ScoreRun:
    """The default scorer for ``which_said``: the aligner adapter's ``score``."""

    def run(audio: str, texts: Sequence[str], *, start=None, end=None):
        worker = _ALIGN_POOL.get(model, threads, rootdir)
        reply = worker.call(
            "score",
            audio=str(Path(audio).resolve()),
            texts=list(texts),
            start=start,
            end=end,
        )
        return reply["scores"]

    return run


def which_said(
    audio: str | os.PathLike | bytes,
    candidates: Sequence[str],
    *,
    start: float | None = None,
    end: float | None = None,
    run: ScoreRun | None = None,
    decisive_margin: float = DFLT_DECISIVE_MARGIN,
) -> list[dict]:
    """The ``candidates`` (texts) ranked by how well the audio supports each, best first:
    ``[{"text", "score", "margin", "decisive"}]``: ``score`` the aligner's
    log-likelihood (minus its CTC loss; ``-inf`` for a text too long for the crop),
    ``margin`` the lead over the next candidate, ``decisive`` whether that lead is at
    least ``decisive_margin``.

    A forced choice, so it cannot hallucinate as transcribing a short crop can: use it
    to check an edit ("witch" or the old word?) on a crop around it (``start``/``end``,
    s). Cut the crop at silences and include the neighbouring words: a crop that ends
    inside a word, or holds one word alone, favours shorter texts by small margins,
    which ``decisive`` flags. Candidates are spelled in letters (numbers spelled out).

    >>> fake = lambda audio, texts, start=None, end=None: [-5.0, -2.0]
    >>> [(c["text"], c["margin"], c["decisive"])
    ...  for c in which_said(b"RIFF", ["bitch", "witch"], run=fake)]
    [('witch', 3.0, True), ('bitch', None, None)]
    """
    run = run or worker_score()
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(audio, bytes):
            path = Path(tmp) / "audio.wav"
            path.write_bytes(audio)
            audio = path
        scores = run(str(audio), list(candidates), start=start, end=end)
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda ts: -ts[1])
    out = []
    for i, (text, score) in enumerate(ranked):
        margin = score - ranked[i + 1][1] if i + 1 < len(ranked) else None
        if margin is not None and math.isfinite(margin):
            margin = round(margin, 4)
        decisive = None if margin is None else bool(margin >= decisive_margin)
        out.append(
            {"text": text, "score": score, "margin": margin, "decisive": decisive}
        )
    return out
