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

    A word with no letter the aligner knows (a number, a symbol) comes back with
    ``start``/``end`` None.
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
