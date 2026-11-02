"""
Split a recording into the voice and the rest (music, noise) before using the voice.

A song or a noisy interview has to lose its backing before its voice can be cloned,
transcribed or edited. ``separate`` runs Demucs (MIT; ``htdemucs_ft`` by default, the
fine-tuned hybrid transformer) in its own environment, built on first use under the
data root like the other local models, and returns the two stems as WAV bytes:

    >>> from voxy.separate import separate                              # doctest: +SKIP
    >>> stems = separate("song.mp3")                                   # doctest: +SKIP
    >>> open("vocals.wav", "wb").write(stems["vocals"])                # doctest: +SKIP

``run`` is the seam (``(audio_path, out_dir, **options) -> {stem: path}``); tests and
other separators plug in there.

>>> import tempfile, pathlib
>>> def fake(audio, out_dir, **options):
...     paths = {n: pathlib.Path(out_dir) / f"{n}.wav" for n in ("vocals", "rest")}
...     for p in paths.values():
...         p.write_bytes(b"RIFF")
...     return paths
>>> sorted(separate(b"RIFF....", run=fake))
['rest', 'vocals']
"""

import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from voxy.envs import EnvSpec
from voxy.worker import WorkerPool

DFLT_SEPARATION_MODEL = "htdemucs_ft"
#: The stem kept as "the voice"; Demucs also has drums, bass and other.
DFLT_STEM = "vocals"
REST_STEM = "rest"
SEPARATION_ENV = EnvSpec(
    "demucs",
    ("demucs==4.0.1", "torch", "torchaudio", "soundfile", "numpy"),
    env_vars={"TORCH_HOME": "{models}"},
)

#: The Demucs workers, one per (model, threads, data root), made on demand.
_DEMUCS_POOL = WorkerPool(SEPARATION_ENV, "demucs")


def close_separation() -> None:
    """Stop the Demucs workers (free their memory)."""
    _DEMUCS_POOL.close()


def worker_run(
    *,
    model: str = DFLT_SEPARATION_MODEL,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> Callable[..., Mapping[str, str]]:
    """The default ``run``: the Demucs adapter in its own environment."""

    def run(audio: str, out_dir: str, **options) -> Mapping[str, str]:
        worker = _DEMUCS_POOL.get(model, threads, rootdir)
        # the worker runs in its own working directory: hand it absolute paths
        reply = worker.call(
            "separate",
            audio=str(Path(audio).resolve()),
            out_dir=str(Path(out_dir).resolve()),
            **options,
        )
        return reply["paths"]

    return run


def separate(
    audio: str | os.PathLike | bytes,
    *,
    stem: str = DFLT_STEM,
    run: Callable[..., Mapping[str, str]] | None = None,
    **options,
) -> dict[str, bytes]:
    """``{stem: wav, "rest": wav}``: ``audio`` split into ``stem`` and everything else.

    Args:
        audio: A path to any format soundfile reads (WAV, FLAC, OGG, MP3 with a
            recent libsndfile), or WAV bytes.
        stem: The stem to keep apart (``"vocals"``; Demucs also has drums, bass,
            other).
        run: ``(audio_path, out_dir, **options) -> {stem: path}``; default
            ``worker_run()`` (Demucs ``htdemucs_ft`` on CPU).
        **options: Passed to ``run`` (Demucs: ``shifts``, ``overlap``).
    """
    run = run or worker_run()
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(audio, bytes):
            path = Path(tmp) / "input.wav"
            path.write_bytes(audio)
            audio = path
        paths = run(str(audio), tmp, stem=stem, **options)
        return {name: Path(paths[name]).read_bytes() for name in (stem, REST_STEM)}
