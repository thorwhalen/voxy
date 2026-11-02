"""
Split a recording into the voice and the rest (music, noise) before using the voice.

A song or a noisy interview has to lose its backing before its voice can be cloned,
transcribed or edited. ``separate`` runs Demucs (MIT; ``htdemucs_ft`` by default, the
fine-tuned hybrid transformer) in its own environment, built on first use under the
data root like the other local models, and returns the two stems as WAV bytes:
the voice, and ``rest``, the input minus the voice (so the two add back up to it).

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
_FOUR_SOURCES = ("drums", "bass", "other", "vocals")
#: The stems of the pretrained models whose sources are known before loading them.
MODEL_SOURCES = {
    "htdemucs": _FOUR_SOURCES,
    "htdemucs_ft": _FOUR_SOURCES,
    "htdemucs_6s": (*_FOUR_SOURCES, "guitar", "piano"),
}
#: torch is bounded to the releases Demucs 4.0.1 was checked with (torchaudio, which
#: Demucs imports but voxy never uses for I/O, follows it).
SEPARATION_ENV = EnvSpec(
    "demucs",
    ("demucs==4.0.1", "torch>=2.11,<2.16", "soundfile", "numpy"),
    env_vars={"TORCH_HOME": "{models}"},
)

#: The Demucs workers, one per (model, threads, data root), made on demand.
_DEMUCS_POOL = WorkerPool(SEPARATION_ENV, "demucs")


def close_separation() -> None:
    """Stop the Demucs workers (free their memory)."""
    _DEMUCS_POOL.close()


def check_stem(stem: str, model: str) -> None:
    """Raise ``ValueError`` if ``model`` is known (``MODEL_SOURCES``) and lacks ``stem``.

    >>> check_stem("vocals", "htdemucs_ft")
    >>> check_stem("guitar", "htdemucs_ft")
    Traceback (most recent call last):
      ...
    ValueError: 'guitar' is not a stem of htdemucs_ft: ['drums', 'bass', 'other', 'vocals']
    """
    sources = MODEL_SOURCES.get(model)
    if sources is not None and stem not in sources:
        raise ValueError(f"{stem!r} is not a stem of {model}: {list(sources)}")


def worker_run(
    *,
    model: str = DFLT_SEPARATION_MODEL,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> Callable[..., Mapping[str, str]]:
    """The default ``run``: the Demucs adapter in its own environment.

    A stem the model lacks fails before the worker starts for the models in
    ``MODEL_SOURCES``, and inside the worker (a ``WorkerError`` carrying the same
    message) for the others.
    """

    def run(
        audio: str, out_dir: str, *, stem: str = DFLT_STEM, **options
    ) -> Mapping[str, str]:
        check_stem(stem, model)
        worker = _DEMUCS_POOL.get(model, threads, rootdir)
        # the worker runs in its own working directory: hand it absolute paths
        reply = worker.call(
            "separate",
            audio=str(Path(audio).resolve()),
            out_dir=str(Path(out_dir).resolve()),
            stem=stem,
            **options,
        )
        return reply["paths"]

    return run


def separate(
    audio: str | os.PathLike | bytes,
    *,
    stem: str = DFLT_STEM,
    sample_rate: int | None = None,
    run: Callable[..., Mapping[str, str]] | None = None,
    **options,
) -> dict[str, bytes]:
    """``{stem: wav, "rest": wav}``: ``audio`` split into ``stem`` and everything else.

    ``rest`` is the input minus the voice, so ``stem + rest`` rebuilds the input (as
    the model hears it: resampled and with its channels). Both are 16-bit PCM WAV
    with the model's channels (Demucs: stereo; a mono input is duplicated, the first
    two channels of a wider one are kept) at the model's rate (Demucs: 44.1 kHz)
    unless ``sample_rate`` says otherwise, whatever the input. If either stem would
    clip, both are divided by one common factor (Demucs's ``prevent_clip``), so they
    still add up to the input times that factor.

    The whole file is decoded and held in memory, and Demucs takes about 24 s per
    minute of audio on CPU for one model (``htdemucs_ft`` runs only the sub-model
    fine-tuned for ``stem``); split hours-long recordings first.

    Args:
        audio: A path to any format soundfile reads (WAV, FLAC, OGG, MP3 with a
            recent libsndfile), or WAV bytes.
        stem: The stem to keep apart (``"vocals"``; Demucs also has drums, bass,
            other).
        sample_rate: The rate of the returned WAVs (default: the model's).
        run: ``(audio_path, out_dir, *, stem, sample_rate, **options) ->
            {stem: path}``; default ``worker_run()`` (Demucs ``htdemucs_ft`` on CPU).
        **options: Passed to ``run`` (Demucs: ``shifts``, ``overlap``).
    """
    run = run or worker_run()
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(audio, bytes):
            path = Path(tmp) / "input.wav"
            path.write_bytes(audio)
            audio = path
        paths = run(str(audio), tmp, stem=stem, sample_rate=sample_rate, **options)
        return {name: Path(paths[name]).read_bytes() for name in (stem, REST_STEM)}
