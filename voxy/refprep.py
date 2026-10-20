"""
Prepare a speaker's recordings for cloning and fine-tuning: clean, cut, transcribe, score.

    >>> from voxy import refprep                                          # doctest: +SKIP
    >>> clips = refprep.prepare_reference(["tape1.m4a", "tape2.mp3"], voice="ada")  # doctest: +SKIP
    >>> ref = refprep.best_references(clips)[0]                            # doctest: +SKIP
    >>> ref["path"], ref["text"], ref["score"]                             # doctest: +SKIP

The steps, each a function you can call alone or replace:

1. ``condition`` (ffmpeg): any format to mono PCM WAV at ``sample_rate``, with a
   high-pass, optional denoising (``denoise=``: 'afftdn', 'anlmdn', 'deepfilternet', or None) and
   loudness normalisation (EBU R128, ``loudness_lufs``).
2. ``transcribe``: timed segments and words from a local Whisper (scribed's
   faster-whisper backend, in its own environment, see ``voxy.envs``).
3. ``plan_clips``: cut points at sentence/segment boundaries, so each clip is
   ``min_seconds``-``max_seconds`` long and carries its exact text.
4. ``score_clip``: duration, signal-to-noise estimate, clipping, level, ASR
   confidence and speaking rate, folded into a ``score`` in [0, 1].

``prepare_reference`` runs them all and saves the clips plus ``manifest.jsonl`` in
the voice's samples store (``voxy.stores.samples_store``), so any backend (a
reference clip) and fine-tuning (``voxy.finetune``, the whole set) read the same data.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from functools import partial
from pathlib import Path

import numpy as np

from voxy.envs import EnvSpec, ensure_env, worker_environ
from voxy.wavutil import frame_db, read_wav, write_wav

DFLT_SAMPLE_RATE = 24000
DFLT_LOUDNESS_LUFS = -20.0
DFLT_HIGHPASS_HZ = 70
DFLT_DENOISE = "afftdn"
DEEPFILTERNET = "deepfilternet"
DFLT_MIN_SECONDS = 3.0
DFLT_MAX_SECONDS = 12.0
DFLT_PAD_SECONDS = 0.12
DFLT_ASR_MODEL = "large-v3-turbo"
MANIFEST_KEY = "manifest.jsonl"
CLIPS_PREFIX = "clips/"
CONDITIONED_PREFIX = "conditioned/"

#: ffmpeg filters for each ``denoise`` strategy (None: no denoising).
DENOISE_FILTERS = {
    None: [],
    "afftdn": ["afftdn=nr=12:nf=-50:tn=1"],
    "anlmdn": ["anlmdn=s=0.0005"],
    # No ffmpeg filter: DeepFilterNet runs as its own stage (see ``with_deepfilternet``).
    DEEPFILTERNET: [],
}

#: DeepFilterNet works at 48 kHz; it needs an old torch/torchaudio pair, hence its own env.
DF_SAMPLE_RATE = 48000
DF_ENV = EnvSpec(
    "deepfilternet",
    ("torch==2.0.1", "torchaudio==2.0.2", "deepfilternet", "soundfile"),
)

ASR_ENV = EnvSpec("asr", ("scribed[faster-whisper]>=0.0.5", "soundfile"))


# 1. conditioning --------------------------------------------------------------


def _ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Preparing audio needs ffmpeg: apt-get install ffmpeg (Linux), "
            "brew install ffmpeg (macOS), or https://ffmpeg.org/download.html"
        )
    return ffmpeg


def conditioning_filters(
    *,
    denoise: str | None = DFLT_DENOISE,
    loudness_lufs: float | None = DFLT_LOUDNESS_LUFS,
    highpass_hz: float | None = DFLT_HIGHPASS_HZ,
) -> list[str]:
    """The ffmpeg audio filter chain for ``condition``.

    >>> conditioning_filters(denoise=None, loudness_lufs=-20, highpass_hz=70)
    ['highpass=f=70', 'loudnorm=I=-20:TP=-1.5:LRA=11']
    """
    if denoise not in DENOISE_FILTERS:
        raise ValueError(f"Unknown denoise {denoise!r} (have: {list(DENOISE_FILTERS)})")
    filters = [f"highpass=f={highpass_hz:g}"] if highpass_hz else []
    filters += DENOISE_FILTERS[denoise]
    if loudness_lufs is not None:
        filters.append(f"loudnorm=I={loudness_lufs:g}:TP=-1.5:LRA=11")
    return filters


def condition(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    *,
    sample_rate: int = DFLT_SAMPLE_RATE,
    start: float | None = None,
    duration: float | None = None,
    filters: Iterable[str] | None = None,
    run=subprocess.run,
) -> Path:
    """Convert ``src`` (any format ffmpeg reads) to clean mono 16-bit WAV at ``dst``."""
    filters = conditioning_filters() if filters is None else list(filters)
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(src), "-ac", "1"]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    run(cmd, check=True)
    return Path(dst)


def deepfilternet_denoise(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    *,
    rootdir: str | os.PathLike | None = None,
    threads: int | None = None,
    run=subprocess.run,
    ensure=ensure_env,
) -> Path:
    """Denoise the 48 kHz WAV ``src`` into ``dst`` with DeepFilterNet 3.

    Runs the ``deepFilter`` command of the isolated :data:`DF_ENV` (built once with
    ``uv`` under the data root: see ``voxy.envs``). One short call per file, so it needs
    no resident worker.
    """
    python = ensure(DF_ENV, rootdir=rootdir)
    command = Path(python).parent / "deepFilter"
    with tempfile.TemporaryDirectory() as out:
        run(
            [str(command), str(src), "-o", out],
            check=True,
            env=worker_environ(DF_ENV, threads=threads, rootdir=rootdir),
        )
        results = sorted(Path(out).glob("*.wav"))
        if not results:
            raise RuntimeError(f"deepFilter wrote no WAV for {src}")
        result = results[0]
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result, dst)
    return Path(dst)


def with_deepfilternet(
    conditioner: Callable[..., Path] | None = None,
    *,
    denoiser: Callable[..., Path] | None = None,
    rootdir: str | os.PathLike | None = None,
) -> Callable[..., Path]:
    """A ``conditioner`` that denoises with DeepFilterNet between two ffmpeg passes.

    Pass 1 converts the source to a 48 kHz mono WAV (no filters); DeepFilterNet cleans
    it; pass 2 applies the usual ``filters`` (high-pass, loudness) and resamples to
    ``sample_rate``. ``conditioner`` (default :func:`condition`) does both ffmpeg passes.
    """
    conditioner = conditioner or condition
    denoiser = denoiser or partial(deepfilternet_denoise, rootdir=rootdir)

    def conditioned(src, dst, *, sample_rate=DFLT_SAMPLE_RATE, filters=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            wide = Path(tmp) / "wide.wav"
            clean = Path(tmp) / "clean.wav"
            conditioner(src, wide, sample_rate=DF_SAMPLE_RATE, filters=[], **kwargs)
            denoiser(wide, clean)
            return conditioner(clean, dst, sample_rate=sample_rate, filters=filters)

    return conditioned


# 2. transcription -------------------------------------------------------------


_ASR_WORKERS: dict = {}


def _asr_worker(model: str, threads: int | None, rootdir: str | None):
    """One transcription worker per (model, threads, data root), started on demand."""
    from voxy.local_clone import voxy_cache_dir
    from voxy.worker import Worker
    from voxy.workers import adapter_path

    key = (model, threads, rootdir)
    if key not in _ASR_WORKERS:
        python = ensure_env(ASR_ENV, rootdir=rootdir)
        _ASR_WORKERS[key] = Worker(
            python,
            adapter_path("asr"),
            init={"model": model},
            env=worker_environ(ASR_ENV, threads=threads, rootdir=rootdir),
            log_path=voxy_cache_dir("logs", "asr.log"),
        )
    return _ASR_WORKERS[key]


def transcribe(
    audio: str | os.PathLike,
    *,
    language: str | None = None,
    model: str = DFLT_ASR_MODEL,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> list[dict]:
    """Timed segments ``[{start, end, text, confidence, words: [...]}, ...]`` of ``audio``."""
    worker = _asr_worker(model, threads, str(rootdir) if rootdir else None)
    reply = worker.call("transcribe", audio=str(audio), language=language)
    return reply["segments"]


def transcribe_text(audio: str | os.PathLike, **kwargs) -> str:
    """The transcript of ``audio`` as one string."""
    return " ".join(s["text"] for s in transcribe(audio, **kwargs)).strip()


def close_asr() -> None:
    """Stop the transcription workers (free their memory)."""
    while _ASR_WORKERS:
        _ASR_WORKERS.popitem()[1].close()


# 3. clip planning -------------------------------------------------------------

_SENTENCE_FINAL = re.compile(r"[.!?…][\"'”’)\]]?$")


def _words(segments: Iterable[Mapping]) -> list[dict]:
    """All timed words, in order (segments without word times count as one word)."""
    words = []
    for s in segments:
        timed = [w for w in s.get("words") or () if w.get("start") is not None]
        words += timed or [
            {
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "confidence": None,
            }
        ]
    return words


def _join_words(words: Iterable[Mapping]) -> str:
    """The words' text, keeping Whisper's own spacing (its words carry a leading space).

    >>> _join_words([{"text": " cypress"}, {"text": "-hooded"}, {"text": " hill."}])
    'cypress-hooded hill.'
    >>> _join_words([{"text": "One"}, {"text": "two."}])
    'One two.'
    """
    texts = [w["text"] for w in words]
    whisper_style = any(t[:1].isspace() for t in texts)
    return ("" if whisper_style else " ").join(texts).strip()


def plan_clips(
    segments: Iterable[Mapping],
    *,
    min_seconds: float = DFLT_MIN_SECONDS,
    max_seconds: float = DFLT_MAX_SECONDS,
    pad_seconds: float = DFLT_PAD_SECONDS,
    total_seconds: float | None = None,
) -> list[dict]:
    """Clips ``[{start, end, text, words}]`` cut between words, ending sentences where possible.

    Words accumulate until the clip passes ``min_seconds`` and a word ends a
    sentence (or the next word would push it past ``max_seconds``). Cuts fall in
    the pause between two words, padded by up to ``pad_seconds``.

    >>> w = lambda s, e, t: {"start": s, "end": e, "text": t, "confidence": 0.9}
    >>> segs = [{"words": [w(0, 1, "One"), w(1.1, 2, "two."), w(3, 4, "Three"),
    ...                    w(4.1, 5, "four.")], "start": 0, "end": 5, "text": ""}]
    >>> [c["text"] for c in plan_clips(segs, min_seconds=1.5, max_seconds=3)]
    ['One two.', 'Three four.']
    """
    words = _words(segments)
    clips, current = [], []

    def flush(next_start):
        if not current:
            return
        prev_end = clips[-1]["end"] if clips else 0.0
        start = max(prev_end, current[0]["start"] - pad_seconds)
        end = current[-1]["end"] + pad_seconds
        if next_start is not None:
            end = min(end, (current[-1]["end"] + next_start) / 2)
        elif total_seconds is not None:
            end = min(end, total_seconds)
        clips.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": _join_words(current),
                "words": list(current),
            }
        )
        current.clear()

    for i, word in enumerate(words):
        if current and word["end"] - current[0]["start"] > max_seconds:
            flush(word["start"])
        current.append(word)
        length = word["end"] - current[0]["start"]
        if length >= min_seconds and _SENTENCE_FINAL.search(word["text"].strip()):
            flush(words[i + 1]["start"] if i + 1 < len(words) else None)
    flush(None)
    return [c for c in clips if c["end"] - c["start"] >= min_seconds * 0.5]


# 4. scoring -------------------------------------------------------------------


def score_clip(
    samples: np.ndarray,
    sample_rate: int,
    *,
    words: Iterable[Mapping] = (),
    text: str = "",
    min_seconds: float = DFLT_MIN_SECONDS,
    max_seconds: float = DFLT_MAX_SECONDS,
) -> dict:
    """Quality measures of a clip, and their combination ``score`` in [0, 1].

    - ``snr_db``: loud-frame level minus quiet-frame level (speech vs floor).
    - ``clipped``: fraction of samples at full scale.
    - ``level_db``: level of the loud frames.
    - ``asr_confidence``: mean Whisper word probability (low: mumbling, noise, music).
    - ``chars_per_second``: speaking rate (outliers: bad alignment).

    >>> rng = np.random.default_rng(0)
    >>> speechy = np.concatenate([0.3 * rng.standard_normal(40000), 0.001 * rng.standard_normal(8000)])
    >>> s = score_clip(speechy, 8000, text="x" * 70)
    >>> s["snr_db"] > 30, 0 < s["score"] <= 1
    (True, True)
    """
    duration = len(samples) / sample_rate
    levels = frame_db(samples, sample_rate)
    loud = float(np.percentile(levels, 95)) if levels.size else -100.0
    floor = float(np.percentile(levels, 10)) if levels.size else -100.0
    snr = loud - floor
    clipped = float(np.mean(np.abs(samples) >= 0.999)) if len(samples) else 0.0
    confidences = [w["confidence"] for w in words if w.get("confidence") is not None]
    asr = float(np.mean(confidences)) if confidences else None
    rate = len(text) / duration if duration else 0.0

    def ramp(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0, 1))

    parts = {
        "snr": ramp(snr, 15, 40),
        "clipping": 1 - ramp(clipped, 0, 0.002),
        "level": 1 - ramp(abs(loud + 18), 6, 18),
        "rate": 1 - ramp(abs(rate - 15), 5, 10),
        "duration": float(min_seconds <= duration <= max_seconds * 1.25),
    }
    if asr is not None:
        parts["asr"] = ramp(asr, 0.5, 0.95)
    score = float(np.prod(list(parts.values())) ** (1 / len(parts)))
    return {
        "duration": round(duration, 3),
        "snr_db": round(snr, 1),
        "clipped": round(clipped, 5),
        "level_db": round(loud, 1),
        "asr_confidence": None if asr is None else round(asr, 3),
        "chars_per_second": round(rate, 1),
        "score": round(score, 3),
    }


# the pipeline -----------------------------------------------------------------


def _slug(path: str | os.PathLike) -> str:
    """A key-safe stem for a source file.

    >>> _slug("/x/My Tape (1).m4a")
    'my-tape-1'
    """
    return re.sub(r"[^a-z0-9]+", "-", Path(path).stem.lower()).strip("-") or "source"


def prepare_reference(
    sources: Iterable[str | os.PathLike],
    *,
    voice: str | None = None,
    samples: MutableMapping | None = None,
    sample_rate: int = DFLT_SAMPLE_RATE,
    denoise: str | None = DFLT_DENOISE,
    loudness_lufs: float | None = DFLT_LOUDNESS_LUFS,
    min_seconds: float = DFLT_MIN_SECONDS,
    max_seconds: float = DFLT_MAX_SECONDS,
    language: str | None = None,
    transcriber: Callable[..., list[dict]] | None = None,
    conditioner: Callable[..., Path] | None = None,
    rootdir: str | os.PathLike | None = None,
    log=print,
) -> list[dict]:
    """Condition, transcribe, cut and score ``sources``; save clips and manifest; return it.

    Args:
        sources: Recordings, any format ffmpeg reads.
        voice: Voice name: clips go to ``samples_store(voice)``.
        samples: A store to write to instead (``filename -> bytes``).
        sample_rate, denoise, loudness_lufs: Conditioning (see ``condition``).
        min_seconds, max_seconds: Clip length bounds.
        language: Force the ASR language (default: detected).
        transcriber: ``(wav_path, language=) -> segments`` (default ``transcribe``).
        conditioner: ``(src, dst, sample_rate=, filters=) -> path`` (default ``condition``).

    Returns:
        The manifest: one dict per clip (``key``, ``path`` if on disk, ``text``,
        ``source``, ``start``, ``end``, and the ``score_clip`` measures), best first.
    """
    if samples is None:
        if voice is None:
            raise ValueError("Pass voice= (a name) or samples= (a store)")
        from voxy.stores import samples_store

        samples = samples_store(voice, rootdir=rootdir)
    transcriber = transcriber or partial_transcribe(rootdir)
    conditioner = conditioner or condition
    if denoise == DEEPFILTERNET:
        conditioner = with_deepfilternet(conditioner, rootdir=rootdir)
    filters = conditioning_filters(denoise=denoise, loudness_lufs=loudness_lufs)
    manifest = []
    with tempfile.TemporaryDirectory() as tmp:
        for src in sources:
            slug = _slug(src)
            wav_path = Path(tmp) / f"{slug}.wav"
            log(f"voxy.refprep: conditioning {src}")
            conditioner(src, wav_path, sample_rate=sample_rate, filters=filters)
            audio, sr = read_wav(wav_path)
            samples[f"{CONDITIONED_PREFIX}{slug}.wav"] = wav_path.read_bytes()
            log(f"voxy.refprep: transcribing {slug} ({len(audio) / sr:.0f} s)")
            segments = transcriber(wav_path, language=language)
            plan = plan_clips(
                segments,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                total_seconds=len(audio) / sr,
            )
            for i, clip in enumerate(plan, 1):
                piece = audio[int(clip["start"] * sr) : int(clip["end"] * sr)]
                key = f"{CLIPS_PREFIX}{slug}-{i:04d}.wav"
                samples[key] = write_wav(piece, sr)
                entry = {
                    "key": key,
                    "text": clip["text"],
                    "source": str(src),
                    "start": clip["start"],
                    "end": clip["end"],
                    "sample_rate": sr,
                    **score_clip(
                        piece,
                        sr,
                        words=clip["words"],
                        text=clip["text"],
                        min_seconds=min_seconds,
                        max_seconds=max_seconds,
                    ),
                }
                manifest.append(entry)
    manifest.sort(key=lambda e: -e["score"])
    root = getattr(samples, "rootdir", None)
    if root:
        for entry in manifest:
            entry["path"] = os.path.join(root, entry["key"])
    samples[MANIFEST_KEY] = "\n".join(json.dumps(e) for e in manifest).encode()
    log(f"voxy.refprep: {len(manifest)} clips saved")
    return manifest


def partial_transcribe(rootdir=None) -> Callable[..., list[dict]]:
    """The default transcriber, bound to a data root."""

    def transcriber(path, *, language=None):
        return transcribe(path, language=language, rootdir=rootdir)

    return transcriber


def load_manifest(samples: Mapping) -> list[dict]:
    """The manifest saved by ``prepare_reference`` in a samples store."""
    return [json.loads(line) for line in samples[MANIFEST_KEY].decode().splitlines()]


def best_references(
    manifest: Iterable[Mapping],
    *,
    n: int = 3,
    min_seconds: float = 5.0,
    max_seconds: float = 12.0,
    min_score: float = 0.0,
) -> list[dict]:
    """The ``n`` best clips for zero-shot reference (length in range, best score first).

    >>> m = [{"duration": 8, "score": 0.7}, {"duration": 2, "score": 0.9},
    ...      {"duration": 10, "score": 0.8}]
    >>> [c["score"] for c in best_references(m, n=2)]
    [0.8, 0.7]
    """
    fits = [
        dict(c)
        for c in manifest
        if min_seconds <= c["duration"] <= max_seconds and c["score"] >= min_score
    ]
    return sorted(fits, key=lambda c: -c["score"])[:n]


def summarize(manifest: Iterable[Mapping]) -> dict:
    """Totals for a manifest: clips, minutes, minutes above score thresholds.

    >>> summarize([{"duration": 60, "score": 0.9}, {"duration": 30, "score": 0.4}])
    {'clips': 2, 'minutes': 1.5, 'minutes_score_0.5+': 1.0, 'minutes_score_0.7+': 1.0}
    """
    manifest = list(manifest)
    minutes = lambda cs: round(sum(c["duration"] for c in cs) / 60, 2)  # noqa: E731
    return {
        "clips": len(manifest),
        "minutes": minutes(manifest),
        "minutes_score_0.5+": minutes([c for c in manifest if c["score"] >= 0.5]),
        "minutes_score_0.7+": minutes([c for c in manifest if c["score"] >= 0.7]),
    }
