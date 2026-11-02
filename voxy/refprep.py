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
3. ``speaker`` (optional, ``voxy.speakers.speaker_filter``): marks words, and
   untranscribed stretches, spoken by someone other than the target voice.
4. ``plan_clips``: cut points at sentence/segment boundaries, so each clip is
   ``min_seconds``-``max_seconds`` long, carries its exact text, and never
   contains a marked word.
5. ``score_clip``: duration, signal-to-noise estimate, clipping, level, ASR
   confidence, speaking rate and (with a speaker step) the speaker match,
   folded into a ``score`` in [0, 1].

``prepare_reference`` runs them all and saves the clips plus ``manifest.jsonl`` in
the voice's samples store (``voxy.stores.samples_store``), so any backend (a
reference clip) and fine-tuning (``voxy.finetune``, the whole set) read the same data.
It only ever adds to the store, keeping every run; ``latest_runs`` gives each source's
latest run, and ``forget`` is the one function that deletes clips.
"""

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np

from voxy.envs import EnvSpec, ensure_env, worker_environ
from voxy.worker import WorkerPool
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
#: Speaker similarity mapped to the score's ``speaker`` part (0 at the low end, 1 at the high).
SPEAKER_SCORE_RANGE = (0.3, 0.7)
MANIFEST_KEY = "manifest.jsonl"
CLIPS_PREFIX = "clips/"
CONDITIONED_PREFIX = "conditioned/"
#: The slugs this module gives sources (see ``_slug``), and the clip keys made from them.
SLUG_PATTERN = re.compile(r"[a-z0-9-]+")
CLIP_KEY_PATTERN = re.compile(
    rf"{re.escape(CLIPS_PREFIX)}(?P<slug>{SLUG_PATTERN.pattern})-\d{{4,}}\.wav"
)
CONDITIONED_KEY_PATTERN = re.compile(
    rf"{re.escape(CONDITIONED_PREFIX)}(?P<slug>{SLUG_PATTERN.pattern})\.wav"
)
#: A run id (``run`` in each manifest entry): its UTC start time to the microsecond, plus
#: random hex characters, so ids sort by time and two runs never share one.
RUN_ID_TIME_FORMAT = "%Y%m%dT%H%M%S.%fZ"
RUN_ID_SUFFIX_LEN = 6
#: ``forget(..., run=LEGACY_RUN)`` selects entries written before runs were recorded
#: (no ``run`` field), matching even a relative ``source`` exactly as it was typed.
LEGACY_RUN = "legacy"

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


#: The asr workers, one per (model, threads, data root), made on demand.
_ASR_POOL = WorkerPool(ASR_ENV, "asr")


def transcribe(
    audio: str | os.PathLike,
    *,
    language: str | None = None,
    model: str = DFLT_ASR_MODEL,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> list[dict]:
    """Timed segments ``[{start, end, text, confidence, words: [...]}, ...]`` of ``audio``."""
    worker = _ASR_POOL.get(model, threads, rootdir)
    # The worker runs in its own working directory: hand it an absolute path.
    reply = worker.call(
        "transcribe", audio=str(Path(audio).resolve()), language=language
    )
    return reply["segments"]


def transcribe_text(audio: str | os.PathLike, **kwargs) -> str:
    """The transcript of ``audio`` as one string."""
    return " ".join(s["text"] for s in transcribe(audio, **kwargs)).strip()


def close_asr() -> None:
    """Stop the transcription workers (free their memory)."""
    _ASR_POOL.close()


# 3. clip planning -------------------------------------------------------------

_SENTENCE_FINAL = re.compile(r"[.!?…][\"'”’)\]]?$")


def _words(segments: Iterable[Mapping]) -> list[dict]:
    """All timed words, in time order (segments without word times count as one word)."""
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
    return sorted(words, key=lambda w: w["start"])


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
    the pause between two words, padded by up to ``pad_seconds``. A word marked
    ``keep=False`` (someone else's, see ``voxy.speakers``) ends the clip before it,
    and the next clip starts after it.

    >>> w = lambda s, e, t: {"start": s, "end": e, "text": t, "confidence": 0.9}
    >>> segs = [{"words": [w(0, 1, "One"), w(1.1, 2, "two."), w(3, 4, "Three"),
    ...                    w(4.1, 5, "four.")], "start": 0, "end": 5, "text": ""}]
    >>> [c["text"] for c in plan_clips(segs, min_seconds=1.5, max_seconds=3)]
    ['One two.', 'Three four.']
    >>> segs[0]["words"].insert(2, {**w(2.2, 2.6, " yes"), "keep": False})
    >>> [(c["text"], c["start"]) for c in plan_clips(segs, min_seconds=1.5, max_seconds=3)]
    [('One two.', 0.0), ('Three four.', 2.88)]
    """
    words = _words(segments)
    clips, current = [], []
    floor = (
        0.0  # no clip may start before this (the end of the last clip or foreign word)
    )

    def flush(next_start):
        nonlocal floor
        if not current:
            return
        start = max(floor, current[0]["start"] - pad_seconds)
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
        floor = end
        current.clear()

    for i, word in enumerate(words):
        if word.get("keep") is False:
            flush(word["start"])
            floor = max(floor, word["end"])
            continue
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
    - ``speaker_similarity``: lowest match of a word to the target voice (words
      annotated by ``voxy.speakers``; None otherwise).

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
    matches = [
        w["speaker_similarity"]
        for w in words
        if w.get("speaker_similarity") is not None
    ]
    speaker = float(min(matches)) if matches else None
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
    if speaker is not None:
        parts["speaker"] = ramp(speaker, *SPEAKER_SCORE_RANGE)
    score = float(np.prod(list(parts.values())) ** (1 / len(parts)))
    return {
        "duration": round(duration, 3),
        "snr_db": round(snr, 1),
        "clipped": round(clipped, 5),
        "level_db": round(loud, 1),
        "asr_confidence": None if asr is None else round(asr, 3),
        "chars_per_second": round(rate, 1),
        "speaker_similarity": None if speaker is None else round(speaker, 3),
        "score": round(score, 3),
    }


# the pipeline -----------------------------------------------------------------


def _slug(path: str | os.PathLike, *, parents: int = 0) -> str:
    """A key-safe stem for a source file, prefixed with its last ``parents`` folders.

    >>> _slug("/x/My Tape (1).m4a")
    'my-tape-1'
    >>> _slug("/songs/Track A/vocals.wav", parents=1)
    'track-a-vocals'
    >>> _slug("/s/日本.wav", parents=1)  # a stem with nothing key-safe in it
    's-source'
    """

    def key_safe(text):
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    path = Path(path)
    folders = path.parent.parts[len(path.parent.parts) - parents :] if parents else ()
    return key_safe("-".join([*folders, key_safe(path.stem) or "source"]))


def _deepened(paths: list[Path]) -> list[str]:
    """Slugs for same-stem ``paths``, each with the fewest parent folders that tell it
    apart from the others.

    A path adds a folder only while it still matches another path that more folders
    could tell apart from it (not one in the same folder, a repeat, or one whose
    folders only differ in case or characters slugs drop: those are numbered later),
    and keeps what it has once it runs out of folders.

    >>> _deepened([Path("/s/one/vocals.wav"), Path("/s/two/vocals.wav")])
    ['one-vocals', 'two-vocals']
    >>> _deepened([Path("/home/x/talk.m4a"), Path("/home/x/talk.wav")])
    ['talk', 'talk']
    >>> _deepened([Path("a/v.wav"), Path("v.wav"), Path("b/v.wav")])
    ['a-v', 'v', 'b-v']
    >>> _deepened([Path("/s/a/x/v.wav"), Path("/s/b/x/v.wav"), Path("/s/c/y/v.wav")])
    ['a-x-v', 'b-x-v', 'y-v']
    """
    # what a path could ever become: folders that only slug alike (case, accents,
    # scripts) can never tell paths apart, so they do not count as different
    whole = [_slug(p, parents=len(p.parent.parts)) for p in paths]
    depth = [0] * len(paths)
    while True:
        slugs = [_slug(p, parents=d) for p, d in zip(paths, depth)]
        futures: dict[str, set[str]] = {}
        for slug, full in zip(slugs, whole):
            futures.setdefault(slug, set()).add(full)
        deeper = [
            i
            for i, path in enumerate(paths)
            if depth[i] < len(path.parent.parts) and len(futures[slugs[i]]) > 1
        ]
        if not deeper:
            return slugs
        for i in deeper:
            depth[i] += 1


def _numbered(slugs: list[str]) -> list[str]:
    """Distinct slugs: a slug's first occurrence stays, repeats get ``-2``, ``-3``, ...,
    skipping any slug already in the list.

    >>> _numbered(["a", "a", "a-2", "a"])
    ['a', 'a-3', 'a-2', 'a-4']
    """
    taken, out = set(slugs), []
    for slug in slugs:
        if slug in out:
            n = 2
            while f"{slug}-{n}" in taken:
                n += 1
            slug = f"{slug}-{n}"
            taken.add(slug)
        out.append(slug)
    return out


def unique_slugs(
    sources: Iterable[str | os.PathLike], *, reserved: Iterable[str] = ()
) -> list[str]:
    """One distinct slug per source: the file's stem, plus parent folders only within a
    group of sources sharing a stem (stem tracks: every one is ``vocals.wav``), numbered
    where folders cannot tell them apart. Slugs never climb past what separates a group,
    and a source whose stem is unique keeps it.

    >>> unique_slugs(["/s/one/vocals.wav", "/s/two/vocals.wav", "/s/talk.m4a"])
    ['one-vocals', 'two-vocals', 'talk']
    >>> unique_slugs(["/Users/x/audio/talk.m4a", "/Users/x/audio/talk.wav"])
    ['talk', 'talk-2']
    >>> unique_slugs(["/s/a.wav", "/s/a.wav", "/s/a-2.wav"])
    ['a', 'a-3', 'a-2']

    ``reserved`` slugs (already used in the store) are never given out:

    >>> unique_slugs(["/t/vocals.wav"], reserved=["vocals"])
    ['vocals-2']
    """
    paths = [Path(s) for s in sources]
    stems = [_slug(p) for p in paths]
    groups: dict[str, list[int]] = {}
    for i, stem in enumerate(stems):
        groups.setdefault(stem, []).append(i)
    slugs = list(stems)
    for group in groups.values():
        if len(group) > 1:
            for i, slug in zip(group, _deepened([paths[i] for i in group])):
                slugs[i] = slug
    # Plain stems claim their slug before deepened ones, so a unique stem keeps its key.
    order = sorted(range(len(slugs)), key=lambda i: slugs[i] != stems[i])
    reserved = sorted(set(reserved))  # they claim their slugs before any source
    numbered = _numbered(reserved + [slugs[i] for i in order])[len(reserved) :]
    final = dict(zip(order, numbered))
    return [final[i] for i in range(len(slugs))]


def _key_slug(key) -> str | None:
    """The source slug in a key this module writes (``clips/<slug>-0001.wav`` or
    ``conditioned/<slug>.wav``), or None for any other key (or a non-string).

    >>> _key_slug("clips/song-a-vocals-0012.wav"), _key_slug("conditioned/talk-2.wav")
    ('song-a-vocals', 'talk-2')
    >>> _key_slug("tape1.wav") is None
    True
    """
    if not isinstance(key, str):
        return None
    key = key.replace("\\", "/")  # a store lists OS separators on Windows
    match = CLIP_KEY_PATTERN.fullmatch(key) or CONDITIONED_KEY_PATTERN.fullmatch(key)
    return match and match["slug"]


def _conditioned_key(slug: str) -> str:
    return f"{CONDITIONED_PREFIX}{slug}.wav"


def _entry_slugs(entry: Mapping) -> set[str]:
    """Every slug a manifest entry claims: its ``slug`` field (if slug-shaped) and the
    slug in its ``key``. Both, since a hand-edited entry may say one and name another.

    >>> sorted(_entry_slugs({"key": "clips/vocals-0001.wav", "slug": "x"}))
    ['vocals', 'x']
    >>> _entry_slugs({"key": "tape1.wav", "slug": 3})
    set()
    """
    slug = entry.get("slug")
    claimed = (
        {slug} if isinstance(slug, str) and SLUG_PATTERN.fullmatch(slug) else set()
    )
    return claimed | ({_key_slug(entry.get("key"))} - {None})


def _used_slugs(samples: Mapping, entries: Iterable[Mapping]) -> set[str]:
    """Slugs a new run must not use: those any manifest entry claims, and those in the
    store's own clip and conditioned keys (orphans in no manifest included).

    >>> store = {"clips/old-0001.wav": b"", "conditioned/lost.wav": b"", "a.wav": b""}
    >>> sorted(_used_slugs(store, [{"key": "clips/talk-0002.wav"}]))
    ['lost', 'old', 'talk']
    """
    from_entries = {s for e in entries for s in _entry_slugs(e)}
    return from_entries | ({_key_slug(k) for k in samples} - {None})


def _source_id(source: str | os.PathLike) -> str:
    """What tells sources apart: the real path, so a relative and an absolute path (or a
    symlink) to one file are one source."""
    return os.path.realpath(source)


def _entry_source_id(entry: Mapping) -> str | None:
    """The real path of the file a manifest entry came from, or None if it cannot be
    known: ``source_path`` if recorded; else ``source`` if it is an absolute path.
    Older voxy recorded ``source`` as typed, and a relative one names whatever file had
    that name in the folder the run happened to start in, so it identifies nothing.

    >>> _entry_source_id({"source_path": "/s/a.wav", "source": "a.wav"})
    '/s/a.wav'
    >>> _entry_source_id({"source": "vocals.wav"}) is None
    True
    """
    if isinstance(path := entry.get("source_path"), str):
        return path
    src = entry.get("source")
    if isinstance(src, str) and os.path.isabs(src):
        return _source_id(src)
    return None


def _is_from(entry: Mapping, source: str | os.PathLike) -> bool:
    """Whether a manifest entry came from ``source``, by real path (see
    ``_entry_source_id``). An entry whose source cannot be known never matches."""
    entry_id = _entry_source_id(entry)
    return entry_id is not None and entry_id == _source_id(source)


def _distinct_sources(sources: Iterable[str | os.PathLike]) -> list:
    """``sources`` with later repeats of one file (by ``_source_id``) dropped."""
    first = {}
    for src in sources:
        first.setdefault(_source_id(src), src)
    return list(first.values())


def _checked_manifest(samples: Mapping) -> list[dict]:
    """The store's manifest (empty if none), refusing one whose lines are not objects:
    raised before a run writes anything. Other oddities (no ``score``, no ``key``, a
    wrong ``slug``) are tolerated: such entries are kept as they are."""
    entries = load_manifest(samples) if MANIFEST_KEY in samples else []
    bad = [i for i, e in enumerate(entries, 1) if not isinstance(e, dict)]
    if bad:
        raise ValueError(f"{MANIFEST_KEY}: lines {bad} are not JSON objects; fix them")
    return entries


def _put_new(samples: MutableMapping, key: str, data: bytes) -> None:
    """``samples[key] = data``, refusing to overwrite: voice samples may exist nowhere
    else, and saved reference profiles point at their keys."""
    if key in samples:
        raise FileExistsError(f"voxy.refprep: {key} already exists; not overwriting it")
    samples[key] = data


def _number(x) -> float | None:
    """``x`` if it is a real number (not a bool), else None."""
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _score_order(entry: Mapping) -> float:
    """Sort key putting the best ``score`` first and entries without one last."""
    score = entry.get("score")
    return -score if isinstance(score, (int, float)) else math.inf


def _new_run_id() -> str:
    """A run id: when it started (UTC, to the microsecond, so ids sort by time), plus a
    few random characters so two runs never share one.

    >>> a, b = _new_run_id(), _new_run_id()
    >>> a != b and len(a) == len(b) and a[15:16] == "." and a[22:24] == "Z-"
    True
    """
    stamp = datetime.now(timezone.utc).strftime(RUN_ID_TIME_FORMAT)
    return f"{stamp}-{uuid.uuid4().hex[:RUN_ID_SUFFIX_LEN]}"


def latest_runs(manifest: Iterable[Mapping]) -> list[Mapping]:
    """``manifest`` with, per source, only the entries of its most recent run.

    ``prepare_reference`` keeps every run, so a source processed twice is in the
    manifest twice (the same audio, cut again); consumers that want one copy of each
    recording (training sets, reference picks, summaries) read this. Sources are told
    apart by real path (``source_path``, or an absolute ``source``); entries with no
    ``run`` count as one run older than any recorded one; entries whose source cannot
    be known (none, or a relative path from an older voxy) are all kept. Order is kept.

    >>> m = [{"key": "a1", "source_path": "/a", "run": "20260101T000000.000000Z-x"},
    ...      {"key": "a2", "source_path": "/a", "run": "20261101T000000.000000Z-y"},
    ...      {"key": "a0", "source": "/a"},
    ...      {"key": "b1", "source_path": "/b"},
    ...      {"key": "t", "source": "tape.wav"}]
    >>> [e["key"] for e in latest_runs(m)]
    ['a2', 'b1', 't']
    """
    manifest = list(manifest)

    def run_rank(entry):  # no run (or a non-string one) ranks below every real id
        run = entry.get("run")
        return (1, run) if isinstance(run, str) else (0, "")

    latest = {}
    for entry in manifest:
        if (sid := _entry_source_id(entry)) is not None:
            latest[sid] = max(latest.get(sid, (0, "")), run_rank(entry))
    return [
        e
        for e in manifest
        if (sid := _entry_source_id(e)) is None or run_rank(e) == latest[sid]
    ]


def _dump_manifest(entries: Iterable[Mapping]) -> bytes:
    return "\n".join(json.dumps(e) for e in entries).encode()


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
    speaker: Callable[..., list[dict]] | None = None,
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
        speaker: ``(wav_path, segments) -> segments`` marking other voices' words
            (``voxy.speakers.speaker_filter``); None keeps every voice.

    Returns:
        The manifest: one dict per clip (``key``, ``path`` if on disk, ``text``,
        ``source`` as given, ``source_path`` (its real path), ``slug``, ``run`` (one id
        per call), ``start``, ``end``, and the ``score_clip`` measures), best first,
        earlier runs' entries included.

    It never deletes or overwrites anything in the store. Run again into the same store
    (another recording, or the same file again), it gives this run's sources fresh slugs
    that no manifest entry and no clip or conditioned key in the store already uses, so
    every earlier clip keeps its key and its bytes, and saved reference profiles
    (``ref_audio`` + ``ref_text``) stay valid. The manifest keeps every earlier entry and
    adds the new ones (re-read just before writing, so a ``forget`` or another run that
    finished meanwhile is not undone). Re-processing a source therefore adds clips beside
    its earlier ones; consumers read ``latest_runs(manifest)`` to see one run per source,
    and ``forget(samples, source, run=...)`` removes the others explicitly. A file given
    twice in one call (by real path) is processed once.
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
    sources = _distinct_sources(sources)
    earlier = _checked_manifest(samples)
    _log_earlier_runs(sources, earlier, log)
    slugs = unique_slugs(sources, reserved=_used_slugs(samples, earlier))
    run = _new_run_id()
    manifest = []
    with tempfile.TemporaryDirectory() as tmp:
        for src, slug in zip(sources, slugs):
            wav_path = Path(tmp) / f"{slug}.wav"
            log(f"voxy.refprep: conditioning {src}")
            conditioner(src, wav_path, sample_rate=sample_rate, filters=filters)
            audio, sr = read_wav(wav_path)
            _put_new(samples, _conditioned_key(slug), wav_path.read_bytes())
            log(f"voxy.refprep: transcribing {slug} ({len(audio) / sr:.0f} s)")
            segments = transcriber(wav_path, language=language)
            if speaker is not None:
                log(f"voxy.refprep: marking other speakers in {slug}")
                segments = speaker(wav_path, segments)
            plan = plan_clips(
                segments,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                total_seconds=len(audio) / sr,
            )
            for i, clip in enumerate(plan, 1):
                piece = audio[int(clip["start"] * sr) : int(clip["end"] * sr)]
                key = f"{CLIPS_PREFIX}{slug}-{i:04d}.wav"
                _put_new(samples, key, write_wav(piece, sr))
                entry = {
                    "key": key,
                    "slug": slug,
                    "run": run,
                    "text": clip["text"],
                    "source": str(src),
                    "source_path": _source_id(src),
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
    new = len(manifest)
    current = _checked_manifest(samples)  # again: it may have changed while we ran
    manifest = sorted(manifest + current, key=_score_order)
    root = getattr(samples, "rootdir", None)
    if root:
        for entry in manifest:
            if isinstance(entry.get("key"), str):
                entry["path"] = os.path.join(root, entry["key"])
    samples[MANIFEST_KEY] = _dump_manifest(manifest)
    log(f"voxy.refprep: {new} clips saved (run {run}), {len(current)} earlier kept")
    return manifest


def _log_earlier_runs(sources: Iterable, earlier: list[Mapping], log) -> None:
    """Tell, per source already in the manifest, that its earlier clips stay."""
    for src in sources:
        n = sum(_is_from(e, src) for e in earlier)
        if n:
            log(
                f"voxy.refprep: {src} already has {n} clips from earlier runs; they are "
                f"kept (refprep.forget(samples, source, run=...) removes them)"
            )


def forget(
    samples: MutableMapping,
    source: str | os.PathLike,
    *,
    run: str | None = None,
    dry_run: bool = False,
    log=print,
) -> list[str]:
    """Remove ``source``'s clips (only those of one ``run``, if given) from a samples
    store: its manifest entries, their clip keys and the source's conditioned key.
    Returns the deleted keys (with ``dry_run=True``: the keys it would delete, changing
    nothing).

    The only function in voxy that deletes voice samples (``prepare_reference`` never
    does), so it is careful about what it deletes:

    - entries match ``source`` by real path: their ``source_path``, or, for entries an
      older voxy wrote without one, their ``source`` only if it is absolute. A relative
      ``source`` names whatever file had that name where that run started, so such
      entries never match; they are logged, and ``run=LEGACY_RUN`` removes them
      deliberately (matching ``source`` exactly as recorded, as in
      ``forget(samples, "vocals.wav", run=LEGACY_RUN)``);
    - only keys of the form this module writes (``clips/<slug>-NNNN.wav``,
      ``conditioned/<slug>.wav``), with a slug taken from a matched entry's clip key, are
      deleted, never a top-level sample;
    - a key a remaining entry names, or whose slug a remaining entry uses, is kept;
    - the manifest is read just before it is written, and written first, so an
      interruption leaves unlisted files, never a manifest naming deleted ones.

    >>> store = {"clips/a-0001.wav": b"A", "conditioned/a.wav": b"C",
    ...          "clips/b-0001.wav": b"B", "tape1.wav": b"T"}
    >>> store[MANIFEST_KEY] = _dump_manifest([
    ...     {"key": "clips/a-0001.wav", "slug": "a", "source": "/s/a.wav"},
    ...     {"key": "clips/b-0001.wav", "slug": "b", "source": "/s/b.wav"},
    ...     {"key": "tape1.wav", "source": "/s/a.wav"}])
    >>> forget(store, "/s/a.wav", dry_run=True)
    ['clips/a-0001.wav', 'conditioned/a.wav']
    >>> len(load_manifest(store)), len(store)
    (3, 5)
    >>> forget(store, "/s/a.wav")
    ['clips/a-0001.wav', 'conditioned/a.wav']
    >>> [e["key"] for e in load_manifest(store)], sorted(store)
    (['clips/b-0001.wav'], ['clips/b-0001.wav', 'manifest.jsonl', 'tape1.wav'])
    """

    def matches(entry):
        if run == LEGACY_RUN:
            if "run" in entry:
                return False
            if _entry_source_id(entry) is None:  # asked for: as recorded
                return entry.get("source") == str(source)
            return _is_from(entry, source)
        return _is_from(entry, source) and (run is None or entry.get("run") == run)

    def str_keys(entries):
        return [k for e in entries if isinstance(k := e.get("key"), str)]

    entries = _checked_manifest(samples)
    if run != LEGACY_RUN:
        _log_unmatched_legacy(entries, source, log)
    gone = [e for e in entries if matches(e)]
    remaining = [e for e in entries if not matches(e)]
    named = set(str_keys(remaining))
    used = {s for e in remaining for s in _entry_slugs(e)}
    slugs = {_key_slug(k) for k in str_keys(gone)} - {None} - used
    candidates = str_keys(gone) + sorted(map(_conditioned_key, slugs))
    keys = [
        k for k in dict.fromkeys(candidates) if _key_slug(k) in slugs and k not in named
    ]
    if dry_run:
        return [k for k in keys if k in samples]
    if not gone:
        return []
    samples[MANIFEST_KEY] = _dump_manifest(remaining)
    return [k for k in keys if samples.pop(k, None) is not None]


def _log_unmatched_legacy(entries: Iterable[Mapping], source, log) -> None:
    """Tell about entries an older voxy wrote with a relative ``source`` that may be
    ``source`` (same file name): ``forget`` cannot know, so it keeps them."""
    name = os.path.basename(os.fspath(source))
    counts = {}
    for e in entries:
        src = e.get("source")
        if _entry_source_id(e) is None and isinstance(src, str):
            if os.path.basename(src) == name:
                counts[src] = counts.get(src, 0) + 1
    for src, n in counts.items():
        log(
            f"voxy.refprep: {n} manifest entries record their source as the relative "
            f"path {src!r} (an older voxy), so which file they came from is unknown; "
            f"kept. To remove them: forget(samples, {src!r}, run=refprep.LEGACY_RUN) "
            f"(with dry_run=True first to see what it deletes), or edit {MANIFEST_KEY}."
        )


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
    latest_only: bool = True,
) -> list[dict]:
    """The ``n`` best clips for zero-shot reference (length in range, best score first).

    With ``latest_only`` (the default) it picks from ``latest_runs(manifest)``: an older
    run of a source is the same audio cut again, so its clips would come back as
    near-duplicates of the newer ones, and the newer run is the one made with the
    current settings. A reference saved from an older run stays valid either way (its
    key and bytes are kept); pass ``latest_only=False`` to choose among every run.

    >>> m = [{"duration": 8, "score": 0.7}, {"duration": 2, "score": 0.9},
    ...      {"duration": 10, "score": 0.8}]
    >>> [c["score"] for c in best_references(m, n=2)]
    [0.8, 0.7]
    """
    if latest_only:
        manifest = latest_runs(manifest)
    fits = [
        dict(c)
        for c in manifest
        if _number(c.get("duration")) is not None
        and min_seconds <= c["duration"] <= max_seconds
        and (_number(c.get("score")) or 0) >= min_score
    ]
    return sorted(fits, key=_score_order)[:n]


def summarize(manifest: Iterable[Mapping]) -> dict:
    """Totals for a manifest: clips, minutes, minutes above score thresholds.

    >>> summarize([{"duration": 60, "score": 0.9}, {"duration": 30, "score": 0.4}])
    {'clips': 2, 'minutes': 1.5, 'minutes_score_0.5+': 1.0, 'minutes_score_0.7+': 1.0}
    """
    manifest = list(manifest)
    minutes = lambda cs: round(sum(c.get("duration", 0) for c in cs) / 60, 2)  # noqa: E731
    scored = lambda t: [c for c in manifest if c.get("score", 0) >= t]  # noqa: E731
    return {
        "clips": len(manifest),
        "minutes": minutes(manifest),
        "minutes_score_0.5+": minutes(scored(0.5)),
        "minutes_score_0.7+": minutes(scored(0.7)),
    }
