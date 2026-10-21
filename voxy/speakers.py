"""
Keep one speaker: find their voice in a recording and mark everyone else's words.

Recordings of a person are rarely just that person: an interviewer asks the
questions, someone says "yes" or "uh-huh" in the middle of an answer. A clone or a
fine-tune trained on those clips learns the other voices too. This module
compares the voice in short sliding windows with a **voice print** (the mean
ECAPA speaker embedding of the target) and marks every transcribed word, and
every untranscribed stretch of someone else's voice, that does not match it:

    >>> from voxy import refprep, speakers                                  # doctest: +SKIP
    >>> print_ = speakers.voice_print("clean_answer.wav")                  # doctest: +SKIP
    >>> refprep.prepare_reference(["interview.m4a"], voice="ada",
    ...                           speaker=speakers.speaker_filter(print_))  # doctest: +SKIP

``plan_clips`` then never puts a marked word, or the pause around one, inside a
clip. With no voice print, ``speaker_filter()`` takes each recording's
**dominant** voice (an iterated centroid over its loud windows), which is right
for a talk or an interview where the target does most of the talking.

The embeddings come from SpeechBrain's ECAPA-TDNN (Apache-2.0, VoxCeleb weights)
running in voxy's isolated ``speaker`` environment (:data:`SPEAKER_ENV`, built on
first use, see ``voxy.envs``); everything after the embeddings is numpy, so the
logic below is testable with made-up vectors.
"""

import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voxy.envs import EnvSpec, ensure_env, worker_environ
from voxy.wavutil import frame_db, read_wav

DFLT_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DFLT_WINDOW_SECONDS = 1.0
DFLT_HOP_SECONDS = 0.1
#: Cosine similarity to the voice print below which a window is someone else.
DFLT_MATCH_SIMILARITY = 0.45
#: Windows this far below the recording's speech level are silence: neither voice.
DFLT_SILENCE_MARGIN_DB = 25.0
#: Untranscribed foreign stretches shorter than this are ignored (one window's wobble).
DFLT_MIN_FOREIGN_SECONDS = 0.2
#: Margin added around a foreign stretch, so a cut never clips its edge into a clip.
DFLT_FOREIGN_PAD_SECONDS = 0.15
#: A word with this share of its length inside a foreign stretch is not kept.
DFLT_WORD_OVERLAP = 0.25
#: Inside a transcribed turn, a stretch is someone else's only below this (smoothed) match:
#: one-second windows of old, noisy audio dip well below ``DFLT_MATCH_SIMILARITY`` on the
#: target's own voice, so a turn that matches as a whole is only cut where the dip is deep.
DFLT_DIP_SIMILARITY = 0.15
#: Window similarities are averaged over this many neighbours before finding stretches.
DFLT_SMOOTH_WINDOWS = 3
#: Turns shorter than this are too short to attribute ("Yes.", "Uh-huh"): never kept.
DFLT_MIN_TURN_SECONDS = 1.0
DOMINANT_ITERATIONS = 6
DOMINANT_QUANTILE = 0.5

SPEAKER_ENV = EnvSpec(
    "speaker",
    ("speechbrain>=1.0", "torch", "torchaudio", "librosa", "soundfile", "numpy"),
    env_vars={"VOXY_SPEAKER_MODELS": "{models}"},
)


@dataclass(frozen=True)
class Windows:
    """Speaker embeddings of sliding windows: ``times`` (n, 2) seconds, unit ``vectors`` (n, d)."""

    times: np.ndarray
    vectors: np.ndarray

    @property
    def centers(self) -> np.ndarray:
        return self.times.mean(axis=1) if len(self.times) else np.zeros(0)

    def select(self, mask) -> "Windows":
        return Windows(self.times[mask], self.vectors[mask])

    def similarity(self, voice_print: np.ndarray) -> np.ndarray:
        """Cosine similarity of each window to ``voice_print``."""
        return self.vectors @ voice_print


# embeddings --------------------------------------------------------------------

_SPEAKER_WORKERS: dict = {}


def _speaker_worker(model: str, threads: int | None, rootdir: str | None):
    """One embedding worker per (model, threads, data root), started on demand."""
    from voxy.local_clone import voxy_cache_dir
    from voxy.worker import Worker
    from voxy.workers import adapter_path

    key = (model, threads, rootdir)
    if key not in _SPEAKER_WORKERS:
        python = ensure_env(SPEAKER_ENV, rootdir=rootdir)
        _SPEAKER_WORKERS[key] = Worker(
            python,
            adapter_path("speaker"),
            init={"model": model},
            env=worker_environ(SPEAKER_ENV, threads=threads, rootdir=rootdir),
            log_path=voxy_cache_dir("logs", "speaker.log"),
        )
    return _SPEAKER_WORKERS[key]


def close_speaker() -> None:
    """Stop the embedding workers (free their memory)."""
    while _SPEAKER_WORKERS:
        _SPEAKER_WORKERS.popitem()[1].close()


def embed_windows(
    audio: str | os.PathLike,
    *,
    spans: Iterable[Sequence[float]] | None = None,
    window: float = DFLT_WINDOW_SECONDS,
    hop: float = DFLT_HOP_SECONDS,
    model: str = DFLT_SPEAKER_MODEL,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> Windows:
    """ECAPA embeddings of ``window``-second windows every ``hop`` s of ``audio``.

    ``spans`` restricts the windows to those ``(start, end)`` stretches (default:
    the whole file). Any sample rate; the worker resamples to 16 kHz.
    """
    worker = _speaker_worker(model, threads, str(rootdir) if rootdir else None)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "windows.npz")
        worker.call(
            "embed",
            audio=str(Path(audio).resolve()),  # the worker has its own cwd
            out_path=out,
            spans=None if spans is None else [list(map(float, s)) for s in spans],
            window=window,
            hop=hop,
        )
        with np.load(out) as data:
            return Windows(data["times"], data["vectors"])


# voice prints -------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def dominant_centroid(
    vectors: np.ndarray,
    *,
    iterations: int = DOMINANT_ITERATIONS,
    quantile: float = DOMINANT_QUANTILE,
) -> np.ndarray:
    """The voice heard most: re-centre on the windows most like the centroid, repeatedly.

    Clustering fragments a noisy voice into several clusters; this converges on
    the majority voice instead (it needs that voice to be more than half the windows
    the first mean is computed from).

    >>> rng = np.random.default_rng(0)
    >>> a, b = np.eye(4)[0], np.eye(4)[1]
    >>> vs = np.array([_unit(a + 0.1 * rng.standard_normal(4)) for _ in range(30)]
    ...             + [_unit(b + 0.1 * rng.standard_normal(4)) for _ in range(10)])
    >>> bool(dominant_centroid(vs) @ a > 0.95)
    True
    """
    if not len(vectors):
        raise ValueError(
            "No speech windows to find a voice in (silent or too short audio)"
        )
    c = _unit(vectors.mean(axis=0))
    for _ in range(iterations):
        sims = vectors @ c
        c = _unit(vectors[sims >= np.quantile(sims, quantile)].mean(axis=0))
    return c


def loud_mask(
    windows: Windows,
    samples: np.ndarray,
    sample_rate: int,
    *,
    margin_db: float = DFLT_SILENCE_MARGIN_DB,
) -> np.ndarray:
    """Windows with speech in them: loud frames within ``margin_db`` of the speech level.

    >>> w = Windows(np.array([[0, 1], [1, 2]]), np.zeros((2, 3)))
    >>> x = np.concatenate([0.3 * np.ones(8000), np.zeros(8000)])
    >>> loud_mask(w, x, 8000).tolist()
    [True, False]
    """
    levels = frame_db(samples, sample_rate)
    if not levels.size or not len(windows.times):
        return np.zeros(len(windows.times), dtype=bool)
    speech_level = float(np.percentile(levels, 95))
    frame_s = len(samples) / sample_rate / len(levels)
    mask = []
    for start, end in windows.times:
        frames = levels[
            int(start / frame_s) : max(int(start / frame_s) + 1, int(end / frame_s))
        ]
        mask.append(
            frames.size > 0 and np.percentile(frames, 90) > speech_level - margin_db
        )
    return np.array(mask, dtype=bool)


def voice_print(
    audio: str | os.PathLike | Iterable[str | os.PathLike],
    *,
    spans: Iterable[Sequence[float]] | None = None,
    embed: Callable[..., Windows] | None = None,
    **embed_kwargs,
) -> np.ndarray:
    """The voice print of ``audio`` (one recording or several): the dominant voice's centroid.

    Pass recordings (or ``spans`` of one) where the target speaks alone, and this is
    simply their voice; over mixed recordings it is whoever speaks most.
    """
    embed = embed or embed_windows
    paths = [audio] if isinstance(audio, (str, os.PathLike)) else list(audio)
    vectors = []
    for path in paths:
        windows = embed(path, spans=spans, **embed_kwargs)
        samples, sr = (
            read_wav(path) if str(path).lower().endswith(".wav") else (None, None)
        )
        if samples is not None:
            windows = windows.select(loud_mask(windows, samples, sr))
        vectors.append(windows.vectors)
    return dominant_centroid(np.concatenate(vectors))


# marking words ------------------------------------------------------------------


def _merge(spans: Iterable[Sequence[float]], *, gap: float = 0.0) -> list[list[float]]:
    """Union of intervals (touching within ``gap`` merge).

    >>> _merge([(0, 1), (0.9, 2), (3, 4)])
    [[0, 2], [3, 4]]
    """
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def foreign_spans(
    windows: Windows,
    similarity: np.ndarray,
    *,
    threshold: float | np.ndarray = DFLT_MATCH_SIMILARITY,
    loud: np.ndarray | None = None,
    min_seconds: float = DFLT_MIN_FOREIGN_SECONDS,
    pad: float = DFLT_FOREIGN_PAD_SECONDS,
) -> list[list[float]]:
    """Stretches where a loud window does not match the voice print, padded by ``pad``.

    ``threshold`` is one similarity, or one per window.

    Each window speaks for the ``hop``-wide slot around its centre, so a short "yes"
    marks about the length of the dip it causes, not a whole window.

    >>> t = np.array([[i * 0.1, i * 0.1 + 1] for i in range(20)])
    >>> sim = np.array([0.8] * 8 + [0.1] * 4 + [0.8] * 8)
    >>> [[round(x, 2) for x in s] for s in foreign_spans(Windows(t, None), sim, pad=0)]
    [[1.25, 1.65]]
    """
    if not len(windows.times):
        return []
    centers = windows.centers
    hop = float(np.median(np.diff(centers))) if len(centers) > 1 else 0.0
    bad = similarity < threshold
    if loud is not None:
        bad &= loud
    slots = [(c - hop / 2, c + hop / 2) for c in centers[bad]]
    return [
        [max(0.0, float(s) - pad), float(e) + pad]
        for s, e in _merge(slots, gap=hop * 0.5)
        if e - s >= min_seconds
    ]


def smooth(similarity: np.ndarray, windows: int = DFLT_SMOOTH_WINDOWS) -> np.ndarray:
    """Moving average of ``similarity`` over ``windows`` neighbours (edges: the shorter mean).

    >>> smooth(np.array([1.0, 1.0, 0.0, 1.0, 1.0])).round(2).tolist()
    [1.0, 0.67, 0.67, 0.67, 1.0]
    """
    windows = min(windows, len(similarity))
    if windows <= 1:
        return np.asarray(similarity, dtype=float)
    kernel = np.ones(windows)
    total = np.convolve(similarity, kernel, mode="same")
    count = np.convolve(np.ones(len(similarity)), kernel, mode="same")
    return total / count


def turn_similarity(
    segments: Iterable[Mapping],
    windows: Windows,
    voice_print: np.ndarray,
    *,
    loud: np.ndarray | None = None,
) -> list[float | None]:
    """Match of each transcribed segment (a turn) to ``voice_print``, from its mean window.

    Averaging the embeddings over a whole turn is far steadier than any one window
    on noisy audio: it is what tells an interviewer's question from an answer. None
    for a segment with no (loud) window centred in it.

    >>> t = np.array([[i * 0.1, i * 0.1 + 1] for i in range(40)])
    >>> v = np.array([np.eye(2)[0]] * 20 + [np.eye(2)[1]] * 20)
    >>> turn_similarity([{"start": 0, "end": 2}, {"start": 2.6, "end": 4.4}],
    ...                 Windows(t, v), np.eye(2)[0])
    [1.0, 0.0]
    """
    centers = windows.centers
    out = []
    for seg in segments:
        inside = (centers >= seg["start"]) & (centers <= seg["end"])
        if loud is not None:
            inside &= loud
        if not inside.any():
            out.append(None)
            continue
        out.append(
            round(float(_unit(windows.vectors[inside].mean(axis=0)) @ voice_print), 3)
        )
    return out


def _overlap(start: float, end: float, spans: Iterable[Sequence[float]]) -> float:
    """Seconds of ``[start, end]`` covered by ``spans``.

    >>> _overlap(1, 3, [(0, 1.5), (2.5, 4)])
    1.0
    """
    return sum(max(0.0, min(end, e) - max(start, s)) for s, e in spans)


def _word_similarity(start, end, centers, similarity) -> float | None:
    """Mean similarity of the windows centred in ``[start, end]`` (else the nearest one)."""
    inside = (centers >= start) & (centers <= end)
    if not inside.any() and len(centers):
        distance = np.abs(centers - (start + end) / 2)
        inside = distance == distance.min()
    return float(similarity[inside].mean()) if inside.any() else None


def _ends_sentence(word: Mapping) -> bool:
    return word.get("text", "").rstrip().endswith((".", "!", "?", "…"))


def mark_speaker(
    segments: Iterable[Mapping],
    windows: Windows,
    similarity: np.ndarray,
    *,
    threshold: float = DFLT_MATCH_SIMILARITY,
    loud: np.ndarray | None = None,
    turns: Sequence[float | None] | None = None,
    dip_threshold: float = DFLT_DIP_SIMILARITY,
    min_turn_seconds: float = DFLT_MIN_TURN_SECONDS,
    word_overlap: float = DFLT_WORD_OVERLAP,
    pad: float = DFLT_FOREIGN_PAD_SECONDS,
    min_seconds: float = DFLT_MIN_FOREIGN_SECONDS,
) -> list[dict]:
    """Annotate each timed word with ``speaker_similarity`` and ``keep``; add gap markers.

    Without ``turns``, a word is kept when the windows centred inside it match the
    voice print on average. With ``turns`` (one match per segment, see
    :func:`turn_similarity`), a word is kept when its whole segment matches and lasts
    at least ``min_turn_seconds``, and its ``speaker_similarity`` is the segment's;
    inside a segment only a deep dip (below ``dip_threshold``) counts as someone else,
    and a one-word sentence ("Yes.") must match the voice print on its own.
    Either way a word is dropped when ``word_overlap`` of it lies in a foreign
    stretch. Every foreign stretch is also added as an empty word with
    ``keep=False`` (an untranscribed "uh-huh" has no word of its own), padded by up
    to ``pad`` and trimmed back to the kept words around it, so ``refprep.plan_clips``
    never puts it inside a clip. A segment without word timings is judged as one word.

    >>> t = np.array([[i * 0.1, i * 0.1 + 1] for i in range(40)])
    >>> sim = np.array([0.8] * 15 + [0.1] * 5 + [0.8] * 20)
    >>> w = lambda s, e, x: {"start": s, "end": e, "text": x}
    >>> segs = [{"start": 0, "end": 4, "text": "a b",
    ...          "words": [w(0.5, 1.4, " a"), w(2.5, 3.8, " b")]}]
    >>> out = mark_speaker(segs, Windows(t, None), sim)
    >>> [(x["text"], x["keep"]) for x in out[0]["words"]], out[1]["words"][0]["end"]
    ([(' a', True), (' b', True)], 2.5)
    """
    segments = list(segments)
    centers = windows.centers
    floor = np.full(len(centers), threshold, dtype=float)
    if turns is not None:
        for seg in segments:
            floor[(centers >= seg["start"]) & (centers <= seg["end"])] = dip_threshold
    foreign = foreign_spans(
        windows, similarity, threshold=floor, loud=loud, min_seconds=min_seconds, pad=0
    )
    out, kept = [], []
    for i, seg in enumerate(segments):
        seg = dict(seg)
        turn = None if turns is None else turns[i]
        turn_ok = turn is None or turn >= threshold
        if turns is not None:
            turn_ok = turn_ok and seg["end"] - seg["start"] >= min_turn_seconds
        timed = [w for w in seg.get("words") or () if w.get("start") is not None]
        # Untimed segment: judge it as one word, or plan_clips would keep it unmarked.
        source = (
            seg.get("words")
            if timed
            else [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg.get("text", ""),
                    "confidence": None,
                }
            ]
        )
        words = []
        for word in source:
            word = dict(word)
            if word.get("start") is None:
                words.append(word)
                continue
            start, end = word["start"], word["end"]
            in_foreign = _overlap(start, end, foreign) / max(end - start, 1e-6)
            sim = _word_similarity(start, end, centers, similarity)
            word_ok = sim is None or sim >= threshold
            if turns is not None:
                word["speaker_similarity"] = turn
                # A one-word sentence ("Yes.") inside a turn is often the listener's.
                alone = _ends_sentence(word) and (
                    not words or _ends_sentence(words[-1])
                )
                word["keep"] = (
                    turn_ok and (word_ok or not alone) and in_foreign < word_overlap
                )
            else:
                word["speaker_similarity"] = None if sim is None else round(sim, 3)
                word["keep"] = word_ok and in_foreign < word_overlap
            if word["keep"]:
                kept.append((start, end))
            words.append(word)
        seg["words"] = words
        out.append(seg)
    markers = []
    for s, e in foreign:
        # Never into a kept word: plan_clips cuts between a word and the marker after it.
        lo = max([s - pad, 0.0] + [we for ws, we in kept if ws < s and we > s - pad])
        hi = min([e + pad] + [ws for ws, we in kept if we > e and ws < e + pad])
        if lo >= hi:  # inside one kept word, within its word_overlap tolerance
            continue
        markers.append(
            {"start": lo, "end": hi, "text": "", "confidence": None, "keep": False}
        )
    if markers:
        out.append(
            {
                "start": markers[0]["start"],
                "end": markers[-1]["end"],
                "text": "",
                "words": markers,
            }
        )
    return out


def speaker_filter(
    target: np.ndarray | None = None,
    *,
    threshold: float = DFLT_MATCH_SIMILARITY,
    dip_threshold: float = DFLT_DIP_SIMILARITY,
    smooth_windows: int = DFLT_SMOOTH_WINDOWS,
    min_turn_seconds: float = DFLT_MIN_TURN_SECONDS,
    embed: Callable[..., Windows] | None = None,
    **embed_kwargs,
) -> Callable[..., list[dict]]:
    """A ``refprep.prepare_reference(speaker=...)`` step: ``(wav_path, segments) -> segments``.

    Decides per transcribed segment (a turn: :func:`turn_similarity`), then cuts
    deep dips inside kept turns and other voices between them (:func:`mark_speaker`).

    Args:
        target: The voice print to keep (from :func:`voice_print`); None takes each
            recording's dominant voice.
        threshold: Cosine similarity a turn (or a stretch between turns) needs to
            count as the target.
        dip_threshold: Smoothed similarity below which a stretch inside a kept turn
            is someone else.
        smooth_windows: Neighbouring windows averaged before finding stretches.
        min_turn_seconds: Shorter turns are never kept.
        embed: ``(path, **embed_kwargs) -> Windows`` (default :func:`embed_windows`).
    """
    embed = embed or embed_windows

    def mark(wav_path, segments):
        segments = list(segments)
        windows = embed(wav_path, **embed_kwargs)
        samples, sr = read_wav(wav_path)
        loud = loud_mask(windows, samples, sr)
        if not loud.any():  # too short or silent: nothing to attribute, no clip either
            return segments
        voice = (
            target if target is not None else dominant_centroid(windows.vectors[loud])
        )
        return mark_speaker(
            segments,
            windows,
            smooth(windows.similarity(voice), smooth_windows),
            threshold=threshold,
            loud=loud,
            turns=turn_similarity(segments, windows, voice, loud=loud),
            dip_threshold=dip_threshold,
            min_turn_seconds=min_turn_seconds,
        )

    return mark
