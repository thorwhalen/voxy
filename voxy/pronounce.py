"""Checking, in the audio, which reading of a flagged word a render used.

:func:`voxy.speechtext.flag_risks` lists the words a model may misread (heteronyms
such as French "est" /ɛ/ vs "Est" /ɛst/). This module listens for the answer: it crops
the word out of a render (word timestamps from ASR), asks a phoneme recogniser for its
frame-wise phoneme log-probabilities, and scores each candidate reading by its CTC
likelihood. The reading the audio supports best is the one the model said; a render
whose word came out as the wrong one can be re-rendered.

The recogniser is ``facebook/wav2vec2-lv-60-espeak-cv-ft``, run in its own torch env
(:data:`PHONEME_ENV`, built by ``voxy.envs``). Its ``vocab.json`` maps espeak IPA
symbols to ids, so candidate readings are written in IPA and tokenised against that
vocabulary: no phonemizer is needed. Everything above the recogniser is pure numpy and
takes the recogniser as an argument (``emitter``), so it is tested without torch.

>>> vocab = ["<pad>", "ɛ", "s", "t"]
>>> emissions = log_probs_from_path([1, 0, 0], vocab_size=4)  # hears /ɛ/ only
>>> best_reading(emissions, vocab, {"verb": "ɛ", "noun": "ɛst"})
'verb'
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from voxy.envs import EnvSpec, ensure_env, worker_environ

#: The phoneme recogniser: wav2vec2 fine-tuned to emit espeak IPA (CTC, 16 kHz).
DFLT_PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
#: Its env: CPU torch plus transformers (the processor's vocab.json is all it needs).
PHONEME_ENV = EnvSpec(
    "phonemes", ("torch", "transformers", "soundfile", "numpy", "librosa")
)
#: Seconds kept on each side of a word's ASR timestamps: ASR word edges are loose.
DFLT_CROP_PADDING = 0.08
#: A reading must beat the runner-up by this much (mean log-likelihood per target
#: symbol) for a verdict to count as decided.
DFLT_MARGIN = 0.5

#: The CTC blank of the recogniser's vocabulary (wav2vec2 uses its pad token).
BLANK_SYMBOL = "<pad>"

#: ``(audio, start, end) -> (log_probs [frames x vocab], vocab)``.
Emitter = Callable[[str, float, float], tuple[np.ndarray, Sequence[str]]]


# 1. scoring --------------------------------------------------------------------


def tokenize_phonemes(reading: str | Sequence[str], vocab: Sequence[str]) -> list[int]:
    """Vocabulary ids of an IPA ``reading``, longest symbol first; spaces are ignored.

    A reading may also be given as a sequence of symbols. Raises ``ValueError`` on a
    symbol the recogniser does not know, since it could never be heard.

    >>> tokenize_phonemes("ɛ̃s", ["<pad>", "ɛ", "ɛ̃", "s"])
    [2, 3]
    """
    index = {symbol: i for i, symbol in enumerate(vocab)}
    if not isinstance(reading, str):
        unknown = [s for s in reading if s not in index]
        if unknown:
            raise ValueError(f"symbols not in the recogniser's vocabulary: {unknown}")
        return [index[s] for s in reading]
    longest = max(map(len, vocab))
    text, ids, i = reading.replace(" ", ""), [], 0
    while i < len(text):
        for size in range(min(longest, len(text) - i), 0, -1):
            if text[i : i + size] in index:
                ids.append(index[text[i : i + size]])
                i += size
                break
        else:
            raise ValueError(
                f"{text[i]!r} in reading {reading!r} is not in the recogniser's "
                f"vocabulary"
            )
    return ids


def ctc_log_likelihood(
    log_probs: np.ndarray, target: Sequence[int], *, blank=0
) -> float:
    """``log P(target | log_probs)`` under CTC (the forward algorithm, in log space).

    ``log_probs`` is ``frames x vocab`` (log-softmax). ``-inf`` when the target cannot
    be aligned (more symbols than frames allow).

    >>> lp = log_probs_from_path([1, 2], vocab_size=3)
    >>> ctc_log_likelihood(lp, [1, 2]) > ctc_log_likelihood(lp, [2, 1])
    True
    """
    log_probs = np.asarray(log_probs, dtype=float)
    frames = log_probs.shape[0]
    extended = [blank]
    for symbol in target:
        extended += [symbol, blank]
    states = len(extended)
    alpha = np.full(states, -np.inf)
    alpha[0] = log_probs[0, blank]
    if states > 1:
        alpha[1] = log_probs[0, extended[1]]
    for t in range(1, frames):
        previous = alpha
        alpha = np.full(states, -np.inf)
        for s in range(states):
            terms = [previous[s]]
            if s >= 1:
                terms.append(previous[s - 1])
            if s >= 2 and extended[s] != blank and extended[s] != extended[s - 2]:
                terms.append(previous[s - 2])
            alpha[s] = np.logaddexp.reduce(terms) + log_probs[t, extended[s]]
    return float(np.logaddexp(alpha[-1], alpha[-2]) if states > 1 else alpha[-1])


def score_readings(
    log_probs: np.ndarray,
    vocab: Sequence[str],
    readings: Mapping[str, str | Sequence[str]],
    *,
    blank: int | None = None,
) -> dict[str, float]:
    """Each reading's CTC log-likelihood per target symbol (higher is better).

    Normalising by length keeps a longer reading ("ɛst") from losing to a shorter one
    ("ɛ") just for having more symbols to pay for. ``blank`` defaults to the id of
    :data:`BLANK_SYMBOL` in ``vocab`` (0 if it has none).
    """
    if blank is None:
        blank = list(vocab).index(BLANK_SYMBOL) if BLANK_SYMBOL in vocab else 0
    scores = {}
    for name, reading in readings.items():
        target = tokenize_phonemes(reading, vocab)
        scores[name] = ctc_log_likelihood(log_probs, target, blank=blank) / max(
            1, len(target)
        )
    return scores


def best_reading(log_probs, vocab, readings, *, blank: int | None = None) -> str:
    """The name of the reading ``log_probs`` support best."""
    scores = score_readings(log_probs, vocab, readings, blank=blank)
    return max(scores, key=scores.get)


def log_probs_from_path(
    path: Sequence[int], *, vocab_size: int, confidence: float = 0.9
) -> np.ndarray:
    """Log-probabilities of a recogniser that hears ``path`` (one id per frame).

    A test and doctest helper: each frame gives ``confidence`` to its id and shares
    the rest evenly.

    >>> log_probs_from_path([0], vocab_size=2).round(2).tolist()
    [[-0.11, -2.3]]
    """
    rest = (1 - confidence) / (vocab_size - 1)
    probs = np.full((len(path), vocab_size), rest)
    probs[np.arange(len(path)), list(path)] = confidence
    return np.log(probs)


# 2. verdicts on a render -------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What the audio says about one flagged word."""

    word: str
    start: float
    end: float
    scores: Mapping[str, float]
    expected: str | None = None
    margin: float = DFLT_MARGIN

    @property
    def heard(self) -> str:
        """The reading the audio supports best."""
        return max(self.scores, key=self.scores.get)

    @property
    def decided(self) -> bool:
        """Whether the best reading beats the runner-up by at least ``margin``."""
        ranked = sorted(self.scores.values(), reverse=True)
        return len(ranked) < 2 or ranked[0] - ranked[1] >= self.margin

    @property
    def ok(self) -> bool | None:
        """True/False when an ``expected`` reading was given and the audio decided."""
        if self.expected is None or not self.decided:
            return None
        return self.heard == self.expected


def word_spans(words: Iterable[Mapping], word: str) -> list[tuple[float, float]]:
    """``(start, end)`` of every ASR word matching ``word`` (case and punctuation aside).

    ``words`` are timed words as ``voxy.refprep.transcribe`` gives them
    (``{"start", "end", "text", ...}``).

    >>> word_spans([{"start": 0.1, "end": 0.3, "text": " Est,"}], "est")
    [(0.1, 0.3)]
    """
    target = word.casefold()
    return [
        (w["start"], w["end"])
        for w in words
        if w["text"].strip().strip(".,;:!?«»\"'()").casefold() == target
    ]


def verify_words(
    audio: str | os.PathLike,
    checks: Iterable[tuple[str, Mapping[str, str], str | None]],
    *,
    words: Iterable[Mapping],
    emitter: Emitter,
    padding: float = DFLT_CROP_PADDING,
    margin: float = DFLT_MARGIN,
) -> list[Verdict]:
    """Verdicts for each ``(word, readings, expected)`` at each place ASR found it.

    Args:
        audio: The render.
        checks: The words to check, each with its candidate readings
            (``{name: IPA}``) and the reading it should have (or None to just
            report). :func:`checks_from_risks` builds them from ``flag_risks``.
        words: Timed words of the render (``transcribe(audio)``'s segments' words).
        emitter: ``(audio, start, end) -> (log_probs, vocab)``; the default worker
            is :func:`phoneme_emitter`.
        padding: Seconds added to each side of the ASR span.
        margin: See :attr:`Verdict.decided`.
    """
    words = list(words)
    verdicts = []
    for word, readings, expected in checks:
        for start, end in word_spans(words, word):
            log_probs, vocab = emitter(
                str(audio), max(0.0, start - padding), end + padding
            )
            verdicts.append(
                Verdict(
                    word,
                    start,
                    end,
                    score_readings(log_probs, vocab, readings),
                    expected,
                    margin,
                )
            )
    return verdicts


def checks_from_risks(
    risks: Iterable,
    *,
    readings: Mapping[str, Mapping[str, str]],
    expected: Mapping[str, str] | None = None,
) -> list[tuple[str, Mapping[str, str], str | None]]:
    """``verify_words`` checks for the flagged words that have known IPA readings.

    ``readings`` maps a word (case-insensitively) to ``{name: IPA}``, e.g. from a
    Lexique-derived heteronym list the caller supplies; ``expected`` names the
    reading each word should have. Words with no entry are skipped. Each word is
    checked once: ``verify_words`` finds every place it is spoken.
    """
    table = {k.casefold(): v for k, v in readings.items()}
    wanted = {k.casefold(): v for k, v in (expected or {}).items()}
    seen, checks = set(), []
    for risk in risks:
        key = risk.word.casefold()
        if key in table and key not in seen:
            seen.add(key)
            checks.append((risk.word, table[key], wanted.get(key)))
    return checks


def failing(verdicts: Iterable[Verdict]) -> list[Verdict]:
    """The verdicts whose word was decidedly read the wrong way (``ok is False``)."""
    return [v for v in verdicts if v.ok is False]


# 3. the recogniser worker ------------------------------------------------------


_PHONEME_WORKERS: dict = {}


def _phoneme_worker(model: str, threads: int | None, rootdir: str | None):
    """One recogniser worker per (model, threads, data root), started on demand."""
    from voxy.local_clone import voxy_cache_dir
    from voxy.worker import Worker
    from voxy.workers import adapter_path

    key = (model, threads, rootdir)
    if key not in _PHONEME_WORKERS:
        python = ensure_env(PHONEME_ENV, rootdir=rootdir)
        _PHONEME_WORKERS[key] = Worker(
            python,
            adapter_path("phonemes"),
            init={"model": model},
            env=worker_environ(PHONEME_ENV, threads=threads, rootdir=rootdir),
            log_path=voxy_cache_dir("logs", "phonemes.log"),
        )
    return _PHONEME_WORKERS[key]


def phoneme_emitter(
    *,
    model: str = DFLT_PHONEME_MODEL,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> Emitter:
    """The default :data:`Emitter`: the wav2vec2 phoneme recogniser in its own env.

    The env (torch, transformers) is built on first use under the data root, and the
    model weights download from the Hugging Face hub then.
    """

    def emit(audio: str, start: float, end: float):
        worker = _phoneme_worker(model, threads, str(rootdir) if rootdir else None)
        reply = worker.call("emissions", audio=audio, start=start, end=end)
        return np.asarray(reply["log_probs"], dtype=float), reply["vocab"]

    return emit
