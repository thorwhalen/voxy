"""Checking, in the audio, which reading of a flagged word a render used.

:func:`voxy.speechtext.flag_risks` lists the words a model may misread (heteronyms
such as French "est" /ɛ/ vs "Est" /ɛst/). This module listens for the answer: it finds
the word in a render (word timestamps from ASR), asks a phoneme recogniser for its
frame-wise phoneme log-probabilities, and scores each candidate reading by its CTC
likelihood. The reading the audio supports best is the one the model said; a render
whose word came out as the wrong one can be re-rendered.

ASR word times are loose, so a word cropped alone carries bits of its neighbours, and a
neighbour's /s/ or /t/ then reads as the longer reading. :func:`audit` (the call for a
finished render), :func:`verify_words` and :func:`render_verified` (the calls while
rendering) therefore score each reading as the whole phrase around it by default, the
neighbours written out by espeak (:func:`phrase_readings`, :func:`espeak_phonemizer`).
They share one seam, a :class:`Scorer`: :func:`phrase_scorer`, :func:`spot_scorer` and
:func:`crop_scorer` make one, each carrying the margin it is calibrated for, and
:func:`reading_scorer` makes one by name (:data:`SCORERS`).

The recogniser is ``facebook/wav2vec2-lv-60-espeak-cv-ft``, run in its own torch env
(:data:`PHONEME_ENV`, built by ``voxy.envs``, with a bundled espeak-ng). Its
``vocab.json`` maps espeak IPA symbols to ids, so readings are written in IPA and
tokenised against that vocabulary. Everything above the recogniser is pure numpy and
takes the recogniser (``emitter``) and the phonemizer as arguments, so it is tested
without torch.

>>> vocab = ["<pad>", "ɛ", "s", "t"]
>>> emissions = log_probs_from_path([1, 0, 0], vocab_size=4)  # hears /ɛ/ only
>>> best_reading(emissions, vocab, {"verb": "ɛ", "noun": "ɛst"})
'verb'
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from voxy.envs import EnvSpec, ensure_env, worker_environ
from voxy.speechtext import _WORD, ELISIONS, NOUN_CLITICS, flag_risks, split_clitic

#: The phoneme recogniser: wav2vec2 fine-tuned to emit espeak IPA (CTC, 16 kHz).
DFLT_PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
#: Its env: CPU torch plus transformers (the processor's vocab.json is all it needs).
PHONEME_ENV = EnvSpec(
    "phonemes",
    (
        "torch",
        "transformers",
        "soundfile",
        "numpy",
        "librosa",
        "phonemizer",
        "espeakng-loader",
    ),
)
#: Seconds kept on each side of a word's ASR timestamps: ASR word edges are loose.
DFLT_CROP_PADDING = 0.08
# Margins: by how much the best reading must beat the runner-up for a verdict to count
# as decided, each on its scorer's own scale. Phrase scoring was calibrated on 250
# flagged words of a French audiobook (one decided misread, a real one); crop scoring
# flagged 46 correct "est" there as the noun.
#: The margin of :func:`crop_scorer` (mean log-likelihood per target symbol).
DFLT_MARGIN = 0.5
#: The margin of :func:`phrase_scorer` (total nats).
DFLT_CONTEXT_MARGIN = 2.0

#: What spotting a word in its context charges per frame left to filler (nats).
DFLT_FILLER_PENALTY = 1.0
#: The language of the texts checked, unless told otherwise.
DFLT_LANGUAGE = "fr"
#: espeak voices for :func:`audit`'s ``language`` codes (others are passed as they are).
ESPEAK_LANGUAGES = {"fr": "fr-fr", "en": "en-us"}

#: ``(texts, language) -> IPA of each``: writes out the words around a flagged one.
Phonemizer = Callable[[Sequence[str], str], Sequence[str]]

#: Words of context on each side of a flagged word when :func:`audit` scores it.
DFLT_CONTEXT_WORDS = 1
#: The margin of :func:`spot_scorer` (total nats).
DFLT_SPOT_MARGIN = 2.0
#: How readings are scored unless told otherwise (a name in :data:`SCORERS`).
DFLT_SCORING = "phrase"

#: In a reading name, what separates a reading from one of its variants (``verb~liaison``).
VARIANT_SEPARATOR = "~"

#: Punctuation stripped from the edges of an ASR or text word before it is matched or spoken.
EDGE_PUNCTUATION = '.,;:!?«»"()…“”'

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


def spot_log_likelihood(
    log_probs: np.ndarray,
    target: Sequence[int],
    *,
    blank: int = 0,
    filler_penalty: float = None,
) -> float:
    """``log P`` of ``target`` said somewhere in ``log_probs``, the rest being filler.

    Keyword spotting under CTC: frames before and after the target are explained by a
    filler that takes each frame's best symbol at a cost of ``filler_penalty`` nats.
    Scoring a word inside a window that holds its neighbours, a reading wins by
    matching phonemes that are really there, not by the window's edges: a crop cut
    from loose ASR word times otherwise favours the longer reading (/ɛst/ over /ɛ/).

    >>> vocab = ["<pad>", "ɛ", "s", "t", "l"]
    >>> lp = log_probs_from_path([4, 1, 0, 4], vocab_size=5)  # "l ɛ l": the verb
    >>> verb, noun = (spot_log_likelihood(lp, tokenize_phonemes(r, vocab)) for r in ("ɛ", "ɛst"))
    >>> verb > noun
    True
    """
    penalty = DFLT_FILLER_PENALTY if filler_penalty is None else filler_penalty
    filler = log_probs.max(axis=1) - penalty
    extended = [blank]
    for symbol in target:
        extended += [symbol, blank]
    states = len(extended)
    neg = -np.inf
    pre, post = filler[0], neg
    alpha = np.full(states, neg)
    alpha[0] = log_probs[0, blank]
    if states > 1:
        alpha[1] = log_probs[0, extended[1]]
    for t in range(1, len(log_probs)):
        previous = alpha
        alpha = np.full(states, neg)
        for s in range(states):
            terms = [previous[s]]
            if s >= 1:
                terms.append(previous[s - 1])
            if s >= 2 and extended[s] != blank and extended[s] != extended[s - 2]:
                terms.append(previous[s - 2])
            if s <= 1:
                terms.append(pre)
            alpha[s] = np.logaddexp.reduce(terms) + log_probs[t, extended[s]]
        ends = [post, previous[-1]] + ([previous[-2]] if states > 1 else [])
        post = np.logaddexp.reduce(ends) + filler[t]
        pre = pre + filler[t]
    finals = [post, alpha[-1]] + ([alpha[-2]] if states > 1 else [])
    return float(np.logaddexp.reduce(finals))


def spot_readings(
    log_probs: np.ndarray,
    vocab: Sequence[str],
    readings: Mapping[str, str | Sequence[str]],
    *,
    blank: int | None = None,
    filler_penalty: float | None = None,
) -> dict[str, float]:
    """Each reading's :func:`spot_log_likelihood`, less that of filler alone (higher is better)."""
    if blank is None:
        blank = list(vocab).index(BLANK_SYMBOL) if BLANK_SYMBOL in vocab else 0
    penalty = DFLT_FILLER_PENALTY if filler_penalty is None else filler_penalty
    baseline = float((log_probs.max(axis=1) - penalty).sum())
    return {
        name: spot_log_likelihood(
            log_probs,
            tokenize_phonemes(reading, vocab),
            blank=blank,
            filler_penalty=penalty,
        )
        - baseline
        for name, reading in readings.items()
    }


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
    """What the audio says about one flagged word.

    ``margin`` defaults to :data:`DFLT_MARGIN`, the crop scale; :func:`verify_words`
    and :func:`audit` give each verdict their scorer's margin (phrase and spot scores
    are totals in nats).
    """

    word: str
    start: float
    end: float
    scores: Mapping[str, float]
    expected: str | None = None
    margin: float = DFLT_MARGIN

    @property
    def heard(self) -> str:
        """The reading (variant included) the audio supports best."""
        return max(self.scores, key=self.scores.get)

    @property
    def reading_scores(self) -> dict[str, float]:
        """``scores`` with each reading's variants (``name~variant``) merged by their best."""
        merged: dict[str, float] = {}
        for name, score in self.scores.items():
            base = reading_name(name)
            merged[base] = max(score, merged.get(base, -np.inf))
        return merged

    @property
    def reading(self) -> str:
        """The reading the audio supports best, variants folded in (``verb~e`` is ``verb``)."""
        return reading_name(self.heard)

    @property
    def decided(self) -> bool:
        """Whether the best reading beats the runner-up reading by at least ``margin``.

        Variants of one reading do not compete with each other.
        """
        ranked = sorted(self.reading_scores.values(), reverse=True)
        return len(ranked) < 2 or ranked[0] - ranked[1] >= self.margin

    @property
    def ok(self) -> bool | None:
        """True/False when an ``expected`` reading was given and the audio decided."""
        if self.expected is None or not self.decided:
            return None
        return self.reading == reading_name(self.expected)


def reading_name(name: str) -> str:
    """The reading a name stands for: ``"verb~liaison"`` is a variant of ``"verb"``.

    >>> reading_name("verb~liaison"), reading_name("noun")
    ('verb', 'noun')
    """
    return name.split(VARIANT_SEPARATOR, 1)[0]


def word_spans(words: Iterable[Mapping], word: str) -> list[tuple[float, float]]:
    """``(start, end)`` of every ASR word matching ``word`` (case and punctuation aside).

    ``words`` are timed words as ``voxy.refprep.transcribe`` gives them
    (``{"start", "end", "text", ...}``).

    >>> word_spans([{"start": 0.1, "end": 0.3, "text": " Est,"}], "est")
    [(0.1, 0.3)]
    """
    words = list(words)
    return [(words[i]["start"], words[i]["end"]) for i in word_indices(words, word)]


def word_indices(words: Sequence[Mapping], word: str) -> list[int]:
    """Indices of the ASR words matching ``word`` (case and punctuation aside).

    >>> word_indices([{"text": " Il"}, {"text": " est,"}, {"text": "Est"}], "est")
    [1, 2]
    >>> word_indices([{"text": " est…"}], "est")
    [0]
    """
    target = _normal_word(word)
    return [i for i, w in enumerate(words) if _normal_word(w["text"]) == target]


def verify_words(
    audio: str | os.PathLike,
    checks: Iterable[tuple[str, Mapping[str, str], str | None]],
    *,
    words: Iterable[Mapping],
    emitter: Emitter | None = None,
    scorer: Scorer | str = DFLT_SCORING,
    margin: float | None = None,
) -> list[Verdict]:
    """Verdicts for each ``(word, readings, expected)`` at each place ASR found it.

    Args:
        audio: The render.
        checks: The words to check, each with its candidate readings
            (``{name: IPA}``) and the reading it should have (or None to just
            report). :func:`checks_from_risks` builds them from ``flag_risks``.
        words: Timed words of the render (``transcribe(audio)``'s segments' words).
        emitter: ``(audio, start, end) -> (log_probs, vocab)`` for a ``scorer``
            given by name; default :func:`phoneme_emitter`.
        scorer: How each reading is scored: a :class:`Scorer`, or a name in
            :data:`SCORERS` (default ``"phrase"``: the whole phrase around the word,
            its neighbours written out by the real espeak worker, so offline tests
            inject ``phrase_scorer(emitter=..., phonemize=fake)`` or use ``"crop"``).
        margin: See :attr:`Verdict.decided`; default ``scorer.margin``.
    """
    words = list(words)
    scorer = _resolve_scorer(scorer, emitter)
    margin = scorer.margin if margin is None else margin
    verdicts = []
    for word, readings, expected in checks:
        for i in word_indices(words, word):
            start, end = words[i]["start"], words[i]["end"]
            scores = scorer(str(audio), words, i, readings)
            verdicts.append(Verdict(word, start, end, scores, expected, margin))
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
    reading each word should have, and otherwise the risks' own ``expected`` (what a
    detector such as ``voxy.speechtext.pos_detector`` found) is used. Words with no
    entry are skipped, and so are words elided onto a clitic (a risk with a ``clitic``,
    "l'Est"; see thorwhalen/voxy#48), whose differing reading then leaves the word's open: ``verify_words``
    matches spellings, not places. Each word is checked once: ``verify_words`` finds every place
    it is spoken, so a word whose risks name *different* readings gets none here
    (group the risks by line with :func:`checks_by_line` to check each one).
    """
    table = {k.casefold(): v for k, v in readings.items()}
    wanted = {k.casefold(): v for k, v in (expected or {}).items()}
    words, found = {}, {}  # first spelling seen; the risks' expected readings
    for risk in risks:
        key = risk.word.casefold()
        if getattr(risk, "clitic", ""):
            # "l'Est": verify_words matches spellings, not places, so it is not checked,
            # but its reading counts: one that differs from the plain word's ("l'Est"
            # noun, "est" verb) leaves the word open, or an ASR hearing "l'" | "Est"
            # would judge that noun by the verb; "c'est" (a verb too) changes nothing
            found.setdefault(key, set()).add(getattr(risk, "expected", None))
            continue
        if key in table:
            words.setdefault(key, risk.word)
            if (reading := getattr(risk, "expected", None)) is not None:
                found.setdefault(key, set()).add(reading)

    def reading_of(key):
        if key in wanted:
            return wanted[key]
        named = found.get(key, ())
        return next(iter(named)) if len(named) == 1 else None

    checks = [(word, table[key], reading_of(key)) for key, word in words.items()]
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


def espeak_phonemizer(
    *,
    model: str = DFLT_PHONEME_MODEL,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> Phonemizer:
    """The default :data:`Phonemizer`: espeak-ng (bundled by ``espeakng-loader``), run in
    the recogniser's env, so its IPA is the recogniser's own alphabet."""

    def phonemize(texts: Sequence[str], language: str) -> list[str]:
        worker = _phoneme_worker(model, threads, str(rootdir) if rootdir else None)
        reply = worker.call(
            "phonemize",
            texts=list(texts),
            language=ESPEAK_LANGUAGES.get(language, language),
        )
        return reply["ipa"]

    return phonemize


def known_symbols(ipa: str, vocab: Sequence[str]) -> list[str]:
    """The symbols of ``ipa`` the recogniser knows, longest first; stress marks, spaces
    and anything else it never emits are dropped.

    >>> known_symbols("ˈtʁist ɛ", ["<pad>", "t", "ʁ", "i", "s", "ɛ"])
    ['t', 'ʁ', 'i', 's', 't', 'ɛ']
    """
    index = set(vocab)
    longest = max(map(len, vocab))
    out, i = [], 0
    while i < len(ipa):
        for size in range(min(longest, len(ipa) - i), 0, -1):
            if ipa[i : i + size] in index and not ipa[i : i + size].isspace():
                out.append(ipa[i : i + size])
                i += size
                break
        else:
            i += 1
    return out


def phrase_readings(
    log_probs: np.ndarray,
    vocab: Sequence[str],
    readings: Mapping[str, str],
    *,
    before: str = "",
    after: str = "",
    blank: int | None = None,
) -> dict[str, float]:
    """Each reading's CTC log-likelihood as the whole phrase ``before + reading + after``.

    The words around the flagged one are written out (IPA) and are the same in every
    candidate, so the comparison isolates the word: its neighbours' sounds cannot be
    credited to a longer reading, as they are when the word is cropped or spotted alone.

    >>> vocab = ["<pad>", "ɛ", "s", "t", "l", "ə"]
    >>> lp = log_probs_from_path([2, 3, 1, 4, 5], vocab_size=6)  # "(trist)e" s t ɛ l ə
    >>> s = phrase_readings(lp, vocab, {"verb": "ɛ", "noun": "ɛst"}, before="st", after="lə")
    >>> s["verb"] > s["noun"]
    True
    """
    if blank is None:
        blank = list(vocab).index(BLANK_SYMBOL) if BLANK_SYMBOL in vocab else 0
    head, tail = known_symbols(before, vocab), known_symbols(after, vocab)
    return {
        name: ctc_log_likelihood(
            log_probs,
            tokenize_phonemes(head + known_symbols(reading, vocab) + tail, vocab),
            blank=blank,
        )
        for name, reading in readings.items()
    }


class Scorer(Protocol):
    """Scores the readings of timed ASR word ``words[i]`` in ``audio`` (higher is better).

    ``margin`` is the margin its scores are calibrated for, on their own scale.
    """

    margin: float

    def __call__(
        self, audio: str, words: Sequence[Mapping], i: int, readings: Mapping[str, str]
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class _Scorer:
    """A :class:`Scorer` made of a scoring function and its margin."""

    score: Callable[[str, Sequence[Mapping], int, Mapping[str, str]], dict]
    margin: float

    def __call__(self, audio, words, i, readings) -> dict[str, float]:
        return self.score(audio, words, i, readings)


def _window(emitter: Emitter, audio, words, i, *, context: int, padding: float):
    """``(log_probs, vocab, lo, hi)`` of ``words[lo..hi]``, the ``context`` words on
    each side of ``words[i]``, cut from ``audio`` with ``padding`` on each side."""
    lo, hi = max(0, i - context), min(len(words) - 1, i + context)
    log_probs, vocab = emitter(
        str(audio), max(0.0, words[lo]["start"] - padding), words[hi]["end"] + padding
    )
    return log_probs, vocab, lo, hi


def crop_scorer(
    *, emitter: Emitter, padding: float = DFLT_CROP_PADDING, margin: float = DFLT_MARGIN
) -> Scorer:
    """The word alone, cut at its ASR times (``padding`` each side), scored by
    :func:`score_readings`; ASR word times are often too loose for this."""

    def score(audio, words, i, readings):
        log_probs, vocab, _, _ = _window(
            emitter, audio, words, i, context=0, padding=padding
        )
        return score_readings(log_probs, vocab, readings)

    return _Scorer(score, margin)


def spot_scorer(
    *,
    emitter: Emitter,
    context: int = DFLT_CONTEXT_WORDS,
    padding: float = DFLT_CROP_PADDING,
    filler_penalty: float = DFLT_FILLER_PENALTY,
    margin: float = DFLT_SPOT_MARGIN,
) -> Scorer:
    """The reading spotted in the window of ``context`` words on each side, the rest
    left to filler (:func:`spot_readings`); a neighbour's /s/ or /t/ can still be
    credited to the longer reading."""

    def score(audio, words, i, readings):
        log_probs, vocab, _, _ = _window(
            emitter, audio, words, i, context=context, padding=padding
        )
        return spot_readings(log_probs, vocab, readings, filler_penalty=filler_penalty)

    return _Scorer(score, margin)


def spoken_context(
    words: Sequence[Mapping], *, elisions: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """``(text, tail)``: what ``words`` say, for a phonemizer, and the IPA of an elided
    clitic they end with.

    A clitic of ``elisions`` (``{clitic: IPA}``, e.g. ``ELISIONS["fr"]``) that ASR split
    from its word is joined back to it, whichever side the apostrophe went
    (``L'`` | ``homme``, or Whisper's ``l`` | ``'aube``). One left at the end (its word
    is the flagged one, or outside the window) is written from ``elisions`` instead of
    being left to the phonemizer, which reads a standalone ``l'`` as /ɛl/. Anything
    not in ``elisions`` (an English quote, ``James'``) is left as it is.

    >>> fr = {"l'": "l", "d'": "d"}
    >>> spoken_context([{"text": " L'"}, {"text": " homme,"}, {"text": " d’"}], elisions=fr)
    ("L'homme", 'd')
    >>> spoken_context([{"text": " avant"}, {"text": " l"}, {"text": "'aube."}], elisions=fr)
    ("avant l'aube", '')
    >>> spoken_context([{"text": " read"}, {"text": " 'War"}], elisions=fr)
    ("read 'War", '')
    """
    elisions = elisions or {}
    text = ""
    for word in words:
        token = word["text"].strip().strip(EDGE_PUNCTUATION).replace("’", "'")
        if not token:
            continue
        last = text.rpartition(" ")[2].casefold()
        joined = last in elisions or (token[0] == "'" and last + "'" in elisions)
        text += token if joined or not text else " " + token
    head, _, last = text.rpartition(" ")
    tail = elisions.get(last.casefold())
    if tail is None:
        return text, ""
    return head, tail


def phrase_scorer(
    *,
    emitter: Emitter,
    phonemize: Phonemizer | None = None,
    language: str = DFLT_LANGUAGE,
    elisions: Mapping[str, str] | None = None,
    context: int = DFLT_CONTEXT_WORDS,
    padding: float = DFLT_CROP_PADDING,
    margin: float = DFLT_CONTEXT_MARGIN,
) -> Scorer:
    """Each reading as the whole phrase around the word (:func:`phrase_readings`), the
    ``context`` words on each side written out by ``phonemize`` in ``language``.

    The neighbours are the same in every candidate, so only the word decides.
    ``phonemize`` defaults to :func:`espeak_phonemizer` (its worker starts on the first
    call); it must give one IPA per text, or a ``ValueError`` is raised. An elided
    clitic next to the word (``l'`` before ``Est``) is written from ``elisions``
    (default ``ELISIONS[language]``, see :func:`spoken_context`).
    """
    if phonemize is None:
        phonemize = espeak_phonemizer()
    if elisions is None:
        elisions = ELISIONS.get(language, {})

    def score(audio, words, i, readings):
        log_probs, vocab, lo, hi = _window(
            emitter, audio, words, i, context=context, padding=padding
        )
        (before, before_tail), (after, after_tail) = (
            spoken_context(ws, elisions=elisions)
            for ws in (words[lo:i], words[i + 1 : hi + 1])
        )
        said = [t for t in (before, after) if t]  # phonemizers drop empty texts
        ipa = list(phonemize(said, language)) if said else []
        if len(ipa) != len(said):
            raise ValueError(
                f"the phonemizer gave {len(ipa)} IPA for {len(said)} texts: {said}"
            )
        written = dict(zip(said, ipa))
        return phrase_readings(
            log_probs,
            vocab,
            readings,
            before=written.get(before, "") + before_tail,
            after=written.get(after, "") + after_tail,
        )

    return _Scorer(score, margin)


#: The ways a reading can be scored in the audio, by name (see :func:`reading_scorer`);
#: each scorer carries the margin it is calibrated for (see :data:`DFLT_MARGIN`).
SCORERS: dict[str, Callable[..., Scorer]] = {
    "phrase": phrase_scorer,
    "spot": spot_scorer,
    "crop": crop_scorer,
}


def reading_scorer(scoring: str = DFLT_SCORING, **options) -> Scorer:
    """The :class:`Scorer` named ``scoring`` in :data:`SCORERS`, made with ``options``.

    >>> reading_scorer("crop", emitter=None).margin
    0.5
    """
    if scoring not in SCORERS:
        raise ValueError(f"scoring must be one of {sorted(SCORERS)}, not {scoring!r}")
    return SCORERS[scoring](**options)


def _resolve_scorer(scorer: Scorer | str, emitter: Emitter | None) -> Scorer:
    """``scorer``, or the one it names, made over ``emitter`` (default
    :func:`phoneme_emitter`)."""
    if isinstance(scorer, str):
        return reading_scorer(scorer, emitter=emitter or phoneme_emitter())
    return scorer


# 4. re-rendering the lines that fail -------------------------------------------

#: Renders of one line at most (the first take included).
DFLT_MAX_ATTEMPTS = 3

#: ``(word, {reading name: IPA}, expected reading or None)``: one word to check.
Check = tuple[str, Mapping[str, str], "str | None"]


@dataclass(frozen=True)
class Take:
    """One render of a line and what the audio says about its flagged words."""

    attempt: int
    audio: str
    verdicts: tuple[Verdict, ...] = ()

    @property
    def failing(self) -> list[Verdict]:
        """The verdicts of this take whose word was decidedly misread."""
        return failing(self.verdicts)


@dataclass(frozen=True)
class LineTakes:
    """Every take of one line (1-based ``line``, as :class:`voxy.speechtext.Risk`).

    A blank line has no take.
    """

    line: int
    text: str
    takes: tuple[Take, ...] = field(default_factory=tuple)
    checks: tuple = ()

    @property
    def chosen(self) -> Take | None:
        """The first take with no failing word, else the one with the fewest."""
        return min(self.takes, key=lambda t: len(t.failing), default=None)

    @property
    def unconfirmed(self) -> list[str]:
        """Checked words with an expected reading that the chosen take did not confirm.

        A word is confirmed when the audio decidedly gave the expected reading at least
        once. It stays unconfirmed when ASR never found it, or when the recogniser
        could not decide.
        """
        verdicts = self.chosen.verdicts if self.chosen else ()
        return [
            word
            for word, _, expected in self.checks
            if expected is not None
            and not any(v.word == word and v.ok for v in verdicts)
        ]

    @property
    def verified(self) -> bool:
        """No checked word misread, and every one with an expected reading confirmed."""
        return not (self.chosen and self.chosen.failing) and not self.unconfirmed


def checks_by_line(
    risks: Iterable,
    *,
    readings: Mapping[str, Mapping[str, str]],
    expected: Mapping[str, str] | None = None,
) -> dict[int, list[Check]]:
    """:func:`checks_from_risks`, grouped by the line each risk is on."""
    by_line: dict[int, list] = {}
    for risk in risks:
        by_line.setdefault(risk.line, []).append(risk)
    grouped = {
        line: checks_from_risks(line_risks, readings=readings, expected=expected)
        for line, line_risks in by_line.items()
    }
    return {line: checks for line, checks in grouped.items() if checks}


def timed_words(segments: Iterable[Mapping]) -> list[Mapping]:
    """The timed words of ASR segments (``voxy.refprep.transcribe``'s output).

    Words without times are dropped; a segment with no timed word counts as one word.

    >>> timed_words([{"words": [{"start": 0.0, "end": 0.2, "text": "a"},
    ...                         {"start": None, "end": None, "text": "b"}]},
    ...              {"start": 1.0, "end": 2.0, "text": "c", "words": []}])
    [{'start': 0.0, 'end': 0.2, 'text': 'a'}, {'start': 1.0, 'end': 2.0, 'text': 'c'}]
    """
    words: list[Mapping] = []
    for s in segments:
        timed = [
            w
            for w in s.get("words") or ()
            if w.get("start") is not None and w.get("end") is not None
        ]
        words += timed or [{"start": s["start"], "end": s["end"], "text": s["text"]}]
    return words


def _transcribed_words(audio: str) -> list[Mapping]:
    from voxy.refprep import transcribe

    return timed_words(transcribe(audio))


def render_verified(
    lines: Iterable[str],
    checks: Mapping[int, Sequence[Check]],
    *,
    render: Callable[[str, int, int], "str | os.PathLike"],
    emitter: Emitter | None = None,
    words_of: Callable[[str], Iterable[Mapping]] = _transcribed_words,
    max_attempts: int = DFLT_MAX_ATTEMPTS,
    scorer: Scorer | str = DFLT_SCORING,
    margin: float | None = None,
) -> list[LineTakes]:
    """Render each line, and re-render the ones whose flagged words came out wrong.

    A line with checks is verified (:func:`verify_words`) after each render and
    rendered again while a word is decidedly misread (:func:`failing`), up to
    ``max_attempts`` renders; a line without checks is rendered once, and a blank
    line not at all (line numbers still count it, as ``flag_risks`` does).
    :attr:`LineTakes.chosen` is the take to keep; :attr:`LineTakes.verified` also
    requires each expected reading to be confirmed (a word ASR did not find, or the
    recogniser could not decide, leaves the line unverified rather than retried).

    Args:
        lines: The lines to speak, in order (line numbers start at 1).
        checks: By line number, the words to check (:func:`checks_by_line`).
        render: ``(text, line, attempt) -> audio path``; each call should give a
            new take (a new seed, or a service's own variation).
            :func:`speech_renderer` adapts a :class:`voxy.base.SpeechModel`.
        emitter: The phoneme recogniser for a ``scorer`` given by name; default
            :func:`phoneme_emitter`.
        words_of: ``audio -> timed words``; the default transcribes with
            ``voxy.refprep.transcribe``.
        max_attempts: Renders of one line at most.
        scorer, margin: As in :func:`verify_words` (default: phrase scoring, its
            neighbours written out by the real espeak worker, so offline tests
            inject ``phrase_scorer(emitter=..., phonemize=fake)`` or use ``"crop"``).

    >>> heard = {"est": "ɛ"}  # a fake recogniser: what each word sounds like
    >>> def emitter(audio, start, end):
    ...     vocab = ["<pad>", "ɛ", "s", "t", "l"]
    ...     path = [vocab.index(s) for s in "l" + heard["est"]]
    ...     return log_probs_from_path(path, vocab_size=5), vocab
    >>> def render(text, line, attempt):
    ...     heard["est"] = "ɛst" if attempt else "ɛ"  # the second take is right
    ...     return f"take-{line}-{attempt}.wav"
    >>> words = lambda audio: [{"start": 0.0, "end": 0.1, "text": "L'"},
    ...                        {"start": 0.1, "end": 0.3, "text": "Est"}]
    >>> phonemize = lambda texts, language: ["l" for _ in texts]  # a fake espeak
    >>> scorer = phrase_scorer(emitter=emitter, phonemize=phonemize)
    >>> checks = {1: [("Est", {"verb": "ɛ", "noun": "ɛst"}, "noun")]}
    >>> [result] = render_verified(["L'Est"], checks, render=render, words_of=words,
    ...                            scorer=scorer)
    >>> [t.audio for t in result.takes], result.chosen.attempt, result.verified
    (['take-1-0.wav', 'take-1-1.wav'], 1, True)
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    scorer = _resolve_scorer(scorer, emitter)
    results = []
    for number, text in enumerate(lines, 1):
        line_checks = list(checks.get(number, ()))
        if not text.strip():
            results.append(LineTakes(number, text))
            continue
        takes: list[Take] = []
        for attempt in range(max_attempts if line_checks else 1):
            audio = str(render(text, number, attempt))
            verdicts = ()
            if line_checks:
                verdicts = tuple(
                    verify_words(
                        audio,
                        line_checks,
                        words=words_of(audio),
                        scorer=scorer,
                        margin=margin,
                    )
                )
            takes.append(Take(attempt, audio, verdicts))
            if not failing(verdicts):
                break
        results.append(LineTakes(number, text, tuple(takes), tuple(line_checks)))
    return results


def speech_renderer(
    model,
    voice=None,
    *,
    outdir: str | os.PathLike | None = None,
    take_kwarg: str | None = None,
    **synthesize_kwargs,
) -> Callable[[str, int, int], str]:
    """A ``render`` for :func:`render_verified` over a :class:`voxy.base.SpeechModel`.

    Each take is saved as ``<outdir>/<line>-<attempt>.<format>`` (default outdir: a
    new folder per renderer under ``voxy_cache_dir("takes")``, so runs never
    overwrite each other's takes). With ``take_kwarg`` (e.g. ``"seed"``), the attempt
    number is passed to ``model.synthesize`` under that name, so a seeded model gives
    a new take; without it, the model is simply called again.
    """
    from voxy.local_clone import voxy_cache_dir

    if outdir is None:
        import time
        import uuid

        run = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        outdir = voxy_cache_dir("takes", run)
    root = Path(outdir)

    def render(text: str, line: int, attempt: int) -> str:
        kwargs = dict(synthesize_kwargs)
        if take_kwarg:
            kwargs[take_kwarg] = attempt
        speech = model.synthesize(text, voice, **kwargs)
        return speech.save(str(root / f"{line:04d}-{attempt}.{speech.format}"))

    return render


def load_readings(path: str | os.PathLike) -> dict[str, dict[str, str]]:
    """Heteronym readings ``{word: {reading name: IPA}}`` from a caller's file.

    A ``.json`` file holds that mapping; any other file is tab-separated
    ``word<TAB>reading name<TAB>IPA``, one reading per line (``#`` lines and blank
    lines skipped), e.g. a list derived from Lexique, which voxy does not ship.
    """
    import json

    path = Path(path)
    if path.suffix == ".json":
        return {w: dict(r) for w, r in json.loads(path.read_text("utf-8")).items()}
    table: dict[str, dict[str, str]] = {}
    for number, row in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not row.strip() or row.startswith("#"):
            continue
        fields = row.split("\t")
        if len(fields) != 3:
            raise ValueError(
                f"{path}:{number}: expected word<TAB>reading name<TAB>IPA, got {row!r}"
            )
        word, name, ipa = (f.strip() for f in fields)
        table.setdefault(word, {})[name] = ipa
    return table


# 5. auditing a finished render -------------------------------------------------

#: The heteronym tables voxy ships, one ``<language>.json`` per language.
HETERONYMS_DIR = Path(__file__).parent / "data" / "heteronyms"


def heteronym_readings(language: str = DFLT_LANGUAGE) -> dict[str, dict[str, str]]:
    """voxy's own heteronym table for ``language``: ``{word: {reading name: IPA}}``.

    A short, hand-made list (French: *est*, *fils*, *plus*, *tous*, the *-ent* verbs...);
    merge a fuller one (e.g. derived from Lexique) read with :func:`load_readings`.

    >>> heteronym_readings("fr")["est"]["noun"]
    'ɛst'
    """
    import json

    path = HETERONYMS_DIR / f"{language}.json"
    if not path.exists():
        have = sorted(p.stem for p in HETERONYMS_DIR.glob("*.json"))
        raise ValueError(
            f"voxy has no heteronym table for {language!r} (it has {have}); "
            "pass your own readings (see load_readings)"
        )
    table = json.loads(path.read_text("utf-8"))
    return {w: dict(r) for w, r in table.items() if not w.startswith("_")}


def _normal_word(word: str) -> str:
    return word.strip().strip(EDGE_PUNCTUATION).replace("’", "'").casefold()


def _clitic_key(clitic: str) -> str:
    return clitic.replace("’", "'").casefold()


def _text_words(
    text: str, *, elisions: Collection[str] = ()
) -> list[tuple[int, str, str]]:
    """``(line, word, clitic)`` of every word of ``text``, in order (lines count from 1).

    A word elided onto one of ``elisions`` is two words, as ASR's are once split
    (:func:`split_elisions`): "l'Est" is ``"l'"`` then ``"Est"`` (clitic ``"l'"``).
    """
    out = []
    for number, line in enumerate(text.split("\n"), 1):
        for match in _WORD.finditer(line):
            clitic, word = split_clitic(match.group(0), elisions)
            if clitic:
                out.append((number, clitic, ""))
            out.append((number, word, clitic))
    return out


def split_elisions(
    words: Iterable[Mapping], *, elisions: Collection[str] = ()
) -> list[dict]:
    """Timed words split at each elision of ``elisions``, as :func:`_text_words` splits
    the text, so :func:`align_words` can match them.

    Whisper's ``" l" | "'est"`` becomes ``" l'" | "est"`` (the apostrophe moved back);
    a word heard whole, ``" l'est"``, becomes ``" l'"`` (empty, at its start) and
    ``"est"`` (its whole span). Other apostrophes (an English opening quote) are left
    alone.

    >>> words = [{"text": " à", "start": 0, "end": 1}, {"text": " l", "start": 1, "end": 2},
    ...          {"text": "'est", "start": 2, "end": 3}, {"text": " l'est", "start": 3, "end": 4},
    ...          {"text": " 'War", "start": 4, "end": 5}]
    >>> [(w["text"], w["start"], w["end"]) for w in split_elisions(words, elisions={"l'": "l"})]
    [(' à', 0, 1), (" l'", 1, 2), ('est', 2, 3), (" l'", 3, 3), ('est', 3, 4), (" 'War", 4, 5)]
    """
    out: list[dict] = []
    for word in words:
        word = dict(word)
        text = word["text"].lstrip()
        if (
            out
            and text[:1] in ("'", "’")
            and _clitic_key(out[-1]["text"].strip()) + "'" in elisions
        ):
            out[-1]["text"] = out[-1]["text"].rstrip() + text[0]
            word["text"] = text[1:]
        clitic, rest = split_clitic(word["text"].strip(), elisions)
        if clitic:
            lead = word["text"][: len(word["text"]) - len(word["text"].lstrip())]
            out.append({**word, "text": lead + clitic, "end": word["start"]})
            word["text"] = rest
        out.append(word)
    return out


def align_words(
    text: str, words: Sequence[Mapping], *, elisions: Collection[str] = ()
) -> dict[int, int]:
    """``{index of a word of text: index of the ASR word heard for it}``.

    Words are matched by aligning the two sequences (``difflib``), so a word ASR missed,
    merged or added elsewhere does not shift the others; a text word with no exact
    counterpart is left out. A word elided onto one of ``elisions`` counts as two
    (:func:`_text_words`).

    >>> align_words("Il est là", [{"text": " Il"}, {"text": " et"}, {"text": " est"}, {"text": " là."}])
    {0: 0, 1: 2, 2: 3}
    """
    from difflib import SequenceMatcher

    said = [_normal_word(w) for _, w, _ in _text_words(text, elisions=elisions)]
    heard = [_normal_word(w["text"]) for w in words]
    matcher = SequenceMatcher(None, said, heard, autojunk=False)
    return {
        block.a + k: block.b + k
        for block in matcher.get_matching_blocks()
        for k in range(block.size)
    }


def _risk_places(
    text: str, risks: Sequence, *, elisions: Collection[str] = ()
) -> list[int | None]:
    """The index in ``_text_words(text)`` of each risk (by line, clitic and order on the
    line: the "est" of "c'est" is not the risk "est")."""
    taken: set[int] = set()
    tokens = _text_words(text, elisions=elisions)
    places = []
    for risk in risks:
        clitic = _clitic_key(getattr(risk, "clitic", ""))
        place = next(
            (
                i
                for i, (line, word, elided) in enumerate(tokens)
                if i not in taken
                and line == risk.line
                and word == risk.word
                and _clitic_key(elided) == clitic
            ),
            None,
        )
        if place is not None:
            taken.add(place)
        places.append(place)
    return places


@dataclass(frozen=True)
class Finding:
    """One flagged word of a render: the risk, and what the audio said about it.

    ``verdict`` is None when the word could not be located in the audio (ASR heard it
    a different number of times than the text has it): listen to that one.
    """

    risk: object
    verdict: Verdict | None = None

    @property
    def ok(self) -> bool | None:
        """As :attr:`Verdict.ok`; None when unchecked or undecided."""
        return None if self.verdict is None else self.verdict.ok


def audit(
    text: str,
    audio: str | os.PathLike,
    *,
    readings: Mapping[str, Mapping[str, str]] | None = None,
    language: str = DFLT_LANGUAGE,
    elisions: Mapping[str, str] | None = None,
    clitics: Collection[str] | None = None,
    expected: Mapping[str, str] | Callable[[object], "str | None"] | None = None,
    detectors: Iterable[Callable[[str], Iterable]] = (),
    emitter: Emitter | None = None,
    words_of: Callable[[str], Iterable[Mapping]] = _transcribed_words,
    scoring: str = DFLT_SCORING,
    phonemize: Phonemizer | None = None,
    context: int = DFLT_CONTEXT_WORDS,
    filler_penalty: float = DFLT_FILLER_PENALTY,
    padding: float = DFLT_CROP_PADDING,
    margin: float | None = None,
    scorer: Scorer | None = None,
) -> list[Finding]:
    """Which reading did each heteronym of ``text`` get in the finished ``audio``?

    ASR (WER) checks cannot answer this: Whisper writes "est" whether the model said
    /ɛ/ or /ɛst/. This flags the words of ``text`` that have several readings
    (:func:`voxy.speechtext.flag_risks`), finds each in the audio (ASR word times; the
    n-th time the text has a word is the n-th time ASR heard it) and scores its
    readings with the phoneme recogniser.

    Args:
        text: What the model was given (the speech text, not the display text).
        audio: The render.
        readings: ``{word: {reading name: IPA}}``; default :func:`heteronym_readings`
            for ``language``.
        elisions: ``{clitic: IPA}`` of the language's elisions (default
            ``ELISIONS[language]``): text and ASR words are split there to be aligned.
        clitics: The clitics a heteronym is still flagged after (default
            ``NOUN_CLITICS[language]``): "l'Est" is the risk ``Est``, clitic ``l'``.
        expected: The reading each word should have: ``{word: reading}``, or
            ``risk -> reading`` to decide per occurrence (``risk.line``, ``risk.word``,
            ``risk.clitic``). A mapping names an elided form by its clitic
            (``{"est": "verb", "l'est": "noun"}``); ``"est"`` alone does not cover
            "l'Est". Otherwise what a ``detectors`` entry found for that word on that
            line (e.g. :func:`voxy.speechtext.pos_detector`), else None (just report).
        detectors: ``text -> risks`` proposers of expected readings.
        emitter: The phoneme recogniser; default :func:`phoneme_emitter`.
        words_of: ``audio -> timed words``; default ASR (``voxy.refprep.transcribe``).
        scoring: How a reading is scored in the audio (:data:`SCORERS`).
            ``"phrase"`` (default): as the whole phrase around the word, its
            ``context`` neighbours written out with ``phonemize``
            (:func:`phrase_scorer`). ``"spot"``: spotted in that window
            (:func:`spot_scorer`). ``"crop"``: the word alone, cut at its ASR times
            (:func:`crop_scorer`). Only the options that scoring takes are used.
        phonemize: For ``"phrase"``; default :func:`espeak_phonemizer`.
        context: Words kept on each side of the flagged one.
        margin: See :attr:`Verdict.decided`; default ``scorer.margin``.
        scorer: A ready :class:`Scorer`; wins over ``scoring`` and its options.

    Returns the findings in text order; ``[f for f in findings if f.ok is False]`` are
    the words decidedly misread (re-render those), ``f.ok is None`` ones were not
    decided or not found.
    """
    table = {
        k.casefold(): dict(v)
        for k, v in (readings or heteronym_readings(language)).items()
    }
    if elisions is None:
        elisions = ELISIONS.get(language, {})
    if clitics is None:
        clitics = NOUN_CLITICS.get(language, ())
    risks = flag_risks(text, heteronyms=table, clitics=clitics)

    def place_key(risk):  # where a risk is on its line: "l'Est" is not "est"
        return risk.line, risk.word.casefold(), _clitic_key(getattr(risk, "clitic", ""))

    proposed: dict[tuple[int, str, str], list] = {}
    for detector in detectors:
        for risk in detector(text):
            if getattr(risk, "expected", None) is not None:
                proposed.setdefault(place_key(risk), []).append(risk.expected)

    def expected_of(risk, nth_on_line):
        if callable(expected):
            return expected(risk)
        named = _clitic_key(getattr(risk, "clitic", "")) + risk.word.casefold()
        if expected is not None and named in expected:
            return expected[named]
        found = proposed.get(place_key(risk), [])
        return found[nth_on_line] if nth_on_line < len(found) else None

    if isinstance(expected, Mapping):
        expected = {_clitic_key(k): v for k, v in expected.items()}
    if scorer is None:
        options = dict(
            phonemize=phonemize,
            language=language,
            elisions=elisions,
            context=context,
            filler_penalty=filler_penalty,
            padding=padding,
        )
        factory = SCORERS.get(scoring)
        taken = inspect.signature(factory).parameters if factory else {}
        scorer = reading_scorer(  # each scoring gets the options it takes
            scoring,
            emitter=emitter or phoneme_emitter(),
            **{k: v for k, v in options.items() if k in taken},
        )
    margin = scorer.margin if margin is None else margin
    if not risks:
        return []
    words = split_elisions(words_of(str(audio)), elisions=elisions)
    heard_at = align_words(text, words, elisions=elisions)

    places = _risk_places(text, risks, elisions=elisions)
    on_line: dict[tuple, int] = {}
    findings = []
    for risk, place in zip(risks, places):
        key = risk.word.casefold()
        line_key = place_key(risk)
        nth_on_line, on_line[line_key] = (
            on_line.get(line_key, 0),
            on_line.get(line_key, 0) + 1,
        )
        want = expected_of(risk, nth_on_line)
        if heard_at.get(place) is None:
            findings.append(Finding(risk))
            continue
        i = heard_at[place]
        start, end = words[i]["start"], words[i]["end"]
        scores = scorer(str(audio), words, i, table[key])
        findings.append(
            Finding(risk, Verdict(risk.word, start, end, scores, want, margin))
        )
    return findings
