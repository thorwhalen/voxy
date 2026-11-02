"""
Correct words in a recording, in the speaker's voice: find, render, splice, check.

    >>> from voxy.splice import replace_words                              # doctest: +SKIP
    >>> fixed = replace_words("narration.wav", "Anna Smyth", "Anna Smith",
    ...                       voice="ada")                                  # doctest: +SKIP
    >>> fixed.passed, fixed.save("narration-fixed.wav")                     # doctest: +SKIP

Replace the whole coarticulated phrase, not one word: a surname spliced alone was
heard with the wrong first consonant, because the first name before it anticipates
the old surname's lips; replacing first name and surname worked (voxy#38).

Steps, each a seam (keyword argument of ``replace_words``):

- ``words``: the recording's timed ASR words (default: ``transcriber(audio)``,
  ``voxy.refprep.transcribe``); ``old`` is found among them (``find_phrase``).
- ``render(text) -> (samples, sample_rate)``: the new phrase in the voice
  (default ``tts_render(voice)``: ``voxy.text_to_speech``; a speech-to-speech
  conversion of a spoken correction, or a local clone, plug in the same way).
- The cut lands at the quietest moment near each end of the phrase
  (``cut_points``): in the gap when there is one, and a little past ASR's word
  times when there is none, since those are often early (a final /z/ left behind
  was heard as "Smith's"). The render is trimmed, resampled, loudness-matched to
  what it replaces (``level``: by default the RMS of the loud frames on both
  sides, ``speech_level``, so pauses do not count), and crossfaded. With
  ``keep_pauses`` (default), the pauses the cut removed are put back around the
  render from the cut's own edges, so they keep their length and their background.
- ``background(samples, sample_rate, length)``: what is mixed under the render so
  the recording's floor does not drop out while the new words play (default
  ``quiet_background``: the recording's quietest frames, or nothing when even
  those are not quiet, as in wall-to-wall audio; ``None`` for no background).
- ``verify``: checks of the result (default ``heard_check``: the new words are
  transcribed where they were put, and the old ones are gone; ``voice_check``
  compares the new phrase's voice print with the rest of the recording's,
  ``voxy.speakers``).

>>> import numpy as np
>>> sr = 8000
>>> tone = lambda s, amp: amp * np.sin(np.arange(int(s * sr)) / 3).astype(np.float32)
>>> gap = np.zeros(sr // 2)
>>> speech = np.concatenate([tone(1, .3), gap, tone(1, .3), gap, tone(.5, .3)])
>>> words = [{"text": "Hello", "start": 0.0, "end": 1.0},
...          {"text": "Smyth", "start": 1.5, "end": 2.5},
...          {"text": "there.", "start": 3.0, "end": 3.5}]
>>> fixed = replace_words(speech, "smyth", "Smith", words=words, sample_rate=sr,
...                       render=lambda text: (tone(0.8, 0.9), sr), verify=())
>>> fixed.replaced, round(fixed.start, 3), round(fixed.end, 3), fixed.passed
((1.375, 2.625), 1.48, 2.28, None)
"""

import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from voxy.base import Speech
from voxy.pronounce import normal_word
from voxy.wavutil import FRAME_MS, frame_db, read_wav, resample, trim_silence, write_wav

DFLT_CROSSFADE_MS = 15
#: How far a cut may sit from the phrase when the neighbouring gap is wider.
DFLT_MAX_MARGIN_MS = 250
#: How far past an ASR word boundary the cut may look for a quiet moment.
DFLT_SNAP_MS = 80
#: Frames within this many dB of the quietest count as equally quiet.
DFLT_QUIET_TOLERANCE_DB = 1.0
#: Frames within this many dB of the loudest count as speech when matching levels.
DFLT_LOUD_MARGIN_DB = 20.0
#: Audio heard around the splice when checking it (ASR needs some context).
DFLT_CHECK_CONTEXT_S = 1.0
DFLT_PEAK = 0.99
#: A frame this far below the loudest of a stretch counts as pause, not speech.
DFLT_PAUSE_DROP_DB = 25.0
#: Share of the recording's frames, the quietest, that its background is built from.
DFLT_BACKGROUND_SHARE = 0.05
#: Those frames count as background only this far below the speech level, or further.
DFLT_BACKGROUND_DROP_DB = 20.0
#: ... and its quiet frames must be steady (within this spread), as noise is and
#: speech or music is not.
DFLT_BACKGROUND_SPREAD_DB = 6.0
#: A frame within this much of the recording's floor counts as pause when padding:
#: soft consonants (/f/, /s/, /h/, 25-35 dB below the vowel) stay speech.
DFLT_PAUSE_ABOVE_FLOOR_DB = 10.0
#: The recording's floor: this percentile of its frame levels.
FLOOR_PERCENTILE = 5
#: ElevenLabs returns mp3 by default; raw PCM decodes without ffmpeg.
DFLT_RENDER_FORMATS = {"elevenlabs": "pcm_24000"}
#: Slack for float error when turning a time into a frame index (3.0 s / 10 ms is 300).
_FRAME_EPSILON = 1e-6

Render = Callable[[str], tuple[np.ndarray, int]]
Check = Callable[[np.ndarray, int, "Replacement"], dict]
Level = Callable[[np.ndarray, int], float]
Background = Callable[[np.ndarray, int, int], "np.ndarray | None"]


def _tokens(phrase: str) -> list[str]:
    """The words of ``phrase`` as compared with ASR words (empty ones dropped)."""
    return [t for t in map(normal_word, phrase.split()) if t]


def _runs(seq: Sequence, run: Sequence) -> list[int]:
    """Where ``run`` occurs in ``seq``, as start indices.

    >>> _runs("abcab", "ab")
    [0, 3]
    """
    n = len(run)
    return [i for i in range(len(seq) - n + 1) if list(seq[i : i + n]) == list(run)]


def find_phrase(
    words: Sequence[Mapping], phrase: str, *, occurrence: int = 0
) -> tuple[int, int]:
    """Indices ``(first, last)`` of the ``occurrence``-th run of ASR words saying ``phrase``.

    >>> ws = [{"text": " Anna"}, {"text": " Smyth,"}, {"text": " Anna"}, {"text": " Smyth."}]
    >>> find_phrase(ws, "anna smyth"), find_phrase(ws, "Anna … Smyth", occurrence=1)
    ((0, 1), (2, 3))
    """
    tokens = _tokens(phrase)  # as heard, not as scripted
    if not tokens:
        raise ValueError(f"The phrase to find has no words: {phrase!r}")
    starts = _runs([normal_word(w["text"]) for w in words], tokens)
    if occurrence >= len(starts):
        found = f"found {len(starts)}" if starts else "not found"
        heard = " ".join(w["text"].strip() for w in words)
        raise ValueError(
            f"{phrase!r} (occurrence {occurrence}) is not in the words: {found}. "
            f"Pass the phrase as ASR heard it: {heard!r}"
        )
    return starts[occurrence], starts[occurrence] + len(tokens) - 1


def quietest(
    samples: np.ndarray,
    sample_rate: int,
    lo: float,
    hi: float,
    *,
    tolerance_db: float = DFLT_QUIET_TOLERANCE_DB,
    fallback: float | None = None,
) -> float:
    """The time (s) of the quietest frame in ``[lo, hi]``; of equally quiet frames
    (within ``tolerance_db``), the one nearest the middle of the window. When no
    frame lies there (an empty window, or one past the last full frame):
    ``fallback``, default ``lo``.

    >>> x = np.concatenate([np.ones(800), np.zeros(800), np.ones(800)])
    >>> quietest(x, 8000, 0.0, 0.3)
    0.145
    >>> quietest(x, 8000, 0.3, 0.3), quietest(x, 8000, 0.3, 0.35, fallback=0.31)
    (0.3, 0.31)
    """
    levels = frame_db(samples, sample_rate)
    step = FRAME_MS / 1000
    first = max(0, int(np.floor(lo / step + _FRAME_EPSILON)))
    last = min(
        len(levels) - 1, max(first, int(np.ceil(hi / step - _FRAME_EPSILON)) - 1)
    )
    if hi <= lo or last < first:
        return lo if fallback is None else fallback
    window = levels[first : last + 1]
    quiet = np.flatnonzero(window <= window.min() + tolerance_db)
    middle = (lo + hi) / 2 / step - 0.5 - first
    best = quiet[np.argmin(np.abs(quiet - middle))]
    return round(float((first + best + 0.5) * step), 6)


def cut_points(
    samples: np.ndarray,
    sample_rate: int,
    words: Sequence[Mapping],
    first: int,
    last: int,
    *,
    max_margin: float = DFLT_MAX_MARGIN_MS / 1000,
    snap: float = DFLT_SNAP_MS / 1000,
) -> tuple[float, float]:
    """Where to cut (s): the quietest moment before ``words[first]`` and after
    ``words[last]``, searched in the gap to the neighbouring word (up to
    ``max_margin``) and at least ``snap`` past the ASR boundary. A phrase at an
    edge of the audio is cut at that edge.

    >>> sr, tone = 8000, np.ones(8000)
    >>> x = np.concatenate([tone, np.zeros(1600), tone, np.zeros(8000), tone])
    >>> ws = [{"start": 0, "end": 1}, {"start": 1.2, "end": 2.2}, {"start": 3.2, "end": 4.2}]
    >>> cut_points(x, sr, ws, 1, 1), cut_points(x, sr, ws, 0, 0)[0], cut_points(x, sr, ws, 2, 2)[1]
    ((1.095, 2.325), 0.0, 4.2)
    """
    total = len(samples) / sample_rate
    start = min(float(words[first]["start"]), total)
    end = min(float(words[last]["end"]), total)  # ASR may end the last word late
    before = float(words[first - 1]["end"]) if first > 0 else 0.0
    after = float(words[last + 1]["start"]) if last + 1 < len(words) else total
    lo = max(0.0, min(start - snap, max(before, start - max_margin)))
    hi = min(total, max(end + snap, min(after, end + max_margin)))
    return (
        quietest(samples, sample_rate, lo, start, fallback=start),
        quietest(samples, sample_rate, end, hi, fallback=end),
    )


def speech_level(
    samples: np.ndarray, sample_rate: int, *, margin_db: float = DFLT_LOUD_MARGIN_DB
) -> float:
    """RMS of the loud frames (within ``margin_db`` of the loudest), so that pauses
    do not lower it; plain RMS when there is no full frame.

    >>> x = np.concatenate([0.5 * np.ones(800), np.zeros(8000)])
    >>> round(speech_level(x, 8000), 3), round(float(np.sqrt(np.mean(x**2))), 3)
    (0.5, 0.151)
    """
    size = max(1, int(sample_rate * FRAME_MS / 1000))
    n = len(samples) // size
    if n == 0:
        return float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
    levels = frame_db(samples, sample_rate)
    frames = np.asarray(samples[: n * size], dtype=np.float64).reshape(n, size)
    loud = frames[levels >= levels.max() - margin_db]
    return float(np.sqrt(np.mean(loud**2)))


def recording_floor(samples: np.ndarray, sample_rate: int) -> float:
    """The level (dB) of a recording's background: a low percentile of its frames.

    >>> rng = np.random.default_rng(0)
    >>> x = np.concatenate([0.5 * np.ones(8000), 0.01 * rng.standard_normal(8000)])
    >>> round(recording_floor(x, 8000))
    -41
    """
    levels = frame_db(samples, sample_rate)
    return float(np.percentile(levels, FLOOR_PERCENTILE)) if len(levels) else -np.inf


def edge_pauses(
    samples: np.ndarray,
    sample_rate: int,
    *,
    quiet_db: float | None = None,
    drop_db: float = DFLT_PAUSE_DROP_DB,
) -> tuple[float, float]:
    """Seconds of pause at the start and at the end of ``samples``: frames at or below
    ``quiet_db`` when given (e.g. the recording's floor plus a margin), else more than
    ``drop_db`` below its loudest frame.

    >>> x = np.concatenate([np.zeros(800), 0.5 * np.ones(1600), np.zeros(2400)])
    >>> edge_pauses(x, 8000)
    (0.1, 0.3)
    """
    levels = frame_db(samples, sample_rate)
    if not len(levels):
        return 0.0, 0.0
    threshold = levels.max() - drop_db if quiet_db is None else quiet_db
    loud = np.flatnonzero(levels > threshold)
    if not len(loud):  # all pause
        return round(len(levels) * FRAME_MS / 1000, 6), 0.0
    step = FRAME_MS / 1000
    return round(float(loud[0] * step), 6), round(
        float((len(levels) - 1 - loud[-1]) * step), 6
    )


def _overlap_add(frames: np.ndarray) -> np.ndarray:
    """``frames`` (one per row) joined with sine-windowed 50% overlaps, the middle
    part only: uncorrelated frames (noise) keep a steady power, with no clicks.

    >>> rng = np.random.default_rng(0)
    >>> y = _overlap_add(rng.standard_normal((400, 80)))
    >>> len(y), round(float(np.sqrt(np.mean(y**2))), 1)
    (15960, 1.0)
    """
    size = frames.shape[1] - frames.shape[1] % 2
    hop = size // 2
    windowed = frames[:, :size] * np.sin(np.pi * (np.arange(size) + 0.5) / size)
    out = np.zeros(hop * (len(windowed) + 1))
    evens, odds = windowed[::2].ravel(), windowed[1::2].ravel()
    out[: len(evens)] += evens
    out[hop : hop + len(odds)] += odds
    return out[hop:-hop]


def quiet_background(
    samples: np.ndarray,
    sample_rate: int,
    length: int,
    *,
    share: float = DFLT_BACKGROUND_SHARE,
    drop_db: float = DFLT_BACKGROUND_DROP_DB,
    spread_db: float = DFLT_BACKGROUND_SPREAD_DB,
) -> np.ndarray | None:
    """``length`` samples of the recording's background, or ``None`` when it has none
    quiet enough to trust.

    The background is made of the quietest ``share`` of its 10 ms frames, in time
    order, overlap-added (``_overlap_add``) and cycled to ``length``. It is used only
    when every one of those frames is at least ``drop_db`` below ``speech_level`` and
    they are steady (within ``spread_db`` of each other): otherwise they would carry
    speech (a recording with no pause) or music (wall-to-wall audio), and nothing is
    better than that under the new words.
    O(len(samples)).

    >>> rng = np.random.default_rng(0)
    >>> x = np.concatenate([np.full(4000, 0.5), 0.005 * rng.standard_normal(2400)])
    >>> bg = quiet_background(x, 8000, 5000)
    >>> len(bg), round(float(np.sqrt(np.mean(bg**2))), 4)  # the floor's quietest frames
    (5000, 0.0044)
    >>> print(quiet_background(np.full(8000, 0.5), 8000, 5000))  # wall-to-wall
    None
    """
    size = max(1, int(sample_rate * FRAME_MS / 1000))
    levels = frame_db(samples, sample_rate)
    speech = speech_level(samples, sample_rate)
    if not len(levels) or speech == 0:
        return None
    count = max(1, int(share * len(levels)))
    quiet = np.sort(np.argpartition(levels, count - 1)[:count])
    chosen = levels[quiet]
    if chosen.max() > 20 * np.log10(speech) - drop_db:
        return None
    if chosen.max() - chosen.min() > spread_db:
        return None
    frames = np.asarray(samples[: len(levels) * size], np.float64).reshape(-1, size)
    needed = -(-length // (size // 2)) + 2  # ``_overlap_add`` gives a half frame each
    return _overlap_add(frames[np.resize(quiet, needed)])[:length].astype(np.float32)


def cap_peak(samples: np.ndarray, peak: float = DFLT_PEAK) -> np.ndarray:
    """``samples``, scaled down if needed so that no sample exceeds ``peak``.

    >>> cap_peak(np.array([0.5, -2.0]), 1.0).tolist()
    [0.25, -1.0]
    """
    top = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return samples * (peak / top) if top > peak else samples


def match_loudness(
    samples: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    *,
    level: Level = speech_level,
    peak: float = DFLT_PEAK,
) -> np.ndarray:
    """``samples`` scaled to ``reference``'s ``level``, its peak capped at ``peak``.

    >>> x = match_loudness(0.1 * np.ones(160), np.r_[0.5 * np.ones(160), np.zeros(800)], 8000)
    >>> round(float(x.max()), 3)
    0.5
    """
    own, target = level(samples, sample_rate), level(reference, sample_rate)
    if own == 0 or target == 0:
        return samples
    return cap_peak(samples * (target / own), peak)


def crossfade_join(
    head: np.ndarray, middle: np.ndarray, tail: np.ndarray, *, fade: int
) -> tuple[np.ndarray, int]:
    """``(head + middle + tail, where middle starts in it)``, with linear crossfades
    of up to ``fade`` samples at each join (shorter where a piece is too short; none
    at an edge of the audio).

    >>> joined, at = crossfade_join(np.ones(10), np.ones(6), np.ones(10), fade=2)
    >>> len(joined), at
    (22, 8)
    >>> joined, at = crossfade_join(np.ones(0), np.ones(6), np.ones(10), fade=2)
    >>> len(joined), at
    (14, 0)
    """
    into = min(fade, len(head), len(middle) // 2)
    out_of = min(fade, len(tail), len(middle) // 2)
    pieces = [head[: len(head) - into]]
    if into:
        ramp = np.linspace(0.0, 1.0, into, dtype=np.float32)
        pieces.append(head[-into:] * ramp[::-1] + middle[:into] * ramp)
    pieces.append(middle[into : len(middle) - out_of])
    if out_of:
        ramp = np.linspace(0.0, 1.0, out_of, dtype=np.float32)
        pieces.append(middle[-out_of:] * ramp[::-1] + tail[:out_of] * ramp)
    pieces.append(tail[out_of:])
    return np.concatenate(pieces).astype(np.float32), len(head) - into


def tts_render(
    voice=None,
    *,
    backend: str | None = None,
    voices: Mapping | None = None,
    use_library: bool = True,
    model=None,
    **tts_kwargs,
) -> Render:
    """A ``render`` speaking the text with ``voxy.text_to_speech`` in ``voice``.

    The voice is resolved once, here (``voxy.facade.resolve_voice``, with
    ``voices``, ``use_library`` and ``model``'s backend), and every render
    synthesizes with what it resolved to. Backends whose default output is
    compressed get a WAV/PCM format (``DFLT_RENDER_FORMATS``) unless
    ``output_format=`` is given; other ``tts_kwargs`` go to ``text_to_speech``.
    """
    from voxy import facade

    backend = backend or (model.name if model is not None else None)
    backend, resolved = facade.resolve_voice(
        voice, backend=backend, voices=voices, use_library=use_library
    )
    kwargs = dict(tts_kwargs)
    if backend in DFLT_RENDER_FORMATS:
        kwargs.setdefault("output_format", DFLT_RENDER_FORMATS[backend])

    def render(text: str) -> tuple[np.ndarray, int]:
        speech = facade.text_to_speech(
            text, resolved, backend=backend, model=model, use_library=False, **kwargs
        )
        return _decode(speech, backend)

    return render


def _decode(speech: Speech, backend: str) -> tuple[np.ndarray, int]:
    if speech.format != "wav":
        raise ValueError(
            f"The {backend} render came back as {speech.format!r}; pass tts_render "
            f"an output_format= for which {backend} returns WAV or PCM audio"
        )
    return read_wav(speech.audio)


@contextmanager
def _temp_wav(samples: np.ndarray, sample_rate: int):
    """The path of a temporary WAV of ``samples``, removed on exit."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "audio.wav")
        with open(path, "wb") as f:
            f.write(write_wav(samples, sample_rate))
        yield path


def _asr_words(audio_path: str, transcriber: Callable | None) -> list[dict]:
    if transcriber is None:
        from voxy.refprep import transcribe as transcriber
    return [w for segment in transcriber(audio_path) for w in segment["words"]]


def heard_check(
    *,
    transcriber: Callable | None = None,
    context: float = DFLT_CHECK_CONTEXT_S,
    heard_as: str | None = None,
) -> Check:
    """A check: are the new words transcribed where they were spliced in, and the old gone?

    ``transcriber(wav_path) -> segments`` (default ``voxy.refprep.transcribe``) runs
    on the splice plus ``context`` seconds each side. ``heard_as`` is the new phrase
    as ASR writes it, when that differs from the text rendered ("Dr. Patterson"
    for "Doctor Patterson"); default: the replacement's ``new``. The old phrase
    counts as heard only as a whole run of its words ("anna jones", not "anna"),
    and not at all when the new phrase contains it.
    """

    def check(samples: np.ndarray, sample_rate: int, replacement) -> dict:
        lo = max(0, int((replacement.start - context) * sample_rate))
        hi = int((replacement.end + context) * sample_rate)
        with _temp_wav(samples[lo:hi], sample_rate) as path:
            heard = [normal_word(w["text"]) for w in _asr_words(path, transcriber)]
        new, old = _tokens(heard_as or replacement.new), _tokens(replacement.old)
        new_heard = bool(_runs(heard, new))
        old_heard = bool(_runs(heard, old)) and not _runs(new, old)
        return {
            "check": "heard",
            "passed": new_heard and not old_heard,
            "heard": " ".join(heard),
            "new_heard": new_heard,
            "old_heard": old_heard,
        }

    return check


def voice_check(
    *,
    embed: Callable | None = None,
    min_similarity: float | None = None,
    window: float | None = None,
) -> Check:
    """A check: does the new phrase sound like the rest of the recording's speaker?

    Cosine similarity of the voice print of the new phrase and of the rest
    (``voxy.speakers.voice_print``, ECAPA in its own environment unless ``embed``
    is given). ``min_similarity`` defaults to ``speakers.DFLT_MATCH_SIMILARITY``
    (below it, a window counts as someone else); ``window`` (s) defaults to the
    speakers' window, shortened to the phrase. Fails, with a ``reason``, when the
    new phrase is the whole recording.
    """

    def check(samples: np.ndarray, sample_rate: int, replacement) -> dict:
        from voxy import speakers

        threshold = (
            speakers.DFLT_MATCH_SIMILARITY if min_similarity is None else min_similarity
        )
        start, end = replacement.start, replacement.end
        size = window or min(speakers.DFLT_WINDOW_SECONDS, end - start)
        total = len(samples) / sample_rate
        # a span shorter than the window gets one window of its own length
        rest = [(a, b) for a, b in ((0.0, start), (end, total)) if b > a]
        if not rest:  # no spans would mean the whole file to the embedder
            reason = "no audio outside the new phrase to compare it with"
            return {"check": "voice", "passed": False, "reason": reason}
        with _temp_wav(samples, sample_rate) as path:
            try:
                new, old = (
                    speakers.voice_print(path, spans=spans, embed=embed, window=size)
                    for spans in ([(start, end)], rest)
                )
            except ValueError as e:  # no speech windows (silence around the phrase)
                return {"check": "voice", "passed": False, "reason": str(e)}
        similarity = round(float(new @ old), 3)
        return {
            "check": "voice",
            "passed": similarity >= threshold,
            "similarity": similarity,
        }

    return check


@dataclass
class Replacement:
    """A corrected recording, where the new phrase is, and what the checks found."""

    speech: Speech  # the corrected recording, as WAV
    old: str
    new: str
    start: float  # seconds: the new speech (the trimmed render) in the result
    end: float
    replaced: tuple[float, float]  # seconds: the cut in the original
    checks: list[dict] = field(default_factory=list)

    @property
    def audio(self) -> bytes:
        """The WAV bytes."""
        return self.speech.audio

    @property
    def sample_rate(self) -> int:
        return self.speech.sample_rate

    @property
    def passed(self) -> bool | None:
        """True if every check passed; None when no check ran."""
        return all(c.get("passed") for c in self.checks) if self.checks else None

    def save(self, path) -> str:
        """Write the WAV to ``path``."""
        return self.speech.save(str(path))


def _as_float(samples: np.ndarray) -> np.ndarray:
    """Float samples: integer PCM scaled to ``[-1, 1)``, floats as they are.

    >>> _as_float(np.array([-32768, 0, 16384], dtype=np.int16)).tolist()
    [-1.0, 0.0, 0.5]
    >>> _as_float(np.array([0, 128], dtype=np.uint8)).tolist()
    [-1.0, 0.0]
    """
    if not np.issubdtype(samples.dtype, np.integer):
        return samples.astype(np.float32)
    info = np.iinfo(samples.dtype)
    half = (int(info.max) - int(info.min) + 1) / 2
    return ((samples.astype(np.float64) - (int(info.min) + half)) / half).astype(
        np.float32
    )


def _read(audio, sample_rate: int | None) -> tuple[np.ndarray, int]:
    """``(samples, sample_rate)`` of a WAV (``read_wav`` averages its channels) or
    of raw mono samples, which need ``sample_rate``."""
    if not isinstance(audio, np.ndarray):
        return read_wav(audio)
    if sample_rate is None:
        raise ValueError("Raw samples need sample_rate=")
    if audio.ndim != 1:
        raise ValueError(
            f"Raw samples must be mono (1-D), not of shape {audio.shape}: "
            "mix them down first (e.g. samples.mean(axis=1))"
        )
    return _as_float(audio), sample_rate


def _render_for(voice, render: Render | None) -> Render:
    if render is not None:
        return render
    if voice is None:
        raise ValueError(
            "Pass voice= (the speaker's voice) or render=: without them the new "
            "words would be in the default backend's default voice"
        )
    return tts_render(voice)


def _recording_words(samples, sample_rate, words, transcriber) -> list:
    if words is not None:
        return list(words)
    with _temp_wav(samples, sample_rate) as path:
        return _asr_words(path, transcriber)


def _insert(
    render: Render, text: str, sample_rate: int, *, reference: np.ndarray, level: Level
) -> np.ndarray:
    """``text`` rendered, trimmed, at ``sample_rate`` and at ``reference``'s level."""
    samples, rate = render(text)
    trimmed = trim_silence(np.asarray(samples, np.float32), rate)
    if not len(trimmed):
        raise ValueError(f"The render of {text!r} is silent: nothing to splice in")
    insert = resample(trimmed, rate, sample_rate)
    return match_loudness(insert, reference, sample_rate, level=level)


def _pause_padding(
    cut: np.ndarray, insert: np.ndarray, sample_rate: int, *, floor_db: float
) -> tuple[np.ndarray, np.ndarray]:
    """The audio to put before and after ``insert`` so that its pauses match those at
    the edges of ``cut`` (what it replaces): the cut's own edges, which are the
    recording's true background at that spot.

    The cut's pauses are frames near the recording's floor (``floor_db``), so the old
    word's soft consonants are never carried back in as "pause".

    >>> cut = np.r_[np.full(800, 0.01), np.ones(800), np.full(400, 0.02)]
    >>> before, after = _pause_padding(cut, np.ones(800), 8000, floor_db=-40)
    >>> len(before), float(before.max()), len(after), float(after.max())
    (800, 0.01, 400, 0.02)
    """
    quiet_db = floor_db + DFLT_PAUSE_ABOVE_FLOOR_DB
    cut_lead, cut_tail = edge_pauses(cut, sample_rate, quiet_db=quiet_db)
    ins_lead, ins_tail = edge_pauses(insert, sample_rate)
    lead = max(0, int((cut_lead - ins_lead) * sample_rate))
    tail = max(0, int((cut_tail - ins_tail) * sample_rate))
    return cut[:lead], cut[len(cut) - tail :]


def _checks(verify, transcriber) -> tuple[Check, ...]:
    checks = (heard_check(transcriber=transcriber),) if verify is None else verify
    return (checks,) if callable(checks) else tuple(checks)


def replace_words(
    audio,
    old: str,
    new: str,
    *,
    voice=None,
    render: Render | None = None,
    words: Iterable[Mapping] | None = None,
    transcriber: Callable | None = None,
    sample_rate: int | None = None,
    occurrence: int = 0,
    crossfade_ms: float = DFLT_CROSSFADE_MS,
    max_margin_ms: float = DFLT_MAX_MARGIN_MS,
    snap_ms: float = DFLT_SNAP_MS,
    level: Level = speech_level,
    keep_pauses: bool = True,
    background: Background | None = quiet_background,
    verify: Check | Sequence[Check] | None = None,
) -> Replacement:
    """Replace the spoken phrase ``old`` with ``new``, rendered in ``voice``.

    Args:
        audio: A WAV path or bytes (channels averaged), or raw mono samples (with
            ``sample_rate=``; integer PCM is scaled to ``[-1, 1)``).
        old: The phrase as spoken (several words: see the module docstring).
        new: What it should say.
        voice: The speaker's voice, for the default ``render`` (a library name, a
            backend voice, a profile). ``voice`` or ``render`` is required.
        render: ``text -> (samples, sample_rate)``; default ``tts_render(voice)``.
            A render that is silent raises ``ValueError``.
        words: Timed ASR words of ``audio``; default ``transcriber(audio)``.
        transcriber: ``wav_path -> segments`` (default ``voxy.refprep.transcribe``),
            for ``words`` and the default check.
        occurrence: Which occurrence of ``old`` (0: the first).
        crossfade_ms: Length of each crossfade.
        max_margin_ms: The furthest a cut may sit from the phrase, in a gap.
        snap_ms: How far past an ASR boundary a cut may look for quiet.
        level: ``(samples, sample_rate) -> level`` the render is matched on;
            default ``speech_level`` (e.g. ``partial(speech_level, margin_db=10)``).
        keep_pauses: Put back, around the render, the pauses at the edges of the
            cut (taken from the cut itself, background included), so the pauses
            keep their length; ``False`` for the tighter splice.
        background: ``(samples, sample_rate, length) -> samples or None``: what to
            mix under the render (``None`` from it: nothing); default
            ``quiet_background``, which gives nothing for audio with no quiet
            frames (wall-to-wall speech or music); ``None`` for no background.
        verify: A check or checks (``(samples, sample_rate, replacement) -> dict``
            with a ``passed`` key); default ``heard_check``; ``()`` for none.
    """
    samples, sr = _read(audio, sample_rate)
    render = _render_for(voice, render)
    words = _recording_words(samples, sr, words, transcriber)
    first, last = find_phrase(words, old, occurrence=occurrence)
    cut = cut_points(
        samples,
        sr,
        words,
        first,
        last,
        max_margin=max_margin_ms / 1000,
        snap=snap_ms / 1000,
    )
    a, b = int(cut[0] * sr), int(cut[1] * sr)
    speech = _insert(render, new, sr, reference=samples[a:b], level=level)
    empty = np.zeros(0, np.float32)
    before, after = (  # measured on the bare render, before any background
        _pause_padding(samples[a:b], speech, sr, floor_db=recording_floor(samples, sr))
        if keep_pauses
        else (empty, empty)
    )
    bed = background(samples, sr, len(speech)) if background is not None else None
    if bed is not None:  # the recording's floor under the new words, no drop-out
        speech = cap_peak(speech + bed)
    insert = np.concatenate([before, speech, after]).astype(np.float32)
    fade = int(sr * crossfade_ms / 1000)
    joined, at = crossfade_join(samples[:a], insert, samples[b:], fade=fade)
    result = Replacement(
        Speech(write_wav(joined, sr), format="wav", sample_rate=sr),
        old,
        new,
        start=(at + len(before)) / sr,
        end=(at + len(before) + len(speech)) / sr,
        replaced=cut,
    )
    result.checks = [
        check(joined, sr, result) for check in _checks(verify, transcriber)
    ]
    return result
