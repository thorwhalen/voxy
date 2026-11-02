"""
Correct words in a recording, in the speaker's voice: find, render, splice, check.

    >>> from voxy.splice import replace_words                              # doctest: +SKIP
    >>> fixed = replace_words("narration.wav", "Anna Smyth", "Anna Smith",
    ...                       voice="ada")                                  # doctest: +SKIP
    >>> fixed.heard, fixed.save("narration-fixed.wav")                      # doctest: +SKIP

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
  what it replaces, and crossfaded.
- ``verify``: checks of the result (default ``heard_check``: the new words are
  transcribed where they were put; ``voice_check`` compares the new phrase's voice
  print with the rest of the recording's, ``voxy.speakers``).

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
>>> fixed.replaced, round(fixed.start, 3), round(fixed.end, 3), fixed.checks
((1.375, 2.625), 1.36, 2.16, [])
"""

import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from voxy.base import Speech
from voxy.wavutil import FRAME_MS, frame_db, read_wav, resample, trim_silence, write_wav

DFLT_CROSSFADE_MS = 15
#: How far a cut may sit from the phrase when the neighbouring gap is wider.
DFLT_MAX_MARGIN_MS = 250
#: How far past an ASR word boundary the cut may look for a quiet moment.
DFLT_SNAP_MS = 80
#: Frames within this many dB of the quietest count as equally quiet.
DFLT_QUIET_TOLERANCE_DB = 1.0
#: Audio heard around the splice when checking it (ASR needs some context).
DFLT_CHECK_CONTEXT_S = 1.0
DFLT_PEAK = 0.99
#: ElevenLabs returns mp3 by default; raw PCM decodes without ffmpeg.
DFLT_RENDER_FORMATS = {"elevenlabs": "pcm_24000"}

Render = Callable[[str], tuple[np.ndarray, int]]
Check = Callable[[np.ndarray, int, "Replacement"], dict]


def _normal(token: str) -> str:
    from voxy.pronounce import _normal_word

    return _normal_word(token)


def find_phrase(
    words: Sequence[Mapping], phrase: str, *, occurrence: int = 0
) -> tuple[int, int]:
    """Indices ``(first, last)`` of the ``occurrence``-th run of ASR words saying ``phrase``.

    >>> ws = [{"text": " Anna"}, {"text": " Smyth,"}, {"text": " Anna"}, {"text": " Smyth."}]
    >>> find_phrase(ws, "anna smyth"), find_phrase(ws, "Anna Smyth", occurrence=1)
    ((0, 1), (2, 3))
    """
    tokens = [_normal(t) for t in phrase.split()]  # as heard, not as scripted
    texts = [_normal(w["text"]) for w in words]
    n = len(tokens)
    starts = [i for i in range(len(texts) - n + 1) if texts[i : i + n] == tokens]
    if occurrence >= len(starts):
        found = f"found {len(starts)}" if starts else "not found"
        heard = " ".join(w["text"].strip() for w in words)
        raise ValueError(
            f"{phrase!r} (occurrence {occurrence}) is not in the words: {found}. "
            f"Pass the phrase as ASR heard it: {heard!r}"
        )
    return starts[occurrence], starts[occurrence] + n - 1


def quietest(
    samples: np.ndarray,
    sample_rate: int,
    lo: float,
    hi: float,
    *,
    tolerance_db: float = DFLT_QUIET_TOLERANCE_DB,
) -> float:
    """The time (s) of the quietest frame in ``[lo, hi]``; of equally quiet frames
    (within ``tolerance_db``), the one nearest the middle of the window.

    >>> x = np.concatenate([np.ones(800), np.zeros(800), np.ones(800)])
    >>> quietest(x, 8000, 0.0, 0.3)
    0.145
    """
    levels = frame_db(samples, sample_rate)
    step = FRAME_MS / 1000
    first = max(0, int(lo / step))
    last = min(len(levels) - 1, max(first, int(np.ceil(hi / step)) - 1))
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
    ``max_margin``) and at least ``snap`` past the ASR boundary.

    >>> sr, tone = 8000, np.ones(8000)
    >>> x = np.concatenate([tone, np.zeros(1600), tone, np.zeros(8000), tone])
    >>> ws = [{"start": 0, "end": 1}, {"start": 1.2, "end": 2.2}, {"start": 3.2, "end": 4.2}]
    >>> cut_points(x, sr, ws, 1, 1)
    (1.095, 2.325)
    """
    total = len(samples) / sample_rate
    start, end = float(words[first]["start"]), float(words[last]["end"])
    before = float(words[first - 1]["end"]) if first > 0 else 0.0
    after = float(words[last + 1]["start"]) if last + 1 < len(words) else total
    lo = max(0.0, min(start - snap, max(before, start - max_margin)))
    hi = min(total, max(end + snap, min(after, end + max_margin)))
    return (
        quietest(samples, sample_rate, lo, start),
        quietest(samples, sample_rate, end, hi),
    )


def match_loudness(
    samples: np.ndarray, reference: np.ndarray, *, peak: float = DFLT_PEAK
) -> np.ndarray:
    """``samples`` scaled to ``reference``'s RMS level, its peak capped at ``peak``.

    >>> x = match_loudness(np.array([0.1, -0.1]), np.array([0.5, -0.5]))
    >>> [round(float(v), 3) for v in x]
    [0.5, -0.5]
    """
    rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
    target = float(np.sqrt(np.mean(reference**2))) if len(reference) else 0.0
    if rms == 0 or target == 0:
        return samples
    scaled = samples * (target / rms)
    top = float(np.max(np.abs(scaled)))
    return scaled * (peak / top) if top > peak else scaled


def fade_lengths(fade: int, head: int, middle: int, tail: int) -> tuple[int, int]:
    """The crossfades (in samples) that fit: into ``middle``, and out of it.

    >>> fade_lengths(120, head=1000, middle=500, tail=0)
    (120, 0)
    """
    return min(fade, head, middle // 2), min(fade, tail, middle // 2)


def crossfade_join(
    head: np.ndarray, middle: np.ndarray, tail: np.ndarray, *, fade: int
) -> np.ndarray:
    """``head + middle + tail`` with linear crossfades of up to ``fade`` samples at
    each join (shorter where a piece is too short; none at an edge of the audio).

    >>> len(crossfade_join(np.ones(10), np.ones(6), np.ones(10), fade=2))
    22
    >>> len(crossfade_join(np.ones(10), np.ones(6), np.ones(0), fade=2))
    14
    """
    into, out_of = fade_lengths(fade, len(head), len(middle), len(tail))
    pieces = [head[: len(head) - into]]
    if into:
        ramp = np.linspace(0.0, 1.0, into, dtype=np.float32)
        pieces.append(head[-into:] * ramp[::-1] + middle[:into] * ramp)
    pieces.append(middle[into : len(middle) - out_of])
    if out_of:
        ramp = np.linspace(0.0, 1.0, out_of, dtype=np.float32)
        pieces.append(middle[-out_of:] * ramp[::-1] + tail[:out_of] * ramp)
    pieces.append(tail[out_of:])
    return np.concatenate(pieces).astype(np.float32)


def tts_render(
    voice=None,
    *,
    backend: str | None = None,
    voices: Mapping | None = None,
    **tts_kwargs,
) -> Render:
    """A ``render`` speaking the text with ``voxy.text_to_speech`` in ``voice``.

    ``voices`` is the voice library (default ``voxy.voices_store()``). Backends
    whose default output is compressed get a WAV/PCM format
    (``DFLT_RENDER_FORMATS``) unless ``output_format=`` is given.
    """

    def render(text: str) -> tuple[np.ndarray, int]:
        from voxy import facade

        name, _ = facade.resolve_voice(voice, backend=backend, voices=voices)
        kwargs = dict(tts_kwargs)
        if name in DFLT_RENDER_FORMATS:
            kwargs.setdefault("output_format", DFLT_RENDER_FORMATS[name])
        if voices is not None:
            kwargs["voices"] = voices
        speech = facade.text_to_speech(text, voice, backend=backend, **kwargs)
        return _decode(speech)

    return render


def _decode(speech: Speech) -> tuple[np.ndarray, int]:
    if speech.format != "wav":
        raise ValueError(
            f"The render came back as {speech.format!r}; pass a WAV or PCM "
            "output_format to tts_render (e.g. output_format='pcm_24000')"
        )
    return read_wav(speech.audio)


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
    """A check: are the new words transcribed where they were spliced in?

    ``transcriber(wav_path) -> segments`` (default ``voxy.refprep.transcribe``) runs
    on the splice plus ``context`` seconds each side. ``heard_as`` is the new phrase
    as ASR writes it, when that differs from the text rendered ("Dr. Patterson"
    for "Doctor Patterson"); default: the replacement's ``new``.
    """

    def check(samples: np.ndarray, sample_rate: int, replacement) -> dict:
        lo = max(0, int((replacement.start - context) * sample_rate))
        hi = int((replacement.end + context) * sample_rate)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "splice.wav")
            with open(path, "wb") as f:
                f.write(write_wav(samples[lo:hi], sample_rate))
            heard = [_normal(w["text"]) for w in _asr_words(path, transcriber)]
        wanted = [_normal(t) for t in (heard_as or replacement.new).split()]
        found = any(
            heard[i : i + len(wanted)] == wanted
            for i in range(len(heard) - len(wanted) + 1)
        )
        return {"check": "heard", "passed": found, "heard": " ".join(heard)}

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
    speakers' window, shortened to the phrase.
    """

    def check(samples: np.ndarray, sample_rate: int, replacement) -> dict:
        from voxy import speakers

        threshold = (
            speakers.DFLT_MATCH_SIMILARITY if min_similarity is None else min_similarity
        )
        start, end = replacement.start, replacement.end
        size = window or min(speakers.DFLT_WINDOW_SECONDS, end - start)
        total = len(samples) / sample_rate
        rest = [(a, b) for a, b in ((0.0, start), (end, total)) if b - a >= size]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "result.wav")
            with open(path, "wb") as f:
                f.write(write_wav(samples, sample_rate))
            new, old = (
                speakers.voice_print(path, spans=spans, embed=embed, window=size)
                for spans in ([(start, end)], rest)
            )
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

    audio: bytes  # WAV
    sample_rate: int
    old: str
    new: str
    start: float  # seconds: the new phrase (crossfades included) in the result
    end: float
    replaced: tuple[float, float]  # seconds: the cut in the original
    checks: list[dict] = field(default_factory=list)

    @property
    def heard(self) -> bool:
        """True if every check passed."""
        return all(c.get("passed") for c in self.checks)

    def save(self, path) -> str:
        """Write the WAV to ``path``."""
        return Speech(self.audio, format="wav").save(str(path))


def _read(audio, sample_rate: int | None) -> tuple[np.ndarray, int]:
    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("Raw samples need sample_rate=")
        return audio.astype(np.float32), sample_rate
    return read_wav(audio)


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
    verify: Check | Sequence[Check] | None = None,
) -> Replacement:
    """Replace the spoken phrase ``old`` with ``new``, rendered in ``voice``.

    Args:
        audio: A WAV path or bytes, or raw samples (with ``sample_rate=``).
        old: The phrase as spoken (several words: see the module docstring).
        new: What it should say.
        voice: The voice for the default ``render`` (a library name, a backend
            voice, a profile).
        render: ``text -> (samples, sample_rate)``; default ``tts_render(voice)``.
        words: Timed ASR words of ``audio``; default ``transcriber(audio)``.
        transcriber: ``wav_path -> segments`` (default ``voxy.refprep.transcribe``),
            for ``words`` and the default check.
        occurrence: Which occurrence of ``old`` (0: the first).
        crossfade_ms: Length of each crossfade.
        max_margin_ms: The furthest a cut may sit from the phrase, in a gap.
        snap_ms: How far past an ASR boundary a cut may look for quiet.
        verify: A check or checks (``(samples, sample_rate, replacement) -> dict``
            with a ``passed`` key); default ``heard_check``; ``()`` for none.
    """
    samples, sr = _read(audio, sample_rate)
    if words is None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "audio.wav")
            with open(path, "wb") as f:
                f.write(write_wav(samples, sr))
            words = _asr_words(path, transcriber)
    words = list(words)
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
    insert, insert_sr = (render or tts_render(voice))(new)
    insert = resample(
        trim_silence(np.asarray(insert, np.float32), insert_sr), insert_sr, sr
    )
    insert = match_loudness(insert, samples[a:b])
    fade = int(sr * crossfade_ms / 1000)
    joined = crossfade_join(samples[:a], insert, samples[b:], fade=fade)
    into, _ = fade_lengths(fade, a, len(insert), len(samples) - b)
    result = Replacement(
        write_wav(joined, sr),
        sr,
        old,
        new,
        start=(a - into) / sr,
        end=(a - into + len(insert)) / sr,
        replaced=cut,
    )
    checks = (heard_check(transcriber=transcriber),) if verify is None else verify
    checks = (checks,) if callable(checks) else tuple(checks)
    result.checks = [check(joined, sr, result) for check in checks]
    return result
