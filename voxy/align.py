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
over a long recording drifts, sung audio most. For a whole recording (an episode),
``align_long`` finds the stretches itself: a rough speech-recognition pass gives anchor
words, and each stretch between anchors is aligned exactly.

``run`` is the seam: ``(audio_path, words, *, start, end) -> [span or None, ...]``.

>>> fake = lambda audio, words, **crop: [{"start": i, "end": i + 0.5, "score": 0.9, "chars": []}
...                                      for i, _ in enumerate(words)]
>>> [s["word"] for s in force_align(b"RIFF....", "Bonjour  le monde", run=fake)]
['Bonjour', 'le', 'monde']
"""

import math
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

#: ``(audio_path, words, *, start, end) -> [span or None, ...]``
Run = Callable[..., Sequence["dict | None"]]


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

    def run(audio: str, words: Sequence[str], *, start=None, end=None):
        worker = _ALIGN_POOL.get(model, threads, rootdir)
        # the worker runs in its own working directory: an absolute path
        reply = worker.call(
            "align",
            audio=str(Path(audio).resolve()),
            words=list(words),
            start=start,
            end=end,
        )
        return reply["words"]

    return run


def force_align(
    audio: str | os.PathLike | bytes,
    text: str | Sequence[str],
    *,
    start: float | None = None,
    end: float | None = None,
    run: Run | None = None,
) -> list[dict]:
    """Each word of ``text``, in order, with its ``start``, ``end`` (s), ``score`` (mean
    character confidence) and ``chars`` (``char``, ``start``, ``end``, ``score``).

    Args:
        audio: A WAV (or any format soundfile reads) path, or WAV bytes.
        text: The words spoken, as a string (split on whitespace) or a list.
        start, end: The stretch of ``audio`` (s) the words are in (default: all of
            it); times stay on the recording's clock.
        run: ``(audio_path, words, *, start, end) -> [span or None, ...]``; default
            ``worker_run()``.

    A word with no letter the aligner knows (a number, a symbol) is placed as one
    "any sound" token. Low scores (under about 0.5) mean the text is not what is said
    there. Audio over 30 s is refused by the default aligner: align in stretches. A
    run that returns None for a word gives ``start``/``end`` None.
    """
    words = text.split() if isinstance(text, str) else list(text)
    run = run or worker_run()
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(audio, bytes):
            path = Path(tmp) / "audio.wav"
            path.write_bytes(audio)
            audio = path
        spans = run(str(audio), words, start=start, end=end)
    if len(spans) != len(words):
        raise ValueError(
            f"the aligner returned {len(spans)} spans for {len(words)} words"
        )
    empty = {"start": None, "end": None, "score": None, "chars": []}
    return [{"word": w, **(span or empty)} for w, span in zip(words, spans)]


ScoreRun = Callable[..., Sequence[float]]
#: A lead (in log-likelihood) under this is a tie: tight crops can favour a slightly
#: shorter text by fractions of a point.
DFLT_DECISIVE_MARGIN = 1.0


def worker_score(
    *,
    model: str = DFLT_ALIGNER,
    threads: int | None = None,
    rootdir: str | os.PathLike | None = None,
) -> ScoreRun:
    """The default scorer for ``which_said``: the aligner adapter's ``score``."""

    def run(audio: str, texts: Sequence[str], *, start=None, end=None):
        worker = _ALIGN_POOL.get(model, threads, rootdir)
        reply = worker.call(
            "score",
            audio=str(Path(audio).resolve()),
            texts=list(texts),
            start=start,
            end=end,
        )
        return reply["scores"]

    return run


def which_said(
    audio: str | os.PathLike | bytes,
    candidates: Sequence[str],
    *,
    start: float | None = None,
    end: float | None = None,
    run: ScoreRun | None = None,
    decisive_margin: float = DFLT_DECISIVE_MARGIN,
) -> list[dict]:
    """The ``candidates`` (texts) ranked by how well the audio supports each, best first:
    ``[{"text", "score", "margin", "decisive"}]``: ``score`` the aligner's
    log-likelihood (minus its CTC loss; ``-inf`` for a text too long for the crop),
    ``margin`` the lead over the next candidate, ``decisive`` whether that lead is at
    least ``decisive_margin``.

    A forced choice, so it cannot hallucinate as transcribing a short crop can: use it
    to check an edit ("witch" or the old word?) on a crop around it (``start``/``end``,
    s). Cut the crop at silences and include the neighbouring words: a crop that ends
    inside a word, or holds one word alone, favours shorter texts by small margins,
    which ``decisive`` flags. Candidates are spelled in letters (numbers spelled out).

    >>> fake = lambda audio, texts, start=None, end=None: [-5.0, -2.0]
    >>> [(c["text"], c["margin"], c["decisive"])
    ...  for c in which_said(b"RIFF", ["bitch", "witch"], run=fake)]
    [('witch', 3.0, True), ('bitch', None, None)]
    """
    run = run or worker_score()
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(audio, bytes):
            path = Path(tmp) / "audio.wav"
            path.write_bytes(audio)
            audio = path
        scores = run(str(audio), list(candidates), start=start, end=end)
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda ts: -ts[1])
    out = []
    for i, (text, score) in enumerate(ranked):
        margin = score - ranked[i + 1][1] if i + 1 < len(ranked) else None
        if margin is not None and math.isfinite(margin):
            margin = round(margin, 4)
        decisive = None if margin is None else bool(margin >= decisive_margin)
        out.append(
            {"text": text, "score": score, "margin": margin, "decisive": decisive}
        )
    return out


# long recordings ---------------------------------------------------------------

#: The longest stretch handed to the aligner (s): under its 30 s limit, with margin.
DFLT_MAX_STRETCH_S = 25.0
#: A stretch whose mean word score is under this is reported as unreliable.
DFLT_MIN_STRETCH_SCORE = 0.5
#: A cut prefers a pause at least this long (s) between two anchored words, so a rough
#: pass's timing error (Whisper's is a few tenths) cannot put the cut inside a word.
DFLT_MIN_CUT_GAP_S = 0.25
#: How far past its outer anchored words (s) a stretch reaches into a long pause, the
#: recording's lead-in or its tail: a long music break or intro is not aligned against.
DFLT_EDGE_S = 1.0
#: Each stretch's crop reaches this far (s) past its bounds, for the same reason.
DFLT_CROP_PAD_S = 0.3
#: What the aligner refuses (s): a crop must stay under it.
ALIGNER_MAX_S = 30.0
#: The rate the recording is converted to once, before it is cut (the aligner's own).
ALIGN_RATE = 16000

#: ``audio_path -> [{"text" (or "word"), "start", "end"}, ...]``: timed words heard.
Heard = Callable[[str], Sequence[dict]]


def plan_stretches(
    n_words: int,
    anchor_times: dict[int, tuple[float, float]],
    duration: float,
    *,
    max_stretch_s: float = DFLT_MAX_STRETCH_S,
    min_gap_s: float = DFLT_MIN_CUT_GAP_S,
    edge_s: float = DFLT_EDGE_S,
    transparent: frozenset[int] = frozenset(),
) -> list[dict]:
    """Cut ``n_words`` text words, ``[0, duration)`` s of audio, into stretches.

    ``anchor_times`` maps a text word's index to the ``(start, end)`` heard for it. A
    cut falls only between two anchored text words with nothing between them but
    ``transparent`` words (punctuation no recogniser hears), so every word between two
    cuts is known to be in that stretch of audio. A stretch ends ``edge_s`` after its
    last anchored word and the next starts ``edge_s`` before its first, or both meet
    in the middle of a short pause; likewise the first stretch starts ``edge_s`` before
    an anchored first word and the last ends ``edge_s`` after an anchored last word, so
    an intro, an outro or a long break is not part of any stretch. Greedy: each stretch
    reaches the farthest cut within ``max_stretch_s`` in a pause of at least
    ``min_gap_s`` (else the widest pause in reach). A stretch no cut can keep that short
    is marked ``too_long`` (the default aligner refuses it), never cut on a guess.

    >>> times = {0: (0.0, 0.4), 1: (0.5, 0.9), 2: (10.0, 10.4), 3: (10.5, 10.9), 4: (20.0, 20.4)}
    >>> [(s["words"], s["start"], s["end"]) for s in plan_stretches(5, times, 21.0, max_stretch_s=12)]
    [((0, 4), 0.0, 11.9), ((4, 5), 19.0, 21.0)]
    """

    def before(j):  # the anchored word at or before j, skipping transparent ones
        while j >= 0 and j in transparent:
            j -= 1
        return j if j in anchor_times else None

    def after(j):
        while j < n_words and j in transparent:
            j += 1
        return j if j in anchor_times else None

    cuts = []  # (first word of the next stretch, this one's end, the next one's start, pause)
    for j in range(n_words - 1):
        a, b = before(j), after(j + 1)
        if a is None or b is None or j in transparent:
            continue
        gap = anchor_times[b][0] - anchor_times[a][1]
        if gap > 2 * edge_s:
            end, start = anchor_times[a][1] + edge_s, anchor_times[b][0] - edge_s
        else:
            end = start = (anchor_times[a][1] + anchor_times[b][0]) / 2
        cuts.append((j + 1, end, start, gap))
    # the edges: past transparent words, and past unheard words (each gets an extra
    # ``edge_s`` of room before the first anchor / after the last one)
    heard = sorted(anchor_times)
    if heard:
        lead = sum(1 for j in range(heard[0]) if j not in transparent)
        trail = sum(1 for j in range(heard[-1] + 1, n_words) if j not in transparent)
        at = max(0.0, anchor_times[heard[0]][0] - edge_s * (1 + lead))
        tail = min(duration, anchor_times[heard[-1]][1] + edge_s * (1 + trail))
    else:
        at, tail = 0.0, duration
    stretches, word = [], 0
    while word < n_words:
        if tail - at <= max_stretch_s:
            stretches.append(
                {"words": (word, n_words), "start": at, "end": tail, "too_long": False}
            )
            break
        ahead = [c for c in cuts if c[0] > word and c[1] > at]
        reach = [c for c in ahead if c[1] - at <= max_stretch_s]
        if reach:
            pauses = [c for c in reach if c[3] >= min_gap_s]
            i, end, start, _ = pauses[-1] if pauses else max(reach, key=lambda c: c[3])
            too_long = False
        elif ahead:
            i, end, start, _ = ahead[0]
            too_long = True
        else:
            stretches.append(
                {"words": (word, n_words), "start": at, "end": tail, "too_long": True}
            )
            break
        stretches.append(
            {"words": (word, i), "start": at, "end": end, "too_long": too_long}
        )
        word, at = i, start
    return stretches


def _as_16k_wav(path: str, tmp: str) -> str:
    """``path`` as a 16 kHz mono 16-bit WAV in ``tmp`` (ffmpeg, file to file), so it is
    decoded once, its length is in its header, and a stretch is cut without decoding the
    rest. An already conforming WAV is used as it is."""
    import subprocess
    import wave

    from voxy.wavutil import _ffmpeg_path

    try:
        with wave.open(path, "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (
                ALIGN_RATE,
                1,
                2,
            ):
                return path
    except (wave.Error, EOFError, OSError):
        pass
    out = str(Path(tmp) / "audio-16k.wav")
    cmd = [_ffmpeg_path(None), "-hide_banner", "-loglevel", "error", "-y", "-i", path]
    subprocess.run(
        [*cmd, "-ac", "1", "-ar", str(ALIGN_RATE), "-c:a", "pcm_s16le", out], check=True
    )
    return out


def _crop(wav_path: str, start: float, end: float, out: str) -> str:
    """Frames ``start``-``end`` s of a WAV written to ``out`` (read by seeking, not decoding)."""
    import wave

    with wave.open(wav_path, "rb") as src:
        rate = src.getframerate()
        src.setpos(int(start * rate))
        frames = src.readframes(int((end - start) * rate))
        with wave.open(out, "wb") as dst:
            dst.setparams(src.getparams())
            dst.writeframes(frames)
    return out


def _shifted(span: dict, by: float) -> dict:
    """A word span on a crop's clock moved to the recording's."""
    move = lambda t: None if t is None else t + by
    chars = [
        {**c, "start": move(c.get("start")), "end": move(c.get("end"))}
        for c in span.get("chars") or ()
    ]
    return {
        **span,
        "start": move(span.get("start")),
        "end": move(span.get("end")),
        "chars": chars,
    }


def _transcribed_words(path: str, *, language: str | None = None) -> list[dict]:
    """The rough pass: timed words heard by voxy's speech recognition (``voxy.refprep``),
    its worker stopped after, so it does not hold memory while the aligner runs."""
    from voxy.refprep import _words, close_asr, transcribe

    try:
        return _words(transcribe(path, language=language))
    finally:
        close_asr()


def align_long(
    audio: str | os.PathLike | bytes,
    text: str | Sequence[str],
    *,
    heard: Heard | None = None,
    run: Run | None = None,
    language: str | None = None,
    max_stretch_s: float = DFLT_MAX_STRETCH_S,
    min_score: float = DFLT_MIN_STRETCH_SCORE,
    pad_s: float = DFLT_CROP_PAD_S,
) -> list[dict]:
    """:func:`force_align` for a recording of any length (a whole episode).

    1. A rough pass of speech recognition (``heard``: default voxy's ASR) gives
       approximate times for the words it heard.
    2. The known text is matched to those words as two sequences
       (``mixing.transcript.align_words``, as ``voxy.pronounce`` does); matched words
       are anchors, and the text between two anchors belongs to the audio between them.
    3. The recording is cut between anchored words into stretches of at most
       ``max_stretch_s`` (:func:`plan_stretches`), and each is aligned exactly with
       :func:`force_align`, on a crop reaching ``pad_s`` past its bounds so a rough time
       a few tenths off cannot clip a word. The recording is converted once to a 16 kHz
       mono WAV (ffmpeg) and each crop is cut from it, so an hour is not decoded once
       per stretch.

    Each word comes back as :func:`force_align` gives it (``word``, ``start``, ``end``,
    ``score``, ``chars``), on the recording's clock, plus ``stretch`` (its index) and
    ``reliable``: False when its stretch's mean score is under ``min_score``, or when no
    cut could keep the stretch short enough (its times are then None: reported, not
    guessed).

    >>> heard = lambda path: [{"text": w, "start": 2.0 * i, "end": 2.0 * i + 1}
    ...                       for i, w in enumerate("one two three four".split())]
    >>> fake = lambda audio, words, *, start, end: [
    ...     {"start": i, "end": i + 0.5, "score": 0.9, "chars": []}
    ...     for i, _ in enumerate(words)]
    >>> import io, wave
    >>> buf = io.BytesIO()
    >>> with wave.open(buf, "wb") as w:
    ...     w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(bytes(2 * 16000 * 8))
    >>> out = align_long(buf.getvalue(), "One two, three four.", heard=heard, run=fake, max_stretch_s=4)
    >>> [(w["word"], w["stretch"], w["reliable"]) for w in out]
    [('One', 0, True), ('two,', 0, True), ('three', 1, True), ('four.', 2, True)]
    """
    from mixing.transcript import align_words
    from mixing.transcript.align import TEXT_KEYS
    from voxy.pronounce import normal_word

    if max_stretch_s + 2 * pad_s >= ALIGNER_MAX_S:
        raise ValueError(
            f"max_stretch_s={max_stretch_s} plus 2 x pad_s={pad_s} must stay under the "
            f"aligner's {ALIGNER_MAX_S:g} s"
        )
    words = text.split() if isinstance(text, str) else list(text)
    text_of = lambda w: next((w[k] for k in TEXT_KEYS if k in w), "")
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(audio, bytes):
            path = Path(tmp) / "audio"
            path.write_bytes(audio)
            audio = path
        timed = list(
            heard(str(audio))
            if heard
            else _transcribed_words(str(audio), language=language)
        )
        timed = sorted(
            (
                w
                for w in timed
                if w.get("start") is not None
                and w.get("end") is not None
                and normal_word(text_of(w))
            ),
            key=lambda w: w["start"],
        )
        anchors = align_words(words, timed, normalize=normal_word, split=None).anchors
        anchor_times = {
            j: (timed[h]["start"], timed[h]["end"]) for j, h in anchors.items()
        }
        wav = _as_16k_wav(str(audio), tmp)
        import wave

        with wave.open(wav, "rb") as w:
            duration = w.getnframes() / w.getframerate()
        transparent = frozenset(j for j, w in enumerate(words) if not normal_word(w))
        plan = plan_stretches(
            len(words),
            anchor_times,
            duration,
            max_stretch_s=max_stretch_s,
            transparent=transparent,
        )
        out: list[dict] = []
        for k, s in enumerate(plan):
            lo, hi = s["words"]
            if s["too_long"]:
                spans = [
                    {"word": w, "start": None, "end": None, "score": None, "chars": []}
                    for w in words[lo:hi]
                ]
                reliable = False
            else:
                start, end = (
                    max(0.0, s["start"] - pad_s),
                    min(duration, s["end"] + pad_s),
                )
                crop = _crop(wav, start, end, str(Path(tmp) / f"stretch-{k}.wav"))
                spans = [
                    _shifted(x, start) for x in force_align(crop, words[lo:hi], run=run)
                ]
                scores = [x["score"] for x in spans if x["score"] is not None]
                reliable = bool(scores) and sum(scores) / len(scores) >= min_score
            out += [{**x, "stretch": k, "reliable": reliable} for x in spans]
    return out
