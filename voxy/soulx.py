"""
SoulX-Singer (Soul AI Lab, 2026; code and weights Apache-2.0): zero-shot singing
voice synthesis and conversion, as the ``soulx`` singing backend.

    >>> import voxy                                                         # doctest: +SKIP
    >>> voxy.sing("melody.mid", "Light a can-dle in the win-dow", backend="soulx")  # doctest: +SKIP

How it runs: in its own environment (``python -m voxy setup soulx``: Python 3.11,
torch 2.2, transformers 4.41, the upstream pins). SoulX-Singer has no
``pyproject.toml``, so pip can't install it: voxy fetches its repository at a pinned
commit (``SOULX_SOURCE``) into ``models/soulx/source/<commit>`` on first use, and the
adapter puts that folder on ``sys.path``. Weights (two 2.8 GB checkpoints, SVS and
SVC, plus RMVPE for F0) come from the Hugging Face Hub at pinned revisions; SVC also
loads ``openai/whisper-base`` (Apache-2.0) on first use. On Apple Silicon it runs on
MPS, with the vocoder on the CPU; elsewhere on CUDA or the CPU (about 8x real time).

What it reads: SoulX metadata, a list of *segments*, one entry per note in every
field, made here from ``voxy.score`` notes by ``soulx_metadata``:

    {"index": "target_0", "language": "English", "time": [0, 6080],
     "duration": "0.30 0.60 ...", "text": "<SP> Light a candle candle ...",
     "phoneme": "", "note_pitch": "0 64 64 67 ...", "note_type": "1 2 2 3 ...", "f0": ""}

``note_type`` is 1 for a rest (``<SP>``), 2 where a word starts, 3 where it
continues; a word sung over several notes is written on each. Phonemes are left
empty and filled in the worker (English, g2p_en), so the host needs no phonemizer.
A synthesis prompt needs the same metadata, aligned to its clip
(``soulx_prompt_metadata``).

Usage disclaimer (upstream README; not a licence term): do not use the model to
impersonate anyone without authorisation or to create deceptive audio; respect
intellectual property, privacy and personal consent.

>>> from voxy.score import Note
>>> meta = soulx_metadata([Note(0.0, 0.5, 64, "hel"), Note(0.5, 0.5, 67, "hel", False)])
>>> meta[0]["text"], meta[0]["note_type"], meta[0]["duration"]
('<SP> hel hel <SP>', '1 2 3 1', '0.30 0.50 0.50 0.40')
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from voxy.base import Speech, VoiceProfile
from voxy.envs import EnvSpec, GitSource
from voxy.score import Note
from voxy.singing import (
    DFLT_LANGUAGE,
    FIT_VOICE,
    SingingBackend,
    SingingWorker,
    register_singing_backend,
)
from voxy.wavutil import write_wav

SP = "<SP>"
REST_TYPE, ONSET_TYPE, CONTINUATION_TYPE = 1, 2, 3
#: Rest before the first note and after the last (upstream's examples start with <SP>).
DFLT_LEAD_S = 0.3
DFLT_TAIL_S = 0.4
#: A gap shorter than this joins the previous note (legato); a longer one is a rest.
DFLT_MIN_REST_S = 0.2
#: A rest longer than this ends a segment.
DFLT_SPLIT_REST_S = 2.0
#: No segment longer than this (upstream allows 60 s; shorter is safer on MPS).
DFLT_MAX_SEGMENT_S = 30.0
DURATION_DIGITS = 2

SOULX_SOURCE = GitSource(
    "https://github.com/Soul-AILab/SoulX-Singer",
    "81aeb3ae772c70093c3de74dc23c92d983801ae4",
)
SOULX_WEIGHTS = {
    "repo_id": "Soul-AILab/SoulX-Singer",
    "revision": "40493ad90286056c7a9095035164434a79daa8c9",
}
SOULX_PREPROCESS = {
    "repo_id": "Soul-AILab/SoulX-Singer-Preprocess",
    "revision": "83dc50289d22a81b1e9998f5b9e111aef7c1fdcd",
}
#: Upstream's inference pins (its requirements.txt minus the preprocessing, web UI
#: and training stacks).
SOULX_ENV = EnvSpec(
    "soulx",
    (
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "numpy<2",
        "transformers==4.41.2",
        "accelerate==1.11.0",
        "librosa==0.11.0",
        "soundfile",
        "omegaconf==2.3.0",
        "scipy",
        "tqdm",
        "huggingface_hub>=0.20,<1",
        "g2p_en==2.1.0",
        "nltk==3.9.2",
    ),
    env_vars={"PYTORCH_ENABLE_MPS_FALLBACK": "1", "NLTK_DATA": "{models}/nltk_data"},
)
SOULX_CONSENT = (
    "Upstream usage disclaimer (not a licence term): do not use the model to "
    "impersonate individuals without authorisation or to create deceptive audio; "
    "respect intellectual property, privacy and personal consent. Clone only your own "
    "voice, a consenting singer's, or a synthetic one."
)


# --------------------------------------------------------------------------- metadata


def _with_rests(
    notes: Sequence[Note], *, lead_s: float | None, tail_s: float, min_rest_s: float
) -> list[Note]:
    """The line with rests (pitch 0) in its gaps, a lead-in and a tail.

    ``lead_s=None`` keeps the notes' own times (a rest from 0 to the first note),
    which is what a prompt aligned to its clip needs.
    """
    if not notes:
        raise ValueError("No notes to sing")
    notes = sorted(notes, key=lambda n: n.start)
    shift = 0.0 if lead_s is None else lead_s - notes[0].start
    line: list[Note] = []
    first = notes[0].start + shift
    if first > 0:
        line.append(Note(0.0, first, 0, SP))
    for note in notes:
        note = replace(note, start=note.start + shift)
        if line and line[-1].pitch:
            gap = note.start - line[-1].end
            if gap >= min_rest_s:
                line.append(Note(line[-1].end, gap, 0, SP))
            else:  # legato, or an overlap: the previous note lasts until this one
                line[-1] = replace(line[-1], dur=note.start - line[-1].start)
        line.append(note)
    if tail_s > 0:
        line.append(Note(line[-1].end, tail_s, 0, SP))
    return line


def _segments(
    line: list[Note], *, split_rest_s: float, max_segment_s: float
) -> list[list[Note]]:
    """``line`` cut into segments: after a rest longer than ``split_rest_s``, and
    before a segment would pass ``max_segment_s``, at its last rest if it has one
    (so the join lands in silence), else at that note."""
    segments, current = [], []
    for note in line:
        if current and not current[-1].pitch and current[-1].dur > split_rest_s:
            segments.append(current)
            current = []
        elif current and note.end - current[0].start > max_segment_s:
            rests = [k for k in range(1, len(current)) if not current[k].pitch]
            cut = rests[-1] + 1 if rests else len(current)
            segments.append(current[:cut])
            current = current[cut:]
        current.append(note)
    return segments + [current] if current else segments


def _check_word(note: Note) -> None:
    """One sung note, one word: no spaces, at least one letter (else the phonemes
    and the notes would not line up)."""
    word = note.word
    if not word or any(c.isspace() for c in word) or not any(c.isalpha() for c in word):
        raise ValueError(
            f"The note at {note.start:.2f} s sings {word!r}: each sung note needs one "
            "word, without spaces and with letters (hyphenate a word over notes: "
            "'can-dle')"
        )


def _durations(segment: Sequence[Note]) -> list[float]:
    """Durations rounded on the cumulative grid, so rounding never drifts."""
    t0 = segment[0].start
    edges = [round(n.start - t0, DURATION_DIGITS) for n in segment]
    edges.append(round(segment[-1].end - t0, DURATION_DIGITS))
    return [round(b - a, DURATION_DIGITS) for a, b in zip(edges, edges[1:])]


def _segment(name: str, i: int, segment: Sequence[Note], language: str) -> dict:
    durations = _durations(segment)
    start_ms = round(segment[0].start * 1000)
    return {
        "index": f"{name}_{i}",
        "language": language,
        "time": [start_ms, start_ms + round(sum(durations) * 1000)],
        "duration": " ".join(f"{d:.{DURATION_DIGITS}f}" for d in durations),
        "text": " ".join(n.word if n.pitch else SP for n in segment),
        "phoneme": "",
        "note_pitch": " ".join(str(n.pitch) for n in segment),
        "note_type": " ".join(
            str(
                REST_TYPE
                if not n.pitch
                else ONSET_TYPE
                if n.starts_word
                else CONTINUATION_TYPE
            )
            for n in segment
        ),
        "f0": "",
    }


def soulx_metadata(
    notes: Sequence[Note],
    *,
    name: str = "target",
    language: str = DFLT_LANGUAGE,
    lead_s: float = DFLT_LEAD_S,
    tail_s: float = DFLT_TAIL_S,
    min_rest_s: float = DFLT_MIN_REST_S,
    split_rest_s: float = DFLT_SPLIT_REST_S,
    max_segment_s: float = DFLT_MAX_SEGMENT_S,
) -> list[dict]:
    """SoulX-Singer metadata (segments, phonemes left empty) for sung ``notes``.

    The line starts ``lead_s`` after a rest, gaps of ``min_rest_s`` or more become
    rests (shorter ones join the previous note), and a segment ends at a rest
    longer than ``split_rest_s`` or before it would pass ``max_segment_s``.
    """
    if language != "English":
        raise NotImplementedError(
            f"SoulX singing in {language} needs its phonemes (only English is done here)"
        )
    for note in notes:
        _check_word(note)
    line = _with_rests(notes, lead_s=lead_s, tail_s=tail_s, min_rest_s=min_rest_s)
    segments = _segments(line, split_rest_s=split_rest_s, max_segment_s=max_segment_s)
    return [_segment(name, i, s, language) for i, s in enumerate(segments)]


def soulx_prompt_metadata(
    notes: Sequence[Note],
    *,
    language: str = DFLT_LANGUAGE,
    tail_s: float = 0.0,
    min_rest_s: float = DFLT_MIN_REST_S,
) -> list[dict]:
    """The one-segment metadata of a prompt clip whose notes are ``notes``, at the
    clip's own times (SoulX crops the clip to the metadata's length).

    >>> from voxy.score import Note
    >>> soulx_prompt_metadata([Note(0.3, 0.6, 67, "la")])[0]["duration"]
    '0.30 0.60'
    """
    infinite = float("inf")
    return soulx_metadata(
        notes,
        name="prompt",
        language=language,
        lead_s=None,
        tail_s=tail_s,
        min_rest_s=min_rest_s,
        split_rest_s=infinite,
        max_segment_s=infinite,
    )


# --------------------------------------------------------------------------- model


def _lower_median(values: Sequence[int]) -> int:
    """The lower middle value (``torch.median``'s rule, which upstream uses)."""
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def fit_voice_shift(target: Sequence[Note], prompt: Sequence[Note]) -> int:
    """Semitones that move ``target`` into the register of the voice singing
    ``prompt``: median prompt pitch minus median target pitch, over notes.

    Upstream computes this per segment (and weights notes by their phoneme count),
    so a long song would change key between segments; voxy computes it once.

    >>> fit_voice_shift([Note(0, 1, 60), Note(1, 1, 64)], [Note(0, 1, 55)])
    -5
    """
    return _lower_median([n.pitch for n in prompt]) - _lower_median(
        [n.pitch for n in target]
    )


def _shift(transpose, *, target=None, prompt=None) -> dict:
    """The adapter's ``auto_shift`` / ``pitch_shift`` for a ``transpose`` value.

    With ``target`` and ``prompt`` notes, 'fit_voice' is computed here, once for
    every segment; without (conversion), the adapter fits it from the audio.

    >>> _shift(0), _shift(-12), _shift("fit_voice")
    ({'auto_shift': False, 'pitch_shift': 0}, {'auto_shift': False, 'pitch_shift': -12}, {'auto_shift': True, 'pitch_shift': 0})
    """
    if transpose == FIT_VOICE:
        if target and prompt:
            return {"auto_shift": False, "pitch_shift": fit_voice_shift(target, prompt)}
        return {"auto_shift": True, "pitch_shift": 0}
    if isinstance(transpose, bool) or not isinstance(transpose, int):
        raise ValueError(f"transpose is semitones or {FIT_VOICE!r}, not {transpose!r}")
    return {"auto_shift": False, "pitch_shift": transpose}


def _prompt(voice: VoiceProfile) -> tuple[str, list[Note] | None]:
    segment = getattr(voice, "segment", voice)
    if not isinstance(segment, Mapping) or "prompt_audio" not in segment:
        raise ValueError(
            "A singing voice is a sung clip: voxy.singing_voice(audio, score, lyrics)"
        )
    notes = segment.get("prompt_notes")
    audio = str(Path(segment["prompt_audio"]).resolve())
    return audio, None if notes is None else [Note(**n) for n in notes]


class SoulXSingingModel:
    """SoulX-Singer as a singing model (``voxy.sing(..., backend="soulx")``).

    Args:
        run: ``(op, **request) -> (samples, sample_rate)``, where ``op`` is the
            adapter's 'sing' or 'convert'; default: a ``SingingWorker`` running
            ``workers/soulx_adapter.py`` in the ``soulx`` environment (built, and the
            source fetched, on first use). Tests inject a fake.
        init: Keyword arguments of the adapter's ``load`` over the backend's
            (``device='cpu'``, ``weights_dir=`` local weights, ``preload=``).
        threads, rootdir, auto_setup, env_vars: As ``voxy.local_clone.WorkerGenerate``.
    """

    name = "soulx"

    def __init__(
        self,
        *,
        run=None,
        init: Mapping | None = None,
        threads: int | None = None,
        rootdir: str | os.PathLike | None = None,
        auto_setup: bool = True,
        env_vars: Mapping[str, str] | None = None,
    ):
        self.backend = SOULX_BACKEND
        self._run = run or SingingWorker(
            self.backend,
            init=init,
            threads=threads,
            rootdir=rootdir,
            env_vars=env_vars,
            auto_setup=auto_setup,
        )

    def _speech(self, op: str, request: dict, params: dict, *, voice: str, text=None):
        if clash := sorted(request.keys() & params.keys()):
            raise TypeError(f"{clash} are set by voxy, not by params")
        samples, sample_rate = self._run(op, **request, **params)
        return Speech(
            write_wav(samples, sample_rate),
            format="wav",
            backend=self.name,
            voice=voice,
            sample_rate=sample_rate,
            text=text,
        )

    def sing(
        self,
        notes: Sequence[Note],
        voice: VoiceProfile,
        *,
        transpose=0,
        language: str = DFLT_LANGUAGE,
        **params,
    ) -> Speech:
        """Sing ``notes`` (with words) in ``voice`` (a clip and the notes it sings)."""
        prompt_audio, prompt_notes = _prompt(voice)
        if not prompt_notes:
            raise ValueError(
                "SoulX synthesis needs the notes and words of the voice's clip: "
                "voxy.singing_voice(audio, score, lyrics)"
            )
        request = {
            "meta": soulx_metadata(notes, language=language),
            "prompt_audio": prompt_audio,
            "prompt_meta": soulx_prompt_metadata(prompt_notes),
            **_shift(transpose, target=notes, prompt=prompt_notes),
        }
        text = " ".join(n.word for n in notes if n.starts_word)
        return self._speech("sing", request, params, voice=prompt_audio, text=text)

    def convert(
        self, audio: str, voice: VoiceProfile, *, transpose=FIT_VOICE, **params
    ) -> Speech:
        """Re-voice the sung clip at ``audio`` with ``voice``'s timbre."""
        prompt_audio, _ = _prompt(voice)
        request = {
            "audio": str(Path(audio).resolve()),
            "prompt_audio": prompt_audio,
            **_shift(transpose),
        }
        return self._speech("convert", request, params, voice=prompt_audio)

    def close(self) -> None:
        """Stop the worker (free the weights)."""
        close = getattr(self._run, "close", None)
        if close:
            close()


SOULX_BACKEND = register_singing_backend(
    SingingBackend(
        "soulx",
        SOULX_ENV,
        SoulXSingingModel,
        init={
            "weights": SOULX_WEIGHTS,
            "preprocess": SOULX_PREPROCESS,
            "device": "auto",
        },
        params={
            "n_steps": 32,
            "cfg": 3.0,
            "control": "score",
            "seed": None,
            "peak": 0.95,
            "crossfade_s": 0.02,
        },
        source=SOULX_SOURCE,
        licence="code and weights Apache-2.0 (RMVPE and whisper-base: Apache-2.0)",
        consent=SOULX_CONSENT,
        notes=(
            "Zero-shot SVS (score control: exact pitch) and SVC; 24 kHz; English, "
            "Mandarin, Cantonese upstream (English only here); ~1.4x real time on "
            "an M1 Max, 6 GB; n_steps=16 halves the time but blurs words."
        ),
    ),
    replace=False,
)
