"""
Singing: sing a score with lyrics in a voice (SVS), or re-voice a sung take (SVC).

    >>> import voxy                                                          # doctest: +SKIP
    >>> voxy.sing("melody.mid", "Light a can-dle in the win-dow").save("sung.wav")  # doctest: +SKIP
    >>> voice = voxy.singing_voice("me.wav", "phrase.mid", "la la la la")   # doctest: +SKIP
    >>> voxy.sing("song.musicxml", voice=voice).save("song.wav")           # doctest: +SKIP
    >>> voxy.convert_singing("take.wav", voice=voice).save("take-in-my-voice.wav")  # doctest: +SKIP

A singing *voice* is a sung clip, plus, for synthesis, the notes and words it sings
(``singing_voice``): a zero-shot singer copies the clip's timbre, and SoulX-Singer
also aligns the clip with its notes and phonemes. The simplest way to get that
alignment is to make the clip from a known score: sing a short given phrase (its
MIDI or MusicXML, with its lyrics) and pass the recording with that score. Without
a voice, a synthetic one is used (``synthetic_singing_voice``: a formant
synthesiser singing "la", which belongs to no one). Conversion needs only the clip.
Only clone a singing voice with its owner's consent; each backend records its own
usage restrictions in ``singing_backends()[name].consent``.

Scores are MIDI or MusicXML files, music21 streams, or lists of notes
(``voxy.score.read_score``, which reads files with ``audiate.sung_line``); lyrics
are one token per note (``can-dle`` spreads a word over two notes, ``-`` holds it
one more).

Backends are ``SingingBackend`` records (``singing_backends()``), each running in
its own environment (``python -m voxy setup soulx``); ``soulx`` (SoulX-Singer,
Apache-2.0) is the built-in one. A model is the seam: ``sing(..., model=...)``
takes any object with ``sing(notes, voice, *, transpose, language, **params)`` and
``convert(audio_path, voice, *, transpose, **params)`` returning a ``Speech``.

>>> class Echo:
...     name = "echo"
...     def sing(self, notes, voice, **kw):
...         return Speech(b"RIFF", "wav", "echo", text=" ".join(n.word for n in notes))
>>> from voxy.score import notes_from_pitches
>>> sing(notes_from_pitches([(60, 1), (62, 1)]), "do re", voice=VoiceProfile(
...     {"prompt_audio": "v.wav", "prompt_notes": []}, 0, "echo", 24000), model=Echo()).text
'do re'
"""

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from voxy import stores
from voxy.base import Speech, VoiceProfile
from voxy.envs import EnvSpec, GitSource, ensure_git_source
from voxy.library import find_voice, profile_from_record
from voxy.local_clone import WorkerGenerate
from voxy.score import Note, notes_from_pitches, read_score
from voxy.wavutil import write_wav
from voxy.worker import Worker

DFLT_SINGING_BACKEND = "soulx"
#: The samples store that keeps singing clips given as bytes.
DFLT_SINGING_SAMPLES = "singing-voices"
DFLT_LANGUAGE = "English"
#: ``transpose`` value meaning "move the melody into the voice's register".
FIT_VOICE = "fit_voice"
Transpose = int | Literal["fit_voice"]
#: Modules that register a built-in singing backend when imported.
BUILTIN_SINGING_MODULES = ("voxy.soulx",)
SYNTHETIC_PROMPT_LEAD_S = 0.3
SYNTHETIC_PROMPT_TAIL_S = 0.4
SYNTHETIC_PROMPT_BPM = 100
#: The phrase the synthetic voices sing ("la" on each note): ``(midi, beats)``.
SYNTHETIC_PROMPT_PHRASE = (
    (67, 1),
    (64, 1),
    (65, 1),
    (62, 1),
    (64, 1),
    (60, 1),
    (62, 1),
    (60, 2),
)
#: Semitones each synthetic voice preset sings the phrase at, from as written.
SYNTHETIC_PRESET_TRANSPOSE = {"high": 0, "low": -12}
SYNTHETIC_SAMPLE_RATE = 24000


class SingingModel(Protocol):
    """What ``sing`` and ``convert_singing`` call."""

    name: str

    def sing(self, notes: list[Note], voice: VoiceProfile, **params) -> Speech: ...

    def convert(self, audio: str, voice: VoiceProfile, **params) -> Speech: ...


@dataclass(frozen=True)
class SingingBackend:
    """What voxy knows about one singing model.

    Args:
        name: Registry name (also its adapter's, ``workers/{name}_adapter.py``).
        env: The environment it runs in.
        factory: ``(**kwargs) -> SingingModel``.
        init: Default keyword arguments of the adapter's ``load``.
        params: Documented parameters and their defaults (pass any to ``sing``).
        source: Code fetched at a pinned commit and handed to the adapter as
            ``repo_dir`` (for a model that pip can't install), or None.
        licence: Code and weights licences.
        consent: The model's own restrictions on cloning and impersonation.
        notes: One line on strengths and caveats.
        sings, converts: Whether it does SVS, SVC.
    """

    name: str
    env: EnvSpec
    factory: Callable[..., "SingingModel"]
    init: Mapping = field(default_factory=dict)
    params: Mapping = field(default_factory=dict)
    source: GitSource | None = None
    licence: str = ""
    consent: str = ""
    notes: str = ""
    sings: bool = True
    converts: bool = True


_SINGING_BACKENDS: dict[str, SingingBackend] = {}
#: One model per backend, made on first use.
_singing_models: dict[str, "SingingModel"] = {}


def register_singing_backend(
    backend: SingingBackend, *, replace: bool = True
) -> SingingBackend:
    """Add a singing backend (``replace=False`` keeps one already registered)."""
    if replace or backend.name not in _SINGING_BACKENDS:
        _SINGING_BACKENDS[backend.name] = backend
    return backend


def singing_backends() -> dict[str, SingingBackend]:
    """Every singing backend, by name (the built-in ones registered first)."""
    import importlib

    for module in BUILTIN_SINGING_MODULES:
        importlib.import_module(module)
    return _SINGING_BACKENDS


def get_singing_model(
    backend: str = DFLT_SINGING_BACKEND, *, models: MutableMapping | None = None
) -> "SingingModel":
    """The (cached) model of ``backend``; ``models`` replaces the shared cache."""
    cache = _singing_models if models is None else models
    if backend not in cache:
        backends = singing_backends()
        if backend not in backends:
            raise KeyError(
                f"No singing backend {backend!r} (have: {', '.join(sorted(backends))})"
            )
        cache[backend] = backends[backend].factory()
    return cache[backend]


def close_singing_models(models: MutableMapping | None = None) -> None:
    """Stop the cached singing models' workers and free their memory (a model stays
    loaded, about 6 GB for SoulX, until this is called or the process ends)."""
    cache = _singing_models if models is None else models
    for model in list(cache.values()):
        close = getattr(model, "close", None)
        if close:
            close()
    cache.clear()


# --------------------------------------------------------------------------- worker


class SingingWorker(WorkerGenerate):
    """The default model call of a singing backend: its adapter in its own
    environment, given the backend's fetched ``source`` as ``repo_dir``.

    ``worker(op, **request)`` runs operation ``op``, which writes a WAV to the
    ``out_path`` it is given, and returns ``(samples, sample_rate)``.
    """

    def _make_worker(self) -> Worker:
        source = getattr(self.backend, "source", None)
        if source is not None and "repo_dir" not in self.init:
            path = ensure_git_source(source, self.backend.name, rootdir=self.rootdir)
            self.init = {**self.init, "repo_dir": str(path)}
        return super()._make_worker()

    def __call__(self, op: str, **request):
        return self._run(op, **request)


# --------------------------------------------------------------------------- voices


def _notes_data(notes) -> list[dict] | None:
    return None if notes is None else [asdict(n) for n in notes]


def singing_voice(
    audio: str | os.PathLike | bytes,
    score=None,
    lyrics: str | None = None,
    *,
    backend: str = DFLT_SINGING_BACKEND,
    sample_rate: int = SYNTHETIC_SAMPLE_RATE,
    name: str = DFLT_SINGING_SAMPLES,
    rootdir: str | os.PathLike | None = None,
    **metadata,
) -> VoiceProfile:
    """A singing voice: a sung clip, and (for synthesis) the score it sings.

    Args:
        audio: The clip, 5 to 30 s of one voice singing, without accompaniment: a
            path, or WAV bytes (kept in the samples store of voice ``name``, under
            the data root ``rootdir``, so the voice outlives the session).
        score: What the clip sings, timed as in the clip (a MIDI or MusicXML file,
            a music21 stream, or notes, such as ``audiate.sung_line`` returns for
            a line its defaults don't pick; see ``voxy.score.read_score``). Needed by
            ``sing``; ``convert_singing`` needs only the clip.
        lyrics: Its words, if the score has none.
        backend: The backend the voice is for (its ``model_type``).
        **metadata: Kept with the voice (e.g. ``consent=``, ``description=``).

    The profile's ``segment`` is plain data (``prompt_audio`` path, ``prompt_notes``),
    so it can be saved in the voice library: ``voxy.save_voice(name, voice,
    consent=...)``.
    """
    if isinstance(audio, bytes):
        key = f"prompt-{hashlib.sha256(audio).hexdigest()[:16]}.wav"
        stores.samples_store(name, rootdir=rootdir)[key] = audio
        audio = stores.kind_dir(stores.SAMPLES_KIND, name, rootdir=rootdir) / key
    notes = read_score(score, lyrics) if score is not None else None
    return VoiceProfile(
        {
            "prompt_audio": str(Path(audio).resolve()),
            "prompt_notes": _notes_data(notes),
        },
        0,
        backend,
        sample_rate,
        metadata or None,
    )


def synthetic_singing_voice(
    preset: str = "high", *, backend: str = DFLT_SINGING_BACKEND
) -> VoiceProfile:
    """A voice that belongs to no one: a formant synthesiser singing "la" on a short
    phrase (``voxy.formant_singer``), rendered once into voxy's cache (the file is
    named by a hash of everything it is rendered from, so a change re-renders it).

    ``preset`` is 'high' or 'low' (an octave down, darker vowels).
    """
    from voxy.formant_singer import FORMANT_PRESETS, SYNTH_VERSION, sing_la

    notes = notes_from_pitches(
        [
            (p + SYNTHETIC_PRESET_TRANSPOSE[preset], b)
            for p, b in SYNTHETIC_PROMPT_PHRASE
        ],
        bpm=SYNTHETIC_PROMPT_BPM,
        start=SYNTHETIC_PROMPT_LEAD_S,
    )
    notes = [Note(n.start, n.dur, n.pitch, "la") for n in notes]
    settings = json.dumps(
        [
            SYNTH_VERSION,
            preset,
            FORMANT_PRESETS[preset],
            [asdict(n) for n in notes],
            SYNTHETIC_PROMPT_TAIL_S,
            SYNTHETIC_SAMPLE_RATE,
        ]
    )
    digest = hashlib.sha256(settings.encode()).hexdigest()[:12]
    path = stores.voxy_cache_dir("singing", f"synthetic-{preset}-{digest}.wav")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = sing_la(
            notes,
            sample_rate=SYNTHETIC_SAMPLE_RATE,
            preset=preset,
            tail_s=SYNTHETIC_PROMPT_TAIL_S,
        )
        partial = path.with_suffix(f".{os.getpid()}-{threading.get_ident()}.part")
        partial.write_bytes(write_wav(samples, SYNTHETIC_SAMPLE_RATE))
        partial.replace(path)
    return singing_voice(path, notes, backend=backend, synthetic=preset)


def resolve_singing_voice(
    voice: VoiceProfile | str | os.PathLike | None,
    *,
    backend: str = DFLT_SINGING_BACKEND,
    voices: Mapping | None = None,
) -> VoiceProfile:
    """The singing voice ``voice`` names: a profile as is; a library name or alias
    (its ``backend`` profile); a path to a sung clip (for conversion); None, the
    synthetic voice."""
    if voice is None:
        return synthetic_singing_voice(backend=backend)
    if isinstance(voice, VoiceProfile):
        return voice
    if isinstance(voice, str):
        voices = stores.voices_store() if voices is None else voices
        key = find_voice(voice, voices=voices)
        if key is not None:
            return profile_from_record(voices[key], backend)
    if isinstance(voice, (str, os.PathLike)) and Path(voice).is_file():
        return singing_voice(voice, backend=backend)
    raise ValueError(
        f"{voice!r} is neither a library voice nor a clip: make one with "
        "voxy.singing_voice(audio, score, lyrics)"
    )


# --------------------------------------------------------------------------- facade


def sing(
    score,
    lyrics: str | None = None,
    *,
    voice: VoiceProfile | str | None = None,
    backend: str = DFLT_SINGING_BACKEND,
    output_path: str | os.PathLike | None = None,
    transpose: Transpose = 0,
    language: str = DFLT_LANGUAGE,
    track: int | None = None,
    channel: int | None = None,
    part: int | str | None = None,
    verse: int | str | None = None,
    model: "SingingModel | None" = None,
    models: MutableMapping | None = None,
    voices: Mapping | None = None,
    **params,
) -> Speech:
    """Sing ``score`` with ``lyrics`` in ``voice`` and return the audio (``.save(path)``).

    Args:
        score: A MIDI or MusicXML file, a music21 stream, or notes
            (``voxy.score.Note``, or ``audiate.SungNote``).
        lyrics: One token per note (``can-dle``: a word over two notes; ``-``: hold
            the word); default: the score's own lyrics.
        voice: A singing voice (``singing_voice``), a library voice saved for
            ``backend``, or None (the synthetic voice).
        backend: The singing backend (``singing_backends()``).
        output_path: Also save the audio there.
        transpose: Semitones to move the melody (0: as written), or 'fit_voice'
            (into the voice's register).
        language: Of the lyrics (SoulX here: English).
        track, channel: Which line of a MIDI file to sing (``track``: index in
            the file; ``channel``: 1 to 16).
        part, verse: Which part (an index, a name or an id) and lyric verse of a
            MusicXML score to sing. When the score leaves the choice open, reading
            it raises and names the candidates (see ``audiate.sung``).
        model: A ready model (its backend is then the backend).
        models: Model cache to use instead of the shared one.
        voices: Voice library store (default ``voxy.voices_store()``).
        **params: The backend's parameters (``singing_backends()[b].params``).
    """
    notes = read_score(
        score, lyrics, track=track, channel=channel, part=part, verse=verse
    )
    model = model or get_singing_model(backend, models=models)
    voice = resolve_singing_voice(voice, backend=model.name, voices=voices)
    speech = model.sing(notes, voice, transpose=transpose, language=language, **params)
    if output_path:
        speech.save(str(output_path))
    return speech


def convert_singing(
    audio: str | os.PathLike | bytes,
    *,
    voice: VoiceProfile | str | None = None,
    backend: str = DFLT_SINGING_BACKEND,
    output_path: str | os.PathLike | None = None,
    transpose: Transpose = FIT_VOICE,
    model: "SingingModel | None" = None,
    models: MutableMapping | None = None,
    voices: Mapping | None = None,
    **params,
) -> Speech:
    """Re-voice the sung ``audio`` in ``voice``, keeping its melody, words and timing.

    Args:
        audio: A sung clip without accompaniment (a path, or WAV bytes); separate a
            song first (``voxy.separate.separate``).
        voice: A singing voice, a library voice, a path to a clip of the target
            voice, or None (the synthetic voice).
        transpose: 'fit_voice' (default: move the melody into the voice's register)
            or semitones (0: keep it).
        Other arguments: as ``sing``.
    """
    model = model or get_singing_model(backend, models=models)
    voice = resolve_singing_voice(voice, backend=model.name, voices=voices)
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(audio, bytes):
            path = Path(tmp) / "input.wav"
            path.write_bytes(audio)
            audio = path
        speech = model.convert(
            str(Path(audio).resolve()), voice, transpose=transpose, **params
        )
    if output_path:
        speech.save(str(output_path))
    return speech
