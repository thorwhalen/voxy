"""
Local voice cloning: speak any text in a voice learned from a short reference clip.

Each backend here (F5-TTS, Chatterbox, XTTS-v2, ...) is a model that runs on your
own machine, CPU included. A *voice* is a reference: a clean clip of the speaker
plus its transcript, as a ``VoiceProfile`` whose segment is plain data
(``{"ref_audio": path, "ref_text": text}``), so it can be saved in the voice
library and reused by every local backend.

    >>> import voxy                                                     # doctest: +SKIP
    >>> model = voxy.create_speech_model("f5tts")                      # doctest: +SKIP
    >>> voice = model.clone_voice("ref.wav", "What the clip says.")     # doctest: +SKIP
    >>> model.synthesize(chapter_text, voice, nfe_step=32).save("ch1.wav")  # doctest: +SKIP

Long text is split into paragraphs and sentence-packed chunks (``voxy.textsplit``),
each chunk is synthesized, trimmed of edge silence, and joined with ``gap_ms`` of
pause (``paragraph_gap_ms`` between paragraphs).

Seams (keyword arguments of every model):

- ``generate(text, *, ref_audio, ref_text, **params) -> (samples, sample_rate)``: the
  model call. Default: a ``voxy.worker.Worker`` running the backend's adapter in its
  own environment (built on first use under ``$VOXY_DATA_DIR/envs``). Tests inject a fake.
- ``chunker(text) -> [[chunk, ...], ...]``: default ``split_text`` at the backend's
  ``max_chars``.

>>> import numpy as np
>>> fake = lambda text, **kw: (np.full(800, 0.1, dtype=np.float32), 8000)
>>> model = LocalCloneSpeechModel("f5tts", generate=fake)
>>> voice = reference_profile("ref.wav", "Hello there.", backend="f5tts")
>>> speech = model.synthesize("One. Two.\\n\\nThree.", voice)
>>> speech.format, speech.sample_rate, speech.backend
('wav', 8000, 'f5tts')
"""

import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np

from voxy.base import Speech, SpeechModel, VoiceProfile, _resolve_text_input
from voxy.envs import EnvSpec, ensure_env, worker_environ
from voxy.textsplit import split_text
from voxy.wavutil import read_wav, silence, trim_silence, write_wav
from voxy.worker import Worker
from voxy.workers import adapter_path

VOXY_CACHE_DIR_ENVVAR = "VOXY_CACHE_DIR"
DFLT_GAP_MS = 250
DFLT_PARAGRAPH_GAP_MS = 700
REFERENCE_SPEAKER_ID = 0
DFLT_REFERENCE_SAMPLE_RATE = 24000


def voxy_cache_dir(*parts: str) -> Path:
    """Intermediates (worker scratch audio, logs): ``$VOXY_CACHE_DIR`` or ``~/.cache/voxy``."""
    root = os.environ.get(VOXY_CACHE_DIR_ENVVAR) or "~/.cache/voxy"
    return Path(root).expanduser().joinpath(*parts)


@dataclass(frozen=True)
class LocalCloneBackend:
    """What voxy knows about one local cloning model.

    Args:
        name: Registry name (also the adapter's name, ``workers/{name}_adapter.py``).
        env: The environment it runs in.
        init: Default keyword arguments of the adapter's ``load`` (model variant,
            checkpoint, device); override per model with ``init=`` or ``**init``.
        params: Documented synthesis parameters and their defaults (pass any of
            them to ``synthesize``; the diagnosis grid sweeps them).
        max_chars: Chunk size for long text.
        licence: Code and weights licences (flag non-commercial ones).
        notes: One line on strengths and caveats.
    """

    name: str
    env: EnvSpec
    init: Mapping = field(default_factory=dict)
    params: Mapping = field(default_factory=dict)
    max_chars: int = 300
    licence: str = ""
    notes: str = ""


LOCAL_CLONE_BACKENDS: dict[str, LocalCloneBackend] = {}


def register_local_clone_backend(backend: LocalCloneBackend) -> LocalCloneBackend:
    """Add (or replace) a local cloning backend description."""
    LOCAL_CLONE_BACKENDS[backend.name] = backend
    return backend


register_local_clone_backend(
    LocalCloneBackend(
        "f5tts",
        EnvSpec("f5tts", ("f5-tts==1.1.22", "soundfile")),
        init={"model": "F5TTS_v1_Base"},
        params={
            "nfe_step": 32,
            "cfg_strength": 2.0,
            "sway_sampling_coef": -1.0,
            "speed": 1.0,
            "cross_fade_duration": 0.15,
            "seed": None,
        },
        max_chars=250,
        licence="code MIT; weights CC-BY-NC-4.0 (non-commercial)",
        notes="Flow matching; strong similarity; needs an accurate ref_text.",
    )
)
register_local_clone_backend(
    LocalCloneBackend(
        "chatterbox",
        EnvSpec(
            "chatterbox",
            # setuptools<81: the watermarker still imports pkg_resources
            ("chatterbox-tts==0.1.7", "soundfile", "setuptools<81"),
        ),
        init={"variant": "original"},
        params={
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
            "min_p": 0.05,
            "top_p": 1.0,
            "seed": None,
        },
        max_chars=300,
        licence="code and weights MIT (outputs carry a Perth watermark)",
        notes="Natural prosody; 'exaggeration' sets expressiveness; variant='turbo' is faster.",
    )
)
register_local_clone_backend(
    LocalCloneBackend(
        "xtts",
        EnvSpec(
            "xtts",
            (
                "coqui-tts==0.27.5",
                "transformers<4.58",
                "torch",
                "torchaudio",
                "torchcodec",
                "soundfile",
            ),
            env_vars={"TTS_HOME": "{models}"},
        ),
        init={},
        params={
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 5.0,
            "top_k": 50,
            "top_p": 0.85,
            "speed": 1.0,
            "language": "en",
            "seed": None,
        },
        max_chars=240,
        licence="code MPL-2.0; weights Coqui Public Model Licence (non-commercial)",
        notes="Mature fine-tuning recipe; several reference clips can be given at once.",
    )
)


register_local_clone_backend(
    LocalCloneBackend(
        "voxcpm",
        EnvSpec("voxcpm", ("voxcpm==2.0.3", "soundfile")),
        init={"model_id": "openbmb/VoxCPM2"},
        params={
            "cfg_value": 2.0,
            "inference_timesteps": 10,
            "prompt_mode": "continuation",
            "retry_badcase": True,
            "seed": None,
        },
        max_chars=220,
        licence="code and weights Apache-2.0",
        notes="Best similarity among permissive models with a CPU path; 48 kHz; "
        "distorts past ~300 characters, so chunks stay short.",
    )
)
register_local_clone_backend(
    LocalCloneBackend(
        "qwen3tts",
        EnvSpec("qwen3tts", ("qwen-tts==0.1.1", "soundfile")),
        init={"model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "dtype": "float32"},
        params={
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "non_streaming_mode": True,
            "language": "English",
            "seed": None,
        },
        max_chars=300,
        licence="code and weights Apache-2.0",
        notes="Most stable permissive cloner (lowest word errors); "
        "model_id ...-0.6B-Base for speed.",
    )
)


def reference_profile(
    ref_audio: str | os.PathLike | list,
    ref_text: str,
    *,
    backend: str = "f5tts",
    sample_rate: int = DFLT_REFERENCE_SAMPLE_RATE,
    **metadata,
) -> VoiceProfile:
    """A voice for local backends: a reference clip (or clips) and what it says.

    >>> reference_profile("ref.wav", "Hi.", backend="xtts").segment
    {'ref_audio': 'ref.wav', 'ref_text': 'Hi.'}
    """
    audio = (
        [str(a) for a in ref_audio] if isinstance(ref_audio, list) else str(ref_audio)
    )
    return VoiceProfile(
        {"ref_audio": audio, "ref_text": ref_text},
        REFERENCE_SPEAKER_ID,
        backend,
        sample_rate,
        metadata or None,
    )


def _reference(voice) -> dict:
    """``{"ref_audio", "ref_text"}`` from a profile or a plain mapping."""
    segment = getattr(voice, "segment", voice)
    if not isinstance(segment, Mapping) or "ref_audio" not in segment:
        raise ValueError(
            "A local cloning backend needs a reference voice: "
            "reference_profile(ref_audio, ref_text) or model.clone_voice(...)"
        )
    return {"ref_audio": segment["ref_audio"], "ref_text": segment.get("ref_text", "")}


class WorkerGenerate:
    """The default ``generate``: the backend's adapter in its own environment.

    Builds the environment on first call (``auto_setup``), starts one worker,
    keeps it (and the weights) alive until ``close()``.
    """

    def __init__(
        self,
        backend: LocalCloneBackend,
        *,
        init: Mapping | None = None,
        threads: int | None = None,
        rootdir: str | os.PathLike | None = None,
        env_vars: Mapping[str, str] | None = None,
        auto_setup: bool = True,
    ):
        self.backend = backend
        self.init = {**backend.init, **(init or {})}
        self.threads, self.rootdir = threads, rootdir
        self.env_vars = dict(env_vars or {})
        self.auto_setup = auto_setup
        self._worker = None

    def _make_worker(self) -> Worker:
        from voxy.envs import env_python, is_env_ready

        spec = self.backend.env
        if self.auto_setup:
            python = ensure_env(spec, rootdir=self.rootdir)
        elif is_env_ready(spec, rootdir=self.rootdir):
            python = env_python(spec, rootdir=self.rootdir)
        else:
            raise RuntimeError(
                f"The {spec.name!r} environment isn't built: "
                f"python -m voxy setup {self.backend.name}"
            )
        env = worker_environ(spec, threads=self.threads, rootdir=self.rootdir)
        env.update(self.env_vars)
        return Worker(
            python,
            adapter_path(self.backend.name),
            init=self.init,
            env=env,
            log_path=voxy_cache_dir("logs", f"{self.backend.name}.log"),
        )

    def __call__(self, text: str, *, ref_audio, ref_text: str, **params):
        if self._worker is None:
            self._worker = self._make_worker()
        scratch = voxy_cache_dir("tmp")
        scratch.mkdir(parents=True, exist_ok=True)
        out_path = scratch / f"{self.backend.name}-{uuid.uuid4().hex}.wav"
        try:
            self._worker.call(
                "synthesize",
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                out_path=str(out_path),
                **params,
            )
            return read_wav(out_path)
        finally:
            out_path.unlink(missing_ok=True)

    def close(self) -> None:
        """Stop the worker and free its memory."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None


class LocalCloneSpeechModel(SpeechModel):
    """A local voice-cloning model as a voxy backend.

    Args:
        backend: A ``LocalCloneBackend`` or its name (subclasses fix it).
        generate: The model call (see the module docstring); default: worker.
        chunker: ``text -> [[chunk, ...], ...]``; default ``split_text``.
        gap_ms: Pause between chunks of a paragraph.
        paragraph_gap_ms: Pause between paragraphs.
        trim: Trim each chunk's leading and trailing silence before joining.
        threads: Torch threads of the worker (default ``$VOXY_NUM_THREADS`` or 8).
        rootdir: Data root for environments and weights (default ``$VOXY_DATA_DIR``).
        auto_setup: Build the environment on first use (else raise if missing).
        env_vars: Extra environment variables for the worker.
        init: Keyword arguments of the model's loader (e.g. ``variant='turbo'``).
    """

    backend_name: str = ""

    def __init__(
        self,
        backend: LocalCloneBackend | str | None = None,
        *,
        generate: Callable | None = None,
        chunker: Callable | None = None,
        gap_ms: float = DFLT_GAP_MS,
        paragraph_gap_ms: float = DFLT_PARAGRAPH_GAP_MS,
        trim: bool = True,
        threads: int | None = None,
        rootdir: str | os.PathLike | None = None,
        auto_setup: bool = True,
        env_vars: Mapping[str, str] | None = None,
        device: str = "cpu",
        init: Mapping | None = None,
        **init_kwargs,
    ):
        super().__init__(device=device)
        backend = backend or self.backend_name
        self.backend = (
            LOCAL_CLONE_BACKENDS[backend] if isinstance(backend, str) else backend
        )
        self.name = self.backend.name
        self.init = {**(init or {}), **init_kwargs}
        if device != "cpu":
            self.init.setdefault("device", device)
        self._generate = generate or WorkerGenerate(
            self.backend,
            init=self.init,
            threads=threads,
            rootdir=rootdir,
            env_vars=env_vars,
            auto_setup=auto_setup,
        )
        self.chunker = chunker or partial(split_text, max_chars=self.backend.max_chars)
        self.gap_ms, self.paragraph_gap_ms, self.trim = gap_ms, paragraph_gap_ms, trim

    def clone_voice(
        self,
        audio_input,
        transcript: str | None = None,
        speaker_id: int = REFERENCE_SPEAKER_ID,
        **metadata,
    ) -> VoiceProfile:
        """A reference voice from a clip (path, or WAV bytes) and its transcript.

        Without ``transcript`` the clip is transcribed locally (``voxy.refprep``).
        """
        if isinstance(audio_input, bytes):
            folder = voxy_cache_dir("references")
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"ref-{uuid.uuid4().hex}.wav"
            path.write_bytes(audio_input)
            audio_input = path
        if transcript is None:
            from voxy.refprep import transcribe_text

            transcript = transcribe_text(audio_input)
        return reference_profile(audio_input, transcript, backend=self.name, **metadata)

    def synthesize(self, text, voice=None, **params) -> Speech:
        """Speak ``text`` (any length) in the reference ``voice``.

        ``params`` go to the model (see ``self.backend.params`` for the knobs).
        """
        text = _resolve_text_input(text)
        reference = _reference(voice)
        pieces, sample_rate = [], None
        for p, paragraph in enumerate(self.chunker(text)):
            for c, chunk in enumerate(paragraph):
                samples, sr = self._generate(chunk, **reference, **params)
                samples = np.asarray(samples, dtype=np.float32).reshape(-1)
                if self.trim:
                    samples = trim_silence(samples, sr)
                sample_rate = sample_rate or sr
                if pieces:
                    gap = self.gap_ms if c else self.paragraph_gap_ms
                    pieces.append(silence(gap, sample_rate))
                pieces.append(samples)
        if sample_rate is None:
            raise ValueError("Nothing to say: the text is empty")
        return Speech(
            write_wav(np.concatenate(pieces), sample_rate),
            format="wav",
            backend=self.name,
            voice=str(reference["ref_audio"]),
            sample_rate=sample_rate,
            text=text,
        )

    def close(self) -> None:
        """Free the model's memory (the next call reloads it)."""
        close = getattr(self._generate, "close", None)
        if close:
            close()


class F5TTSSpeechModel(LocalCloneSpeechModel):
    """F5-TTS: flow-matching zero-shot cloning (``params``: nfe_step, cfg_strength, speed...)."""

    backend_name = "f5tts"


class ChatterboxSpeechModel(LocalCloneSpeechModel):
    """Chatterbox: expressive zero-shot cloning (``variant='original'|'turbo'``)."""

    backend_name = "chatterbox"


class VoxCPMSpeechModel(LocalCloneSpeechModel):
    """VoxCPM2: high-similarity cloning (``params``: cfg_value, inference_timesteps, prompt_mode)."""

    backend_name = "voxcpm"


class Qwen3TTSSpeechModel(LocalCloneSpeechModel):
    """Qwen3-TTS Base: stable cloning (``init``: model_id 1.7B or 0.6B)."""

    backend_name = "qwen3tts"


class XTTSSpeechModel(LocalCloneSpeechModel):
    """XTTS-v2 through coqui-tts (weights under the non-commercial Coqui licence).

    The weights' licence must be accepted once: ``accept_licence=True`` or
    ``$VOXY_ACCEPT_COQUI_CPML=1`` (else loading would stop to ask on a terminal).
    """

    backend_name = "xtts"
    LICENCE_ENVVAR = "VOXY_ACCEPT_COQUI_CPML"

    def __init__(self, *args, accept_licence: bool | None = None, **kwargs):
        if accept_licence is None:
            accept_licence = os.environ.get(self.LICENCE_ENVVAR) == "1"
        env_vars = dict(kwargs.pop("env_vars", None) or {})
        if accept_licence:
            env_vars["COQUI_TOS_AGREED"] = "1"
        super().__init__(*args, env_vars=env_vars, **kwargs)
        self.accept_licence = accept_licence

    def synthesize(self, text, voice=None, **params) -> Speech:
        """As ``LocalCloneSpeechModel.synthesize``, once the licence is accepted."""
        if not self.accept_licence and not self.init.get("model_path"):
            raise PermissionError(
                "XTTS-v2 weights are under the Coqui Public Model Licence "
                "(non-commercial, https://coqui.ai/cpml). Accept it with "
                f"accept_licence=True or {self.LICENCE_ENVVAR}=1."
            )
        return super().synthesize(text, voice, **params)
