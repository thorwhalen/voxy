"""
Voice generation in one call, whatever the service behind it.

    >>> import voxy                                                    # doctest: +SKIP
    >>> voxy.text_to_speech("Hello!", voice="cora").save("hi.mp3")    # doctest: +SKIP
    >>> voxy.text_to_speech("Hello!", voice="Daniel", backend="say")  # doctest: +SKIP
    >>> voxy.list_voices()               # the library: our named voices  # doctest: +SKIP
    >>> voxy.list_voices("elevenlabs")   # a backend's own voices          # doctest: +SKIP
    >>> voxy.voice_id("coco")            # 'rcrK...' (aliases work)        # doctest: +SKIP

How ``voice`` is understood, first match wins:

1. a ``VoiceProfile``: used as is, on its own backend;
2. a name or alias in the voice library (``voxy.voices_store()``): that voice's
   saved profile for ``backend`` (or, if ``backend`` is not given, its first one);
3. anything else: the backend's own voice id or name (e.g. 'nova', 'Daniel');
4. ``None``: the backend's default voice, if it has one.

Pass ``use_library=False`` to reach a backend voice whose name is also a library
name or alias. With no ``backend``, a library voice uses its profile for the
default backend if it has one, else its ``default_backend``, else its first.

Backends are entries of ``voxy.speech_model_factories``; add one with
``voxy.register_speech_model``. The default backend is ``$VOXY_TTS_BACKEND``
(read at call time), else 'elevenlabs'.
"""

import os
from collections.abc import Mapping, MutableMapping

from voxy import stores
from voxy.base import (
    Speech,
    SpeechModel,
    VoiceInfo,
    VoiceProfile,
    canonical_backend,
    create_speech_model,
)
from voxy.library import find_voice, profile_from_record

VOXY_TTS_BACKEND_ENVVAR = "VOXY_TTS_BACKEND"
FALLBACK_TTS_BACKEND = "elevenlabs"
LIBRARY_BACKEND = "voxy"  # the ``backend`` of library entries in ``list_voices()``

#: One model per backend, built on first use (clients are reusable).
_models: dict[str, SpeechModel] = {}


def dflt_tts_backend() -> str:
    """``$VOXY_TTS_BACKEND``, else 'elevenlabs' (read on every call)."""
    return canonical_backend(
        os.environ.get(VOXY_TTS_BACKEND_ENVVAR) or FALLBACK_TTS_BACKEND
    )


def get_speech_model(
    backend: str | None = None, *, models: MutableMapping | None = None
) -> SpeechModel:
    """The (cached) model for ``backend``; ``models`` replaces the shared cache."""
    cache = _models if models is None else models
    key = canonical_backend(backend or dflt_tts_backend())
    if key not in cache:
        cache[key] = create_speech_model(key)
    return cache[key]


def clear_speech_models(models: MutableMapping | None = None) -> None:
    """Forget cached models (e.g. after changing keys or settings)."""
    (_models if models is None else models).clear()


def resolve_voice(
    voice: VoiceProfile | str | None,
    *,
    backend: str | None = None,
    voices: Mapping | None = None,
    use_library: bool = True,
) -> tuple[str, VoiceProfile | str | None]:
    """``(backend, voice)`` to synthesize with (see the module docstring for the rules).

    >>> lib = {"cora": {"name": "cora", "aliases": ["Coco"], "profiles": {
    ...     "elevenlabs": {"segment": "v1", "speaker_id": 1, "model_type": "elevenlabs",
    ...                    "sample_rate": 24000}}}}
    >>> b, v = resolve_voice("coco", voices=lib)
    >>> b, v.segment
    ('elevenlabs', 'v1')
    >>> resolve_voice("Daniel", backend="say", voices=lib)
    ('say', 'Daniel')
    """
    backend = canonical_backend(backend) if backend else None
    if isinstance(voice, VoiceProfile):
        profile_backend = canonical_backend(voice.model_type)
        if backend is not None and backend != profile_backend:
            raise ValueError(
                f"A {voice.model_type!r} voice can't be used with backend {backend!r}"
            )
        return profile_backend, voice
    if use_library and isinstance(voice, str):
        voices = stores.voices_store() if voices is None else voices
        key = find_voice(voice, voices=voices)
        if key is not None:
            profile = profile_from_record(
                voices[key], backend, prefer=dflt_tts_backend()
            )
            return canonical_backend(profile.model_type), profile
    return backend or dflt_tts_backend(), voice


def text_to_speech(
    text: str,
    voice: VoiceProfile | str | None = None,
    *,
    backend: str | None = None,
    output_path: str | None = None,
    voices: Mapping | None = None,
    model: SpeechModel | None = None,
    models: MutableMapping | None = None,
    use_library: bool = True,
    **kwargs,
) -> Speech:
    """Speak ``text`` in ``voice`` and return the encoded audio (``.save(path)``).

    Args:
        text: What to say.
        voice: A library name or alias ('cora', 'Coco'), a backend's own voice
            ('nova', 'Daniel', an ElevenLabs id), a ``VoiceProfile``, or None.
        backend: Service to use ('elevenlabs', 'say', 'aix', 'fal', 'csm', or
            any registered). Inferred from library voices.
        output_path: Also save the audio there.
        voices: Voice library store (default ``voxy.voices_store()``).
        model: A ready model to use (its backend is then the backend).
        models: Model cache to use instead of the shared one.
        use_library: Look ``voice`` up in the library first (False: always
            the backend's own voice of that name).
        **kwargs: Backend-specific options (e.g. ``output_format=`` for
            ElevenLabs, ``speed=`` for aix, ``quality=`` for fal).
    """
    if model is not None:
        backend = backend or model.name
    backend, voice = resolve_voice(
        voice, backend=backend, voices=voices, use_library=use_library
    )
    model = model or get_speech_model(backend, models=models)
    if voice is None:
        voice = model.dflt_voice
    speech = model.synthesize(text, voice, **kwargs)
    if output_path:
        speech.save(output_path)
    return speech


def list_voices(
    backend: str | None = None,
    *,
    voices: Mapping | None = None,
    model: SpeechModel | None = None,
    models: MutableMapping | None = None,
    **kwargs,
) -> list[VoiceInfo]:
    """Our named voices (``backend=None``), or the voices a backend offers.

    >>> lib = {"cora": {"name": "cora", "aliases": ["Coco"], "description": "d",
    ...                 "profiles": {"elevenlabs": {"segment": "v1"}}}}
    >>> [(v.name, v.labels["backends"]) for v in list_voices(voices=lib)]
    [('cora', ['elevenlabs'])]
    """
    if backend is None and model is None:
        voices = stores.voices_store() if voices is None else voices
        return [
            VoiceInfo(
                voice_id=key,
                name=key,
                backend=LIBRARY_BACKEND,
                description=voices[key].get("description", ""),
                labels={
                    "aliases": list(voices[key].get("aliases", [])),
                    "backends": list(voices[key].get("profiles", {})),
                },
            )
            for key in sorted(voices)
        ]
    return (model or get_speech_model(backend, models=models)).list_voices(**kwargs)


def voice_id(
    name: str, *, backend: str = "elevenlabs", voices: Mapping | None = None
) -> str:
    """The provider's id for library voice ``name`` (for code that calls a provider).

    >>> lib = {"cora": {"name": "cora", "aliases": ["Coco"], "profiles": {
    ...     "elevenlabs": {"segment": "v1", "speaker_id": 1, "model_type": "elevenlabs",
    ...                    "sample_rate": 24000}}}}
    >>> voice_id("Coco", voices=lib)
    'v1'
    """
    voices = stores.voices_store() if voices is None else voices
    key = find_voice(name, voices=voices)
    if key is None:
        raise KeyError(f"No voice named {name!r} (have: {', '.join(sorted(voices))})")
    return str(profile_from_record(voices[key], canonical_backend(backend)).segment)
