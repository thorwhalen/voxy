"""
A persistent voice library: clone a voice once, then use it by name.

A voice record (one per person, in ``voxy.stores.voices_store``) holds what voxy
knows about a voice, plus one saved profile per backend:

    {"name": "ada", "aliases": [...], "consent": "...",
     "profiles": {"elevenlabs": {"segment": "<voice_id>", "model_type": "elevenlabs", ...}}}

Only profiles whose ``segment`` is plain data (an id, as for ElevenLabs) can be
saved; a local model's in-memory segment (CSM) cannot.

>>> import tempfile
>>> from voxy.base import VoiceProfile
>>> voices = voices_store(rootdir=tempfile.mkdtemp())
>>> _ = save_voice("ada", VoiceProfile("v1", 999, "elevenlabs", 24000), voices=voices,
...                aliases=["Ada"])
>>> load_voice("ada", voices=voices).segment
'v1'
>>> voices["ada"]["aliases"]
['Ada']
"""

import io
import json
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from voxy import stores
from voxy.base import SpeechModel, VoiceProfile, create_speech_model
from voxy.stores import AUDIO_EXTS, voices_store

RESERVED_RECORD_FIELDS = frozenset({"name", "profiles"})

DFLT_LIBRARY_MODEL_TYPE = "elevenlabs"


def profile_to_dict(profile: VoiceProfile) -> dict:
    """A JSON-ready dict of ``profile``; refuses segments that aren't plain data."""
    record = asdict(profile)
    try:
        json.dumps(record)
    except TypeError as e:
        raise TypeError(
            f"A {profile.model_type!r} profile holds an in-memory segment and can't "
            "be saved to the voice library; only id-based (remote) profiles can."
        ) from e
    return record


def save_voice(
    name: str,
    profile: VoiceProfile,
    *,
    voices: MutableMapping | None = None,
    **record_fields: Any,
) -> dict:
    """Save ``profile`` under ``name`` (merged into any existing record) and return the record.

    ``record_fields`` (e.g. ``aliases=``, ``description=``, ``consent=``) are set on
    the record itself. Aliases must not already name another voice.
    """
    reserved = RESERVED_RECORD_FIELDS & set(record_fields)
    if reserved:
        raise ValueError(f"Record fields {sorted(reserved)} are managed by voxy")
    voices = stores.voices_store() if voices is None else voices
    _check_aliases_free(name, record_fields.get("aliases"), voices)
    record = dict(voices[name]) if name in voices else {"name": name}
    record.update(record_fields)
    entry = profile_to_dict(profile)
    entry["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["profiles"] = {**record.get("profiles", {}), profile.model_type: entry}
    voices[name] = record
    return record


def find_voice(name: str, *, voices: Mapping | None = None) -> str | None:
    """The library key of the voice called ``name`` (case-insensitive, aliases too).

    >>> lib = {"cora": {"aliases": ["Cora", "Coco"]}, "vanessa": {"aliases": ["Ness"]}}
    >>> find_voice("coco", voices=lib), find_voice("Vanessa", voices=lib), find_voice("x", voices=lib)
    ('cora', 'vanessa', None)
    """
    voices = stores.voices_store() if voices is None else voices
    if not isinstance(name, str):
        return None
    keys = list(voices)  # the stored spelling, even on case-insensitive disks
    wanted = name.casefold()
    for match in (lambda k: k == name, lambda k: k.casefold() == wanted):
        if found := next((k for k in keys if match(k)), None):
            return found
    for key in keys:
        if any(
            isinstance(a, str) and a.casefold() == wanted
            for a in voices[key].get("aliases", [])
        ):
            return key
    return None


def _check_aliases_free(name: str, aliases, voices: Mapping) -> None:
    """Refuse aliases that already name, or alias, another voice."""
    for alias in aliases or []:
        owner = find_voice(alias, voices=voices)
        if owner is not None and owner.casefold() != name.casefold():
            raise ValueError(f"Alias {alias!r} already refers to voice {owner!r}")


def profile_from_record(
    record: Mapping, model_type: str | None = None, *, prefer: str | None = None
) -> VoiceProfile:
    """The record's saved ``model_type`` profile.

    With ``model_type=None``: the ``prefer`` profile if the record has one, else
    the record's ``default_backend``, else its first.
    """
    profiles = record.get("profiles", {})
    if not profiles:
        raise KeyError(f"Voice {record.get('name')!r} has no saved profiles")
    if model_type is None:
        model_type = next(
            (b for b in (prefer, record.get("default_backend")) if b in profiles),
            next(iter(profiles)),
        )
    if model_type not in profiles:
        raise KeyError(
            f"Voice {record.get('name')!r} has no {model_type!r} profile "
            f"(has: {', '.join(profiles)})"
        )
    fields = {
        k: v
        for k, v in profiles[model_type].items()
        if k in VoiceProfile.__annotations__
    }
    return VoiceProfile(**fields)


def load_voice(
    name: str,
    *,
    model_type: str | None = DFLT_LIBRARY_MODEL_TYPE,
    voices: Mapping | None = None,
) -> VoiceProfile:
    """The saved ``model_type`` profile of the voice called ``name`` (or an alias).

    ``model_type=None`` takes the voice's first saved profile.
    """
    voices = stores.voices_store() if voices is None else voices
    key = find_voice(name, voices=voices)
    if key is None:
        raise KeyError(f"No voice named {name!r} (have: {', '.join(voices)})")
    return profile_from_record(voices[key], model_type)


def _refuse_existing(name, voices, backend: str, overwrite: bool) -> None:
    """Refuse to replace a saved ``backend`` profile of ``name`` unless asked to."""
    if overwrite:
        return
    voices = stores.voices_store() if voices is None else voices
    key = find_voice(name, voices=voices)
    if key is not None and backend in voices[key].get("profiles", {}):
        raise ValueError(
            f"Voice {key!r} already has a {backend!r} profile; pass overwrite=True "
            "to replace it"
        )


def _is_sample_key(key: str) -> bool:
    """Top-level audio files only (no notes, no subfolders).

    >>> [k for k in ["a.wav", "b.MP3", "notes.txt", "sub/c.wav"] if _is_sample_key(k)]
    ['a.wav', 'b.MP3']
    """
    k = key.replace("\\", "/")
    return "/" not in k and k.lower().endswith(AUDIO_EXTS)


def _named_file(key: str, content: bytes) -> io.BytesIO:
    f = io.BytesIO(content)
    f.name = key.replace("\\", "/").rsplit("/", 1)[-1]
    return f


def clone_from_samples(
    name: str,
    *,
    model: SpeechModel | None = None,
    model_type: str = DFLT_LIBRARY_MODEL_TYPE,
    samples: Mapping[str, bytes] | None = None,
    voices: MutableMapping | None = None,
    record_fields: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    **clone_kwargs: Any,
) -> VoiceProfile:
    """Clone the voice ``name`` from its stored samples and save the profile.

    Args:
        name: Voice name: the key in the samples and voices stores.
        model: A speech model; defaults to ``create_speech_model(model_type)``.
        model_type: Backend used when ``model`` is not given.
        samples: ``filename -> bytes``; defaults to ``samples_store(name)``. Only
            its top-level audio files are uploaded.
        voices: Voice records store; defaults to ``voices_store()``.
        record_fields: Extra fields for the voice record (aliases, consent...).
        overwrite: Replace an existing profile of this backend for ``name``
            (otherwise refused before anything is uploaded).
        **clone_kwargs: Passed to ``model.clone_voice`` (e.g. ``labels=``,
            ``remove_background_noise=``). ``name=`` defaults to ``name``.
    """
    _refuse_existing(
        name, voices, model_type if model is None else model.name, overwrite
    )
    samples = stores.samples_store(name) if samples is None else samples
    keys = sorted(k for k in samples if _is_sample_key(k))
    if not keys:
        raise ValueError(f"No audio samples stored for voice {name!r}")
    files = [_named_file(k, samples[k]) for k in keys]
    model = model or create_speech_model(model_type)
    clone_kwargs.setdefault("name", name)
    profile = model.clone_voice(files, **clone_kwargs)
    save_voice(name, profile, voices=voices, **dict(record_fields or {}))
    return profile


def design_from_description(
    name: str,
    description: str,
    *,
    preview=None,
    model: SpeechModel | None = None,
    model_type: str = DFLT_LIBRARY_MODEL_TYPE,
    voices: MutableMapping | None = None,
    record_fields: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    **design_kwargs: Any,
) -> VoiceProfile:
    """Design a new voice from a text ``description`` and save it as ``name``.

    Args:
        name: Library name for the voice.
        description: What it sounds like (age, accent, tone, pace, character).
        preview: The chosen preview (or its id) from the model's
            ``design_voice_previews``; if omitted, the first generated one.
        model: A speech model that can design voices (default: ``model_type``'s).
        record_fields: Extra fields for the voice record (aliases...).
        overwrite: Replace an existing profile of this backend for ``name``
            (otherwise refused before any paid call).
        **design_kwargs: Passed to ``model.design_voice`` (``labels=``, ``seed=``...).
    """
    stores.check_voice_name(name)
    _refuse_existing(
        name, voices, model_type if model is None else model.name, overwrite
    )
    model = model or create_speech_model(model_type)
    design_kwargs.setdefault("name", name)
    profile = model.design_voice(description, preview=preview, **design_kwargs)
    fields = {"description": description, **dict(record_fields or {})}
    save_voice(name, profile, voices=voices, **fields)
    return profile
