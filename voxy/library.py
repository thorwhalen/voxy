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

from voxy.base import SpeechModel, VoiceProfile, create_speech_model
from voxy.stores import samples_store, voices_store

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
    the record itself.
    """
    voices = voices_store() if voices is None else voices
    record = dict(voices[name]) if name in voices else {"name": name}
    record.update(record_fields)
    entry = profile_to_dict(profile)
    entry["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["profiles"] = {**record.get("profiles", {}), profile.model_type: entry}
    voices[name] = record
    return record


def load_voice(
    name: str,
    *,
    model_type: str = DFLT_LIBRARY_MODEL_TYPE,
    voices: Mapping | None = None,
) -> VoiceProfile:
    """The saved ``model_type`` profile of the voice called ``name``."""
    voices = voices_store() if voices is None else voices
    if name not in voices:
        raise KeyError(f"No voice named {name!r} (have: {', '.join(voices)})")
    profiles = voices[name].get("profiles", {})
    if model_type not in profiles:
        raise KeyError(
            f"Voice {name!r} has no {model_type!r} profile (has: {', '.join(profiles)})"
        )
    fields = {
        k: v
        for k, v in profiles[model_type].items()
        if k in VoiceProfile.__annotations__
    }
    return VoiceProfile(**fields)


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
    **clone_kwargs: Any,
) -> VoiceProfile:
    """Clone the voice ``name`` from its stored samples and save the profile.

    Args:
        name: Voice name: the key in the samples and voices stores.
        model: A speech model; defaults to ``create_speech_model(model_type)``.
        model_type: Backend used when ``model`` is not given.
        samples: ``filename -> bytes``; defaults to ``samples_store(name)``.
        voices: Voice records store; defaults to ``voices_store()``.
        record_fields: Extra fields for the voice record (aliases, consent...).
        **clone_kwargs: Passed to ``model.clone_voice`` (e.g. ``labels=``,
            ``remove_background_noise=``). ``name=`` defaults to ``name``.
    """
    model = model or create_speech_model(model_type)
    samples = samples_store(name) if samples is None else samples
    files = [_named_file(k, samples[k]) for k in sorted(samples)]
    if not files:
        raise ValueError(f"No samples stored for voice {name!r}")
    clone_kwargs.setdefault("name", name)
    profile = model.clone_voice(files, **clone_kwargs)
    save_voice(name, profile, voices=voices, **dict(record_fields or {}))
    return profile
