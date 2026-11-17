"""
A content-addressed cache of synthesized speech, so re-rendering unchanged text is free.

The key covers everything that reaches the audio: the backend, its model settings, the
voice, the text and the synthesis options. A store is any ``MutableMapping`` from key to
:class:`~voxy.base.Speech`; :class:`SpeechCache` is the default, a folder of audio files
with a small JSON record each.

>>> import tempfile
>>> cache = SpeechCache(tempfile.mkdtemp())
>>> calls = []
>>> class Fake:
...     name, model_id = "fake", "m1"
...     def synthesize(self, text, voice=None, **kw):
...         from voxy.base import Speech
...         calls.append(text)
...         return Speech(text.encode(), format="wav", backend="fake", voice=voice, text=text)
>>> synth = cached_synthesizer(Fake(), cache)
>>> synth("hi", "v").cached, synth("hi", "v").cached, len(calls)
(False, True, 1)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from collections.abc import Callable, Iterator, MutableMapping

from voxy.base import Speech

SPEECH_CACHE_ENVVAR = "VOXY_SPEECH_CACHE_DIR"
#: Bump when the meaning of a key changes, so old entries stop matching.
KEY_VERSION = 1
_MODEL_ATTRS = ("model_id", "model", "quality")


def default_speech_cache_dir() -> Path:
    """``$VOXY_SPEECH_CACHE_DIR``, else ``$XDG_CACHE_HOME/voxy/speech``, else ``~/.cache/voxy/speech``."""
    if os.environ.get(SPEECH_CACHE_ENVVAR):
        return Path(os.environ[SPEECH_CACHE_ENVVAR]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "voxy" / "speech"


def speech_key(model, text: str, voice, options: dict) -> str:
    """The cache key: a SHA-256 over backend, model settings, voice, text and options."""
    model_settings = {a: getattr(model, a, None) for a in _MODEL_ATTRS}
    voice_id = getattr(voice, "segment", None) or getattr(voice, "name", None) or voice
    payload = {
        "v": KEY_VERSION,
        "backend": getattr(model, "name", type(model).__name__),
        "model": model_settings,
        "voice": str(voice_id) if voice_id is not None else None,
        "text": text,
        "options": options,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class SpeechCache(MutableMapping):
    """Speech on disk: ``<key>.<format>`` audio plus ``<key>.json`` (everything else)."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root).expanduser() if root else default_speech_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def _meta(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def __getitem__(self, key: str) -> Speech:
        try:
            meta = json.loads(self._meta(key).read_text())
            audio = (self.root / f"{key}.{meta['format']}").read_bytes()
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            raise KeyError(key) from None
        return Speech(audio=audio, **{k: v for k, v in meta.items() if k != "cached"})

    def __setitem__(self, key: str, speech: Speech) -> None:
        meta = {k: v for k, v in asdict(speech).items() if k not in ("audio", "cached")}
        (self.root / f"{key}.{speech.format}").write_bytes(speech.audio)
        self._meta(key).write_text(
            json.dumps(meta)
        )  # written last: its presence means complete

    def __delitem__(self, key: str) -> None:
        meta = self[key]
        (self.root / f"{key}.{meta.format}").unlink(missing_ok=True)
        self._meta(key).unlink(missing_ok=True)

    def __iter__(self) -> Iterator[str]:
        return (p.stem for p in self.root.glob("*.json"))

    def __len__(self) -> int:
        return sum(1 for _ in self)


def resolve_cache(cache) -> MutableMapping | None:
    """``None``/``False``: no cache; ``True``: the default folder; a path: that folder;
    a mapping: used as is."""
    if cache is None or cache is False:
        return None
    if cache is True:
        return SpeechCache()
    if isinstance(cache, (str, os.PathLike)):
        return SpeechCache(cache)
    return cache


class TakeRejected(RuntimeError):
    """An engine kept returning takes that failed validation (never cached)."""


#: Speaking-rate bounds (characters per second) for a take of a long text. Speech
#: runs about 12-17 chars/s; a take far slower is noise, breath or silence (seen: a
#: 177-character request answered with 29 s of breathing), far faster is a truncation.
MIN_CHARS_PER_S, MAX_CHARS_PER_S, MIN_CHECKED_CHARS = 7.0, 30.0, 60


def speaking_rate_check(
    speech: Speech,
    text: str,
    *,
    min_chars_per_s: float = MIN_CHARS_PER_S,
    max_chars_per_s: float = MAX_CHARS_PER_S,
    min_chars: int = MIN_CHECKED_CHARS,
) -> str | None:
    """A problem description if ``speech`` cannot be ``text`` spoken, else ``None``.

    Only texts of ``min_chars`` or more are judged (a short line's lead-in and
    tail dominate its duration). Needs no model: the duration comes from the audio.
    """
    if len(text) < min_chars:
        return None
    from voxy.pauses import decode_speech

    samples, rate = decode_speech(speech)
    seconds = len(samples) / rate if rate else 0.0
    cps = len(text) / seconds if seconds else float("inf")
    if cps < min_chars_per_s:
        return f"{seconds:.1f}s for {len(text)} chars ({cps:.1f}/s): too slow to be this text (noise, breath, silence?)"
    if cps > max_chars_per_s:
        return (
            f"{seconds:.1f}s for {len(text)} chars ({cps:.1f}/s): too fast (truncated?)"
        )
    return None


def validated(
    synthesize: Callable[..., Speech], validate, *, retries: int = 2
) -> Callable[..., Speech]:
    """``synthesize`` that re-asks (up to ``retries`` times) while ``validate(speech, text)``
    reports a problem, then raises :class:`TakeRejected`."""

    def synth(text, voice=None, **options) -> Speech:
        problems = []
        for _ in range(retries + 1):
            speech = synthesize(text, voice, **options)
            problem = validate(speech, text)
            if problem is None:
                return speech
            problems.append(problem)
        raise TakeRejected(
            f"{len(problems)} takes of {text[:60]!r} rejected: {problems}"
        )

    return synth


def cached_synthesizer(
    model, cache: MutableMapping, *, validate=None, retries: int = 2
) -> Callable[..., Speech]:
    """``model.synthesize``, served from ``cache`` when the same request was made before.

    A served result has ``cached=True``, so callers can tell spend from reuse.
    ``validate(speech, text) -> problem | None`` (e.g. :func:`speaking_rate_check`)
    runs on every new take before it is cached: a bad take is re-asked up to
    ``retries`` times and never stored, so a cache cannot freeze a broken take.
    """
    fresh = (
        validated(model.synthesize, validate, retries=retries)
        if validate
        else model.synthesize
    )

    def synthesize(text, voice=None, **options) -> Speech:
        key = speech_key(model, text, voice, options)
        try:
            return replace(cache[key], cached=True)
        except KeyError:
            speech = fresh(text, voice, **options)
            cache[key] = speech
            return speech

    return synthesize
