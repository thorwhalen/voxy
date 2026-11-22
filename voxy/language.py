"""
Say the language once, and have each engine get it in its own option.

Engines disagree on how a caller pins the language of a take: Seed Speech on fal takes
``language``, ElevenLabs' newer models take ``language_code``, OpenAI's voices take
none. Left unset, an engine that detects the language per request can flip: an English
sentence carrying a year and two percentages came back read in Tagalog. So the caller
says ``language="en"`` once (``voxy.text_to_speech(..., language="en")``) and this
module maps it, by backend and model id, the way ``voxy.direction`` maps delivery
directions. A new model only needs :func:`register_language_option`.

>>> language_option("fal", "fal-ai/bytedance/seed-speech/tts/v2")
LanguageOption(param='language', detects=True)
>>> language_options("elevenlabs", "eleven_flash_v2_5", "en")
{'language_code': 'en'}
>>> language_options("say", None, "en")  # no option: nothing is sent
{}
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageOption:
    """How a model takes a language.

    ``param``: the synthesis option that pins the language (``None``: the model takes
    none, so a requested language is ignored). ``detects``: the model guesses the
    language per request when it is not pinned, so an unpinned call can come back in
    the wrong language.
    """

    param: str | None = None
    detects: bool = False


NO_OPTION = LanguageOption()

# backend -> [(model-id glob, option)], first match wins; '*' matches a missing model id
_LANGUAGE_OPTIONS: dict[str, list[tuple[str, LanguageOption]]] = {
    "fal": [("fal-ai/bytedance/seed-speech/*", LanguageOption("language", detects=True))],
    # ElevenLabs returns an error when a model cannot enforce a language, so only the
    # models documented to accept ``language_code`` get it.
    "elevenlabs": [
        ("eleven_turbo_v2_5*", LanguageOption("language_code", detects=True)),
        ("eleven_flash_v2_5*", LanguageOption("language_code", detects=True)),
        ("eleven_v3*", LanguageOption("language_code", detects=True)),
    ],
}


def register_language_option(backend: str, model_glob: str, option: LanguageOption) -> None:
    """Declare how ``backend``'s models matching ``model_glob`` take a language
    (checked before the ones already registered)."""
    _LANGUAGE_OPTIONS.setdefault(backend, []).insert(0, (model_glob, option))


def language_option(backend: str, model_id: str | None = None) -> LanguageOption:
    """The language option of ``backend``'s model ``model_id`` (:data:`NO_OPTION` when unknown).

    >>> language_option("elevenlabs", "eleven_multilingual_v2").param is None
    True
    """
    for glob, option in _LANGUAGE_OPTIONS.get(backend, ()):
        if fnmatch.fnmatchcase(model_id or "", glob):
            return option
    return NO_OPTION


def language_options(backend: str, model_id: str | None, language: str | None) -> dict:
    """The synthesis options that pin ``language`` on this model: ``{}`` when no
    language is asked or the model takes none.

    >>> language_options("fal", "fal-ai/bytedance/seed-speech/tts/v2", "en")
    {'language': 'en'}
    >>> language_options("fal", "fal-ai/bytedance/seed-speech/tts/v2", None)
    {}
    """
    param = language_option(backend, model_id).param
    return {param: language} if language and param else {}
