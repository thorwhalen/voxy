"""
Say the language once, and have each engine get it in its own option.

Engines disagree on how a caller pins the language of a take: Seed Speech on fal takes
``language``, ElevenLabs' newer models take ``language_code``, Qwen3-TTS takes
``language`` spelled out ("English"), OpenAI's voices take none. Left unset, an engine
that detects the language per request can flip: an English sentence carrying a year and
two percentages came back read in Tagalog. So the caller says ``language="en"`` once
(``voxy.text_to_speech(..., language="en")``) and this module maps it, by backend and
model id, the way ``voxy.direction`` maps delivery directions. A model nobody registered
gets ``language`` passed through unchanged, as before this seam existed. A new model only
needs :func:`register_language_option`.

>>> language_option("fal", "fal-ai/bytedance/seed-speech/tts/v2")
LanguageOption(param='language', detects=True, to_value=None)
>>> language_options("elevenlabs", "eleven_flash_v2_5", "en")
{'language_code': 'en'}
>>> language_options("elevenlabs", "eleven_multilingual_v2", "en")  # takes none: nothing sent
{}
>>> language_options("qwen3tts", None, "fr")
{'language': 'French'}
>>> language_options("someone-new", None, "de")  # unregistered: passed through
{'language': 'de'}
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageOption:
    """How a model takes a language.

    ``param``: the synthesis option that pins the language (``None``: the model takes
    none, so a requested language is dropped, with a warning). ``detects``: the model
    guesses the language per request when it is not pinned, so an unpinned call can
    come back in the wrong language. ``to_value``: turns the caller's code ('en') into
    what the model wants ('English'); ``None`` sends it as given.
    """

    param: str | None = None
    detects: bool = False
    to_value: Callable[[str], str] | None = None


class LanguageUnsetWarning(UserWarning):
    """No ``language=`` on a model that guesses the language of each request."""


class LanguageIgnoredWarning(UserWarning):
    """``language=`` asked of a model that takes no language option (dropped)."""


NO_OPTION = LanguageOption()
#: What a model nobody registered gets: ``language`` passed through as given.
PASSTHROUGH = LanguageOption("language")

#: ISO 639-1 codes to the English names some models want ("en" -> "English").
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}


def language_name(code: str) -> str:
    """'English' for 'en' (and 'en-US'); anything else unchanged.

    >>> language_name("en-US"), language_name("English")
    ('English', 'English')
    """
    return LANGUAGE_NAMES.get(code.split("-")[0].lower(), code)


# backend -> [(model-id glob, option)], first match wins; '*' matches a missing model id
_LANGUAGE_OPTIONS: dict[str, list[tuple[str, LanguageOption]]] = {
    "fal": [("fal-ai/bytedance/seed-speech/*", LanguageOption("language", detects=True))],
    # ElevenLabs returns an error when a model cannot enforce a language, so only the
    # models documented to accept ``language_code`` get it; the rest take none.
    "elevenlabs": [
        ("eleven_turbo_v2_5*", LanguageOption("language_code", detects=True)),
        ("eleven_flash_v2_5*", LanguageOption("language_code", detects=True)),
        ("eleven_v3*", LanguageOption("language_code", detects=True)),
        ("*", NO_OPTION),
    ],
    # OpenAI's speech endpoint (through aix/LiteLLM) has no language parameter.
    "aix": [("*", NO_OPTION)],
    "chatterbox": [("*", LanguageOption("language"))],
    "xtts": [("*", LanguageOption("language"))],
    "qwen3tts": [("*", LanguageOption("language", to_value=language_name))],
}


def register_language_option(backend: str, model_glob: str, option: LanguageOption) -> None:
    """Declare how ``backend``'s models matching ``model_glob`` take a language
    (checked before the ones already registered)."""
    _LANGUAGE_OPTIONS.setdefault(backend, []).insert(0, (model_glob, option))


def language_option(backend: str, model_id: str | None = None) -> LanguageOption:
    """The language option of ``backend``'s model ``model_id``: the registered one, else
    :data:`PASSTHROUGH` (``language`` sent as given, as before this seam existed).

    >>> language_option("elevenlabs", "eleven_multilingual_v2").param is None
    True
    """
    for glob, option in _LANGUAGE_OPTIONS.get(backend, ()):
        if fnmatch.fnmatchcase(model_id or "", glob):
            return option
    return PASSTHROUGH


def options_for(option: LanguageOption, language: str | None) -> dict:
    """The synthesis options ``option`` sends for ``language`` (``{}`` for none)."""
    if not language or option.param is None:
        return {}
    return {option.param: option.to_value(language) if option.to_value else language}


def language_options(backend: str, model_id: str | None, language: str | None) -> dict:
    """The synthesis options that pin ``language`` on this model: ``{}`` when no
    language is asked or the model takes none.

    >>> language_options("fal", "fal-ai/bytedance/seed-speech/tts/v2", "en")
    {'language': 'en'}
    >>> language_options("fal", "fal-ai/bytedance/seed-speech/tts/v2", None)
    {}
    """
    return options_for(language_option(backend, model_id), language)
