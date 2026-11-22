"""The requests one model needs for a directed text: the text and options of each, and the
silence to splice after it, without synthesizing anything.

A caller that sends its own requests (its own client, cache key and billing) still needs
voxy's knowledge of how a model takes a directed line: its markup dialect
(:mod:`voxy.direction`) and the longest pause it renders itself (:mod:`voxy.pauses`).
:func:`plan_requests` combines the two, as ``voxy.text_to_speech`` does before speaking.

>>> [(r.text, r.options, r.pause_s) for r in plan_requests('Hello. <break time="5s"/> [whispers] Bye.', "elevenlabs", "eleven_v3")]
[('Hello.', {}, 5.0), ('[whispers] Bye.', {}, 0.0)]
>>> [(r.text, r.pause_s) for r in plan_requests('A. <break time="1s"/> B.', "elevenlabs", "eleven_multilingual_v2")]
[('A. <break time="1s" /> B.', 0.0)]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from voxy.direction import Dialect, dialect_for, plan_directions
from voxy.language import LanguageOption, options_for
from voxy.language import language_option as lookup_language_option
from voxy.pauses import native_break_for, plan_sections


@dataclass(frozen=True)
class Request:
    """One request to send: its ``text`` (SSML breaks the model renders itself kept
    inline), the synthesis ``options`` carrying its direction, and the ``pause_s`` of
    silence to splice after its audio. A request with no text is a pause alone."""

    text: str
    options: dict = field(default_factory=dict)
    pause_s: float = 0.0


def plan_requests(
    text: str,
    backend: str,
    model_id: str | None = None,
    *,
    instruction: str | None = None,
    dialect: Dialect | None = None,
    native_break_s: float | None = None,
    language: str | None = None,
    language_option: LanguageOption | None = None,
) -> list[Request]:
    """The requests that speak ``text`` on ``backend``'s model ``model_id``, in order.

    ``text`` may hold delivery directions (``[dryly]``, ``[beat]``) and SSML
    ``<break>``s; ``instruction`` is a direction for the whole text. The model's
    dialect (:func:`voxy.direction.dialect_for`) decides how directions are sent, and
    the longest break it renders (:func:`voxy.pauses.native_break_for`) decides which
    breaks stay in the text and which become silence between requests. ``dialect`` and
    ``native_break_s`` override the two lookups (a model object passes its own). Pass
    the model id the caller will send: an unknown model is planned conservatively (plain
    text, every break cut into silence). ``language`` ('en') is added to every request
    in the model's own option (:func:`voxy.language.language_options`), or not at all
    on a model that takes none. ``language_option`` overrides that lookup, as
    ``dialect`` does (a model object passes its own ``language_option()``).

    >>> plan_requests("Hi.", "fal", "fal-ai/bytedance/seed-speech/tts/v2", language="en")[0].options
    {'language': 'en'}
    """
    dialect = dialect_for(backend, model_id) if dialect is None else dialect
    native = (
        native_break_for(backend, model_id)
        if native_break_s is None
        else native_break_s
    )
    option = (
        lookup_language_option(backend, model_id)
        if language_option is None
        else language_option
    )
    pinned = options_for(option, language)
    return [
        Request(section.text, {**pinned, **part.options}, section.pause_s)
        for part in plan_directions(text, dialect, instruction=instruction)
        for section in plan_sections(part.text, native_break_s=native)
    ]
