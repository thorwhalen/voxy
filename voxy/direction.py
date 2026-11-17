"""
Delivery directions: one way to write *how* a line is said, rendered for whatever voice says it.

A script marks delivery inline, in the bracket notation ElevenLabs' v3 models read:
``[dryly] We all believe in agility. [beat] But then...``. That notation is voxy's
authoring form for any backend. Each backend has a **dialect**, the way it takes
direction, and :func:`plan_directions` turns a directed text into what that dialect
accepts:

=================  ===========================================================
dialect            what happens to ``[dryly]``, ``[beat]``, ``[laughs]``
=================  ===========================================================
``inline-tags``    sent as written: the model performs them (eleven_v3, eleven_v4)
``instruction``    the text is cut where the direction changes; each part is sent
                   with a natural-language instruction (Seed Speech's
                   ``voice_instruction``, OpenAI's ``instructions``) built from its
                   tags; pause tags become real silence
``plain``          tags removed, pause tags become real silence (every other model)
=================  ===========================================================

Which dialect a backend speaks is :func:`dialect_for` (``backend``, ``model_id``), a
registry you can extend with :func:`register_dialect`; ``SpeechModel.markup_dialect``
asks it. The vocabulary (what each tag means as an instruction, and how long a pause
tag is) is :data:`DIRECTIONS`; an unknown tag still becomes an instruction
("Deliver this <tag>."), so writers are not limited to the table.

>>> t = "[dryly] We all agree. [beat] Then we disagree."
>>> [(d.text, d.options) for d in plan_directions(t, PLAIN)]
[('We all agree. <break time="0.45s" /> Then we disagree.', {})]
>>> [(d.text, d.options) for d in plan_directions(t, INLINE)]
[('[dryly] We all agree. [beat] Then we disagree.', {})]
>>> for d in plan_directions(t, instruction_dialect("voice_instruction")):
...     print(repr(d.text), d.options)
'We all agree. <break time="0.45s" /> Then we disagree.' {'voice_instruction': 'Delivery: dry, understated, a little ironic.'}
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping

#: An inline direction: ``[word]`` or ``[a few words]``, lowercase, as eleven_v3 reads them.
DIRECTION_RE = re.compile(r"\[([a-z][a-z \-]{1,24})\]")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


@dataclass(frozen=True)
class Direction:
    """What one tag asks for: a ``pause_s`` of silence, or an ``instruction`` phrase."""

    kind: str  # 'pause' | 'delivery' | 'nonverbal'
    instruction: str = ""
    pause_s: float = 0.0


_D, _N = "delivery", "nonverbal"
#: The vocabulary: tag -> meaning. Pause tags are silence everywhere but on an
#: inline-tags model (which performs them itself); the rest become instructions.
DIRECTIONS: Mapping[str, Direction] = MappingProxyType(
    {
        "pause": Direction("pause", pause_s=0.7),
        "short pause": Direction("pause", pause_s=0.3),
        "long pause": Direction("pause", pause_s=1.4),
        "beat": Direction("pause", pause_s=0.45),
        "dryly": Direction(_D, "dry, understated, a little ironic"),
        "wryly": Direction(_D, "wry, with a knowing half-smile"),
        "deadpan": Direction(_D, "deadpan: flat, letting the line land by itself"),
        "mock-solemn": Direction(_D, "mock-solemn, pretending to be grave"),
        "emphatic": Direction(_D, "emphatic, leaning on the key words"),
        "earnest": Direction(_D, "earnest and sincere"),
        "thoughtful": Direction(_D, "thoughtful and unhurried, thinking it through"),
        "curious": Direction(_D, "curious, genuinely asking"),
        "amused": Direction(_D, "amused, smiling"),
        "sarcastic": Direction(_D, "lightly sarcastic"),
        "excited": Direction(_D, "excited, with energy"),
        "incredulous": Direction(_D, "incredulous"),
        "serious": Direction(_D, "serious and grounded"),
        "warmly": Direction(_D, "warm"),
        "firmly": Direction(_D, "firm"),
        "quietly": Direction(_D, "quieter, softer"),
        "softly": Direction(_D, "soft"),
        "slowly": Direction(_D, "slow and deliberate"),
        "rushed": Direction(_D, "quick, a little rushed"),
        "whispers": Direction(_D, "whispered"),
        "laughs": Direction(_N, "with a laugh in the voice"),
        "laughs softly": Direction(_N, "with a soft laugh in the voice"),
        "chuckles": Direction(_N, "with a chuckle in the voice"),
        "sighs": Direction(_N, "with a sigh"),
    }
)


def direction(tag: str) -> Direction:
    """The meaning of ``tag`` (unknown tags read as a delivery instruction).

    >>> direction("beat").pause_s, direction("tenderly").instruction
    (0.45, 'tenderly')
    """
    return DIRECTIONS.get(tag) or Direction(_D, tag)


def directions(text: str) -> list[str]:
    """The inline direction tags in ``text``, in order.

    >>> directions("[dryly] Fine. [beat] Really.")
    ['dryly', 'beat']
    """
    return DIRECTION_RE.findall(text)


def strip_directions(text: str) -> str:
    """``text`` with its direction tags removed and spacing tidied.

    >>> strip_directions("[wryly] I have met , [beat] some.")
    'I have met, some.'
    """
    return _tidy(DIRECTION_RE.sub(" ", text))


def _tidy(text: str) -> str:
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _SPACES.sub(" ", text)).strip()


@dataclass(frozen=True)
class Dialect:
    """How a backend takes direction.

    ``inline_tags``: the model performs ``[tags]`` itself, so the text is sent as written.
    ``instruction_param``: the synthesis option that takes a natural-language delivery
    instruction (``None``: the model takes none).
    """

    name: str
    inline_tags: bool = False
    instruction_param: str | None = None


INLINE = Dialect("inline-tags", inline_tags=True)
PLAIN = Dialect("plain")


def instruction_dialect(param: str) -> Dialect:
    """A dialect taking a natural-language instruction in the option ``param``."""
    return Dialect(f"instruction:{param}", instruction_param=param)


# backend -> [(model-id glob, dialect)], first match wins; '*' matches a missing model id
_DIALECTS: dict[str, list[tuple[str, Dialect]]] = {
    "elevenlabs": [("eleven_v3*", INLINE), ("eleven_v4*", INLINE)],
    "fal": [
        ("fal-ai/bytedance/seed-speech/*", instruction_dialect("voice_instruction"))
    ],
    "aix": [("*gpt-4o-mini-tts*", instruction_dialect("instructions"))],
}


def register_dialect(backend: str, model_glob: str, dialect: Dialect) -> None:
    """Declare that ``backend``'s models matching ``model_glob`` speak ``dialect``
    (checked before the ones already registered)."""
    _DIALECTS.setdefault(backend, []).insert(0, (model_glob, dialect))


def dialect_for(backend: str, model_id: str | None = None) -> Dialect:
    """The dialect of ``backend``'s model ``model_id`` (:data:`PLAIN` when unknown).

    >>> dialect_for("elevenlabs", "eleven_v3").name, dialect_for("elevenlabs", "eleven_multilingual_v2").name
    ('inline-tags', 'plain')
    >>> dialect_for("fal", "fal-ai/bytedance/seed-speech/tts/v2").instruction_param
    'voice_instruction'
    """
    for glob, dialect in _DIALECTS.get(backend, ()):
        if fnmatch.fnmatchcase(model_id or "", glob):
            return dialect
    return PLAIN


@dataclass(frozen=True)
class Directed:
    """One request's worth of a directed text: its text (SSML breaks allowed) and the
    extra synthesis ``options`` carrying its direction."""

    text: str
    options: dict = field(default_factory=dict)


def _pauses_to_breaks(text: str) -> str:
    def sub(m):
        d = direction(m.group(1))
        return f' <break time="{d.pause_s:g}s" /> ' if d.kind == "pause" else m.group(0)

    return DIRECTION_RE.sub(sub, text)


def compose_instruction(instruction: str | None, tags: list[str]) -> str:
    """The instruction for a part: the voice's standing ``instruction``, then its tags.

    >>> compose_instruction("A wry essayist.", ["dryly", "laughs softly"])
    'A wry essayist. Delivery: dry, understated, a little ironic; with a soft laugh in the voice.'
    """
    phrases = [direction(t).instruction for t in tags if direction(t).kind != "pause"]
    parts = [instruction.strip()] if instruction else []
    if phrases:
        parts.append("Delivery: " + "; ".join(phrases) + ".")
    return " ".join(parts)


def plan_directions(
    text: str, dialect: Dialect, *, instruction: str | None = None
) -> list[Directed]:
    """The requests that say ``text`` in ``dialect`` (see the module docstring).

    ``instruction`` is a standing delivery instruction (a character, a register) sent
    with every part on an instruction dialect, and ignored by the others.

    >>> d = instruction_dialect("instructions")
    >>> [(p.text, p.options["instructions"]) for p in plan_directions(
    ...     "[curious] What gives? [emphatic] Never.", d, instruction="An essayist.")]
    [('What gives?', 'An essayist. Delivery: curious, genuinely asking.'), ('Never.', 'An essayist. Delivery: emphatic, leaning on the key words.')]
    """
    if dialect.inline_tags:
        return [Directed(text)]
    if not dialect.instruction_param:
        return [Directed(_tidy(DIRECTION_RE.sub(" ", _pauses_to_breaks(text))))]
    text = _pauses_to_breaks(text)  # pauses are silence, and do not change direction
    parts: list[Directed] = []

    def flush(segment: str, seg_tags: list[str]):
        segment = _tidy(segment)
        if not segment:
            return
        options = {}
        instr = compose_instruction(instruction, seg_tags)
        if instr:
            options[dialect.instruction_param] = instr
        if parts and parts[-1].options == options:  # same direction: one request
            parts[-1] = Directed(_tidy(parts[-1].text + " " + segment), options)
        else:
            parts.append(Directed(segment, options))

    # a tag, or a run of tags with no words between them, governs the words after it
    current: list[str] = []
    fresh, pos = True, 0
    for m in DIRECTION_RE.finditer(text):
        before = text[pos : m.start()]
        if before.strip():
            flush(before, current)
            fresh = True
        current = [m.group(1)] if fresh else current + [m.group(1)]
        fresh, pos = False, m.end()
    flush(text[pos:], current)
    return parts or [Directed("")]
