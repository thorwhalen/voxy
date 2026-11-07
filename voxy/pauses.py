"""
Pauses in text to speak: SSML breaks, honoured on every backend.

The standard way to ask a speech engine for silence is the W3C SSML 1.1 break
element, written into the text [1]:

    Breathe in.<break time="4s"/> And out, slowly.<break time="6s"/>

Engines honour it unevenly: ElevenLabs' v2-family models up to 3 s, ``eleven_v3``
and ``eleven_v4`` not at all [2], local models never. So ``voxy.text_to_speech``
resolves breaks itself. A break the backend renders (``SpeechModel.native_break_s``)
stays in the text; at any other break the text is cut, each part is spoken on its
own, and the parts are joined with exactly that much silence. Text with no markup
takes the old path, untouched.

The structure underneath is a list of :class:`Section` (``parse_sections``): the
text, cut at every ``<break>`` and every SSML ``<mark name="..."/>``, each section
carrying the silence that follows it and the mark that ends it. A linear render
reads ``plan_sections``, which cuts only where the backend needs a cut and drops
marks. A player that holds at a mark until the listener continues (SMIL 3.0's
``end="<id>.activateEvent"`` [3]) reads ``parse_sections`` and speaks section by
section: no other change is needed for that.

>>> [(s.text, s.pause_s) for s in parse_sections('In.<break time="4s"/> Out.')]
[('In.', 4.0), ('Out.', 0.0)]

[1] W3C. Speech Synthesis Markup Language (SSML) Version 1.1, section 3.2.3.
    https://www.w3.org/TR/speech-synthesis11/#edef_break
[2] ElevenLabs. Eleven v4 / Eleven v3 prompting guides ("SSML is not supported").
    https://elevenlabs.io/docs/overview/capabilities/text-to-speech/eleven-v4
[3] W3C. SMIL 3.0 Timing and Synchronization (event-based end values).
    https://www.w3.org/TR/SMIL3/smil-timing.html
"""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

#: Seconds for each SSML ``strength`` (SSML leaves them to the engine; these are
#: Azure Speech's documented values). A bare ``<break/>`` is ``medium``.
BREAK_STRENGTH_S: Mapping[str, float] = {
    "none": 0.0,
    "x-weak": 0.25,
    "weak": 0.5,
    "medium": 0.75,
    "strong": 1.0,
    "x-strong": 1.25,
}
DFLT_BREAK_STRENGTH = "medium"

_TAG = re.compile(
    r"<(?P<name>break|mark)\b(?P<attrs>[^>]*?)/?>(?:\s*</(?P=name)>)?", re.I
)
_ATTR = re.compile(r"""([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_TIME = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)\s*(ms|s)\s*$", re.I)


@dataclass(frozen=True)
class Section:
    """A stretch of text to speak, and how it ends.

    ``pause_s`` is the silence after it; ``mark`` the name of the SSML mark that
    ends it, if one does (a point a player may stop at).
    """

    text: str
    pause_s: float = 0.0
    mark: str | None = None


def has_pause_markup(text) -> bool:
    """Whether ``text`` holds an SSML ``<break>`` or ``<mark>``.

    >>> has_pause_markup('Hi <break time="1s"/>'), has_pause_markup("Hi [softly]")
    (True, False)
    """
    return isinstance(text, str) and _TAG.search(text) is not None


def break_seconds(attrs: str) -> float:
    """The silence a ``<break>`` asks for, from its attribute string.

    ``time`` (``"3s"``, ``"250ms"``) wins over ``strength``; neither means medium.

    >>> break_seconds(' time="250ms"'), break_seconds(" strength='strong'"), break_seconds("")
    (0.25, 1.0, 0.75)
    """
    found = {k.lower(): a if a or not b else b for k, a, b in _ATTR.findall(attrs)}
    if "time" in found:
        m = _TIME.match(found["time"])
        if not m:
            raise ValueError(
                f"<break time={found['time']!r}>: SSML times are like '3s' or '250ms'"
            )
        value, unit = float(m.group(1)), m.group(2).lower()
        return value / 1000 if unit == "ms" else value
    strength = found.get("strength", DFLT_BREAK_STRENGTH).lower()
    if strength not in BREAK_STRENGTH_S:
        raise ValueError(
            f"<break strength={strength!r}>: use one of {sorted(BREAK_STRENGTH_S)}"
        )
    return BREAK_STRENGTH_S[strength]


def _mark_name(attrs: str) -> str:
    for key, a, b in _ATTR.findall(attrs):
        if key.lower() == "name":
            return a or b
    raise ValueError('<mark> needs a name: <mark name="..."/>')


def _clean(text: str) -> str:
    return " ".join(text.split())


def parse_sections(text: str) -> list[Section]:
    """``text`` cut at every SSML ``<break>`` and ``<mark>``.

    Breaks in a row add up; a mark keeps the pause of breaks around it. A leading
    break gives a first section with no text (silence first); a trailing one is
    kept (silence last).

    >>> parse_sections('<break time="2s"/>One.<mark name="m1"/> Two.<break/><break/>')
    [Section(text='', pause_s=2.0, mark=None), Section(text='One.', pause_s=0.0, mark='m1'), Section(text='Two.', pause_s=1.5, mark=None)]
    """
    sections: list[Section] = []
    pos, pause, mark = 0, 0.0, None
    pending_text = ""
    for m in _TAG.finditer(text):
        chunk = _clean(text[pos : m.start()])
        if chunk:
            if pending_text or pause or mark:
                sections.append(Section(pending_text, round(pause, 6), mark))
            pending_text, pause, mark = chunk, 0.0, None
        if m.group("name").lower() == "break":
            pause += break_seconds(m.group("attrs"))
        else:
            if mark is not None:  # two marks with nothing between: keep both points
                sections.append(Section(pending_text, round(pause, 6), mark))
                pending_text, pause = "", 0.0
            mark = _mark_name(m.group("attrs"))
        pos = m.end()
    tail = _clean(text[pos:])
    if tail:
        if pending_text or pause or mark:
            sections.append(Section(pending_text, round(pause, 6), mark))
        pending_text, pause, mark = tail, 0.0, None
    if pending_text or pause or mark or not sections:
        sections.append(Section(pending_text, round(pause, 6), mark))
    return sections


def render_break(seconds: float) -> str:
    """The SSML break for ``seconds``, in the form ElevenLabs documents.

    >>> render_break(1.5)
    '<break time="1.5s" />'
    """
    return f'<break time="{seconds:g}s" />'


def plan_sections(text: str, *, native_break_s: float = 0.0) -> list[Section]:
    """The sections to synthesize separately, for a backend that renders breaks of
    up to ``native_break_s`` seconds itself.

    A break that short stays in the text (as :func:`render_break`); a mark is
    dropped (a linear render has nothing to do at it); every other break ends a
    section.

    >>> t = 'A.<break time="1s"/> B.<break time="5s"/> C.'
    >>> [(s.text, s.pause_s) for s in plan_sections(t, native_break_s=3)]
    [('A. <break time="1s" /> B.', 5.0), ('C.', 0.0)]
    >>> [s.text for s in plan_sections(t)]
    ['A.', 'B.', 'C.']
    """
    out: list[Section] = []
    text_parts: list[str] = []
    for section in parse_sections(text):
        if section.text:
            text_parts.append(section.text)
        if section.pause_s == 0:
            continue
        if 0 < section.pause_s <= native_break_s and section.text:
            text_parts.append(render_break(section.pause_s))
            continue
        out.append(Section(" ".join(text_parts), section.pause_s))
        text_parts = []
    if text_parts or not out:
        out.append(Section(" ".join(text_parts)))
    # a break kept inline at the very end of a section is better as silence
    return [_trailing_break_to_pause(s) for s in out]


def _trailing_break_to_pause(section: Section) -> Section:
    m = re.search(r'\s*<break time="([\d.]+)s" />$', section.text)
    if not m:
        return section
    return Section(
        section.text[: m.start()], round(section.pause_s + float(m.group(1)), 6)
    )


def join_sections(
    sections: Iterable[Section],
    synthesize: Callable[[str], "Speech"],
    *,
    backend: str = "",
):
    """One WAV ``Speech``: each section spoken by ``synthesize``, then its silence.

    ``synthesize(text) -> Speech`` must return WAV (or anything ``read_wav``
    reads); sections at different rates are resampled to the first one's.
    """
    from voxy.base import Speech
    from voxy.wavutil import read_wav, resample, silence, write_wav

    pieces: list[np.ndarray] = []
    rate = None
    voice = None
    for section in sections:
        if section.text:
            speech = synthesize(section.text)
            if speech.format != "wav":
                raise ValueError(
                    f"Joining pauses needs WAV, and the {speech.backend or backend!r} "
                    f"backend returned {speech.format!r}: pass an output format "
                    "that gives WAV or PCM (for ElevenLabs, omit output_format or "
                    "pass e.g. 'pcm_44100')"
                )
            samples, sr = read_wav(speech.audio)
            voice = voice or speech.voice
            if rate is None:
                rate = sr
            elif sr != rate:
                samples = resample(samples, sr, rate)
            pieces.append(samples)
        if section.pause_s:
            pieces.append(silence(section.pause_s * 1000, rate or DFLT_SILENCE_RATE))
    rate = rate or DFLT_SILENCE_RATE
    audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    return Speech(
        write_wav(audio, rate),
        format="wav",
        backend=backend,
        voice=voice,
        sample_rate=rate,
    )


#: The rate of a render that is nothing but silence (no section had text).
DFLT_SILENCE_RATE = 24000
