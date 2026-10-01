"""
Speech through fal.ai, via falaw: many hosted TTS models, chosen by quality tier.

falaw picks a model for ``quality`` ('fast', 'balanced', 'best'...) unless
``model_id`` is given; ``voice`` means whatever that model calls a voice. falaw is
imported lazily (``pip install 'voxy[fal]'``).

>>> FalSpeechModel(tts=lambda text, **kw: None, fetch=lambda url: b"").name
'fal'
"""

import os
from collections.abc import Callable

from voxy.base import Speech, SpeechModel, VoiceProfile, _resolve_text_input

FAL_MODEL_TYPE = "fal"
DFLT_FAL_QUALITY = "balanced"
_AUDIO_SUBTYPES = {
    "mpeg": "mp3",
    "mp3": "mp3",
    "wav": "wav",
    "x-wav": "wav",
    "wave": "wav",
    "ogg": "ogg",
    "opus": "opus",
    "flac": "flac",
    "x-flac": "flac",
    "aac": "aac",
    "mp4": "m4a",
    "x-m4a": "m4a",
    "webm": "webm",
}


def _falaw_text_to_speech(text, **kwargs):
    try:
        from falaw.operations.audio import text_to_speech
    except ImportError as e:
        raise ImportError(
            "The 'fal' backend needs falaw (pip install 'voxy[fal]')."
        ) from e
    return text_to_speech(text, **kwargs)


def _fetch_url(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url) as response:
        return response.read()


def _audio_format(content_type: str, url: str) -> str:
    """'mp3', 'wav'... from a content type, else from the URL's extension.

    >>> _audio_format("audio/wav", "x"), _audio_format("", "https://h/a.mp3?sig=1")
    ('wav', 'mp3')
    """
    subtype = (content_type or "").split(";")[0].split("/")[-1].strip().lower()
    if subtype in _AUDIO_SUBTYPES:
        return _AUDIO_SUBTYPES[subtype]
    ext = os.path.splitext(url.split("?")[0])[1].lstrip(".").lower()
    return ext or "mp3"


class FalSpeechModel(SpeechModel):
    """fal.ai text-to-speech (through falaw) as a voxy backend.

    Args:
        quality: falaw quality tier used to pick a model.
        model_id: A specific fal model (overrides ``quality``).
        tts: ``(text, **kw) -> falaw.Result`` (tests inject a fake).
        fetch: ``url -> bytes`` to download the result.
    """

    name = FAL_MODEL_TYPE

    def __init__(
        self,
        *,
        quality: str = DFLT_FAL_QUALITY,
        model_id: str | None = None,
        tts: Callable | None = None,
        fetch: Callable[[str], bytes] | None = None,
    ):
        super().__init__(device="cpu")
        self.quality = quality
        self.model_id = model_id
        self._tts = tts or _falaw_text_to_speech
        self._fetch = fetch or _fetch_url

    def synthesize(
        self, text, voice: VoiceProfile | str | None = None, **kwargs
    ) -> Speech:
        """Speech from fal; ``extra=`` passes model-specific arguments."""
        text = _resolve_text_input(text)
        voice = getattr(voice, "segment", voice)
        result = self._tts(
            text,
            quality=kwargs.pop("quality", self.quality),
            voice=voice,
            model_id=kwargs.pop("model_id", self.model_id),
            **kwargs,
        )
        asset = result.first
        if asset is None:
            raise RuntimeError(
                f"fal returned no audio (application {result.application!r})"
            )
        return Speech(
            self._fetch(asset.url),
            format=_audio_format(asset.content_type, asset.url),
            backend=self.name,
            voice=voice,
            text=text,
        )
