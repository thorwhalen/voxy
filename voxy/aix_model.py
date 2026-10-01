"""
Speech through aix (LiteLLM): OpenAI's TTS voices and the other providers LiteLLM routes.

The provider is picked by the ``model`` string (e.g. ``"gpt-4o-mini-tts"``,
``"tts-1-hd"``); keys and defaults come from aix's own config. aix is imported
lazily (``pip install 'voxy[aix]'``).

>>> AixSpeechModel(tts=lambda text, **kw: None).name
'aix'
"""

from collections.abc import Callable

from voxy.base import Speech, SpeechModel, VoiceInfo, VoiceProfile, _resolve_text_input

AIX_MODEL_TYPE = "aix"
#: OpenAI's built-in TTS voices (which ones a model accepts depends on the model).
OPENAI_TTS_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
)


def _aix_text_to_speech(text, **kwargs):
    try:
        from aix.audio import text_to_speech
    except ImportError as e:
        raise ImportError(
            "The 'aix' backend needs aix (pip install 'voxy[aix]')."
        ) from e
    return text_to_speech(text, **kwargs)


class AixSpeechModel(SpeechModel):
    """aix/LiteLLM text-to-speech as a voxy backend.

    Args:
        model: TTS model (None: aix's configured default).
        response_format: 'mp3', 'opus', 'aac', 'flac', 'wav'...
        tts: ``(text, **kw) -> GeneratedAudio`` (tests inject a fake).
    """

    name = AIX_MODEL_TYPE

    def __init__(
        self,
        *,
        model: str | None = None,
        response_format: str = "mp3",
        tts: Callable | None = None,
    ):
        super().__init__(device="cpu")
        self.model = model
        self.response_format = response_format
        self._tts = tts or _aix_text_to_speech

    def list_voices(self) -> list[VoiceInfo]:
        """OpenAI's built-in voices (other LiteLLM providers have their own)."""
        return [
            VoiceInfo(v, v, self.name, labels={"provider": "openai"})
            for v in OPENAI_TTS_VOICES
        ]

    def synthesize(
        self, text, voice: VoiceProfile | str | None = None, **kwargs
    ) -> Speech:
        """Speech from aix; extra kwargs (``speed=``, ``api_key=``...) go to aix."""
        text = _resolve_text_input(text)
        voice = getattr(voice, "segment", voice)
        fmt = kwargs.pop("response_format", self.response_format)
        result = self._tts(
            text, model=self.model, voice=voice, response_format=fmt, **kwargs
        )
        audio = result if isinstance(result, bytes) else result.data
        return Speech(
            audio,
            format=fmt,
            backend=self.name,
            voice=voice or getattr(result, "voice", None),
            text=text,
        )
