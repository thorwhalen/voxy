"""
Kokoro: a neutral local voice that needs no reference clip, no key and no GPU.

Kokoro-82M (weights Apache-2.0) through ``kokoro-onnx`` (MIT) speaks in 54 stock
voices (``af_heart``, ``am_michael``, ``bf_emma``, ...) and runs faster than
realtime on a CPU. It runs in its own small environment, like the local cloners
(``voxy.local_clone``); ``python -m voxy setup kokoro`` builds it and fetches the
model files (about 340 MB) into ``$VOXY_DATA_DIR/models/kokoro/``.

    >>> import voxy                                                       # doctest: +SKIP
    >>> voxy.text_to_speech("Hello.", "bf_emma", backend="kokoro").save("hi.wav")  # doctest: +SKIP

A voice's first letter is its language (``a`` American English, ``b`` British,
``f`` French, ...); ``lang`` follows it unless you pass one.

On macOS the espeak-ng bundled with kokoro-onnx (espeakng-loader) has its data
folder compiled in and cannot find it, so the worker exits on load; use ``say``
there (thorwhalen/voxy#61).

>>> import numpy as np
>>> fake = lambda text, **kw: (np.full(800, 0.1, dtype=np.float32), 24000)
>>> speech = KokoroSpeechModel(generate=fake).synthesize("One. Two.", "bm_george")
>>> speech.backend, speech.voice, speech.sample_rate
('kokoro', 'bm_george', 24000)
"""

import os
from collections.abc import Callable

from voxy.base import VoiceInfo
from voxy.envs import EnvSpec
from voxy.local_clone import LocalCloneBackend, LocalSpeechModel

KOKORO_MODEL_TYPE = "kokoro"
KOKORO_VOICE_ENVVAR = "VOXY_KOKORO_VOICE"
DFLT_KOKORO_VOICE = "af_heart"
KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
KOKORO_RELEASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
#: A voice name's first letter -> the language kokoro-onnx phonemizes in.
KOKORO_LANGUAGES = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}
#: A voice name's second letter -> the voice's gender.
KOKORO_GENDERS = {"f": "female", "m": "male"}

KOKORO = LocalCloneBackend(
    KOKORO_MODEL_TYPE,
    EnvSpec(
        KOKORO_MODEL_TYPE,
        ("kokoro-onnx==0.6.1", "soundfile"),
        torch_backend=None,
        assets={
            name: f"{KOKORO_RELEASE_URL}/{name}"
            for name in (KOKORO_MODEL_FILE, KOKORO_VOICES_FILE)
        },
    ),
    init={"model_file": KOKORO_MODEL_FILE, "voices_file": KOKORO_VOICES_FILE},
    params={"speed": 1.0},
    max_chars=400,
    licence="code MIT (kokoro-onnx); weights Apache-2.0",
    notes="Stock voices only (no cloning); CPU, faster than realtime; no torch.",
)


def voice_language(voice: str) -> str:
    """The language a stock voice speaks, from its first letter.

    >>> voice_language("bf_emma")
    'en-gb'
    """
    try:
        return KOKORO_LANGUAGES[voice[:1]]
    except KeyError:
        raise ValueError(
            f"Not a kokoro voice: {voice!r} (see voxy.list_voices('kokoro'))"
        ) from None


def voice_info(voice: str) -> VoiceInfo:
    """What a stock voice's name says about it.

    >>> voice_info("am_michael").labels
    {'language': 'en-us', 'gender': 'male'}
    """
    return VoiceInfo(
        voice_id=voice,
        name=voice,
        backend=KOKORO_MODEL_TYPE,
        labels={
            "language": KOKORO_LANGUAGES.get(voice[:1], ""),
            "gender": KOKORO_GENDERS.get(voice[1:2], ""),
        },
    )


class KokoroSpeechModel(LocalSpeechModel):
    """Kokoro-82M as a voxy backend (``params``: speed; lang, else the voice's).

    Args:
        voice: Default stock voice (else ``$VOXY_KOKORO_VOICE``, else 'af_heart').
        generate: The model call (default: the worker, see ``LocalSpeechModel``).
        **kwargs: The other ``LocalSpeechModel`` options (threads, rootdir, ...).
    """

    def __init__(
        self,
        *,
        voice: str | None = None,
        generate: Callable | None = None,
        **kwargs,
    ):
        super().__init__(KOKORO, generate=generate, **kwargs)
        self.dflt_voice = (
            voice or os.environ.get(KOKORO_VOICE_ENVVAR) or DFLT_KOKORO_VOICE
        )

    def _voice_request(self, voice) -> tuple[dict, str]:
        voice = getattr(voice, "segment", voice) or self.dflt_voice
        return {"voice": voice, "lang": voice_language(voice)}, voice

    def list_voices(self) -> list[VoiceInfo]:
        """The stock voices (loads the model on first call)."""
        call = getattr(self._generate, "call", None)
        if call is None:
            raise self._unsupported("list voices with an injected generate")
        return [voice_info(v) for v in call("voices")["voices"]]
