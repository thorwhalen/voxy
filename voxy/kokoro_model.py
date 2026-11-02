"""
Kokoro: a neutral local voice that needs no reference clip, no key and no GPU.

Kokoro-82M (weights Apache-2.0) through ``kokoro-onnx`` (MIT) speaks in 54 stock
voices (``af_heart``, ``am_michael``, ``bf_emma``, ...) and runs faster than
realtime on a CPU. It runs in its own small environment, like the local cloners
(``voxy.local_clone``): ``python -m voxy setup kokoro`` builds it, and the model
files (about 354 MB) are fetched from the Hugging Face hub on first use into
``$VOXY_DATA_DIR/models/hf``, like every other local backend's weights.

    >>> import voxy                                                       # doctest: +SKIP
    >>> voxy.text_to_speech("Hello.", "bf_emma", backend="kokoro").save("hi.wav")  # doctest: +SKIP

A voice's first letter is its language (``a`` American English, ``b`` British,
``f`` French, ...) and its second its gender; ``lang`` follows the voice unless
you pass one. The voice list is static (``KOKORO_VOICES``), so listing voices
needs no model.

On macOS the espeak-ng bundled with kokoro-onnx (espeakng-loader) has its data
folder compiled in and cannot find it, so the first synthesis fails; use ``say``
there (thorwhalen/voxy#61).

>>> import numpy as np
>>> fake = lambda text, **kw: (np.full(800, 0.1, dtype=np.float32), 24000)
>>> speech = KokoroSpeechModel(generate=fake).synthesize("One. Two.", "bm_george")
>>> speech.backend, speech.voice, speech.sample_rate
('kokoro', 'bm_george', 24000)
"""

import os
from collections.abc import Callable, Iterable

from voxy.base import Speech, VoiceInfo
from voxy.envs import EnvSpec
from voxy.local_clone import LocalBackend, LocalSpeechModel, register_local_backend

KOKORO_MODEL_TYPE = "kokoro"
KOKORO_VOICE_ENVVAR = "VOXY_KOKORO_VOICE"
DFLT_KOKORO_VOICE = "af_heart"
KOKORO_DEVICE = "cpu"
# fastrtc/kokoro-onnx is a Hugging Face mirror of the files kokoro-onnx publishes as
# the GitHub release model-files-v1.0. At this revision both are byte-identical to
# that release: kokoro-v1.0.onnx sha256 7d5df8ec…6a6c5, voices-v1.0.bin sha256
# bca610b8…1fbf7d. Pinning the revision keeps a changed mirror from reaching users.
KOKORO_HF_REPO = "fastrtc/kokoro-onnx"
KOKORO_HF_REVISION = "8d07950c9b6c87ce6809e9bba7bd494336217c2a"
KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
#: The voices in ``voices-v1.0.bin``.
KOKORO_VOICES = (
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zf_xiaoyi",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
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

KOKORO = register_local_backend(
    LocalBackend(
        KOKORO_MODEL_TYPE,
        EnvSpec(
            KOKORO_MODEL_TYPE,
            ("kokoro-onnx==0.6.1", "soundfile", "huggingface_hub"),
            torch_backend=None,
        ),
        init={
            "repo_id": KOKORO_HF_REPO,
            "revision": KOKORO_HF_REVISION,
            "model_file": KOKORO_MODEL_FILE,
            "voices_file": KOKORO_VOICES_FILE,
        },
        params={"speed": 1.0},
        max_chars=400,
        licence="code MIT (kokoro-onnx); weights Apache-2.0",
        notes="Stock voices only (no cloning); CPU, faster than realtime; no torch.",
        clones=False,
    ),
    replace=False,
)


def _not_a_voice(voice) -> ValueError:
    return ValueError(f"Not a kokoro voice: {voice!r} (see voxy.list_voices('kokoro'))")


def parse_voice(voice: str) -> tuple[str, str]:
    """``(language, gender)`` of a stock voice, from its name's first two letters.

    >>> parse_voice("bf_emma")
    ('en-gb', 'female')
    """
    try:
        return KOKORO_LANGUAGES[voice[0]], KOKORO_GENDERS[voice[1]]
    except (KeyError, IndexError, TypeError):
        raise _not_a_voice(voice) from None


def voice_info(voice: str) -> VoiceInfo:
    """What a stock voice's name says about it.

    >>> voice_info("am_michael").labels
    {'language': 'en-us', 'gender': 'male'}
    """
    language, gender = parse_voice(voice)
    return VoiceInfo(
        voice_id=voice,
        name=voice,
        backend=KOKORO_MODEL_TYPE,
        labels={"language": language, "gender": gender},
    )


class KokoroSpeechModel(LocalSpeechModel):
    """Kokoro-82M as a voxy backend (``params``: speed; lang, else the voice's).

    Args:
        voice: Default stock voice (else ``$VOXY_KOKORO_VOICE``, else 'af_heart').
        voices: The voices the model accepts (default ``KOKORO_VOICES``). For a
            newer voices file, also pass ``init={'voices_file': ..., 'revision': ...}``;
            names must start with a known language letter (``KOKORO_LANGUAGES``).
        generate: The model call (default: the worker, see ``LocalSpeechModel``).
        device: Only 'cpu' (kokoro-onnx runs on onnxruntime's CPU provider).
        **kwargs: The other ``LocalSpeechModel`` options (threads, rootdir, ...).
    """

    def __init__(
        self,
        *,
        voice: str | None = None,
        voices: Iterable[str] = KOKORO_VOICES,
        generate: Callable | None = None,
        device: str = KOKORO_DEVICE,
        **kwargs,
    ):
        if device != KOKORO_DEVICE:
            raise ValueError(f"kokoro runs on {KOKORO_DEVICE!r} only, not {device!r}")
        super().__init__(KOKORO, generate=generate, **kwargs)
        self.voices = tuple(voices)
        self.dflt_voice = (
            voice or os.environ.get(KOKORO_VOICE_ENVVAR) or DFLT_KOKORO_VOICE
        )

    def _stock_voice(self, voice) -> str:
        """The voice's name, checked against ``self.voices``."""
        voice = getattr(voice, "segment", voice) or self.dflt_voice
        if voice not in self.voices:
            raise _not_a_voice(voice)
        return voice

    def _voice_request(self, voice) -> tuple[dict, str]:
        return {"voice": voice}, voice

    def synthesize(self, text, voice=None, *, lang=None, **params) -> Speech:
        """Speak ``text`` in a stock ``voice``, in ``lang`` (default: the voice's)."""
        voice = self._stock_voice(voice)
        lang = lang or parse_voice(voice)[0]
        return super().synthesize(text, voice, lang=lang, **params)

    def list_voices(self) -> list[VoiceInfo]:
        """The stock voices (no model needed)."""
        return [voice_info(v) for v in self.voices]
