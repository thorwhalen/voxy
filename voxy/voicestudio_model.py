"""
Local description-prompted speech through VoiceStudio (Parler-TTS and friends).

A voice here is a *text description* ("a warm, low female voice, close to the
microphone"), not a recording: ``voice`` may be that string or a ``VoiceProfile``
made by ``design_voice``. Nothing is uploaded; weights run locally (CPU works,
slowly) and are fetched from Hugging Face on first use (``model_id=``; the default
is several GB, so ``pip install 'voxy[voicestudio]'`` does not download them).

The model call is a seam: ``generate(text, description) -> (waveform, sample_rate)``
is injectable, so tests (and other VoiceStudio architectures) never load weights.

>>> model = VoiceStudioSpeechModel(generate=lambda text, description: ([0.0] * 8, 8000))
>>> model.synthesize("Hi", "a calm voice").format
'wav'
"""

import io
import wave
from collections.abc import Callable, Sequence

from voxy.base import Speech, SpeechModel, VoiceProfile, _resolve_text_input

VOICESTUDIO_MODEL_TYPE = "voicestudio"
DFLT_VOICESTUDIO_MODEL_ID = "parler-tts/parler-tts-mini-v1"
DFLT_DESCRIPTION = "A clear, natural voice speaking at a moderate pace with very clean audio."
PCM_SAMPLE_WIDTH_BYTES = 2
PCM_MAX = 32767


def _pcm_wav(samples: Sequence[float], sample_rate: int) -> bytes:
    """Mono 16-bit WAV bytes from floats in [-1, 1].

    >>> _pcm_wav([0.0, 1.0], 8000)[:4]
    b'RIFF'
    """
    frames = b"".join(
        int(max(-1.0, min(1.0, float(s))) * PCM_MAX).to_bytes(2, "little", signed=True)
        for s in samples
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    return buffer.getvalue()


def _parler_generate(model_id: str, device: str) -> Callable:
    """The default ``generate``: Parler-TTS through voicestudio, loaded on first call."""
    state: dict = {}

    def generate(text, description):
        if not state:
            try:
                import voicestudio.models.parler_tts  # noqa: F401 (registers the classes)
                from transformers import AutoModel, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "The 'voicestudio' backend needs voicestudio and torch "
                    "(pip install 'voxy[voicestudio]')."
                ) from e
            state["model"] = AutoModel.from_pretrained(model_id).to(device)
            state["tokenizer"] = AutoTokenizer.from_pretrained(model_id)
        model, tokenizer = state["model"], state["tokenizer"]
        description_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        audio = model.generate(input_ids=description_ids, prompt_input_ids=prompt_ids)
        return audio.cpu().float().numpy().squeeze(), model.config.sampling_rate

    return generate


class VoiceStudioSpeechModel(SpeechModel):
    """VoiceStudio (description-prompted TTS) as a voxy backend.

    Args:
        model_id: Hugging Face id of the base model (default Parler-TTS mini).
        device: Torch device for the default ``generate``.
        generate: ``(text, description) -> (waveform, sample_rate)`` (tests inject a fake).
    """

    name = VOICESTUDIO_MODEL_TYPE
    dflt_voice = DFLT_DESCRIPTION

    def __init__(
        self,
        *,
        model_id: str = DFLT_VOICESTUDIO_MODEL_ID,
        device: str = "cpu",
        generate: Callable | None = None,
    ):
        super().__init__(device=device)
        self.model_id = model_id
        self._generate = generate or _parler_generate(model_id, device)

    def design_voice(self, description: str, *, speaker_id: int = 0, **_) -> VoiceProfile:
        """A profile for a voice described in words (no model call, nothing stored remotely)."""
        return VoiceProfile(description, speaker_id, self.name, 0, {"description": description})

    def synthesize(
        self, text, voice: VoiceProfile | str | None = None, **kwargs
    ) -> Speech:
        """Speech in the voice described by ``voice`` (a description or a profile)."""
        text = _resolve_text_input(text)
        description = getattr(voice, "segment", voice) or self.dflt_voice
        waveform, sample_rate = self._generate(text, description, **kwargs)
        return Speech(
            _pcm_wav(waveform, sample_rate),
            format="wav",
            backend=self.name,
            voice=description,
            sample_rate=sample_rate,
            text=text,
        )
