"""
ElevenLabs backend for voxy: instant voice cloning and synthesis through the API.

Select it with ``create_speech_model("elevenlabs")``. Cloning uploads one or more
audio samples to ElevenLabs' Instant Voice Cloning (IVC) endpoint and returns a
``VoiceProfile`` whose ``segment`` is the new ``voice_id``; synthesis renders text
in that voice. ElevenLabs keeps the voice, so a later session can rebuild the
profile from the id alone with ``ElevenLabsSpeechModel.voice_profile(voice_id)``.

The key is read from ``api_key=`` or, failing that, the ``ELEVEN_API_KEY`` /
``ELEVENLABS_API_KEY`` environment variables (the same order the fleet's other
ElevenLabs clients use). The ``elevenlabs`` SDK is imported lazily, so it is only
needed when a request is actually made (``pip install voxy[elevenlabs]``).

ElevenLabs requires that whoever clones a voice has the right and the consent of
the voice's owner (or their guardian) to do so.

>>> model = ElevenLabsSpeechModel(api_key="unused", client_factory=lambda key: None)
>>> model.sample_rate
24000
"""

import io
import json
import os
import wave
from dataclasses import dataclass
from collections.abc import Callable, Iterable, Mapping
from functools import cached_property
from typing import Any, BinaryIO

import numpy as np
import torch

from voxy.base import (
    DFLT_ASSUMED_SAMPLE_RATE,
    DFLT_VOXY_DEVICE,
    Speech,
    SpeechModel,
    VoiceInfo,
    VoiceProfile,
    _resolve_audio_input,
    _resolve_text_input,
    tensor_to_wav_bytes,
)

ELEVENLABS_MODEL_TYPE = "elevenlabs"
ELEVENLABS_API_KEY_ENVVARS = ("ELEVEN_API_KEY", "ELEVENLABS_API_KEY")
DFLT_ELEVENLABS_MODEL_ID = os.environ.get(
    "VOXY_ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"
)
# Raw PCM is available on every ElevenLabs tier and decodes without ffmpeg.
DFLT_ELEVENLABS_OUTPUT_FORMAT = "pcm_24000"
DFLT_CLONE_NAME_TEMPLATE = "voxy-{speaker_id}"
# What synthesize() returns by default: a playable file, on every tier.
DFLT_ELEVENLABS_SYNTH_FORMAT = "mp3_44100_128"
DFLT_VOICE_DESIGN_MODEL_ID = "eleven_multilingual_ttv_v2"
# The ElevenLabs IVC form accepts at most this many sample files per voice.
DFLT_MAX_CLONE_FILES = 25

AudioInput = str | os.PathLike | bytes | BinaryIO | torch.Tensor | np.ndarray


def resolve_elevenlabs_api_key(
    api_key: str | None = None,
    *,
    envvars: Iterable[str] = ELEVENLABS_API_KEY_ENVVARS,
) -> str | None:
    """The explicit ``api_key`` if given, else the first set env var, else None.

    >>> resolve_elevenlabs_api_key("explicit")
    'explicit'
    >>> resolve_elevenlabs_api_key(envvars=["VOXY_SURELY_UNSET_VAR"]) is None
    True
    """
    if api_key:
        return api_key
    return next((os.environ[v] for v in envvars if os.environ.get(v)), None)


def default_elevenlabs_client_factory(api_key: str):
    """Build the official SDK client (imported lazily)."""
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise ImportError(
            "The elevenlabs package is required for voxy's ElevenLabs backend. "
            "Install with 'pip install voxy[elevenlabs]' (or 'pip install elevenlabs')."
        ) from e
    return ElevenLabs(api_key=api_key)


# -----------------------------------------------------------------------------
# Audio encoding helpers
# -----------------------------------------------------------------------------


_wav_bytes = tensor_to_wav_bytes  # one WAV encoder, in voxy.base


def _pcm16_to_tensor(pcm: bytes) -> torch.Tensor:
    """Decode little-endian 16-bit mono PCM into a float tensor in [-1, 1].

    >>> _pcm16_to_tensor(b"\\x00\\x00\\xff\\x7f").tolist()
    [0.0, 0.999969482421875]
    >>> _pcm16_to_tensor(b"\\x00\\x00\\xff").tolist()
    [0.0]
    """
    pcm = pcm[: len(pcm) - len(pcm) % 2]  # a truncated stream can end mid-sample
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(samples)


def _wav_to_tensor(data: bytes) -> tuple[torch.Tensor, int]:
    """Decode 16-bit mono WAV bytes to ``(tensor, sample_rate)``.

    Reads the samples from the ``data`` chunk directly, so a streamed WAV whose
    header carries a placeholder size still decodes.

    >>> audio, sr = _wav_to_tensor(_wav_bytes(torch.zeros(10), 16000))
    >>> tuple(audio.shape), sr
    ((10,), 16000)
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV payload")
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        chunk_id, size = (
            data[pos : pos + 4],
            int.from_bytes(data[pos + 4 : pos + 8], "little"),
        )
        body = pos + 8
        if chunk_id == b"fmt ":
            channels = int.from_bytes(data[body + 2 : body + 4], "little")
            rate = int.from_bytes(data[body + 4 : body + 8], "little")
            width = int.from_bytes(data[body + 14 : body + 16], "little")
            fmt = (channels, rate, width)
        elif chunk_id == b"data":
            if fmt is None or fmt[0] != 1 or fmt[2] != 16:
                raise ValueError(
                    f"Expected 16-bit mono WAV, got (channels, rate, bits)={fmt}"
                )
            end = body + size if 0 < size <= len(data) - body else len(data)
            return _pcm16_to_tensor(data[body:end]), fmt[1]
        pos = body + size + (size % 2)
    raise ValueError("WAV payload has no data chunk")


def _output_format_parts(output_format: str) -> tuple[str, int]:
    """Split an ElevenLabs output format into ``(codec, sample_rate)``.

    >>> _output_format_parts("pcm_24000")
    ('pcm', 24000)
    >>> _output_format_parts("mp3_44100_128")
    ('mp3', 44100)
    """
    codec, rate, *_ = output_format.split("_")
    return codec, int(rate)


def _to_upload_file(
    audio_input: AudioInput,
    *,
    index: int,
    sample_rate: int,
    cleanup_audio_fn: Callable | None,
) -> tuple[str, bytes]:
    """Turn one voxy audio input into a ``(filename, bytes)`` upload part.

    Encoded inputs (paths, bytes, file-likes) are uploaded untouched, unless a
    ``cleanup_audio_fn`` is given, in which case they are decoded, cleaned, and
    re-encoded as WAV. Raw tensors/arrays are always encoded as WAV.
    """
    default_name = f"sample_{index:03d}"
    if isinstance(audio_input, os.PathLike):
        audio_input = os.fspath(audio_input)

    if cleanup_audio_fn is None and not isinstance(
        audio_input, (torch.Tensor, np.ndarray)
    ):
        if isinstance(audio_input, str):
            if not os.path.isfile(audio_input):
                raise ValueError(f"Audio path does not exist: {audio_input}")
            with open(audio_input, "rb") as f:
                return os.path.basename(audio_input), f.read()
        if isinstance(audio_input, bytes):
            return default_name + _sniff_suffix(audio_input), audio_input
        if hasattr(audio_input, "read"):
            content = audio_input.read()
            name = getattr(audio_input, "name", None)
            if isinstance(name, str) and name:
                return os.path.basename(name), content
            return default_name + _sniff_suffix(content), content
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    audio, rate = _resolve_audio_input(audio_input, assumed_sample_rate=sample_rate)
    if cleanup_audio_fn is not None:
        audio = cleanup_audio_fn(audio, rate)
    return f"{default_name}.wav", _wav_bytes(audio, rate)


def _sniff_suffix(content: bytes) -> str:
    """A file suffix guessed from the first bytes, so the upload has a format hint.

    >>> _sniff_suffix(b"RIFF....WAVEfmt "), _sniff_suffix(b"ID3..."), _sniff_suffix(b"??")
    ('.wav', '.mp3', '')
    """
    head = content[:12]
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head[:4] in (b"OggS", b"fLaC"):
        return ".ogg" if head[:4] == b"OggS" else ".flac"
    return ""


_SINGLE_AUDIO_TYPES = (str, bytes, os.PathLike, torch.Tensor, np.ndarray)


def _as_input_list(audio_input) -> list:
    """A single audio input becomes a one-element list; any other iterable is listed.

    >>> _as_input_list("a.wav")
    ['a.wav']
    >>> _as_input_list(("a.wav", b"..."))
    ['a.wav', b'...']
    >>> _as_input_list(x for x in ["a.wav", "b.wav"])
    ['a.wav', 'b.wav']
    """
    if isinstance(audio_input, _SINGLE_AUDIO_TYPES) or hasattr(audio_input, "read"):
        return [audio_input]
    if isinstance(audio_input, Iterable):
        return list(audio_input)
    return [audio_input]


# -----------------------------------------------------------------------------
# The speech model
# -----------------------------------------------------------------------------


@dataclass
class VoiceDesignPreview:
    """One candidate voice from ``design_voice_previews``: listen, then pick."""

    generated_voice_id: str
    audio: bytes  # an mp3 sample of the voice
    text: str = ""
    duration_s: float | None = None

    def save(self, path: str) -> str:
        _write_bytes(path, self.audio)
        return path


class ElevenLabsSpeechModel(SpeechModel):
    """Speech model backed by the ElevenLabs API (Instant Voice Cloning + TTS).

    Args:
        api_key: ElevenLabs key; defaults to ``ELEVEN_API_KEY`` / ``ELEVENLABS_API_KEY``.
        model_id: TTS model used by ``generate_speech``.
        output_format: ElevenLabs output format. ``pcm_*`` and ``wav_*`` formats
            decode to a tensor with no extra dependency.
        client_factory: ``api_key -> client``. Defaults to the official SDK; tests
            inject a fake so nothing reaches the API.
        device: Device the returned tensors are placed on.
    """

    name = ELEVENLABS_MODEL_TYPE

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = DFLT_ELEVENLABS_MODEL_ID,
        output_format: str = DFLT_ELEVENLABS_OUTPUT_FORMAT,
        client_factory: Callable[[str], Any] | None = None,
        device: str = DFLT_VOXY_DEVICE,
    ):
        super().__init__(device)
        self._api_key = resolve_elevenlabs_api_key(api_key)
        self.model_id = model_id
        self.output_format = output_format
        self.client_factory = client_factory or default_elevenlabs_client_factory

    def __repr__(self):  # never show the key
        return (
            f"{type(self).__name__}(model_id={self.model_id!r}, "
            f"output_format={self.output_format!r})"
        )

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio ``generate_speech`` returns."""
        return _output_format_parts(self.output_format)[1]

    @cached_property
    def client(self):
        """The ElevenLabs client, built on first use."""
        if not self._api_key:
            raise RuntimeError(
                "voxy's ElevenLabs backend needs an API key. Set one of "
                f"{', '.join(ELEVENLABS_API_KEY_ENVVARS)} or pass api_key=."
            )
        return self.client_factory(self._api_key)

    # --- voices -------------------------------------------------------------

    def clone_voice(
        self,
        audio_input: AudioInput | list[AudioInput] | tuple[AudioInput, ...],
        transcript: str | None = None,
        speaker_id: int = 999,
        *,
        cleanup_audio_fn: Callable | None = None,
        name: str | None = None,
        description: str | None = None,
        labels: Mapping[str, str] | None = None,
        remove_background_noise: bool | None = None,
        assumed_sample_rate: int = DFLT_ASSUMED_SAMPLE_RATE,
        max_files: int | None = DFLT_MAX_CLONE_FILES,
    ) -> VoiceProfile:
        """Create an ElevenLabs instant voice clone from one or more audio samples.

        ElevenLabs recommends about 1-2 minutes of clean, single-speaker audio in
        total (at least 1, at most about 3); how it is split across files does
        not matter.

        Args:
            audio_input: One audio input, or an iterable of them (paths, bytes,
                file-likes, tensors, arrays).
            transcript: Ignored: IVC needs no transcript. Kept so every voxy
                backend shares one ``clone_voice`` signature.
            speaker_id: Carried on the profile, and used in the default name.
            cleanup_audio_fn: Optional ``(audio, sample_rate) -> audio`` applied
                before upload. Off by default: ElevenLabs does its own processing,
                and voxy's silence trimmer can chop soft speech.
            name: Voice name shown in ElevenLabs (default ``voxy-<speaker_id>``).
            description: Voice description.
            labels: Voice labels (keys such as language, accent, gender, age).
            remove_background_noise: Ask ElevenLabs to isolate the voice. Can
                make clean samples worse.
            assumed_sample_rate: Sample rate of raw tensor/array inputs.
            max_files: Refuse, before uploading anything, more samples than
                ElevenLabs accepts per voice (``None`` to skip the check).
                Concatenate short clips to stay under it.

        Returns:
            A ``VoiceProfile`` whose ``segment`` (and ``metadata['voice_id']``)
            is the new ElevenLabs voice id.
        """
        inputs = _as_input_list(audio_input)
        if not inputs:
            raise ValueError("clone_voice needs at least one audio sample")
        if max_files is not None and len(inputs) > max_files:
            raise ValueError(
                f"{len(inputs)} samples given; ElevenLabs accepts at most {max_files} "
                "per voice. Concatenate clips into fewer files (total length is "
                "what matters), or pass max_files= if the limit changed."
            )
        files = [
            _to_upload_file(
                x,
                index=i,
                sample_rate=assumed_sample_rate,
                cleanup_audio_fn=cleanup_audio_fn,
            )
            for i, x in enumerate(inputs)
        ]
        name = name or DFLT_CLONE_NAME_TEMPLATE.format(speaker_id=speaker_id)
        optional = {
            "description": description,
            # A multipart form can't carry a dict: the API takes labels as JSON text.
            "labels": json.dumps(dict(labels)) if labels is not None else None,
            "remove_background_noise": remove_background_noise,
        }
        response = self.client.voices.ivc.create(
            name=name,
            files=files,
            **{k: v for k, v in optional.items() if v is not None},
        )
        return self.voice_profile(
            response.voice_id,
            speaker_id=speaker_id,
            metadata={
                "name": name,
                "n_samples": len(files),
                "upload_bytes": sum(len(content) for _, content in files),
                "requires_verification": getattr(
                    response, "requires_verification", None
                ),
            },
        )

    def voice_profile(
        self,
        voice_id: str,
        *,
        speaker_id: int = 999,
        metadata: Mapping[str, Any] | None = None,
    ) -> VoiceProfile:
        """A profile for an existing ElevenLabs voice (a past clone, or a stock voice).

        >>> model = ElevenLabsSpeechModel(api_key="unused")
        >>> model.voice_profile("abc123").segment
        'abc123'
        """
        return VoiceProfile(
            segment=voice_id,
            speaker_id=speaker_id,
            model_type=ELEVENLABS_MODEL_TYPE,
            sample_rate=self.sample_rate,
            metadata={"voice_id": voice_id, **(metadata or {})},
        )

    def list_voices(
        self,
        *,
        search: str | None = None,
        voice_type: str | None = None,
        page_size: int = 100,
    ) -> list[VoiceInfo]:
        """Voices in the account and its library (``voice_type``: e.g. 'default',
        'personal', 'cloned', 'generated')."""
        optional = {"search": search, "voice_type": voice_type}
        response = self.client.voices.search(
            page_size=page_size, **{k: v for k, v in optional.items() if v}
        )
        return [
            VoiceInfo(
                voice_id=v.voice_id,
                name=v.name or "",
                backend=self.name,
                description=getattr(v, "description", None) or "",
                labels=dict(getattr(v, "labels", None) or {}),
            )
            for v in response.voices
        ]

    def design_voice_previews(
        self,
        description: str,
        *,
        text: str | None = None,
        model_id: str = DFLT_VOICE_DESIGN_MODEL_ID,
        seed: int | None = None,
        guidance_scale: float | None = None,
        loudness: float | None = None,
        **design_kwargs,
    ) -> list[VoiceDesignPreview]:
        """Candidate voices for a text ``description`` (ElevenLabs Voice Design).

        ``text`` (100-1000 characters) is what the previews say; without it
        ElevenLabs writes a fitting line. Nothing is added to the account until
        one preview is passed to ``design_voice``.
        """
        import base64

        optional = {
            "text": text,
            "auto_generate_text": None if text else True,
            "model_id": model_id,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "loudness": loudness,
        }
        response = self.client.text_to_voice.design(
            voice_description=description,
            **{k: v for k, v in optional.items() if v is not None},
            **design_kwargs,
        )
        return [
            VoiceDesignPreview(
                generated_voice_id=p.generated_voice_id,
                audio=base64.b64decode(p.audio_base_64),
                text=getattr(response, "text", None) or text or "",
                duration_s=getattr(p, "duration_secs", None),
            )
            for p in response.previews
        ]

    def design_voice(
        self,
        description: str,
        *,
        preview: "VoiceDesignPreview | str | None" = None,
        name: str | None = None,
        labels: Mapping[str, str] | None = None,
        speaker_id: int = 999,
        **preview_kwargs,
    ) -> VoiceProfile:
        """Create a voice from a text ``description`` and return its profile.

        Args:
            description: What the voice sounds like (age, accent, tone, pace...).
            preview: The chosen preview (or its ``generated_voice_id``) from
                ``design_voice_previews``. If omitted, previews are generated
                and the first is used.
            name: Voice name in ElevenLabs (default ``voxy-<speaker_id>``).
            labels: Voice labels (language, accent, gender, age...).
            **preview_kwargs: Passed to ``design_voice_previews`` when
                ``preview`` is omitted.
        """
        if preview is None:
            preview = self.design_voice_previews(description, **preview_kwargs)[0]
        generated_id = getattr(preview, "generated_voice_id", preview)
        name = name or DFLT_CLONE_NAME_TEMPLATE.format(speaker_id=speaker_id)
        optional = {"labels": dict(labels) if labels is not None else None}
        voice = self.client.text_to_voice.create(
            voice_name=name,
            voice_description=description,
            generated_voice_id=generated_id,
            **{k: v for k, v in optional.items() if v is not None},
        )
        return self.voice_profile(
            voice.voice_id,
            speaker_id=speaker_id,
            metadata={"name": name, "designed_from": description},
        )

    def synthesize(
        self,
        text: str | bytes | io.TextIOBase,
        voice: VoiceProfile | str | None = None,
        *,
        output_format: str = DFLT_ELEVENLABS_SYNTH_FORMAT,
        **kwargs,
    ) -> Speech:
        """Encoded speech (mp3 by default) in ``voice`` (a profile or voice id)."""
        if voice is None:
            raise ValueError(
                "ElevenLabs needs a voice: pass a VoiceProfile or a voice id"
            )
        text = _resolve_text_input(text)
        codec, rate = _output_format_parts(output_format)
        payload = self.synthesize_bytes(
            text, voice, output_format=output_format, **kwargs
        )
        if codec == "pcm":  # raw samples aren't a playable file: wrap them as WAV
            payload, codec = _wav_bytes(_pcm16_to_tensor(payload), rate), "wav"
        return Speech(
            payload,
            format=codec,
            backend=self.name,
            voice=_voice_id(voice),
            sample_rate=rate,
            text=text,
        )

    def delete_voice(self, voice: VoiceProfile | str) -> None:
        """Delete a voice from the ElevenLabs account."""
        self.client.voices.delete(_voice_id(voice))

    # --- synthesis ----------------------------------------------------------

    def synthesize_bytes(
        self,
        text: str | bytes | io.TextIOBase,
        voice_profile: VoiceProfile | str,
        *,
        model_id: str | None = None,
        output_format: str | None = None,
        **convert_kwargs,
    ) -> bytes:
        """The raw audio ElevenLabs returns, in ``output_format`` (any format).

        Extra keyword arguments (``voice_settings``, ``seed``, ``language_code``...)
        go straight to ``client.text_to_speech.convert``.
        """
        chunks = self.client.text_to_speech.convert(
            _voice_id(voice_profile),
            text=_resolve_text_input(text),
            model_id=model_id or self.model_id,
            output_format=output_format or self.output_format,
            **convert_kwargs,
        )
        return chunks if isinstance(chunks, bytes) else b"".join(chunks)

    def generate_speech(
        self,
        text: str | bytes | io.TextIOBase,
        voice_profile: VoiceProfile | str | None = None,
        output_path: str | None = None,
        max_length_ms: int = 10000,
        **kwargs,
    ) -> torch.Tensor:
        """Synthesize ``text`` in a cloned (or stock) ElevenLabs voice.

        Args:
            text: Text to synthesize.
            voice_profile: From ``clone_voice`` or ``voice_profile``, or a bare
                voice id. Required: ElevenLabs has no voiceless default.
            output_path: If given, save the audio there (WAV for pcm/wav formats,
                the raw payload otherwise).
            max_length_ms: Ignored; ElevenLabs sizes the audio to the text.
            **kwargs: Passed to ``synthesize_bytes``.

        Returns:
            A mono float tensor at the output format's sample rate (that is
            ``self.sample_rate`` unless ``output_format=`` is passed here).
        """
        if voice_profile is None:
            raise ValueError(
                "ElevenLabs needs a voice: pass a VoiceProfile or a voice id"
            )
        output_format = kwargs.pop("output_format", None) or self.output_format
        codec, rate = _output_format_parts(output_format)
        if codec not in ("pcm", "wav"):
            raise ValueError(
                f"Output format {output_format!r} can't be decoded to a tensor "
                "without ffmpeg; use synthesize_bytes() for compressed formats, "
                "or a pcm_*/wav_* format here."
            )
        payload = self.synthesize_bytes(
            text, voice_profile, output_format=output_format, **kwargs
        )
        if codec == "pcm":
            audio = _pcm16_to_tensor(payload)
        else:
            audio, rate = _wav_to_tensor(payload)
        if output_path:
            _write_bytes(output_path, _wav_bytes(audio, rate))
        return audio.to(self.device)


def _voice_id(voice: VoiceProfile | str) -> str:
    """The ElevenLabs voice id of a profile (or of a bare id).

    >>> _voice_id("abc")
    'abc'
    >>> _voice_id(VoiceProfile("abc", 1, "csm", 24000))
    Traceback (most recent call last):
      ...
    ValueError: Incompatible voice profile type: csm
    """
    if isinstance(voice, str):
        return voice
    if voice.model_type != ELEVENLABS_MODEL_TYPE:
        raise ValueError(f"Incompatible voice profile type: {voice.model_type}")
    return voice.segment


def _write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
