"""
Voxy: A flexible speech synthesis and voice cloning module.

This module provides a plugin architecture for working with different speech synthesis
models, with initial support for the CSM-1B model.
"""

import os
import io
from typing import Union, Optional, List, Dict, Any, BinaryIO, Tuple
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torchaudio
import numpy as np
from huggingface_hub import hf_hub_download

# Try to import whisper for transcription, but don't fail if it's not available
try:
    import whisper

    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False


# TODO: Scan for models and define DFLT accordingly
DFLT_VOXY_MODEL = os.environ.get("DFLT_VOXY_MODEL", "csm")

# Sample rate assumed for raw (already-decoded) audio inputs -- tensors and
# numpy arrays carry no sample rate of their own. Callers that know the real
# rate should pass it explicitly rather than rely on this.
DFLT_ASSUMED_SAMPLE_RATE = 16000

# The sample rate Whisper expects its input audio to be at
WHISPER_SAMPLE_RATE = 16000

# Determine the default device for model inference
DFLT_VOXY_DEVICE = os.environ.get("DFLT_VOXY_DEVICE", None)

# Special case: CSM model has compatibility issues with MPS
# See error: "Output channels > 65536 not supported at the MPS device"
if DFLT_VOXY_DEVICE is None:
    if torch.cuda.is_available():
        DFLT_VOXY_DEVICE = "cuda"
    # Note: We're skipping MPS even if available for CSM compatibility
    else:
        DFLT_VOXY_DEVICE = "cpu"

VOXY_MODELS_CACHE_DIR = os.environ.get("VOXY_MODELS_CACHE_DIR")
if VOXY_MODELS_CACHE_DIR is None:
    standard_huggingface_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(standard_huggingface_cache):
        VOXY_MODELS_CACHE_DIR = standard_huggingface_cache
    else:
        VOXY_MODELS_CACHE_DIR = "~/.cache/voxy/models"
VOXY_MODELS_CACHE_DIR = os.path.expanduser(VOXY_MODELS_CACHE_DIR)


# -----------------------------------------------------------------------------
# Helper functions for input normalization
# -----------------------------------------------------------------------------


def _resolve_audio_input(
    audio_input: str | bytes | BinaryIO | torch.Tensor | np.ndarray,
    *,
    assumed_sample_rate: int = DFLT_ASSUMED_SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """
    Resolves various audio input formats to a torch.Tensor and sample rate.

    Args:
        audio_input: Audio in various formats:
            - str: Path to an audio file
            - bytes: Raw audio data
            - BinaryIO: File-like object containing audio data
            - torch.Tensor: Direct audio tensor
            - np.ndarray: Numpy array of audio samples
        assumed_sample_rate: Sample rate to report for raw tensor/array inputs,
            which carry no sample rate of their own. Ignored for inputs that
            are decoded (paths, bytes, file-like objects), since those have a
            real sample rate.

    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    if isinstance(audio_input, str):
        # Check if it's a file path
        if os.path.isfile(audio_input):
            return torchaudio.load(audio_input)
        else:
            raise ValueError(f"Audio path does not exist: {audio_input}")

    elif isinstance(audio_input, bytes):
        # Convert bytes to file-like object
        byte_stream = io.BytesIO(audio_input)
        return torchaudio.load(byte_stream)

    elif isinstance(audio_input, (io.IOBase, BinaryIO)):
        # File-like object
        return torchaudio.load(audio_input)

    elif isinstance(audio_input, torch.Tensor):
        # A bare tensor carries no sample rate; use the caller's assumption.
        # The tensor shape must be [channels, samples] or [samples]
        if len(audio_input.shape) > 2:
            raise ValueError(f"Invalid audio tensor shape: {audio_input.shape}")
        return audio_input, assumed_sample_rate

    elif isinstance(audio_input, np.ndarray):
        # Convert numpy array to tensor; it carries no sample rate either
        audio_tensor = torch.from_numpy(audio_input)
        if len(audio_tensor.shape) == 1:
            # Add channel dimension if not present
            audio_tensor = audio_tensor.unsqueeze(0)
        return audio_tensor, assumed_sample_rate

    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")


def _resolve_text_input(text_input: str | bytes | io.TextIOBase) -> str:
    """
    Resolves various text input formats to a string.

    Args:
        text_input: Text in various formats:
            - str: Direct text or path to a text file
            - bytes: UTF-8 encoded text
            - TextIOBase: File-like object containing text

    Returns:
        String containing the text

    >>> _resolve_text_input("Hello there.")
    'Hello there.'
    >>> _resolve_text_input(b"Hello there.")
    'Hello there.'
    >>> _resolve_text_input(io.StringIO("Hello there."))
    'Hello there.'
    >>> _resolve_text_input(42)
    Traceback (most recent call last):
      ...
    TypeError: Unsupported text input type: <class 'int'>
    """
    if isinstance(text_input, str):
        # If it starts with / and is a file, read the content
        if text_input.startswith("/") and os.path.isfile(text_input):
            with open(text_input) as f:
                return f.read()
        # Otherwise, use the string directly
        return text_input

    elif isinstance(text_input, bytes):
        # Decode bytes to string
        return text_input.decode("utf-8")

    elif isinstance(text_input, io.TextIOBase):
        # Read from file-like object
        return text_input.read()

    else:
        raise TypeError(f"Unsupported text input type: {type(text_input)}")


# -----------------------------------------------------------------------------
# Audio processing functions
# -----------------------------------------------------------------------------


def cleanup_audio(
    audio: torch.Tensor,
    sample_rate: int,
    normalize: bool = True,
    remove_silence: bool = True,
    silence_threshold: float = 0.02,
    min_silence_duration: float = 0.2,
) -> torch.Tensor:
    """
    Clean up audio by normalizing volume and removing silence.

    Args:
        audio: Audio tensor [channels, samples] or [samples]
        sample_rate: Sample rate of the audio
        normalize: Whether to normalize the audio volume
        remove_silence: Whether to remove silence
        silence_threshold: Threshold for silence detection (0.0-1.0)
        min_silence_duration: Minimum silence duration in seconds

    Returns:
        Processed audio tensor

    A 1-D input is given a channel dimension, and the loudest sample is
    normalized to 1.0:

    >>> audio = torch.tensor([0.0, 0.5, 0.0, 0.0])
    >>> processed = cleanup_audio(audio, sample_rate=8000)
    >>> tuple(processed.shape)
    (1, 4)
    >>> round(float(processed.max()), 3)
    1.0

    Stereo input is mixed down to mono:

    >>> stereo = torch.tensor([[0.0, 0.5, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]])
    >>> tuple(cleanup_audio(stereo, sample_rate=8000).shape)
    (1, 4)
    """
    # Ensure input is 2D with shape [channels, samples]
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    # Convert to mono if not already
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Move to CPU for processing
    device = audio.device
    audio = audio.cpu()

    # Normalize volume
    if normalize:
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / (max_val + 1e-8)

    # Remove silence
    if remove_silence:
        # Convert to numpy for easier processing
        audio_np = audio.squeeze(0).numpy()

        # Calculate energy
        energy = np.abs(audio_np)

        # Find regions above threshold (speech)
        is_speech = energy > silence_threshold

        # Convert min_silence_duration to samples
        min_silence_samples = int(min_silence_duration * sample_rate)

        # Find speech segments
        speech_segments = []
        in_speech = False
        speech_start = 0

        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                # Start of speech segment
                in_speech = True
                speech_start = i
            elif not is_speech[i] and in_speech:
                # Potential end of speech segment
                # Only end if silence is long enough
                silence_count = 0
                for j in range(i, min(len(is_speech), i + min_silence_samples)):
                    if not is_speech[j]:
                        silence_count += 1
                    else:
                        break

                if silence_count >= min_silence_samples:
                    # End of speech segment
                    in_speech = False
                    speech_segments.append((speech_start, i))

        # Handle case where audio ends during speech
        if in_speech:
            speech_segments.append((speech_start, len(is_speech)))

        # Concatenate speech segments
        if not speech_segments:
            # If no speech found, return original audio
            processed_audio = audio
        else:
            # Add small buffer around segments
            buffer_samples = int(0.05 * sample_rate)  # 50ms buffer
            processed_segments = []

            for start, end in speech_segments:
                buffered_start = max(0, start - buffer_samples)
                buffered_end = min(len(audio_np), end + buffer_samples)
                processed_segments.append(audio_np[buffered_start:buffered_end])

            # Concatenate all segments
            processed_audio_np = np.concatenate(processed_segments)
            processed_audio = torch.tensor(processed_audio_np, device="cpu").unsqueeze(
                0
            )
    else:
        processed_audio = audio

    # Return to original device
    return processed_audio.to(device)


def audio_to_text(
    audio_input: str | bytes | BinaryIO | torch.Tensor | np.ndarray,
    model_size: str = "base",
    *,
    sample_rate: int | None = None,
) -> str:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_input: Audio in various formats
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        sample_rate: Sample rate of ``audio_input`` when it is a raw tensor or
            numpy array. Required for correct transcription of raw audio that
            is not at ``DFLT_ASSUMED_SAMPLE_RATE``; ignored when the input is a
            path, bytes or file-like object (those carry their own rate).

    Returns:
        Transcribed text

    Raises:
        ImportError: If whisper is not installed
    """
    if not _HAS_WHISPER:
        raise ImportError(
            "openai-whisper is required for transcription. "
            "Install with 'pip install voxy[transcription]' "
            "(or 'pip install openai-whisper')."
        )

    # Resolve audio input
    audio, resolved_sample_rate = _resolve_audio_input(
        audio_input,
        assumed_sample_rate=(
            sample_rate if sample_rate is not None else DFLT_ASSUMED_SAMPLE_RATE
        ),
    )

    # Load whisper model
    model = whisper.load_model(model_size)

    # If audio is a torch tensor, convert to numpy array
    if isinstance(audio, torch.Tensor):
        # Ensure mono
        if len(audio.shape) > 1 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0)
        else:
            audio = audio.squeeze(0)

        # Convert to numpy
        audio_np = audio.cpu().numpy()
    else:
        audio_np = audio

    # Resample if needed -- Whisper expects WHISPER_SAMPLE_RATE
    if resolved_sample_rate != WHISPER_SAMPLE_RATE:
        # Use torchaudio for resampling
        audio_tensor = torch.tensor(audio_np).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor,
            orig_freq=resolved_sample_rate,
            new_freq=WHISPER_SAMPLE_RATE,
        )
        audio_np = audio_tensor.squeeze(0).numpy()

    # Transcribe
    result = model.transcribe(audio_np)

    return result["text"].strip()


# -----------------------------------------------------------------------------
# Main SpeechModel classes
# -----------------------------------------------------------------------------


@dataclass
class VoiceProfile:
    """Data class to store voice cloning information."""

    segment: Any  # Model-specific voice segment
    speaker_id: int
    model_type: str
    sample_rate: int
    metadata: dict[str, Any] | None = None


@dataclass
class Speech:
    """Synthesized speech: encoded audio plus what produced it.

    >>> import tempfile, os
    >>> speech = Speech(b"RIFF...", format="wav", backend="say", voice="Daniel")
    >>> path = speech.save(os.path.join(tempfile.mkdtemp(), "hi.wav"))
    >>> open(path, "rb").read()[:4]
    b'RIFF'
    """

    audio: bytes
    format: str  # container/codec: 'mp3', 'wav', 'opus', 'flac', 'aac', ...
    backend: str = ""
    voice: str | None = None
    sample_rate: int | None = None
    text: str | None = None

    def save(self, path: str) -> str:
        """Write the audio to ``path`` (folders created) and return the path."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.audio)
        return path


@dataclass
class VoiceInfo:
    """A voice a backend offers (stock, designed, or cloned)."""

    voice_id: str
    name: str
    backend: str
    description: str = ""
    labels: dict[str, Any] | None = None


def tensor_to_wav_bytes(audio: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a tensor ([channels, samples] or [samples]) as mono 16-bit WAV.

    Integer tensors are taken as PCM and scaled to [-1, 1] first.

    >>> data = tensor_to_wav_bytes(torch.zeros(160), 16000)
    >>> data[:4], len(data)
    (b'RIFF', 364)
    >>> tensor_to_wav_bytes(torch.tensor([[0, 32767]], dtype=torch.int16), 8000)[-2:]
    b'\\xff\\x7f'
    """
    import wave

    if not torch.is_floating_point(audio):
        audio = audio.to(torch.float64) / torch.iinfo(audio.dtype).max
    if audio.dim() == 2:
        audio = audio.mean(dim=0)
    samples = audio.detach().cpu().to(torch.float32).clamp(-1.0, 1.0).numpy()
    pcm = (samples * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm)
    return buffer.getvalue()


class SpeechModel:
    """Base class for speech backends (local models or services).

    A backend implements whichever capabilities it has; the rest raise
    ``NotImplementedError`` naming the backend:

    - ``synthesize(text, voice) -> Speech``: the facade's one required method
      (the default renders ``generate_speech`` to WAV);
    - ``list_voices() -> list[VoiceInfo]``;
    - ``clone_voice(samples, ...) -> VoiceProfile``;
    - ``design_voice(description, ...) -> VoiceProfile``;
    - ``generate_speech(text, profile) -> torch.Tensor``.
    """

    #: Registry name of the backend (also each profile's ``model_type``).
    name: str = ""
    #: Voice used when the caller names none (``None``: the caller must).
    dflt_voice: str | None = None

    def _unsupported(self, capability: str):
        return NotImplementedError(
            f"The {self.name or type(self).__name__!r} backend can't {capability}"
        )

    def synthesize(
        self, text: str, voice: "VoiceProfile | str | None" = None, **kwargs
    ) -> Speech:
        """Render ``text`` in ``voice`` and return encoded audio."""
        audio = self.generate_speech(text, voice, **kwargs)
        sample_rate = getattr(voice, "sample_rate", None) or self.sample_rate
        return Speech(
            tensor_to_wav_bytes(audio, sample_rate),
            format="wav",
            backend=self.name,
            voice=None if voice is None else str(getattr(voice, "segment", voice)),
            sample_rate=sample_rate,
            text=_resolve_text_input(text),
        )

    def list_voices(self) -> list[VoiceInfo]:
        """The voices this backend offers."""
        raise self._unsupported("list voices")

    def design_voice(self, description: str, **kwargs) -> VoiceProfile:
        """Create a new voice from a text description."""
        raise self._unsupported("design voices")

    def __init__(self, device: str = DFLT_VOXY_DEVICE):
        """
        Initialize the speech model.

        Args:
            device: Device for model inference ('cuda', 'cpu', 'mps')
        """
        self.device = device

    def clone_voice(
        self,
        audio_input: str | bytes | BinaryIO | torch.Tensor | np.ndarray,
        transcript: str | None = None,
        speaker_id: int = 999,
        *,
        cleanup_audio_fn: Callable | None = cleanup_audio,
    ) -> VoiceProfile:
        """
        Create a voice profile from an audio sample and its transcript.

        Args:
            audio_input: Audio in various formats
            transcript: Text transcription of the audio (if None, auto-transcribed)
            speaker_id: Unique ID for this voice
            cleanup_audio_fn: Function to clean up audio (None to skip)

        Returns:
            VoiceProfile: A packaged voice profile
        """
        raise self._unsupported("clone voices")

    def generate_speech(
        self,
        text: str | bytes | io.TextIOBase,
        voice_profile: VoiceProfile | None = None,
        output_path: str | None = None,
        max_length_ms: int = 10000,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate speech using a voice profile.

        Args:
            text: Text to synthesize
            voice_profile: Voice profile from clone_voice()
            output_path: Path to save the audio (optional)
            max_length_ms: Maximum audio length in milliseconds
            **kwargs: Additional model-specific parameters

        Returns:
            Generated audio tensor
        """
        raise self._unsupported("generate speech tensors")


class CSMSpeechModel(SpeechModel):
    """Speech model implementation using Sesame's CSM-1B model."""

    name = "csm"

    # Class-level cache for model path to avoid repeated downloads
    _model_cache_path = None

    def __init__(self, model_path: str | None = None, device: str = DFLT_VOXY_DEVICE):
        """
        Initialize the CSM speech model.

        Args:
            model_path: Path to the model checkpoint (None to download from HF)
            device: Device for model inference ('cuda', 'cpu')
                    Note: 'mps' is not supported due to model architecture limitations
        """
        # Enforce CPU if MPS is requested, as CSM doesn't work on MPS
        if device == "mps":
            print("Warning: CSM model is not compatible with MPS. Falling back to CPU.")
            device = "cpu"

        super().__init__(device)
        self.model_path = model_path
        self._generator = None  # Lazy initialization

    def _ensure_generator_loaded(self):
        """Ensure the generator is loaded."""
        if self._generator is None:
            # Import here to avoid dependencies if not using CSM
            from generator import load_csm_1b, Segment

            if self.model_path is None:
                # First check if we already have the model in the HF cache
                cache_dir = os.path.expanduser(VOXY_MODELS_CACHE_DIR)
                # Check if model already exists in cache
                possible_model_path = os.path.join(
                    cache_dir, "models--sesame--csm-1b/snapshots", "*", "ckpt.pt"
                )
                import glob

                cached_models = glob.glob(possible_model_path)

                if cached_models:
                    # Use the first match (most recent snapshot typically)
                    self.model_path = cached_models[0]
                    CSMSpeechModel._model_cache_path = self.model_path
                    print(f"Using existing model from cache: {self.model_path}")
                else:
                    # Check class cache
                    if CSMSpeechModel._model_cache_path is not None and os.path.exists(
                        CSMSpeechModel._model_cache_path
                    ):
                        self.model_path = CSMSpeechModel._model_cache_path
                        print(f"Using cached CSM model from: {self.model_path}")
                    else:
                        # Download the model if not provided
                        try:
                            # Create a consistent cache directory
                            os.makedirs(cache_dir, exist_ok=True)

                            print("Downloading CSM-1B model from Hugging Face Hub...")
                            self.model_path = hf_hub_download(
                                repo_id="sesame/csm-1b",
                                filename="ckpt.pt",
                                cache_dir=cache_dir,
                            )
                            # Update the class-level cache
                            CSMSpeechModel._model_cache_path = self.model_path
                            print(f"Model downloaded to: {self.model_path}")
                        except Exception as e:
                            raise RuntimeError(
                                "Failed to download CSM-1B model. Ensure you have huggingface-cli "
                                f"installed and are logged in with appropriate permissions: {e}"
                            )

            # Load the generator
            print(f"Loading CSM model on {self.device}...")
            self._generator = load_csm_1b(self.device)
            print("Model loaded successfully.")

            # Save a reference to the Segment class
            self.Segment = Segment

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio (loads the model)."""
        self._ensure_generator_loaded()
        return self._generator.sample_rate

    def clone_voice(
        self,
        audio_input: str | bytes | BinaryIO | torch.Tensor | np.ndarray,
        transcript: str | None = None,
        speaker_id: int = 999,
        *,
        cleanup_audio_fn: Callable | None = cleanup_audio,
    ) -> VoiceProfile:
        """
        Create a voice profile from an audio sample and its transcript.

        Args:
            audio_input: Audio in various formats
            transcript: Text transcription of the audio (if None, auto-transcribed)
            speaker_id: Unique ID for this voice
            cleanup_audio_fn: Function to clean up audio (None to skip)

        Returns:
            VoiceProfile: A packaged voice profile
        """
        # Load model if not already loaded
        self._ensure_generator_loaded()

        # Resolve audio input
        audio_tensor, sample_rate = _resolve_audio_input(audio_input)

        # Clean up audio if requested
        if cleanup_audio_fn is not None:
            audio_tensor = cleanup_audio_fn(audio_tensor, sample_rate)

        # Convert to mono if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

        # Squeeze out channel dimension if present
        audio_tensor = audio_tensor.squeeze(0)

        # Resample if needed
        if sample_rate != self._generator.sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor,
                orig_freq=sample_rate,
                new_freq=self._generator.sample_rate,
            )

        # Auto-transcribe if no transcript provided.
        # Pass the sample rate explicitly: audio_tensor has just been resampled
        # to the generator's rate, and a bare tensor carries no rate of its own,
        # so without this the audio would be transcribed as if it were at
        # DFLT_ASSUMED_SAMPLE_RATE (i.e. at the wrong speed).
        if transcript is None:
            transcript = audio_to_text(
                audio_tensor, sample_rate=self._generator.sample_rate
            )
        else:
            # Resolve transcript if not a string
            transcript = _resolve_text_input(transcript)

        # Create segment for voice profile
        segment = self.Segment(
            text=transcript, speaker=speaker_id, audio=audio_tensor.to(self.device)
        )

        # Create and return voice profile
        return VoiceProfile(
            segment=segment,
            speaker_id=speaker_id,
            model_type="csm",
            sample_rate=self._generator.sample_rate,
            metadata={
                "transcript_length": len(transcript),
                "audio_length_seconds": len(audio_tensor) / self._generator.sample_rate,
            },
        )

    def generate_speech(
        self,
        text: str | bytes | io.TextIOBase,
        voice_profile: VoiceProfile | None = None,
        output_path: str | None = None,
        max_length_ms: int = 10000,
        temperature: float = 0.7,
        topk: int = 30,
    ) -> torch.Tensor:
        """
        Generate speech using a voice profile.

        Args:
            text: Text to synthesize
            voice_profile: Voice profile from clone_voice()
            output_path: Path to save the audio (optional)
            max_length_ms: Maximum audio length in milliseconds
            temperature: Sampling temperature (lower = more deterministic)
            topk: Top-k sampling parameter

        Returns:
            Generated audio tensor
        """
        # Load model if not already loaded
        self._ensure_generator_loaded()

        # Resolve text input
        text = _resolve_text_input(text)

        # Set up context and speaker ID
        if voice_profile is not None:
            if voice_profile.model_type != "csm":
                raise ValueError(
                    f"Incompatible voice profile type: {voice_profile.model_type}"
                )

            context = [voice_profile.segment]
            speaker_id = voice_profile.speaker_id
        else:
            # No voice profile, use default speaker
            context = []
            speaker_id = 0

        # Add punctuation if missing to help with phrasing
        if not any(p in text for p in [".", ",", "!", "?"]):
            text = text + "."

        # Generate audio
        audio = self._generator.generate(
            text=text,
            speaker=speaker_id,
            context=context,
            max_audio_length_ms=max_length_ms,
            temperature=temperature,
            topk=topk,
        )

        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            torchaudio.save(
                output_path, audio.unsqueeze(0).cpu(), self._generator.sample_rate
            )

        return audio


# -----------------------------------------------------------------------------
# Factory function for creating speech models
# -----------------------------------------------------------------------------


def _lazy_factory(module: str, class_name: str) -> Callable[..., SpeechModel]:
    """A factory importing its backend on first use (backends import this module)."""

    def factory(**kwargs) -> SpeechModel:
        import importlib

        return getattr(importlib.import_module(module), class_name)(**kwargs)

    factory.__name__ = f"create_{class_name}"
    return factory


#: Backend name -> factory (keys lowercase). Add one with ``register_speech_model``.
speech_model_factories: dict[str, Callable[..., SpeechModel]] = {
    "csm": CSMSpeechModel,
    "csm-1b": CSMSpeechModel,
    "elevenlabs": _lazy_factory("voxy.elevenlabs_model", "ElevenLabsSpeechModel"),
    "aix": _lazy_factory("voxy.aix_model", "AixSpeechModel"),
    "fal": _lazy_factory("voxy.fal_model", "FalSpeechModel"),
    "say": _lazy_factory("voxy.say_model", "SaySpeechModel"),
}


def register_speech_model(
    name: str, factory: Callable[..., SpeechModel], *, overwrite: bool = False
) -> Callable[..., SpeechModel]:
    """Register a backend (a class or ``**kwargs -> SpeechModel`` callable).

    Returns ``factory``, so it also works as a class decorator via ``functools.partial``.

    >>> class Echo(SpeechModel):
    ...     name = "echo"
    >>> _ = register_speech_model("echo", Echo)
    >>> type(create_speech_model("echo")).__name__
    'Echo'
    >>> register_speech_model("echo", Echo)
    Traceback (most recent call last):
      ...
    ValueError: A speech backend named 'echo' is already registered (pass overwrite=True)
    >>> del speech_model_factories["echo"]
    """
    key = name.lower()
    if key in speech_model_factories and not overwrite:
        raise ValueError(
            f"A speech backend named {name!r} is already registered "
            "(pass overwrite=True)"
        )
    speech_model_factories[key] = factory
    return factory


def create_speech_model(model_type: str = DFLT_VOXY_MODEL, **kwargs) -> SpeechModel:
    """
    Create a speech model of the specified type.

    Args:
        model_type: A key of ``speech_model_factories`` ('csm', 'csm-1b',
            'elevenlabs', 'aix', 'fal', 'say', or any registered); case-insensitive.
        **kwargs: Additional model-specific parameters

    Returns:
        SpeechModel instance

    Raises:
        ValueError: If the model type is not supported

    The returned model loads its (large) weights lazily, on first use:

    >>> model = create_speech_model("csm")
    >>> type(model).__name__
    'CSMSpeechModel'
    >>> type(create_speech_model("elevenlabs", api_key="unused")).__name__
    'ElevenLabsSpeechModel'
    >>> create_speech_model("no-such-model")
    Traceback (most recent call last):
      ...
    ValueError: Unsupported model type: no-such-model (supported: csm, csm-1b, elevenlabs, aix, fal, say)
    """
    factories = {k.lower(): v for k, v in speech_model_factories.items()}
    factory = factories.get(model_type.lower())
    if factory is None:
        raise ValueError(
            f"Unsupported model type: {model_type} "
            f"(supported: {', '.join(speech_model_factories)})"
        )
    return factory(**kwargs)
