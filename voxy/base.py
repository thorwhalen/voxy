"""
Voxy: A flexible speech synthesis and voice cloning module.

This module provides a plugin architecture for working with different speech synthesis
models, with initial support for the CSM-1B model.
"""

import os
import io
from typing import Union, Optional, List, Dict, Any, Callable, BinaryIO, Tuple
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

# Determine the default device for model inference
DFLT_VOXY_DEVICE = os.environ.get("DFLT_VOXY_DEVICE", None)

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
    standard_huggingface_cache = os.path.expanduser('~/.cache/huggingface/hub')
    if os.path.exists(standard_huggingface_cache):
        VOXY_MODELS_CACHE_DIR = standard_huggingface_cache
    else:
        VOXY_MODELS_CACHE_DIR = '~/.cache/voxy/models'
VOXY_MODELS_CACHE_DIR = os.path.expanduser(VOXY_MODELS_CACHE_DIR)


# -----------------------------------------------------------------------------
# Helper functions for input normalization
# -----------------------------------------------------------------------------


def _resolve_audio_input(
    audio_input: Union[str, bytes, BinaryIO, torch.Tensor, np.ndarray],
) -> Tuple[torch.Tensor, int]:
    """
    Resolves various audio input formats to a torch.Tensor and sample rate.

    Args:
        audio_input: Audio in various formats:
            - str: Path to an audio file
            - bytes: Raw audio data
            - BinaryIO: File-like object containing audio data
            - torch.Tensor: Direct audio tensor
            - np.ndarray: Numpy array of audio samples

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
        # Assume default sample rate of 16000 if directly passed tensor
        # and the tensor shape is [channels, samples] or [samples]
        if len(audio_input.shape) > 2:
            raise ValueError(f"Invalid audio tensor shape: {audio_input.shape}")
        return audio_input, 16000

    elif isinstance(audio_input, np.ndarray):
        # Convert numpy array to tensor
        # Assume default sample rate of 16000
        audio_tensor = torch.from_numpy(audio_input)
        if len(audio_tensor.shape) == 1:
            # Add channel dimension if not present
            audio_tensor = audio_tensor.unsqueeze(0)
        return audio_tensor, 16000

    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")


def _resolve_text_input(text_input: Union[str, bytes, io.TextIOBase]) -> str:
    """
    Resolves various text input formats to a string.

    Args:
        text_input: Text in various formats:
            - str: Direct text or path to a text file
            - bytes: UTF-8 encoded text
            - TextIOBase: File-like object containing text

    Returns:
        String containing the text
    """
    if isinstance(text_input, str):
        # If it starts with / and is a file, read the content
        if text_input.startswith('/') and os.path.isfile(text_input):
            with open(text_input, 'r') as f:
                return f.read()
        # Otherwise, use the string directly
        return text_input

    elif isinstance(text_input, bytes):
        # Decode bytes to string
        return text_input.decode('utf-8')

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
            processed_audio = torch.tensor(processed_audio_np, device='cpu').unsqueeze(
                0
            )
    else:
        processed_audio = audio

    # Return to original device
    return processed_audio.to(device)


def audio_to_text(
    audio_input: Union[str, bytes, BinaryIO, torch.Tensor, np.ndarray],
    model_size: str = "base",
) -> str:
    """
    Transcribe audio to text using Whisper.

    Args:
        audio_input: Audio in various formats
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')

    Returns:
        Transcribed text

    Raises:
        ImportError: If whisper is not installed
    """
    if not _HAS_WHISPER:
        raise ImportError(
            "whisper is required for transcription. Install with 'pip install whisper'"
        )

    # Resolve audio input
    audio, sample_rate = _resolve_audio_input(audio_input)

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

    # Resample if needed
    if sample_rate != 16000:
        # Whisper expects 16kHz
        # Use torchaudio for resampling
        audio_tensor = torch.tensor(audio_np).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=16000
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
    metadata: Dict[str, Any] = None


class SpeechModel:
    """Base class for speech synthesis models."""

    def __init__(self, device: str = DFLT_VOXY_DEVICE):
        """
        Initialize the speech model.

        Args:
            device: Device for model inference ('cuda', 'cpu', 'mps')
        """
        self.device = device

    def clone_voice(
        self,
        audio_input: Union[str, bytes, BinaryIO, torch.Tensor, np.ndarray],
        transcript: Optional[str] = None,
        speaker_id: int = 999,
        *,
        cleanup_audio_fn: Optional[Callable] = cleanup_audio,
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
        raise NotImplementedError("Subclasses must implement this method")

    def generate_speech(
        self,
        text: Union[str, bytes, io.TextIOBase],
        voice_profile: Optional[VoiceProfile] = None,
        output_path: Optional[str] = None,
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
        raise NotImplementedError("Subclasses must implement this method")


class CSMSpeechModel(SpeechModel):
    """Speech model implementation using Sesame's CSM-1B model."""

    # Class-level cache for model path to avoid repeated downloads
    _model_cache_path = None

    def __init__(
        self, model_path: Optional[str] = None, device: str = DFLT_VOXY_DEVICE
    ):
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

    def clone_voice(
        self,
        audio_input: Union[str, bytes, BinaryIO, torch.Tensor, np.ndarray],
        transcript: Optional[str] = None,
        speaker_id: int = 999,
        *,
        cleanup_audio_fn: Optional[Callable] = cleanup_audio,
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

        # Auto-transcribe if no transcript provided
        if transcript is None:
            transcript = audio_to_text(audio_tensor)
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
        text: Union[str, bytes, io.TextIOBase],
        voice_profile: Optional[VoiceProfile] = None,
        output_path: Optional[str] = None,
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
        if not any(p in text for p in ['.', ',', '!', '?']):
            text = text + '.'

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


def create_speech_model(model_type: str = "csm", **kwargs) -> SpeechModel:
    """
    Create a speech model of the specified type.

    Args:
        model_type: Type of speech model ('csm' or 'csm-1b' currently supported)
        **kwargs: Additional model-specific parameters

    Returns:
        SpeechModel instance

    Raises:
        ValueError: If the model type is not supported
    """
    if model_type.lower() in ["csm", "csm-1b"]:
        return CSMSpeechModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
