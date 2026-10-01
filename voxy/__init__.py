"""
Facade for voice cloning and speech synthesis
"""

from voxy.base import (
    create_speech_model,
    speech_model_factories,
    audio_to_text,
    cleanup_audio,
    SpeechModel,
    VoiceProfile,
)
from voxy.elevenlabs_model import ElevenLabsSpeechModel
from voxy.stores import voxy_data_dir, voices_store, samples_store, sources_store
from voxy.library import save_voice, load_voice, clone_from_samples
