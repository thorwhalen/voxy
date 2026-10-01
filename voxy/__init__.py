"""
Facade for voice generation: speech synthesis in any voice, cloning and designing voices.

>>> import voxy
>>> sorted(voxy.speech_model_factories)
['aix', 'csm', 'csm-1b', 'elevenlabs', 'fal', 'say']
"""

from voxy.base import (
    create_speech_model,
    register_speech_model,
    speech_model_factories,
    audio_to_text,
    cleanup_audio,
    Speech,
    SpeechModel,
    VoiceInfo,
    VoiceProfile,
)
from voxy.elevenlabs_model import ElevenLabsSpeechModel
from voxy.stores import voxy_data_dir, voices_store, samples_store, sources_store
from voxy.library import (
    save_voice,
    load_voice,
    find_voice,
    clone_from_samples,
    design_from_description,
)
from voxy.facade import (
    text_to_speech,
    list_voices,
    voice_id,
    resolve_voice,
    get_speech_model,
)
