"""Character voices (#128): a library record = stock voice + acting direction + model."""

import pytest

from voxy import add_character, load_voice, text_to_speech
from voxy.aix_model import AixSpeechModel
from voxy.stores import voices_store

DIRECTION = "Light French accent, theatrical, Paris 1882."


def _aix():
    calls = []

    def tts(text, **kw):
        calls.append(kw)
        return b"mp3"

    return AixSpeechModel(tts=tts), calls


@pytest.fixture
def library(tmp_path):
    voices = voices_store(rootdir=tmp_path)
    add_character("sarah-bernhardt", voice="shimmer", backend="aix", direction=DIRECTION, model="gpt-4o-mini-tts", voices=voices, aliases=["Sarah"])
    return voices


def test_a_character_speaks_with_its_voice_direction_and_model(library):
    model, calls = _aix()
    text_to_speech("Bonjour, Paris.", "sarah", voices=library, models={"aix": model})
    (kw,) = calls
    assert kw["voice"] == "shimmer"
    assert kw["model"] == "gpt-4o-mini-tts"
    assert DIRECTION in kw["instructions"]


def test_the_calls_own_instruction_and_options_win(library):
    model, calls = _aix()
    text_to_speech("Hi.", "sarah-bernhardt", voices=library, models={"aix": model}, instruction="Whispering.", model_id="gpt-4o-mini-tts-2")
    (kw,) = calls
    assert kw["instructions"] == "Whispering." and kw["model"] == "gpt-4o-mini-tts-2"


def test_the_record_is_an_ordinary_library_voice(library):
    record = library["sarah-bernhardt"]
    assert record["kind"] == "character" and record["aliases"] == ["Sarah"]
    profile = load_voice("Sarah", model_type="aix", voices=library)
    assert (profile.segment, profile.model_type, profile.metadata["direction"]) == ("shimmer", "aix", DIRECTION)


def test_model_as_a_string_is_the_backends_model_id():
    # Was: AttributeError "'str' object has no attribute 'synthesize'".
    model, calls = _aix()
    text_to_speech("Hi.", "nova", backend="aix", model="gpt-4o-mini-tts", models={"aix": model}, instruction="Dry.")
    (kw,) = calls
    assert kw["model"] == "gpt-4o-mini-tts"
    assert kw["instructions"] == "Dry."  # the model id reached the dialect lookup too


def test_a_direction_on_a_model_that_takes_none_warns(tmp_path):
    voices = voices_store(rootdir=tmp_path)
    add_character("narrator", voice="nova", backend="aix", direction="Warm.", voices=voices)  # no model named
    model, calls = _aix()
    with pytest.warns(UserWarning, match="takes no delivery instruction"):
        text_to_speech("Hi.", "narrator", voices=voices, models={"aix": model})
    assert "instructions" not in calls[0]


def test_other_profiles_metadata_is_not_read_as_settings():
    from voxy import VoiceProfile

    model, calls = _aix()
    cloned = VoiceProfile("nova", 0, "aix", 0, metadata={"options": {"speed": 9}, "direction": "x"})
    text_to_speech("Hi.", cloned, models={"aix": model})
    assert "speed" not in calls[0] and "instructions" not in calls[0]


def test_add_character_refuses_unknown_backends_and_existing_voices(tmp_path):
    from voxy import VoiceProfile, save_voice

    voices = voices_store(rootdir=tmp_path)
    with pytest.raises(ValueError, match="Unknown backend"):
        add_character("x", voice="v", backend="nope", voices=voices)
    save_voice("ada", VoiceProfile("el-ada", 1, "elevenlabs", 24000), voices=voices)
    with pytest.raises(ValueError, match="not a character"):
        add_character("ada", voice="nova", backend="aix", voices=voices)


def test_model_and_model_id_naming_two_models_is_an_error():
    model, _ = _aix()
    with pytest.raises(ValueError, match="name two models"):
        text_to_speech("Hi.", "nova", backend="aix", model="tts-1", model_id="gpt-4o-mini-tts", models={"aix": model})
