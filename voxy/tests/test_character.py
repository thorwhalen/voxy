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
