"""Delivery directions: rendered per dialect by the facade, and the speech cache."""

import numpy as np
import pytest

from voxy import (
    INLINE,
    PLAIN,
    Speech,
    SpeechModel,
    dialect_for,
    instruction_dialect,
    plan_directions,
    register_dialect,
    text_to_speech,
)
from voxy.direction import _DIALECTS, compose_instruction
from voxy.wavutil import read_wav, write_wav

RATE = 8000


class Voice(SpeechModel):
    """0.1 s of tone per call, recording (text, options); speaks the dialect it is given."""

    def __init__(self, name="tone", model_id=None):
        super().__init__(device="cpu")
        self.name, self.model_id, self.calls = name, model_id, []

    def synthesize(self, text, voice=None, **kwargs):
        self.calls.append((text, kwargs))
        audio = write_wav(np.full(RATE // 10, 0.5, np.float32), RATE)
        return Speech(audio, format="wav", backend=self.name, voice=voice, text=text)


SEED = "fal-ai/bytedance/seed-speech/tts/v2"
TEXT = "[dryly] We all agree. [beat] But [emphatic] never on how."


def test_the_registry_knows_who_speaks_what():
    assert dialect_for("elevenlabs", "eleven_v3") is INLINE
    assert dialect_for("elevenlabs", "eleven_multilingual_v2") is PLAIN
    assert dialect_for("fal", SEED).instruction_param == "voice_instruction"
    assert dialect_for("aix", "gpt-4o-mini-tts").instruction_param == "instructions"
    assert dialect_for("say") is PLAIN and dialect_for("fal", None) is PLAIN


def test_instruction_models_get_one_request_per_direction_with_real_silence():
    model = Voice("fal", SEED)
    speech = text_to_speech(TEXT, "tim_en", model=model, instruction="An essayist.")
    assert [(t, kw["voice_instruction"]) for t, kw in model.calls] == [
        ("We all agree.", "An essayist. Delivery: dry, understated, a little ironic."),
        ("But", "An essayist. Delivery: dry, understated, a little ironic."),
        ("never on how.", "An essayist. Delivery: emphatic, leaning on the key words."),
    ]
    samples, rate = read_wav(speech.audio)
    assert len(samples) / rate == pytest.approx(0.3 + 0.45, abs=0.01)  # 3 parts + [beat]
    assert speech.text == TEXT


def test_plain_models_never_hear_a_tag_and_inline_models_hear_all_of_them():
    plain = Voice("say")
    text_to_speech(TEXT, "Daniel", model=plain)
    assert all("[" not in t for t, _ in plain.calls)
    assert all("voice_instruction" not in kw for _, kw in plain.calls)
    inline = Voice("elevenlabs", "eleven_v3")
    text_to_speech(TEXT, "x", model=inline)
    assert inline.calls == [(TEXT, {})]


def test_keep_and_strip_override_auto():
    model = Voice("fal", SEED)
    text_to_speech(TEXT, "tim_en", model=model, directions="keep")
    assert model.calls[-1][0] == TEXT
    text_to_speech(TEXT, "tim_en", model=model, directions="strip")
    assert model.calls[-1][0] == "We all agree. But never on how."
    with pytest.raises(ValueError, match="directions"):
        text_to_speech(TEXT, "tim_en", model=model, directions="loud")


def test_untagged_text_is_one_plain_request():
    model = Voice("fal", SEED)
    text_to_speech("Just words.", "tim_en", model=model)
    assert model.calls == [("Just words.", {})]


def test_unknown_tags_become_instructions_and_registration_extends():
    assert plan_directions("[tenderly] Hi.", instruction_dialect("i"))[0].options == {
        "i": "Delivery: tenderly."
    }
    assert compose_instruction(None, ["beat"]) == ""
    register_dialect("say", "special*", instruction_dialect("style"))
    try:
        assert dialect_for("say", "special-1").instruction_param == "style"
    finally:
        _DIALECTS["say"].pop(0)


def test_cache_serves_repeats_and_reports_it(tmp_path):
    model = Voice("fal", SEED)
    first = text_to_speech(TEXT, "tim_en", model=model, cache=tmp_path)
    n = len(model.calls)
    again = text_to_speech(TEXT, "tim_en", model=model, cache=tmp_path)
    assert len(model.calls) == n and again.cached and not first.cached
    assert again.audio == first.audio
    text_to_speech(TEXT, "tim_en", model=model, cache=tmp_path, instruction="Other.")
    assert len(model.calls) == 2 * n  # a different instruction is a different request
