"""``voxy.plan_requests``: the requests a model needs for a directed text, without speaking."""

import numpy as np
import pytest

from voxy import Request, Speech, SpeechModel, native_break_for, plan_requests, register_native_breaks, text_to_speech
from voxy.pauses import _NATIVE_BREAKS
from voxy.wavutil import write_wav

SEED = "fal-ai/bytedance/seed-speech/tts/v2"
TEXT = '[dryly] We all agree. <break time="2s"/> [beat] But [emphatic] never on how.'


class Recorder(SpeechModel):
    """Records each (text, options) it is asked to speak; renders breaks up to ``native`` itself."""

    def __init__(self, name, model_id, native=0.0):
        super().__init__(device="cpu")
        self.name, self.model_id, self.native, self.calls = name, model_id, native, []

    def native_break_s(self, **_):
        return self.native

    def synthesize(self, text, voice=None, **kwargs):
        self.calls.append((text, kwargs))
        return Speech(write_wav(np.zeros(80, np.float32), 8000), format="wav", backend=self.name, voice=voice, text=text)


@pytest.mark.parametrize(
    "backend, model_id, native, language",
    [("fal", SEED, 0.0, "en"), ("elevenlabs", "eleven_v3", 0.0, "en"), ("say", None, 0.0, None), ("fal", SEED, 3.0, "fr")],
)
def test_the_plan_is_what_text_to_speech_sends(backend, model_id, native, language):
    model = Recorder(backend, model_id, native)
    text_to_speech(TEXT, "someone", model=model, instruction="An essayist.", language=language)
    planned = plan_requests(TEXT, backend, model_id, instruction="An essayist.", native_break_s=native, language=language)
    spoken = [(r.text, r.options) for r in planned if r.text]
    assert [(t, {k: v for k, v in kw.items() if k in r_opts}) for (t, kw), (_, r_opts) in zip(model.calls, spoken)] == spoken
    assert len(model.calls) == len(spoken)


def test_an_instruction_model_gets_one_request_per_direction_and_the_pauses_between():
    plan = plan_requests(TEXT, "fal", SEED, instruction="An essayist.")
    assert [r.text for r in plan] == ["We all agree.", "But", "never on how."]
    assert [r.pause_s for r in plan] == [pytest.approx(2.45), 0.0, 0.0]  # the break and the [beat], one silence
    assert plan[0].options == {"voice_instruction": "An essayist. Delivery: dry, understated, a little ironic."}
    assert all(isinstance(r, Request) for r in plan)


def test_a_model_that_renders_short_breaks_keeps_them_in_the_text():
    short = 'One. <break time="1s"/> Two. <break time="5s"/> Three.'
    v2 = plan_requests(short, "elevenlabs", "eleven_multilingual_v2")
    v3 = plan_requests(short, "elevenlabs", "eleven_v3")
    assert [(r.text, r.pause_s) for r in v2] == [('One. <break time="1s" /> Two.', 5.0), ("Three.", 0.0)]
    assert [(r.text, r.pause_s) for r in v3] == [("One.", 1.0), ("Two.", 5.0), ("Three.", 0.0)]


def test_a_backend_registers_its_break_table():
    try:
        register_native_breaks("acme", {"acme-1": 2.0})
        assert native_break_for("acme", "acme-1") == 2.0 and native_break_for("acme", "acme-2") == 0.0
        assert [r.text for r in plan_requests('A. <break time="1s"/> B.', "acme", "acme-1")] == ['A. <break time="1s" /> B.']
    finally:
        _NATIVE_BREAKS.pop("acme", None)


def test_the_elevenlabs_table_is_one_object_wherever_it_is_read():
    from voxy.elevenlabs_model import ELEVENLABS_NATIVE_BREAK_S, ElevenLabsSpeechModel

    assert ELEVENLABS_NATIVE_BREAK_S is _NATIVE_BREAKS["elevenlabs"]
    assert ElevenLabsSpeechModel(model_id="eleven_flash_v2_5").native_break_s() == native_break_for("elevenlabs", "eleven_flash_v2_5") == 3.0
