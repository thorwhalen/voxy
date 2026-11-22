"""Tests for the language seam (#131): say the language once, each engine gets its own option."""

import io
import wave
import warnings

import pytest

from voxy import Speech, SpeechModel, register_language_option, text_to_speech
from voxy.language import LanguageOption, _LANGUAGE_OPTIONS
from voxy.speech_cache import speech_key

SEED = "fal-ai/bytedance/seed-speech/tts/v2"


def _silence(n_frames=240, rate=24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"\0\0" * n_frames)
    return buf.getvalue()


class Recorder(SpeechModel):
    def __init__(self, name, model_id=None):
        super().__init__(device="cpu")
        self.name, self.model_id, self.calls = name, model_id, []

    def synthesize(self, text, voice=None, **kwargs):
        self.calls.append(kwargs)
        return Speech(_silence(), format="wav", backend=self.name, voice=None, text=text)


@pytest.mark.parametrize(
    "backend, model_id, sent",
    [
        ("fal", SEED, {"language": "en"}),
        ("elevenlabs", "eleven_flash_v2_5", {"language_code": "en"}),
        ("elevenlabs", "eleven_v3", {"language_code": "en"}),
    ],
)
def test_language_goes_in_each_engines_own_option(backend, model_id, sent):
    model = Recorder(backend, model_id)
    text_to_speech("Hello.", "v", model=model, language="en")
    assert model.calls == [sent]


def test_an_explicit_option_wins_and_the_cache_key_sees_the_language():
    model = Recorder("elevenlabs", "eleven_flash_v2_5")
    text_to_speech("Hello.", "v", model=model, language="en", language_code="fr")
    assert model.calls == [{"language_code": "fr"}]
    assert speech_key(model, "Hi", "v", {"language": "en"}) != speech_key(model, "Hi", "v", {})


def test_a_model_without_the_option_warns_and_sends_nothing():
    model = Recorder("elevenlabs", "eleven_multilingual_v2")  # would error on language_code
    with pytest.warns(UserWarning, match="takes no language option"):
        text_to_speech("Hello.", "v", model=model, language="en")
    assert model.calls == [{}]


def test_an_unpinned_guessing_model_warns_and_a_pinned_one_does_not():
    with pytest.warns(UserWarning, match="guesses the language"):
        text_to_speech("Hello.", "v", model=Recorder("fal", SEED))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text_to_speech("Hello.", "v", model=Recorder("fal", SEED), language="en")
        text_to_speech("Hello.", "v", model=Recorder("rec"))  # takes none, none asked: silent


def test_directed_text_keeps_the_language_on_every_part():
    model = Recorder("fal", SEED)
    text_to_speech("[dryly] Fine. [tenderly] Really.", "v", model=model, language="en")
    assert len(model.calls) >= 2 and all(c.get("language") == "en" for c in model.calls)


def test_a_new_model_registers_its_option(monkeypatch):
    monkeypatch.setitem(_LANGUAGE_OPTIONS, "rec", [])
    register_language_option("rec", "*", LanguageOption("lang"))
    model = Recorder("rec")
    text_to_speech("Hello.", "v", model=model, language="de")
    assert model.calls == [{"lang": "de"}]


def test_a_backend_nobody_registered_gets_language_passed_through_silently():
    # As before the seam: a local model that takes ``language`` itself keeps getting it.
    model = Recorder("someone-new")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text_to_speech("Bonjour.", "v", model=model, language="fr")
    assert model.calls == [{"language": "fr"}]


@pytest.mark.parametrize(
    "backend, sent",
    [("chatterbox", {"language": "fr"}), ("xtts", {"language": "fr"}), ("qwen3tts", {"language": "French"})],
)
def test_local_engines_get_their_own_spelling(backend, sent):
    model = Recorder(backend)
    text_to_speech("Bonjour.", "v", model=model, language="fr")
    assert model.calls == [sent]


def test_openai_takes_none_and_the_warnings_have_their_own_classes():
    from voxy import LanguageIgnoredWarning, LanguageUnsetWarning

    with pytest.warns(LanguageIgnoredWarning):
        text_to_speech("Hello.", "v", model=Recorder("aix", "gpt-4o-mini-tts"), language="en")
    with pytest.warns(LanguageUnsetWarning):
        text_to_speech("Hello.", "v", model=Recorder("elevenlabs", "eleven_v3"))


def test_a_model_that_is_not_a_speech_model_still_works():
    class Bare:
        name, dflt_voice = "bare", "v"

        def __init__(self):
            self.calls = []

        def synthesize(self, text, voice=None, **kwargs):
            self.calls.append(kwargs)
            return Speech(_silence(), format="wav", backend="bare", voice=None, text=text)

    model = Bare()
    text_to_speech("Hello.", "v", model=model, language="en")
    assert model.calls == [{"language": "en"}]
