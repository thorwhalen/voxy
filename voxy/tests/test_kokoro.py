"""Tests for the kokoro backend (stock voices through ``LocalSpeechModel``)."""

import numpy as np
import pytest

import voxy
from voxy.kokoro_model import KOKORO, KOKORO_VOICES, KokoroSpeechModel
from voxy.local_clone import LocalCloneSpeechModel, reference_profile


class FakeGenerate:
    """Records each synthesize request instead of running a model."""

    def __init__(self):
        self.requests = []

    def __call__(self, text, **request):
        self.requests.append({"text": text, **request})
        return np.full(240, 0.1, dtype=np.float32), 24000


def test_language_follows_the_voice_unless_given():
    gen = FakeGenerate()
    model = KokoroSpeechModel(generate=gen)
    model.synthesize("Bonjour.", "ff_siwis")
    model.synthesize("Hello.", "ff_siwis", lang="en-us", speed=1.2)
    assert [(r["voice"], r["lang"]) for r in gen.requests] == [
        ("ff_siwis", "fr-fr"),
        ("ff_siwis", "en-us"),
    ]
    assert gen.requests[1]["speed"] == 1.2


def test_long_text_is_chunked_and_joined():
    gen = FakeGenerate()
    speech = KokoroSpeechModel(generate=gen, trim=False).synthesize(
        "One.\n\nTwo.", "am_michael"
    )
    assert [r["text"] for r in gen.requests] == ["One.", "Two."]
    assert speech.format == "wav" and speech.voice == "am_michael"
    assert speech.audio[:4] == b"RIFF"


def test_default_voice_and_its_override(monkeypatch):
    gen = FakeGenerate()
    monkeypatch.delenv("VOXY_KOKORO_VOICE", raising=False)
    assert KokoroSpeechModel(generate=gen).dflt_voice == "af_heart"
    monkeypatch.setenv("VOXY_KOKORO_VOICE", "bf_emma")
    assert KokoroSpeechModel(generate=gen).dflt_voice == "bf_emma"
    assert KokoroSpeechModel(generate=gen, voice="am_adam").dflt_voice == "am_adam"
    KokoroSpeechModel(generate=gen).synthesize("Hi.")
    assert gen.requests[-1]["voice"] == "bf_emma"


def test_unknown_voice_is_refused_before_the_model_runs():
    gen = FakeGenerate()
    with pytest.raises(ValueError, match=r"voxy\.list_voices\('kokoro'\)"):
        KokoroSpeechModel(generate=gen).synthesize("Hi.", "alex")
    assert gen.requests == []


def test_a_newer_voices_file_needs_no_code_change():
    gen = FakeGenerate()
    model = KokoroSpeechModel(generate=gen, voices=KOKORO_VOICES + ("bf_newone",))
    model.synthesize("Hi.", "bf_newone")
    assert gen.requests[-1]["lang"] == "en-gb"


def test_no_cloning():
    with pytest.raises(NotImplementedError, match="kokoro"):
        KokoroSpeechModel(generate=FakeGenerate()).clone_voice("ref.wav", "Hi.")


def test_only_cpu():
    assert KokoroSpeechModel(generate=FakeGenerate(), device="cpu").device == "cpu"
    with pytest.raises(ValueError, match="cpu"):
        KokoroSpeechModel(generate=FakeGenerate(), device="cuda")


def test_facade_speaks_kokoro_through_its_model_cache():
    gen = FakeGenerate()
    cache = {"kokoro": KokoroSpeechModel(generate=gen)}
    speech = voxy.text_to_speech(
        "Hi.", "bf_emma", backend="kokoro", models=cache, voices={}
    )
    assert speech.backend == "kokoro" and speech.voice == "bf_emma"
    assert gen.requests == [{"text": "Hi.", "voice": "bf_emma", "lang": "en-gb"}]


def test_listing_voices_builds_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXY_DATA_DIR", str(tmp_path))
    voices = voxy.list_voices("kokoro", models={})
    assert len(voices) == len(KOKORO_VOICES) == 54
    assert (voices[0].voice_id, voices[0].labels) == (
        "af_alloy",
        {"language": "en-us", "gender": "female"},
    )
    assert list(tmp_path.iterdir()) == []


def test_kokoro_env_needs_no_torch_and_fetches_through_the_hub():
    assert KOKORO.env.torch_backend is None
    assert not any("torch" in r for r in KOKORO.env.requirements)
    assert "huggingface_hub" in KOKORO.env.requirements
    assert {"repo_id", "revision", "model_file", "voices_file"} <= set(KOKORO.init)


def test_params_cannot_replace_a_cloners_reference():
    gen = FakeGenerate()
    model = LocalCloneSpeechModel("f5tts", generate=gen)
    voice = reference_profile("ref.wav", "Reference words.")
    with pytest.raises(TypeError, match="ref_audio"):
        model.synthesize("Hi.", voice, ref_audio="other.wav")
    assert gen.requests == []
