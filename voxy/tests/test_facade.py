"""Tests for the voice-generation facade and the non-ElevenLabs backends (all faked)."""

import json
from types import SimpleNamespace

import pytest

import voxy
from voxy import (
    ElevenLabsSpeechModel,
    Speech,
    SpeechModel,
    VoiceProfile,
    design_from_description,
    list_voices,
    register_speech_model,
    save_voice,
    speech_model_factories,
    text_to_speech,
    voice_id,
)
from voxy.__main__ import main
from voxy.aix_model import AixSpeechModel
from voxy.fal_model import FalSpeechModel
from voxy.say_model import SaySpeechModel
from voxy.stores import voices_store
from voxy.tests.test_elevenlabs_model import FakeClient


class Recorder(SpeechModel):
    """A backend that records what it was asked to say."""

    name = "rec"
    dflt_voice = "house"

    def __init__(self, **kwargs):
        super().__init__(device="cpu")
        self.calls = []

    def synthesize(self, text, voice=None, **kwargs):
        self.calls.append((text, voice, kwargs))
        return Speech(b"audio", format="wav", backend=self.name, voice=str(getattr(voice, "segment", voice)))


@pytest.fixture
def library(tmp_path):
    voices = voices_store(rootdir=tmp_path)
    save_voice("cora", VoiceProfile("el-cora", 1, "elevenlabs", 24000), voices=voices, aliases=["Cora", "Coco"])
    save_voice("cora", VoiceProfile("Kathy", 1, "rec", 22050), voices=voices)
    return voices


def test_library_voice_uses_its_saved_backend(library):
    model = Recorder()
    speech = text_to_speech("Hi", "coco", voices=library, model=model, backend="rec")
    (text, voice, _), = model.calls
    assert (text, voice.segment, speech.voice) == ("Hi", "Kathy", "Kathy")


def test_library_voice_missing_backend_profile_is_explained(library):
    with pytest.raises(KeyError, match="no 'say' profile"):
        text_to_speech("Hi", "cora", backend="say", voices=library)


def test_native_voice_and_default_voice(library):
    model = Recorder()
    text_to_speech("Hi", "nova", backend="rec", voices=library, model=model)
    text_to_speech("Hi", backend="rec", voices=library, model=model)
    assert [c[1] for c in model.calls] == ["nova", "house"]


def test_profile_backend_mismatch_is_refused():
    with pytest.raises(ValueError, match="can't be used with backend"):
        text_to_speech("Hi", VoiceProfile("x", 1, "elevenlabs", 1), backend="say")


def test_output_path_and_registered_backend(library, tmp_path):
    register_speech_model("rec", Recorder, overwrite=True)
    try:
        out = tmp_path / "o" / "hi.wav"
        text_to_speech("Hi", "Kathy", backend="rec", voices=library, output_path=str(out), model=Recorder())
        assert out.read_bytes() == b"audio"
        assert isinstance(voxy.create_speech_model("REC"), Recorder)
    finally:
        del speech_model_factories["rec"]


def test_list_voices_library_and_voice_id(library):
    (info,) = list_voices(voices=library)
    assert info.name == "cora" and info.labels == {"aliases": ["Cora", "Coco"], "backends": ["elevenlabs", "rec"]}
    assert voice_id("Coco", voices=library) == "el-cora"
    with pytest.raises(KeyError, match="No voice named"):
        voice_id("nobody", voices=library)


def test_elevenlabs_synthesize_and_list_voices():
    client = FakeClient()
    client.voices.search = lambda **kw: SimpleNamespace(voices=[SimpleNamespace(voice_id="a", name="Ann", description=None, labels={"accent": "x"})])
    model = ElevenLabsSpeechModel(api_key="k", client_factory=lambda key: client)
    speech = text_to_speech("Hi", "vid", backend="elevenlabs", model=model)
    assert speech.format == "mp3" and speech.voice == "vid"
    assert client.calls[-1][2]["output_format"] == "mp3_44100_128"
    pcm = model.synthesize("Hi", "vid", output_format="pcm_24000")
    assert pcm.format == "wav" and pcm.audio[:4] == b"RIFF"
    assert [(v.voice_id, v.name, v.labels) for v in list_voices("elevenlabs", model=model)] == [("a", "Ann", {"accent": "x"})]


def test_design_voice_previews_then_create(tmp_path):
    client = FakeClient()
    model = ElevenLabsSpeechModel(api_key="k", client_factory=lambda key: client)
    previews = model.design_voice_previews("a dry, fast comic narrator", seed=3)
    assert [p.generated_voice_id for p in previews] == ["gen0", "gen1", "gen2"]
    assert previews[1].audio == b"ID3\x01" and previews[0].text == "auto text"
    op, kwargs = client.calls[-1]
    assert op == "ttv.design" and kwargs["auto_generate_text"] is True and kwargs["seed"] == 3
    voices = voices_store(rootdir=tmp_path)
    profile = design_from_description(
        "ov", "a dry, fast comic narrator", preview=previews[1], model=model,
        voices=voices, record_fields={"aliases": ["OV"]}, labels={"accent": "neutral"},
    )
    op, kwargs = client.calls[-1]
    assert op == "ttv.create" and kwargs["generated_voice_id"] == "gen1" and kwargs["voice_name"] == "ov"
    assert profile.segment == "designed-id"
    assert voices["ov"]["description"] == "a dry, fast comic narrator"
    assert voice_id("OV", voices=voices) == "designed-id"


def test_design_voice_without_preview_uses_first():
    client = FakeClient()
    model = ElevenLabsSpeechModel(api_key="k", client_factory=lambda key: client)
    assert model.design_voice("calm").segment == "designed-id"
    assert client.calls[-1][1]["generated_voice_id"] == "gen0"


def test_unsupported_capabilities_name_the_backend():
    with pytest.raises(NotImplementedError, match="'say' backend can't design voices"):
        SaySpeechModel(run=lambda *a, **k: None).design_voice("x")
    with pytest.raises(NotImplementedError, match="'fal' backend can't clone voices"):
        FalSpeechModel(tts=None, fetch=None).clone_voice(b"x")


def test_say_backend_with_fake_command():
    seen = []

    def run(cmd, **kwargs):
        seen.append(cmd)
        if "-o" in cmd:
            with open(cmd[cmd.index("-o") + 1], "wb") as f:
                f.write(b"RIFFfake")
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout="Daniel              en_GB    # Hello! My name is Daniel.\n")

    model = SaySpeechModel(run=run)
    assert [v.name for v in model.list_voices()] == ["Daniel"]
    speech = model.synthesize("Hi there", "Daniel")
    assert speech.audio == b"RIFFfake" and speech.format == "wav"
    assert seen[-1][:3] == ["say", "-v", "Daniel"] and seen[-1][-1] == "Hi there"


def test_aix_backend_passes_model_voice_and_format():
    calls = []

    def tts(text, **kwargs):
        calls.append((text, kwargs))
        return SimpleNamespace(data=b"ID3", voice=kwargs["voice"])

    model = AixSpeechModel(model="tts-1-hd", tts=tts)
    speech = model.synthesize("Hi", "nova", speed=1.2)
    assert calls == [("Hi", {"model": "tts-1-hd", "voice": "nova", "response_format": "mp3", "speed": 1.2})]
    assert (speech.audio, speech.format, speech.backend) == (b"ID3", "mp3", "aix")
    assert "nova" in [v.name for v in model.list_voices()]


def test_fal_backend_downloads_first_asset():
    asset = SimpleNamespace(url="https://x/a.wav?sig=1", content_type="audio/wav")
    model = FalSpeechModel(
        tts=lambda text, **kw: SimpleNamespace(first=asset, application="m"),
        fetch=lambda url: b"RIFF" if url == asset.url else b"",
    )
    speech = model.synthesize("Hi", "v1", quality="best")
    assert (speech.audio, speech.format) == (b"RIFF", "wav")
    empty = FalSpeechModel(tts=lambda text, **kw: SimpleNamespace(first=None, application="m"), fetch=None)
    with pytest.raises(RuntimeError, match="no audio"):
        empty.synthesize("Hi")


def test_cli_lists_library_voices(library, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("VOXY_DATA_DIR", str(tmp_path))
    assert main(["voices"]) == 0
    assert capsys.readouterr().out.startswith("cora\tCora, Coco\televenlabs, rec")
