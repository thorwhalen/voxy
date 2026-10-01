"""Tests for the voice library and its stores (no network: the API is faked)."""

import pytest

from voxy import ElevenLabsSpeechModel, clone_from_samples, load_voice, save_voice
from voxy.base import VoiceProfile
from voxy.stores import samples_store, voices_store, voxy_data_dir
from voxy.tests.test_elevenlabs_model import FakeClient


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXY_DATA_DIR", str(tmp_path))
    assert voxy_data_dir() == tmp_path
    assert (tmp_path / "voices").is_dir() is False  # nothing created until used
    voices_store()["x"] = {"name": "x"}
    assert (tmp_path / "voices" / "x.json").is_file()


def test_clone_from_samples_uploads_stored_samples_and_saves(tmp_path):
    client = FakeClient()
    model = ElevenLabsSpeechModel(api_key="k", client_factory=lambda key: client)
    samples = samples_store("ada", rootdir=tmp_path)
    samples["b.wav"] = b"RIFFb"
    samples["a.wav"] = b"RIFFa"
    voices = voices_store(rootdir=tmp_path)
    profile = clone_from_samples(
        "ada", model=model, samples=samples, voices=voices,
        record_fields={"aliases": ["Ada"]}, labels={"language": "en"},
    )
    (_, kwargs), = client.calls
    assert kwargs["name"] == "ada"
    assert [n for n, _ in kwargs["files"]] == ["a.wav", "b.wav"]
    assert profile.segment == "new-voice-id"
    reloaded = load_voice("ada", voices=voices_store(rootdir=tmp_path))
    assert reloaded.segment == "new-voice-id" and reloaded.model_type == "elevenlabs"
    assert voices["ada"]["aliases"] == ["Ada"]


def test_save_voice_merges_profiles_and_fields(tmp_path):
    voices = voices_store(rootdir=tmp_path)
    save_voice("ada", VoiceProfile("v1", 1, "elevenlabs", 24000), voices=voices, consent="yes")
    save_voice("ada", VoiceProfile("v2", 1, "other", 16000), voices=voices)
    record = voices["ada"]
    assert set(record["profiles"]) == {"elevenlabs", "other"} and record["consent"] == "yes"


def test_unsaveable_profile_and_missing_voice(tmp_path):
    voices = voices_store(rootdir=tmp_path)
    with pytest.raises(TypeError, match="in-memory"):
        save_voice("ada", VoiceProfile(object(), 1, "csm", 24000), voices=voices)
    with pytest.raises(KeyError, match="No voice"):
        load_voice("nobody", voices=voices)
    save_voice("ada", VoiceProfile("v1", 1, "elevenlabs", 24000), voices=voices)
    with pytest.raises(KeyError, match="no 'csm' profile"):
        load_voice("ada", model_type="csm", voices=voices)


def test_clone_from_empty_samples_fails(tmp_path):
    model = ElevenLabsSpeechModel(api_key="k", client_factory=lambda key: FakeClient())
    with pytest.raises(ValueError, match="No samples"):
        clone_from_samples("ada", model=model, samples={}, voices={})
