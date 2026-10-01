"""Tests for the ElevenLabs backend, with the API faked (plus opt-in live tests)."""

import io
import json
import os
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from voxy import ElevenLabsSpeechModel, create_speech_model
from voxy.base import VoiceProfile
from voxy.elevenlabs_model import _wav_bytes


class FakeClient:
    """Records calls; answers like the SDK's ``voices`` and ``text_to_speech``."""

    def __init__(self, pcm: bytes = b"\x00\x00\xff\x7f" * 10):
        self.calls = []
        self.pcm = pcm
        self.voices = SimpleNamespace(
            ivc=SimpleNamespace(create=self._ivc_create), delete=self._delete
        )
        self.text_to_speech = SimpleNamespace(convert=self._convert)

    def _ivc_create(self, **kwargs):
        self.calls.append(("ivc.create", kwargs))
        return SimpleNamespace(voice_id="new-voice-id", requires_verification=False)

    def _delete(self, voice_id):
        self.calls.append(("delete", voice_id))

    def _convert(self, voice_id, **kwargs):
        self.calls.append(("convert", voice_id, kwargs))
        if kwargs["output_format"].startswith("wav"):
            return iter([_wav_bytes(torch.zeros(4), 24000)])
        half = len(self.pcm) // 2
        return iter([self.pcm[:half], self.pcm[half:]])  # the SDK streams chunks


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def model(client):
    return ElevenLabsSpeechModel(api_key="test-key", client_factory=lambda key: client)


def test_selected_by_keyword(client):
    m = create_speech_model("ElevenLabs", api_key="k", client_factory=lambda key: client)
    assert isinstance(m, ElevenLabsSpeechModel)


def test_clone_uploads_every_sample_and_returns_profile(model, client, tmp_path):
    path = tmp_path / "clip.m4a"
    path.write_bytes(b"encoded-audio")
    profile = model.clone_voice(
        [str(path), b"raw-bytes", io.BytesIO(b"filelike"), np.zeros(1600)],
        name="cora",
        labels={"language": "en"},
    )
    (op, kwargs), = client.calls
    assert op == "ivc.create"
    assert kwargs["name"] == "cora"
    assert json.loads(kwargs["labels"]) == {"language": "en"}  # multipart needs text
    assert "description" not in kwargs  # unset options are not sent
    names = [n for n, _ in kwargs["files"]]
    contents = [c for _, c in kwargs["files"]]
    assert names[:3] == ["clip.m4a", "sample_001", "sample_002"]
    assert contents[:3] == [b"encoded-audio", b"raw-bytes", b"filelike"]
    assert names[3].endswith(".wav") and contents[3][:4] == b"RIFF"
    assert profile.segment == profile.metadata["voice_id"] == "new-voice-id"
    assert profile.model_type == "elevenlabs"
    assert profile.metadata["n_samples"] == 4


def test_clone_single_input_and_default_name(model, client):
    model.clone_voice(b"x", speaker_id=7)
    (_, kwargs), = client.calls
    assert kwargs["name"] == "voxy-7"
    assert kwargs["files"] == [("sample_000", b"x")]


def test_clone_missing_path_fails_before_any_upload(model, client):
    with pytest.raises(ValueError, match="does not exist"):
        model.clone_voice(["/no/such/file.wav"])
    assert client.calls == []


def test_clone_applies_cleanup_fn(model, client):
    seen = []

    def cleanup(audio, sr):
        seen.append(sr)
        return audio * 0

    model.clone_voice(torch.ones(800), cleanup_audio_fn=cleanup, assumed_sample_rate=8000)
    assert seen == [8000]


def test_generate_speech_decodes_pcm_and_saves_wav(model, client, tmp_path):
    profile = model.voice_profile("vid")
    out = tmp_path / "out" / "phrase.wav"
    audio = model.generate_speech("Hello.", profile, output_path=str(out))
    op, voice_id, kwargs = client.calls[-1]
    assert (op, voice_id, kwargs["text"]) == ("convert", "vid", "Hello.")
    assert kwargs["output_format"] == "pcm_24000"
    assert audio.shape == (20,) and audio.dtype == torch.float32
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnframes()) == (24000, 20)


def test_generate_speech_decodes_wav_format(model):
    audio = model.generate_speech("Hi", "vid", output_format="wav_24000")
    assert audio.shape == (4,)


def test_generate_speech_rejects_compressed_format_before_calling(model, client):
    with pytest.raises(ValueError, match="synthesize_bytes"):
        model.generate_speech("Hi", "vid", output_format="mp3_44100_128")
    assert client.calls == []
    assert model.synthesize_bytes("Hi", "vid", output_format="mp3_44100_128")


def test_generate_speech_needs_a_voice(model):
    with pytest.raises(ValueError, match="needs a voice"):
        model.generate_speech("Hi")


def test_rejects_other_backends_profiles(model):
    with pytest.raises(ValueError, match="Incompatible"):
        model.generate_speech("Hi", VoiceProfile(None, 1, "csm", 24000))


def test_delete_voice(model, client):
    model.delete_voice(model.voice_profile("vid"))
    assert client.calls == [("delete", "vid")]


def test_missing_key_is_reported_without_a_request(monkeypatch):
    for var in ("ELEVEN_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    m = ElevenLabsSpeechModel(client_factory=lambda key: pytest.fail("no client"))
    with pytest.raises(RuntimeError, match="API key"):
        m.generate_speech("Hi", "vid")


def test_repr_hides_the_key():
    assert "secret" not in repr(ElevenLabsSpeechModel(api_key="secret"))


_LIVE = os.environ.get("VOXY_LIVE_ELEVENLABS")
_LIVE_SAMPLE = os.environ.get("VOXY_LIVE_ELEVENLABS_SAMPLE")


@pytest.mark.skipif(
    not _LIVE, reason="live ElevenLabs test; set VOXY_LIVE_ELEVENLABS=1 (uses credits)"
)
def test_live_synthesize_with_stock_voice():
    model = ElevenLabsSpeechModel()
    stock = model.client.voices.search(page_size=1, voice_type="default").voices[0]
    audio = model.generate_speech("Testing, one two three.", stock.voice_id)
    assert audio.numel() > model.sample_rate // 2


@pytest.mark.skipif(
    not (_LIVE and _LIVE_SAMPLE),
    reason=(
        "live clone test; set VOXY_LIVE_ELEVENLABS=1 and VOXY_LIVE_ELEVENLABS_SAMPLE "
        "to a recording of a voice you have consent to clone (ElevenLabs flags "
        "clones of its own stock voices)"
    ),
)
def test_live_clone_synthesize_delete():
    """Clone the given sample, speak with the clone, then delete the clone."""
    model = ElevenLabsSpeechModel()
    profile = model.clone_voice([_LIVE_SAMPLE], name="voxy-live-test")
    try:
        audio = model.generate_speech("Testing, one two three.", profile)
        assert audio.numel() > model.sample_rate // 2
    finally:
        model.delete_voice(profile)


def test_clone_refuses_too_many_files_before_upload(model, client):
    with pytest.raises(ValueError, match="at most 3"):
        model.clone_voice([b"a"] * 4, max_files=3)
    assert client.calls == []
    model.clone_voice([b"a"] * 4, max_files=None)
    assert len(client.calls[-1][1]["files"]) == 4


def test_clone_accepts_generators_and_int_arrays(model, client, tmp_path):
    for i in range(2):
        (tmp_path / f"c{i}.wav").write_bytes(b"RIFF....WAVE")
    model.clone_voice(sorted(tmp_path.glob("*.wav")))
    assert [n for n, _ in client.calls[-1][1]["files"]] == ["c0.wav", "c1.wav"]
    model.clone_voice(np.full(800, 16000, dtype=np.int16), assumed_sample_rate=8000)
    (name, content), = client.calls[-1][1]["files"]
    assert content[:4] == b"RIFF" and b"\x00\x00" * 4 not in content[-8:]


def test_bytes_uploads_get_a_sniffed_suffix(model, client):
    model.clone_voice([b"ID3rest", b"\x00\x00\x00\x18ftypM4A "])
    assert [n for n, _ in client.calls[-1][1]["files"]] == ["sample_000.mp3", "sample_001.m4a"]


def test_streamed_wav_with_placeholder_size_decodes(model, client):
    wav = bytearray(_wav_bytes(torch.zeros(6), 22050))
    data_at = wav.index(b"data")
    wav[data_at + 4 : data_at + 8] = (0xFFFFFFFF).to_bytes(4, "little")
    client._convert = lambda voice_id, **kw: iter([bytes(wav)])
    client.text_to_speech.convert = client._convert
    assert model.generate_speech("Hi", "vid", output_format="wav_22050").shape == (6,)
