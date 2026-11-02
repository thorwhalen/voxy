"""Tests for the kokoro backend and the downloadable assets of an ``EnvSpec``."""

import json

import numpy as np
import pytest

import voxy
from voxy import envs
from voxy.envs import EnvSpec
from voxy.kokoro_model import KOKORO, KokoroSpeechModel


class FakeGenerate:
    """Records each synthesize request; answers ``call('voices')`` like the adapter."""

    def __init__(self, voices=("af_heart", "bm_george", "ff_siwis")):
        self.requests, self._voices = [], list(voices)

    def __call__(self, text, **request):
        self.requests.append({"text": text, **request})
        return np.full(240, 0.1, dtype=np.float32), 24000

    def call(self, op, **request):
        assert op == "voices"
        return {"voices": self._voices}


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
    with pytest.raises(ValueError, match="Not a kokoro voice"):
        KokoroSpeechModel(generate=gen).synthesize("Hi.", "Samantha")
    assert gen.requests == []


def test_no_cloning():
    with pytest.raises(NotImplementedError, match="kokoro"):
        KokoroSpeechModel(generate=FakeGenerate()).clone_voice("ref.wav", "Hi.")


def test_list_voices_reads_the_model_and_labels_them():
    voices = KokoroSpeechModel(generate=FakeGenerate()).list_voices()
    assert [(v.voice_id, v.labels["language"], v.labels["gender"]) for v in voices] == [
        ("af_heart", "en-us", "female"),
        ("bm_george", "en-gb", "male"),
        ("ff_siwis", "fr-fr", "female"),
    ]


def test_facade_reaches_kokoro_by_name():
    gen = FakeGenerate()
    model = voxy.create_speech_model("kokoro", generate=gen)
    speech = voxy.text_to_speech("Hi.", "bf_emma", model=model, voices={})
    assert speech.backend == "kokoro" and gen.requests[0]["lang"] == "en-gb"


def test_kokoro_env_needs_no_torch_and_lists_its_model_files():
    assert KOKORO.env.torch_backend is None
    assert not any("torch" in r for r in KOKORO.env.requirements)
    assert set(KOKORO.env.assets) == set(KOKORO.init.values())


# --- EnvSpec assets ------------------------------------------------------------------


def _quiet(*args):
    pass


def _fake_run(calls):
    def run(argv, **kwargs):
        calls.append(argv)

    return run


def test_ensure_env_fetches_missing_assets_once(tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "_uv", lambda: "uv")
    spec = EnvSpec(
        "demo", ("pkg",), assets={"w-v1.bin": "https://example.com/w-v1.bin"}
    )
    fetched, runs = [], []

    def download(url, path):
        fetched.append(url)
        path.write_bytes(b"weights")

    envs.ensure_env(
        spec, rootdir=tmp_path, run=_fake_run(runs), download=download, log=_quiet
    )
    assert fetched == ["https://example.com/w-v1.bin"]
    assert (tmp_path / "models" / "demo" / "w-v1.bin").read_bytes() == b"weights"
    assert envs.is_env_ready(spec, rootdir=tmp_path)

    envs.ensure_env(
        spec, rootdir=tmp_path, run=_fake_run(runs), download=download, log=_quiet
    )
    assert len(fetched) == 1 and len(runs) == 2  # venv + install, once


def test_a_missing_asset_makes_the_env_not_ready(tmp_path):
    spec = EnvSpec("demo", (), assets={"w.bin": "https://example.com/w.bin"})
    folder = tmp_path / "envs" / "demo"
    folder.mkdir(parents=True)
    (folder / envs.ENV_MARKER).write_text(json.dumps({"digest": spec.digest()}))
    assert not envs.is_env_ready(spec, rootdir=tmp_path)
    assert envs.missing_assets(spec, rootdir=tmp_path) == ["w.bin"]


def test_assets_do_not_change_the_digest():
    plain = EnvSpec("demo", ("pkg",))
    assert plain.digest() == EnvSpec("demo", ("pkg",), assets={"a": "u"}).digest()


def test_a_failed_download_leaves_nothing_behind(tmp_path, monkeypatch):
    class Dropped:
        """A response that breaks after the first chunk."""

        def __init__(self):
            self.chunks = [b"partial"]

        def read(self, *args):
            if self.chunks:
                return self.chunks.pop()
            raise OSError("connection reset")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(envs.urllib.request, "urlopen", lambda url: Dropped())
    with pytest.raises(OSError):
        envs._download("https://example.com/w.bin", tmp_path / "w.bin")
    assert list(tmp_path.iterdir()) == []
