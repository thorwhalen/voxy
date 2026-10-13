"""Tests for the voicestudio backend: WAV encoding and the lazily loaded default generator (all faked)."""

import io
import sys
import wave
from types import ModuleType, SimpleNamespace

import pytest

from voxy.voicestudio_model import (
    VoiceStudioSpeechModel,
    _parler_generate,
    _pcm_wav,
)


def _read(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())


def test_pcm_wav_clips_and_scales():
    channels, width, rate, frames = _read(_pcm_wav([0.0, 1.0, -1.0, 5.0, -5.0], 8000))
    assert (channels, width, rate) == (1, 2, 8000)
    samples = [int.from_bytes(frames[i : i + 2], "little", signed=True) for i in range(0, len(frames), 2)]
    assert samples == [0, 32767, -32767, 32767, -32767]


def test_pcm_wav_accepts_numpy_and_empty():
    import numpy as np

    assert _read(_pcm_wav(np.array([0.5, -0.5], dtype="float32"), 16000))[2] == 16000
    assert _read(_pcm_wav([], 16000))[3] == b""


def test_extra_kwargs_reach_a_custom_generate():
    seen = {}

    def gen(text, description, **kw):
        seen.update(kw)
        return [0.0], 8000

    VoiceStudioSpeechModel(generate=gen).synthesize("Hi", seed=3)
    assert seen == {"seed": 3}


def test_default_generate_reports_missing_extras(monkeypatch):
    monkeypatch.setitem(sys.modules, "voicestudio", None)  # makes the import fail
    with pytest.raises(ImportError, match=r"voxy\[voicestudio\]"):
        _parler_generate("m", "cpu")("Hi", "a voice")


def test_default_generate_loads_once_and_uses_both_prompts(monkeypatch):
    loads = []

    class Ids:
        def __init__(self, tag):
            self.input_ids = self
            self.tag = tag

        def to(self, device):
            return (self.tag, device)

    class Audio:
        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self

        def squeeze(self):
            return [0.0, 0.1]

    class Model:
        config = SimpleNamespace(sampling_rate=22050)

        def to(self, device):
            return self

        def generate(self, *, input_ids, prompt_input_ids):
            self.last = (input_ids, prompt_input_ids)
            return Audio()

    model = Model()
    transformers = ModuleType("transformers")
    transformers.AutoModel = SimpleNamespace(from_pretrained=lambda mid: loads.append(mid) or model)
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda mid: lambda text, return_tensors: Ids(text)
    )
    parent, leaf = ModuleType("voicestudio"), ModuleType("voicestudio.models")
    leaf_mod = ModuleType("voicestudio.models.parler_tts")
    parent.models, leaf.parler_tts = leaf, leaf_mod
    for name, mod in [("voicestudio", parent), ("voicestudio.models", leaf),
                      ("voicestudio.models.parler_tts", leaf_mod), ("transformers", transformers)]:
        monkeypatch.setitem(sys.modules, name, mod)

    generate = _parler_generate("some/model", "cpu")
    assert generate("hello", "a voice") == ([0.0, 0.1], 22050)
    generate("again", "a voice")
    assert loads == ["some/model"]  # weights loaded once
    assert model.last == (("a voice", "cpu"), ("again", "cpu"))
