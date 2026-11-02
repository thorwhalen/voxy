"""A cloud voice must not need torch: importing voxy and speaking through a
service leaves torch, torchaudio and whisper unimported (issue #60)."""

import subprocess
import sys

import pytest

from voxy.base import SpeechModel, optional_import

HEAVY = ("torch", "torchaudio", "whisper")

CLOUD_SESSION = """
import sys
from types import SimpleNamespace
import numpy as np
import voxy
from voxy.elevenlabs_model import ElevenLabsSpeechModel

pcm = b"\\x00\\x00\\xff\\x7f" * 8
client = SimpleNamespace(
    text_to_speech=SimpleNamespace(convert=lambda voice_id, **kw: iter([pcm])),
    voices=SimpleNamespace(ivc=SimpleNamespace(create=lambda **kw: SimpleNamespace(voice_id="v"))),
)
model = ElevenLabsSpeechModel(api_key="k", client_factory=lambda key: client)
voxy.text_to_speech("Hi.", "abc", model=model, voices={}, output_format="pcm_24000")
model.clone_voice([np.zeros(160), b"RIFF....WAVEfmt "], name="x")
voxy.list_voices("kokoro")
print(",".join(m for m in HEAVY if m in sys.modules))
"""


def test_a_cloud_voice_imports_no_torch():
    done = subprocess.run(
        [sys.executable, "-c", f"HEAVY = {HEAVY!r}\n" + CLOUD_SESSION],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == ""


def test_a_missing_optional_module_names_its_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    with pytest.raises(ImportError, match=r"voxy\[local\]"):
        optional_import("torch")


def test_the_device_is_resolved_on_first_use_only():
    class Service(SpeechModel):
        name = "service"

    model = Service()
    assert model._device is None
    model.device = "cpu"
    assert model.device == "cpu"
