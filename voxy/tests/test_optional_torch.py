"""A cloud voice must not need torch: importing voxy and speaking through a
service leaves torch, torchaudio and whisper unimported (issue #60)."""

import os
import subprocess
import sys

import numpy as np
import pytest

from voxy.base import optional_import, tensor_to_wav_bytes

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


NO_TORCH_DEVICE = """
import sys
sys.modules["torch"] = None  # import torch -> ImportError
from voxy.base import SpeechModel

class Service(SpeechModel):
    name = "service"

model = Service()
assert model._device is None, "resolved before first use"
print(model.device)
"""


def test_the_device_is_resolved_on_first_use_and_falls_back_to_cpu():
    done = subprocess.run(
        [sys.executable, "-c", NO_TORCH_DEVICE],
        capture_output=True,
        text=True,
        check=True,
        env={k: v for k, v in os.environ.items() if k != "DFLT_VOXY_DEVICE"},
    )
    assert done.stdout.strip() == "cpu"


def test_a_bfloat16_tensor_encodes():
    torch = pytest.importorskip("torch")
    audio = torch.tensor([0.0, 0.5, -0.5], dtype=torch.bfloat16)
    assert tensor_to_wav_bytes(audio, 8000) == tensor_to_wav_bytes(
        audio.float().numpy(), 8000
    )


def test_3d_audio_is_refused():
    with pytest.raises(ValueError, match="3-D"):
        tensor_to_wav_bytes(np.zeros((1, 2, 3)), 8000)
