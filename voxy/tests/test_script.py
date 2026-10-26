"""Cast-script rendering with a fake model: keyed takes, re-render on change, external voices."""

import shutil

import numpy as np
import pytest

from voxy.base import Speech
from voxy.script import (
    AWAITING,
    AWAITING_TAKE,
    add_take,
    missing_takes,
    plan_script,
    render_script,
    segment_key,
    status,
)
from voxy.wavutil import write_wav

SR = 8000
SCRIPT = [
    {"id": "s1", "segments": [{"role": "guide", "text": "Come."}, {"role": "author", "text": "I come."}]},
    {"id": "s2", "segments": [{"role": "author", "text": "Again."}]},
]
VOICES = {"g": {"voice": "gv"}, "a": {"voice": "av", "params": {"seed": 1}}}
CASTING = {"roles": {"guide": "g", "author": "a"}}


class FakeModel:
    calls = []

    def synthesize(self, text, voice, **params):
        FakeModel.calls.append((text, voice, params))
        tone = 0.2 * np.sin(np.linspace(0, 200, SR)).astype(np.float32)
        return Speech(write_wav(tone, SR), format="wav")


def _render(root, **kw):
    kw.setdefault("casting", CASTING)
    kw.setdefault("voices", VOICES)
    return render_script(
        SCRIPT, root, model_factory=lambda backend, **init: FakeModel(),
        voice_loader=lambda spec, root: spec["voice"], log=lambda m: None, **kw,
    )


needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed"
)


def test_unknown_role_is_refused():
    with pytest.raises(ValueError, match="no voice for role"):
        plan_script([{"id": "x", "segments": [{"role": "ghost", "text": "Boo"}]}], CASTING)


def test_text_prep_changes_the_key_not_the_text():
    plan = plan_script(SCRIPT, CASTING, text_prep=str.upper)
    seg = plan[0]["segments"][0]
    assert (seg["text"], seg["tts_text"]) == ("Come.", "COME.")
    assert segment_key(seg, VOICES["g"]) != segment_key({**seg, "tts_text": "Come."}, VOICES["g"])


@needs_ffmpeg
def test_render_makes_takes_and_files_then_only_what_changed(tmp_path):
    FakeModel.calls = []
    index = _render(tmp_path)
    assert [i["status"] for i in index["items"]] == ["done", "done"]
    assert len(FakeModel.calls) == 3
    assert (tmp_path / "sections" / "s1.mp3").exists()
    assert len(list((tmp_path / "segments").glob("*.wav"))) == 3
    assert status(tmp_path)["segments_by_voice"] == {"g": 1, "a": 2}

    FakeModel.calls = []
    _render(tmp_path)  # nothing changed: nothing synthesised
    assert FakeModel.calls == []

    _render(tmp_path, casting={**CASTING, "takes": {"s2": 2}})  # retake of s2: seed moves
    assert [(c[0], c[2]) for c in FakeModel.calls] == [("Again.", {"seed": 2})]
    assert len(list((tmp_path / "segments").glob("*.wav"))) == 4  # the first take is kept


def test_role_without_a_voice_waits(tmp_path):
    index = _render(tmp_path, casting={"roles": {"guide": None, "author": "a"}}, only=[])
    assert [i["status"] for i in index["items"]] == [AWAITING, "pending"]


def test_external_voice_owes_takes_until_filed(tmp_path):
    voices = {**VOICES, "g": {"voice": "gv", "external": True}}
    index = _render(tmp_path, voices=voices, only=[])
    assert index["items"][0]["status"] == AWAITING_TAKE
    (owed,) = missing_takes(SCRIPT, tmp_path, casting=CASTING, voices=voices)
    assert owed["section"] == "s1" and owed["tts_text"] == "Come."
    add_take(tmp_path, owed, write_wav(np.zeros(SR, np.float32), SR))
    assert missing_takes(SCRIPT, tmp_path, casting=CASTING, voices=voices) == []
    assert _render(tmp_path, voices=voices, only=[])["items"][0]["status"] == "pending"
