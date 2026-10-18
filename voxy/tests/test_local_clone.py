"""Local cloning backends, reference prep and diagnosis pages, with no model loaded."""

import json
import sys

import numpy as np
import pytest

import voxy
from voxy import diagnosis, refprep
from voxy.envs import EnvSpec, ensure_env, is_env_ready
from voxy.library import load_voice, save_voice
from voxy.local_clone import (
    LOCAL_CLONE_BACKENDS,
    LocalCloneSpeechModel,
    XTTSSpeechModel,
    reference_profile,
)
from voxy.stores import samples_store, voices_store
from voxy.wavutil import read_wav, write_wav
from voxy.worker import Worker, WorkerError

SR = 8000


def fake_generate(calls):
    def generate(text, *, ref_audio, ref_text, **params):
        calls.append({"text": text, "ref_audio": ref_audio, **params})
        tone = 0.2 * np.sin(np.linspace(0, 100, SR // 2)).astype(np.float32)
        return np.concatenate([np.zeros(400, np.float32), tone, np.zeros(400, np.float32)]), SR

    return generate


def test_local_backends_are_registered():
    assert {"f5tts", "chatterbox", "xtts"} <= set(voxy.speech_model_factories)
    assert voxy.canonical_backend("F5-TTS") == "f5tts"
    model = voxy.create_speech_model("chatterbox", generate=fake_generate([]))
    assert model.name == "chatterbox"


def test_synthesize_chunks_long_text_and_joins_with_pauses():
    calls = []
    model = LocalCloneSpeechModel(
        "f5tts", generate=fake_generate(calls), gap_ms=100, paragraph_gap_ms=500
    )
    voice = reference_profile("ref.wav", "Reference words.")
    text = "First sentence. Second one!\n\nA new paragraph."
    speech = model.synthesize(text, voice, nfe_step=16)
    assert [c["text"] for c in calls] == ["First sentence. Second one!", "A new paragraph."]
    assert all(c["nfe_step"] == 16 and c["ref_audio"] == "ref.wav" for c in calls)
    samples, sr = read_wav(speech.audio)
    assert sr == SR
    # two tones trimmed to 60 ms of edge silence each side, plus one paragraph pause
    assert abs(len(samples) / sr - (2 * (0.5 + 0.12) + 0.5)) < 0.05


def test_clone_voice_profile_round_trips_through_the_library(tmp_path):
    model = LocalCloneSpeechModel("f5tts", generate=fake_generate([]))
    profile = model.clone_voice(str(tmp_path / "ref.wav"), "Some words.")
    voices = voices_store(rootdir=tmp_path)
    save_voice("ada", profile, voices=voices)
    loaded = load_voice("ada", model_type="f5tts", voices=voices)
    assert loaded.segment == {"ref_audio": str(tmp_path / "ref.wav"), "ref_text": "Some words."}


def test_text_to_speech_resolves_a_library_reference_voice(tmp_path):
    voices = voices_store(rootdir=tmp_path)
    save_voice("ada", reference_profile("r.wav", "Hi.", backend="f5tts"), voices=voices)
    model = LocalCloneSpeechModel("f5tts", generate=fake_generate([]))
    speech = voxy.text_to_speech("Hello.", "ada", voices=voices, model=model)
    assert speech.backend == "f5tts" and speech.format == "wav"


def test_synthesize_without_reference_fails_clearly():
    model = LocalCloneSpeechModel("f5tts", generate=fake_generate([]))
    with pytest.raises(ValueError, match="reference voice"):
        model.synthesize("Hi.", "Daniel")


def test_xtts_needs_its_licence_accepted(monkeypatch):
    monkeypatch.delenv(XTTSSpeechModel.LICENCE_ENVVAR, raising=False)
    model = XTTSSpeechModel(generate=fake_generate([]))
    with pytest.raises(PermissionError, match="Coqui Public Model Licence"):
        model.synthesize("Hi.", reference_profile("r.wav", "Hi."))
    accepted = XTTSSpeechModel(generate=fake_generate([]), accept_licence=True)
    assert accepted.synthesize("Hi.", reference_profile("r.wav", "Hi.")).format == "wav"


def test_every_backend_has_an_adapter_file():
    from voxy.workers import adapter_path

    for name in LOCAL_CLONE_BACKENDS:
        assert adapter_path(name).exists(), name


# worker -----------------------------------------------------------------------


def test_worker_reports_errors_and_keeps_serving(tmp_path):
    adapter = tmp_path / "a_adapter.py"
    adapter.write_text(
        "def load(**kw): return kw\n"
        "def boom(state): raise KeyError('nope')\n"
        "def hello(state, name): print('noise on stdout'); return {'hi': name}\n"
    )
    with Worker(sys.executable, adapter, log_path=tmp_path / "w.log") as worker:
        with pytest.raises(WorkerError, match="nope"):
            worker.call("boom")
        assert worker.call("hello", name="x") == {"hi": "x"}  # stdout noise didn't break it
    assert "noise on stdout" in (tmp_path / "w.log").read_text()


def test_worker_that_fails_to_load_raises_and_stops(tmp_path):
    adapter = tmp_path / "b_adapter.py"
    adapter.write_text("def load(**kw): raise ImportError('no torch here')\n")
    worker = Worker(sys.executable, adapter, log_path=tmp_path / "w.log")
    with pytest.raises(WorkerError, match="no torch here"):
        worker.call("anything")
    assert not worker.running


def test_ensure_env_builds_once_then_reuses(tmp_path):
    spec = EnvSpec("demo", ("numpy",))
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        assert kwargs["env"]["UV_CONSTRAINT"] == ""  # host pins don't leak in

    import shutil

    if shutil.which("uv") is None:
        pytest.skip("uv not installed")
    ensure_env(spec, rootdir=tmp_path, run=run, log=lambda *a: None)
    assert is_env_ready(spec, rootdir=tmp_path)
    n = len(commands)
    ensure_env(spec, rootdir=tmp_path, run=run, log=lambda *a: None)
    assert len(commands) == n


# reference prep ---------------------------------------------------------------


def _word(start, end, text, confidence=0.9):
    return {"start": start, "end": end, "text": text, "confidence": confidence}


def test_prepare_reference_cuts_scores_and_saves_a_manifest(tmp_path):
    rng = np.random.default_rng(1)
    speech = 0.2 * rng.standard_normal(SR * 10).astype(np.float32)
    speech[SR * 4 : SR * 5] *= 0.001  # a pause between two sentences
    source = tmp_path / "tape.wav"
    source.write_bytes(write_wav(speech, SR))

    def conditioner(src, dst, **kwargs):
        dst.write_bytes(open(src, "rb").read())
        return dst

    def transcriber(path, language=None):
        words = [_word(0.2, 1.5, "It"), _word(1.6, 3.9, "began."), _word(5.1, 7, "Then"),
                 _word(7.1, 9.8, "ended.")]
        return [{"start": 0, "end": 10, "text": "", "words": words}]

    samples = samples_store("ada", rootdir=tmp_path)
    manifest = refprep.prepare_reference(
        [source], samples=samples, conditioner=conditioner, transcriber=transcriber,
        min_seconds=2, log=lambda *a: None,
    )
    assert sorted(c["text"] for c in manifest) == ["It began.", "Then ended."]
    assert all(0 <= c["score"] <= 1 and c["path"].endswith(".wav") for c in manifest)
    assert refprep.load_manifest(samples) == manifest
    assert "conditioned/tape.wav" in samples


def test_plan_clips_respects_max_length():
    words = [_word(i, i + 0.9, f"w{i}") for i in range(30)]
    clips = refprep.plan_clips([{"words": words, "start": 0, "end": 30, "text": ""}],
                               min_seconds=3, max_seconds=8)
    assert all(c["end"] - c["start"] <= 8.5 for c in clips)
    assert " ".join(c["text"] for c in clips) == " ".join(w["text"] for w in words)


# diagnosis --------------------------------------------------------------------


def test_render_grid_and_page(tmp_path):
    calls = []

    def factory(backend, **init):
        return LocalCloneSpeechModel(backend, generate=fake_generate(calls), **init)

    variants = diagnosis.expand_grid(
        [{"backend": "f5tts", "params": {"nfe_step": [16, 32]}},
         {"backend": "chatterbox", "init": {"variant": "turbo"}, "params": {"temperature": 0.7}}],
        references=["r1"],
    )
    assert len({v.id for v in variants}) == 3
    refs = {"r1": reference_profile("r.wav", "Hi.")}
    results = diagnosis.render_grid("Hello there.", refs, variants, tmp_path,
                                    model_factory=factory, audio_format="wav", log=lambda *a: None)
    assert all(not r.get("error") and (tmp_path / r["audio"]).exists() for r in results)
    page = diagnosis.build_page(results, tmp_path, title="T", text="Hello there.")
    content = page.read_text()
    assert all(v.id in content for v in variants)
    assert "<b>nfe_step=16</b>" in content and "nfe_step=32" in content
    # resume: nothing re-rendered
    n = len(calls)
    diagnosis.render_grid("Hello there.", refs, variants, tmp_path, model_factory=factory,
                          audio_format="wav", log=lambda *a: None)
    assert len(calls) == n
    saved = json.loads((tmp_path / diagnosis.RESULTS_FILE).read_text())
    assert len(saved["results"]) == 3


def test_a_failing_variant_is_recorded_not_fatal(tmp_path):
    def factory(backend, **init):
        def generate(text, **params):
            if params.get("nfe_step") == 16:
                raise RuntimeError("diverged")
            return fake_generate([])(text, **params)

        return LocalCloneSpeechModel(backend, generate=generate)

    variants = diagnosis.expand_grid([{"backend": "f5tts", "params": {"nfe_step": [16, 32]}}])
    results = diagnosis.render_grid("Hi.", {"": reference_profile("r.wav", "x")}, variants,
                                    tmp_path, model_factory=factory, audio_format="wav",
                                    log=lambda *a: None)
    errors = [r for r in results if r.get("error")]
    assert len(errors) == 1 and "diverged" in errors[0]["error"]


# fine-tuning export -----------------------------------------------------------


def test_export_dataset_writes_each_recipe_layout(tmp_path):
    from voxy.finetune import export_dataset

    samples = {"clips/a.wav": b"RIFFa", "clips/b.wav": b"RIFFb"}
    manifest = [
        {"key": "clips/a.wav", "text": "Good | clip.", "score": 0.8, "duration": 5.0},
        {"key": "clips/b.wav", "text": "Bad clip.", "score": 0.1, "duration": 5.0},
    ]
    lj = export_dataset(manifest, samples, tmp_path / "lj")
    assert lj["clips"] == 1
    assert (tmp_path / "lj/metadata.csv").read_text() == "a|Good , clip.|Good , clip.\n"
    f5 = export_dataset(manifest, samples, tmp_path / "f5", format="f5tts")
    lines = (tmp_path / "f5/metadata.csv").read_text().splitlines()
    assert lines[0] == "audio_file|text" and lines[1].endswith("a.wav|Good , clip.")
    assert f5["clips"] == 1


def test_chatterbox_adapter_routes_language_by_variant(tmp_path):
    import importlib.util

    from voxy.workers import adapter_path

    spec = importlib.util.spec_from_file_location("cb_adapter", adapter_path("chatterbox"))
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    pytest.importorskip("torch")
    pytest.importorskip("soundfile")
    import torch

    class ChatterboxMultilingualTTS:
        sr = SR

        def generate(self, text, **kwargs):
            self.kwargs = kwargs
            return torch.zeros(1, SR // 10)

    class ChatterboxTTS(ChatterboxMultilingualTTS):
        pass

    multi = ChatterboxMultilingualTTS()
    kw = dict(text="Bonjour.", ref_audio="r.wav", ref_text="x", out_path=str(tmp_path / "o.wav"))
    adapter.synthesize(multi, language="fr", **kw)
    assert multi.kwargs["language_id"] == "fr"
    english_only = ChatterboxTTS()
    with pytest.raises(ValueError, match="multilingual"):
        adapter.synthesize(english_only, language="fr", **kw)
    adapter.synthesize(english_only, **kw)
    assert "language_id" not in english_only.kwargs
