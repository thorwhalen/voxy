"""Local cloning backends, reference prep and diagnosis pages, with no model loaded."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import voxy
from voxy import diagnosis, refprep
from voxy.envs import EnvSpec, ensure_env, is_env_ready
from voxy.library import _is_sample_key, load_voice, save_voice
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
        return np.concatenate(
            [np.zeros(400, np.float32), tone, np.zeros(400, np.float32)]
        ), SR

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
    assert [c["text"] for c in calls] == [
        "First sentence. Second one!",
        "A new paragraph.",
    ]
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
    assert loaded.segment == {
        "ref_audio": str(tmp_path / "ref.wav"),
        "ref_text": "Some words.",
    }


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
    with Worker(sys.executable, adapter, stderr=tmp_path / "w.log") as worker:
        with pytest.raises(WorkerError, match="nope"):
            worker.call("boom")
        assert worker.call("hello", name="x") == {
            "hi": "x"
        }  # stdout noise didn't break it
    assert "noise on stdout" in (tmp_path / "w.log").read_text()


def test_worker_that_fails_to_load_raises_and_stops(tmp_path):
    adapter = tmp_path / "b_adapter.py"
    adapter.write_text("def load(**kw): raise ImportError('no torch here')\n")
    worker = Worker(sys.executable, adapter, stderr=tmp_path / "w.log")
    with pytest.raises(WorkerError, match="no torch here"):
        worker.call("anything")
    assert not worker.running


def test_close_from_another_thread_waits_for_the_call_in_flight(tmp_path):
    """close() takes the call lock, so it cannot pull a pipe from under a call (#80)."""
    import threading

    adapter = tmp_path / "e_adapter.py"
    adapter.write_text("def load(**kw): return kw\n")
    worker = Worker(sys.executable, adapter, stderr=tmp_path / "w.log")
    in_call, release = threading.Event(), threading.Event()

    def call_in_flight():  # holds the lock as call() does
        with worker._lock:
            in_call.set()
            release.wait(5)

    caller = threading.Thread(target=call_in_flight)
    caller.start()
    in_call.wait(5)
    closer = threading.Thread(target=worker.close)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive()  # waiting for the call
    release.set()
    closer.join(5)
    caller.join(5)
    assert not closer.is_alive()


def test_a_closed_pipe_is_reported_as_the_worker_exiting(tmp_path):
    adapter = tmp_path / "f_adapter.py"
    adapter.write_text("def load(**kw): return kw\ndef ping(state): return {}\n")
    worker = Worker(sys.executable, adapter, stderr=tmp_path / "w.log")
    worker.call("ping")
    worker._proc.stdin.close()  # as a close() racing the write would leave it
    with pytest.raises(WorkerError, match="exited"):
        worker.call("ping")
    assert not worker.running
    assert worker.call("ping") == {}  # the next call starts a fresh worker
    worker.close()


def test_a_request_that_cannot_be_serialised_leaves_the_worker_alive(tmp_path):
    adapter = tmp_path / "g_adapter.py"
    adapter.write_text("def load(**kw): return kw\ndef ping(state, **kw): return {}\n")
    worker = Worker(sys.executable, adapter, stderr=tmp_path / "w.log")
    worker.call("ping")
    circular = {}
    circular["self"] = circular
    with pytest.raises(ValueError, match="[Cc]ircular"):
        worker.call("ping", x=circular)
    assert worker.running
    worker.close()


def test_worker_pool_makes_each_worker_once_across_threads():
    """Shared ASR/speaker/phoneme workers: one per key, even under threads (#78)."""
    import threading
    import time

    from voxy.worker import WorkerPool

    made, closed = [], []

    class FakeWorker:
        def __init__(self, key):
            self.key = key

        def close(self):
            closed.append(self.key)

    def make(spec, adapter, *, model, threads, rootdir):
        made.append((adapter, model, threads, rootdir))
        time.sleep(0.05)  # an env build: two unguarded threads would both run it
        return FakeWorker((model, threads))

    pool = WorkerPool(EnvSpec("demo", ()), "asr", make=make)
    got = []
    threads = [
        threading.Thread(target=lambda: got.append(pool.get("base"))) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1 and len({id(w) for w in got}) == 1
    assert pool.get("base", rootdir=None) is got[0]
    assert pool.get("small", threads=2) is not got[0]
    pool.close()
    assert sorted(closed) == [("base", None), ("small", 2)]
    pool.get("base")
    assert len(made) == 3  # a fresh worker after close


def test_transcription_from_threads_starts_one_asr_worker(monkeypatch):
    """The real wiring: refprep's pool, four threads transcribing at once (#78)."""
    import threading
    import time

    made = []

    class FakeASR:
        def call(self, op, **request):
            return {"segments": []}

        def close(self):
            pass

    def make(spec, adapter, **key):
        made.append((adapter, key["rootdir"]))
        time.sleep(0.05)
        return FakeASR()

    monkeypatch.setattr(refprep._ASR_POOL, "_make", make)
    refprep.close_asr()
    try:
        threads = [
            threading.Thread(
                target=refprep.transcribe, args=("a.wav",), kwargs={"rootdir": r}
            )
            for r in (None, "", None, "")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert made == [("asr", None)]  # "" is the default data root, like None
    finally:
        refprep.close_asr()


def test_env_worker_is_the_one_factory_for_local_workers(tmp_path, monkeypatch):
    """#89: init, env vars and auto_setup in one place (WorkerGenerate and pools use it)."""
    from voxy import worker as worker_module
    from voxy.local_clone import WorkerGenerate

    spec = EnvSpec("demo", ())
    monkeypatch.setenv("VOXY_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(RuntimeError, match="python -m voxy setup f5tts"):
        worker_module.env_worker(spec, "f5tts", rootdir=tmp_path, auto_setup=False)

    monkeypatch.setattr(
        "voxy.envs.ensure_env", lambda spec, rootdir=None: "/env/python"
    )
    w = worker_module.env_worker(
        spec,
        "asr",
        model="base",
        init={"device": "cpu"},
        rootdir=tmp_path,
        env_vars={"COQUI_TOS_AGREED": "1"},
    )
    # ps.isolated.Worker makes a python path absolute (D:\\env\\python on Windows)
    assert w.python == os.path.abspath("/env/python")
    assert w.init == {"model": "base", "device": "cpu"}
    assert w.env["COQUI_TOS_AGREED"] == "1" and "OMP_NUM_THREADS" in w.env
    assert w.log_path == tmp_path / "cache" / "logs" / "asr.log"  # a kept log

    generate = WorkerGenerate(
        LOCAL_CLONE_BACKENDS["f5tts"], env_vars={"COQUI_TOS_AGREED": "1"}
    )
    made = generate._make_worker()
    assert made.env["COQUI_TOS_AGREED"] == "1" and made.adapter.endswith(
        "f5tts_adapter.py"
    )
    assert made.init == {"model": "F5TTS_v1_Base"}  # the backend's init, unchanged
    monkeypatch.undo()  # the real ensure_env again: the env is missing under tmp_path
    with pytest.raises(RuntimeError, match="setup f5tts"):
        WorkerGenerate(
            LOCAL_CLONE_BACKENDS["f5tts"], rootdir=tmp_path, auto_setup=False
        )._make_worker()


def test_ensure_env_builds_once_then_reuses(tmp_path, monkeypatch):
    from ps import isolated

    from voxy.envs import ENV_MARKER, env_dir, env_python

    monkeypatch.setattr(isolated, "_find_uv", lambda: "uv")
    monkeypatch.setenv("UV_CONSTRAINT", "/host/pins.txt")
    spec = EnvSpec("demo", ("numpy",))
    commands = []

    def run(cmd, **kwargs):
        commands.append(list(cmd))
        assert kwargs["env"]["UV_CONSTRAINT"] == ""  # host pins don't leak in
        if cmd[1] == "venv":  # what uv would leave on disk
            python = env_python(spec, rootdir=tmp_path)
            python.parent.mkdir(parents=True)
            python.touch()

    python = ensure_env(spec, rootdir=tmp_path, run=run, log=lambda *a: None)
    assert python == env_python(spec, rootdir=tmp_path)
    assert is_env_ready(spec, rootdir=tmp_path)
    assert (env_dir(spec, rootdir=tmp_path) / ENV_MARKER).exists()  # voxy's marker
    assert commands[0][:5] == [
        "uv",
        "venv",
        "--python",
        "3.11",
        str(env_dir(spec, rootdir=tmp_path)),
    ]
    assert commands[1][-3:] == ["--torch-backend", "cpu", "numpy"]  # voxy's default
    ensure_env(spec, rootdir=tmp_path, run=run, log=lambda *a: None)
    assert len(commands) == 2  # ready: nothing runs


def test_env_digests_are_the_ones_voxy_built_its_environments_with():
    """Moving onto ps.isolated must not rebuild an environment already built."""
    assert (
        EnvSpec("demo", ("numpy",)).digest() == "9b996d61b155"
    )  # what voxy <= 0.0.68 computed
    assert (
        EnvSpec("demo", ("numpy",), env_vars={"X": "1"}).digest()
        == EnvSpec("demo", ("numpy",)).digest()
    )  # env vars need no rebuild


#: Adapters whose environment has no torch (their threads are capped otherwise).
NON_TORCH_ADAPTERS = {"kokoro", "asr"}


def test_adapters_import_nothing_from_voxy_and_cap_torch_threads_first():
    import ast

    from voxy.workers import WORKERS_DIR

    adapters = sorted(WORKERS_DIR.glob("*_adapter.py"))
    assert len(adapters) >= len(LOCAL_CLONE_BACKENDS)
    for path in adapters:
        source = path.read_text()
        assert "from voxy" not in source and "import voxy" not in source, path.name
        name = path.stem.removesuffix("_adapter")
        if name in NON_TORCH_ADAPTERS:
            continue
        load = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "load"
        )
        body = load.body[1:] if ast.get_docstring(load) else load.body
        imported, called = body[0], body[1]
        assert isinstance(imported, ast.ImportFrom), path.name
        assert imported.module == "_worker_threads", path.name
        assert ast.unparse(called) == "cap_torch_threads()", path.name


def test_the_thread_variable_is_spelled_once():
    from voxy import envs
    from voxy.workers import WORKERS_DIR
    from voxy.workers._worker_threads import THREADS_ENVVAR

    assert envs.VOXY_NUM_THREADS_ENVVAR == THREADS_ENVVAR == "VOXY_NUM_THREADS"
    assert THREADS_ENVVAR in envs.THREAD_ENVVARS
    for path in WORKERS_DIR.glob("*_adapter.py"):
        assert '"VOXY_NUM_THREADS"' not in path.read_text(), path.name


def test_a_worker_caps_torch_threads_through_the_shared_helper(tmp_path):
    """The helper is importable from an adapter (its folder is on sys.path) and caps torch."""
    import shutil

    from voxy.workers import WORKERS_DIR

    shutil.copy(WORKERS_DIR / "_worker_threads.py", tmp_path)
    (tmp_path / "torch.py").write_text(  # a fake torch recording what it is asked
        "calls = []\n"
        "def set_num_threads(n): calls.append(['intra', n])\n"
        "def set_num_interop_threads(n): calls.append(['inter', n])\n"
    )
    adapter = tmp_path / "t_adapter.py"
    adapter.write_text(
        "def load(**kw):\n"
        "    from _worker_threads import cap_torch_threads\n"
        "    cap_torch_threads()\n"
        "def calls(state):\n"
        "    import torch\n"
        "    return {'calls': torch.calls}\n"
    )
    with Worker(sys.executable, adapter, env={"VOXY_NUM_THREADS": "6"}) as worker:
        assert worker.call("calls") == {"calls": [["intra", 6], ["inter", 1]]}
    with Worker(sys.executable, adapter, env={"VOXY_NUM_THREADS": ""}) as worker:
        assert worker.call("calls") == {"calls": []}  # no cap: torch left alone


def test_worker_still_takes_log_path_as_a_deprecated_alias(tmp_path):
    adapter = tmp_path / "h_adapter.py"
    adapter.write_text(
        "def load(**kw): print('to the log')\ndef ping(state): return {}\n"
    )
    with pytest.warns(DeprecationWarning, match="stderr="):
        worker = Worker(sys.executable, adapter, log_path=tmp_path / "old.log")
    with worker:
        worker.call("ping")
    assert "to the log" in (tmp_path / "old.log").read_text()


def test_ensure_env_rebuilds_a_half_built_env_an_older_voxy_left(tmp_path, monkeypatch):
    """Older voxy marked an env only when built: a venv with no marker is voxy's to replace."""
    from ps import isolated

    from voxy.envs import env_dir, env_python

    monkeypatch.setattr(isolated, "_find_uv", lambda: "uv")
    spec = EnvSpec("demo", ("numpy",))
    python = env_python(spec, rootdir=tmp_path)
    python.parent.mkdir(parents=True)
    python.touch()  # the venv an interrupted build left, without .voxy-env.json
    (env_dir(spec, rootdir=tmp_path) / "leftover").touch()
    commands = []

    def run(cmd, **kwargs):
        commands.append(list(cmd))
        if cmd[1] == "venv":
            assert not env_dir(spec, rootdir=tmp_path).exists()  # removed first
            python.parent.mkdir(parents=True)
            python.touch()

    ensure_env(spec, rootdir=tmp_path, run=run, log=lambda *a: None)
    assert is_env_ready(spec, rootdir=tmp_path) and len(commands) == 2


def test_ensure_env_leaves_an_unmarked_folder_that_is_not_a_venv(tmp_path, monkeypatch):
    from ps import isolated

    from voxy.envs import env_dir

    monkeypatch.setattr(isolated, "_find_uv", lambda: "uv")
    spec = EnvSpec("demo", ("numpy",))
    folder = env_dir(spec, rootdir=tmp_path)
    folder.mkdir(parents=True)
    (folder / "notes.txt").write_text("mine")
    with pytest.raises(RuntimeError, match="not built by ps"):
        ensure_env(
            spec, rootdir=tmp_path, run=lambda *a, **k: None, log=lambda *a: None
        )
    assert (folder / "notes.txt").read_text() == "mine"


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
        words = [
            _word(0.2, 1.5, "It"),
            _word(1.6, 3.9, "began."),
            _word(5.1, 7, "Then"),
            _word(7.1, 9.8, "ended."),
        ]
        return [{"start": 0, "end": 10, "text": "", "words": words}]

    samples = samples_store("ada", rootdir=tmp_path)
    manifest = refprep.prepare_reference(
        [source],
        samples=samples,
        conditioner=conditioner,
        transcriber=transcriber,
        min_seconds=2,
        log=lambda *a: None,
    )
    assert sorted(c["text"] for c in manifest) == ["It began.", "Then ended."]
    assert all(0 <= c["score"] <= 1 and c["path"].endswith(".wav") for c in manifest)
    assert refprep.load_manifest(samples) == manifest
    assert "conditioned/tape.wav" in samples


def test_prepare_reference_keeps_same_named_sources_apart(tmp_path):
    """Demucs stems are all ``<track>/vocals.wav``: no clip may overwrite another's."""
    rng = np.random.default_rng(2)
    sources = {}
    for folder, gain in [("song-a", 0.1), ("song-b", 0.4)]:
        (tmp_path / folder).mkdir()
        sources[folder] = tmp_path / folder / "vocals.wav"
        sources[folder].write_bytes(
            write_wav(gain * rng.standard_normal(SR * 4).astype(np.float32), SR)
        )

    def conditioner(src, dst, **kwargs):
        dst.write_bytes(open(src, "rb").read())
        return dst

    def transcriber(path, language=None):
        return [
            {"start": 0, "end": 4, "text": "", "words": [_word(0.2, 3.8, "Hello.")]}
        ]

    samples = samples_store("ada", rootdir=tmp_path / "data")
    manifest = refprep.prepare_reference(
        list(sources.values()),
        samples=samples,
        conditioner=conditioner,
        transcriber=transcriber,
        min_seconds=2,
        log=lambda *a: None,
    )
    assert len({c["key"] for c in manifest}) == len(manifest) == 2
    for clip in manifest:
        folder = Path(clip["source"]).parent.name
        assert folder in clip["key"]
        rms = np.sqrt(np.mean(read_wav(samples[clip["key"]])[0] ** 2))
        assert (rms > 0.2) == (folder == "song-b")  # the bytes are this source's
    for key in ("conditioned/song-a-vocals.wav", "conditioned/song-b-vocals.wav"):
        assert key in samples  # (listing the store gives OS separators on Windows)


def test_unique_slugs_deepens_only_names_that_collide():
    """A unique stem keeps its key, whatever order the sources come in."""
    unique, *stems = ["/s/one-vocals.wav", "/s/one/vocals.wav", "/s/two/vocals.wav"]
    for sources in ([unique, *stems], [*stems, unique]):
        slugs = dict(zip(sources, refprep.unique_slugs(sources)))
        assert slugs[unique] == "one-vocals"
        assert slugs["/s/two/vocals.wav"] == "two-vocals"
        assert len(set(slugs.values())) == 3


def test_unique_slugs_one_repeat_or_folderless_member_does_not_collapse_its_group():
    """#69: a same-folder pair or a bare relative path used to number the whole group."""
    assert refprep.unique_slugs(["/x/talk.wav", "/x/talk.m4a", "/y/talk.wav"]) == [
        "x-talk",
        "x-talk-2",
        "y-talk",  # keeps its folder
    ]
    assert refprep.unique_slugs(["a/v.wav", "v.wav", "b/v.wav"]) == ["a-v", "v", "b-v"]
    # folders that only slug alike never separate paths: no climb to the home folder
    assert refprep.unique_slugs(["/Users/x/Audio/v.wav", "/Users/x/audio/v.wav"]) == [
        "v",
        "v-2",
    ]
    assert refprep.unique_slugs(["/home/me/日本/v.wav", "/home/me/中国/v.wav"]) == [
        "v",
        "v-2",
    ]
    assert refprep.unique_slugs(["/s/a/x/v.wav", "/s/b/x/v.wav", "/s/c/y/v.wav"]) == [
        "a-x-v",
        "b-x-v",
        "y-v",  # the shortest that separates it
    ]


def test_unique_slugs_names_an_unsluggable_stem_source_not_its_folder():
    assert refprep.unique_slugs(["/s/日本.wav", "/s/x.wav"]) == ["source", "x"]
    assert refprep.unique_slugs(["/s/日本.wav", "/t/日本.wav"]) == [
        "s-source",
        "t-source",
    ]


def _stem(folder, root, rng, gain=0.2):
    """A ``<root>/<folder>/vocals.wav`` of noise at ``gain`` (a Demucs-style stem)."""
    (root / folder).mkdir(exist_ok=True)
    path = root / folder / "vocals.wav"
    path.write_bytes(
        write_wav(gain * rng.standard_normal(SR * 6).astype(np.float32), SR)
    )
    return path


def _rerun(samples, *, n_clips=1, offset=0.0, log=lambda *a: None):
    """A ``prepare_reference`` into ``samples`` with fakes cutting ``n_clips`` per
    source (starting ``offset`` seconds in)."""

    def transcriber(path, language=None):
        words = [
            _word(offset + i * 2 + 0.2, offset + i * 2 + 1.8, f"Hello {i}.")
            for i in range(n_clips)
        ]
        return [{"start": 0, "end": 6, "text": "", "words": words}]

    def run(*sources):
        return refprep.prepare_reference(
            list(sources),
            samples=samples,
            conditioner=lambda src, dst, **kw: dst.write_bytes(open(src, "rb").read()),
            transcriber=transcriber,
            min_seconds=1,
            log=log,
        )

    return run


def _keys(samples, prefix=""):
    return {
        k.replace("\\", "/") for k in samples if k.replace("\\", "/").startswith(prefix)
    }


def _snapshot(samples):
    """Every key and its bytes, but the manifest (which a run rewrites)."""
    return {
        k.replace("\\", "/"): samples[k] for k in samples if k != refprep.MANIFEST_KEY
    }


def _assert_untouched(before, samples):
    assert {k: samples[k] for k in before} == before  # nothing deleted or overwritten


def _set_manifest(samples, entries):
    samples[refprep.MANIFEST_KEY] = "\n".join(map(json.dumps, entries)).encode()


def test_running_prepare_reference_again_keeps_earlier_clips_and_adds_new_keys(
    tmp_path,
):
    """#68: a re-run adds; every earlier clip keeps its key, its bytes and its entry."""
    rng = np.random.default_rng(3)
    a, b = _stem("song-a", tmp_path, rng, 0.1), _stem("song-b", tmp_path, rng, 0.4)
    samples = samples_store("ada", rootdir=tmp_path / "data")
    first = _rerun(samples)(a, b)
    before = _snapshot(samples)
    logged = []
    second = _rerun(samples, log=logged.append)(a)
    _assert_untouched(before, samples)
    new = set(_snapshot(samples)) - set(before)
    assert new == {"clips/vocals-0001.wav", "conditioned/vocals.wav"}  # fresh slug
    assert {e["key"] for e in second} == {e["key"] for e in first} | {
        "clips/vocals-0001.wav"
    }
    assert len({e["run"] for e in second}) == 2
    assert refprep.load_manifest(samples) == second
    assert any("forget" in line and str(a) in line for line in logged)


def test_a_new_recording_under_an_old_filename_keeps_the_old_takes_clips(tmp_path):
    rng = np.random.default_rng(4)
    a = _stem("song-a", tmp_path, rng, 0.05)
    samples = samples_store("ada", rootdir=tmp_path / "data")
    _rerun(samples, n_clips=3)(a)
    before = _snapshot(samples)
    _stem("song-a", tmp_path, rng, 0.6)  # a new take, same path
    manifest = _rerun(samples)(a)
    _assert_untouched(before, samples)
    assert len(manifest) == 4


def test_saved_reference_paths_keep_their_bytes_and_text(tmp_path):
    rng = np.random.default_rng(5)
    a = _stem("song-a", tmp_path, rng)
    samples = samples_store("ada", rootdir=tmp_path / "data")
    ref = refprep.best_references(_rerun(samples, n_clips=2)(a), min_seconds=1)[0]
    audio = Path(ref["path"]).read_bytes()
    manifest = _rerun(samples, offset=1.0)(a)  # cut differently this time
    assert Path(ref["path"]).read_bytes() == audio
    assert [e["text"] for e in manifest if e["key"] == ref["key"]] == [ref["text"]]


def test_prepare_reference_never_overwrites_orphan_keys(tmp_path):
    """Clips in no manifest (an older store, a crashed run, a zero-clip source)."""
    a = _stem("song-a", tmp_path, np.random.default_rng(6))
    samples = {
        "clips/vocals-0001.wav": b"orphan a saved profile points at",
        "conditioned/vocals-2.wav": b"a zero-clip source's",
    }
    before = dict(samples)
    manifest = _rerun(samples)(a)
    _assert_untouched(before, samples)
    assert [e["key"] for e in manifest] == ["clips/vocals-3-0001.wav"]


@pytest.mark.parametrize(
    "entries",
    [
        # a slug field that is not a slug, naming a clip-shaped key
        [{"key": "clips/vocals-0001.wav", "slug": "BAD!", "source": "/x/o.wav"}],
        # a slug field that disagrees with the key
        [{"key": "clips/vocals-0001.wav", "slug": "x", "score": 0.5}],
        # two sources sharing one slug
        [
            {"key": "clips/vocals-0001.wav", "slug": "vocals", "source": "/x/a.wav"},
            {"key": "clips/vocals-0002.wav", "slug": "vocals", "source": "/x/b.wav"},
        ],
        # no score, no key, a non-string slug or key
        [{"key": "tape1.wav"}, {"score": 0.1, "source": "/q.wav"}, {"slug": 3}],
        [{"key": 7, "score": "high"}],
    ],
)
def test_a_corrupt_manifest_neither_crashes_nor_overwrites(tmp_path, entries):
    a = _stem("song-a", tmp_path, np.random.default_rng(7))
    samples = samples_store("ada", rootdir=tmp_path / "data")
    for e in entries:
        if isinstance(e.get("key"), str):
            samples[e["key"]] = f"bytes of {e['key']}".encode()
    _set_manifest(samples, entries)
    before = _snapshot(samples)
    manifest = _rerun(samples, n_clips=2)(a)
    _assert_untouched(before, samples)
    assert len(manifest) == len(entries) + 2  # every earlier entry kept
    assert all(e in manifest for e in refprep.load_manifest(samples))
    scored = [isinstance(e.get("score"), (int, float)) for e in manifest]
    assert scored == sorted(scored, reverse=True)  # unscored entries sort last


def test_a_manifest_line_that_is_not_an_object_fails_before_writing(tmp_path):
    a = _stem("song-a", tmp_path, np.random.default_rng(8))
    samples = {refprep.MANIFEST_KEY: b'{"key": "clips/x-0001.wav"}\n[1, 2]'}
    with pytest.raises(ValueError, match="not JSON objects"):
        _rerun(samples)(a)
    assert set(samples) == {refprep.MANIFEST_KEY}


def test_writing_an_existing_key_raises():
    with pytest.raises(FileExistsError):
        refprep._put_new({"clips/a-0001.wav": b"A"}, "clips/a-0001.wav", b"B")


def test_prepare_reference_tells_sources_apart_by_real_path(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    a = _stem("song-a", tmp_path, rng)
    samples = samples_store("ada", rootdir=tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    _rerun(samples, n_clips=2)(Path("song-a/vocals.wav"))
    manifest = _rerun(samples)(a, Path("song-a") / ".." / "song-a" / "vocals.wav")
    assert len([e for e in manifest if e["slug"] == "vocals-2"]) == 1  # once
    assert {e["source_path"] for e in manifest} == {refprep.source_identity(a)}
    assert len(refprep.forget(samples, a)) == 5  # both runs: 3 clips, 2 conditioned
    assert _keys(samples, "clips/") == _keys(samples, "conditioned/") == set()
    assert refprep.load_manifest(samples) == []


def test_forget_removes_one_runs_entries_and_files_and_nothing_else(tmp_path):
    rng = np.random.default_rng(10)
    a, b = _stem("song-a", tmp_path, rng), _stem("song-b", tmp_path, rng)
    samples = samples_store("ada", rootdir=tmp_path / "data")
    first = _rerun(samples, n_clips=2)(a, b)
    _rerun(samples)(a)
    b_files = {k: v for k, v in _snapshot(samples).items() if "song-b" in k}
    samples["tape1.wav"] = b"mine"
    hand = {"key": "tape1.wav", "source": str(a)}  # a top-level sample, same source
    _set_manifest(samples, refprep.load_manifest(samples) + [hand])

    deleted = refprep.forget(samples, a, run=first[0]["run"])
    assert sorted(deleted) == [
        "clips/song-a-vocals-0001.wav",
        "clips/song-a-vocals-0002.wav",
        "conditioned/song-a-vocals.wav",
    ]
    assert {"clips/vocals-0001.wav", "conditioned/vocals.wav"} <= _keys(samples)

    deleted = refprep.forget(samples, a)  # the rest of a
    assert sorted(deleted) == ["clips/vocals-0001.wav", "conditioned/vocals.wav"]
    assert samples["tape1.wav"] == b"mine"  # its entry is gone, never the sample
    _assert_untouched(b_files, samples)
    assert sorted(e["key"] for e in refprep.load_manifest(samples)) == [
        "clips/song-b-vocals-0001.wav",
        "clips/song-b-vocals-0002.wav",
    ]


def test_forget_keeps_files_a_remaining_entry_names_or_shares_a_slug_with():
    samples = {
        k: b"x" for k in ("clips/a-0001.wav", "clips/a-0002.wav", "conditioned/a.wav")
    }
    a, b = (
        os.path.realpath("/s/a.wav"),
        os.path.realpath("/s/b.wav"),
    )  # a drive on Windows
    _set_manifest(
        samples,
        [
            {"key": "clips/a-0001.wav", "slug": "a", "source_path": a},
            {"key": "clips/a-0002.wav", "slug": "a", "source_path": a},
            {"key": "clips/a-0002.wav", "source_path": b},  # corrupt: b names it
        ],
    )
    assert refprep.forget(samples, a) == []
    assert len(_keys(samples, "clips/")) == 2 and "conditioned/a.wav" in samples
    assert [e["source_path"] for e in refprep.load_manifest(samples)] == [b]


def test_forget_matches_old_entries_only_by_absolute_real_path():
    samples = {k: b"x" for k in ("clips/talk-0001.wav", "conditioned/talk.wav")}
    samples["clips/abs-0001.wav"] = b"A"
    entries = [
        {"key": "clips/talk-0001.wav", "source": "talk.m4a"},  # relative: unknown file
        {"key": "clips/abs-0001.wav", "source": "/s/abs.m4a"},
    ]
    _set_manifest(samples, entries)
    logged = []
    assert refprep.forget(samples, "talk.m4a", log=logged.append) == []
    assert refprep.forget(samples, "/elsewhere/talk.m4a", log=logged.append) == []
    assert all("LEGACY_RUN" in line and "'talk.m4a'" in line for line in logged)
    assert len(logged) == 2 and refprep.load_manifest(samples) == entries
    assert refprep.forget(samples, "abs.m4a", log=logged.append) == []  # not /s/
    assert refprep.forget(samples, "/s/abs.m4a") == ["clips/abs-0001.wav"]
    # deliberately, by the string as recorded
    legacy = refprep.LEGACY_RUN
    assert refprep.forget(samples, "talk.m4a", run=legacy) == [
        "clips/talk-0001.wav",
        "conditioned/talk.wav",
    ]
    assert set(samples) == {refprep.MANIFEST_KEY}


def test_forget_of_a_relative_name_never_deletes_another_folders_old_clips(
    tmp_path, monkeypatch
):
    """Review repro: an older voxy stored ``vocals.wav`` (relative) for a stem in X;
    a new run of Y's ``vocals.wav``; ``forget("vocals.wav")`` from Y kept X's clips."""
    rng = np.random.default_rng(20)
    x, y = _stem("x", tmp_path, rng), _stem("y", tmp_path, rng, 0.5)
    samples = {}
    monkeypatch.chdir(x.parent)
    _rerun(samples, n_clips=2)("vocals.wav")
    old = refprep.load_manifest(samples)
    for e in old:  # as the older voxy wrote it
        del e["slug"], e["source_path"], e["run"]
    _set_manifest(samples, old)
    x_files = _snapshot(samples)
    monkeypatch.chdir(y.parent)
    _rerun(samples)("vocals.wav")
    logged = []
    deleted = refprep.forget(samples, "vocals.wav", log=logged.append)
    assert deleted == ["clips/vocals-2-0001.wav", "conditioned/vocals-2.wav"]
    _assert_untouched(x_files, samples)
    assert [e["key"] for e in refprep.load_manifest(samples)] == [e["key"] for e in old]
    assert any("run=refprep.LEGACY_RUN" in line for line in logged)
    legacy = refprep.LEGACY_RUN
    would = refprep.forget(samples, "vocals.wav", run=legacy, dry_run=True)
    assert sorted(would) == sorted(x_files)
    _assert_untouched(x_files, samples)
    assert sorted(refprep.forget(samples, "vocals.wav", run=legacy)) == sorted(would)


def test_forget_dry_run_lists_what_it_would_delete_and_changes_nothing(tmp_path):
    rng = np.random.default_rng(21)
    a, b = _stem("song-a", tmp_path, rng), _stem("song-b", tmp_path, rng)
    samples = samples_store("ada", rootdir=tmp_path / "data")
    _rerun(samples, n_clips=2)(a, b)
    before, manifest = _snapshot(samples), samples[refprep.MANIFEST_KEY]
    would = refprep.forget(samples, a, dry_run=True)
    assert _snapshot(samples) == before and samples[refprep.MANIFEST_KEY] == manifest
    assert refprep.forget(samples, tmp_path / "nope.wav", dry_run=True) == []
    assert refprep.forget(samples, a) == would and len(would) == 3


def test_forget_takes_conditioned_keys_only_from_clip_key_slugs(tmp_path):
    """A hand-edited ``slug`` on a top-level entry must not doom another source's
    conditioned file."""
    a = _stem("song-a", tmp_path, np.random.default_rng(22))
    samples = {"conditioned/zz.wav": b"another source's"}
    _rerun(samples)(a)
    hand = {"key": "tape1.wav", "slug": "zz", "source_path": os.path.realpath(a)}
    _set_manifest(samples, refprep.load_manifest(samples) + [hand])
    assert refprep.forget(samples, a) == [
        "clips/vocals-0001.wav",
        "conditioned/vocals.wav",
    ]
    assert samples["conditioned/zz.wav"] == b"another source's"


def _prep_with_hook(samples, hook, *sources):
    """``prepare_reference`` whose transcriber first calls ``hook`` (another process
    changing the store while this run is in flight)."""

    def transcriber(path, language=None):
        hook()
        return [{"start": 0, "end": 6, "text": "", "words": [_word(0.2, 1.8, "Hi.")]}]

    return refprep.prepare_reference(
        list(sources),
        samples=samples,
        conditioner=lambda src, dst, **kw: dst.write_bytes(open(src, "rb").read()),
        transcriber=transcriber,
        min_seconds=1,
        log=lambda *a: None,
    )


def test_prepare_reference_rereads_the_manifest_before_writing_it(tmp_path):
    """A ``forget`` or another run finishing meanwhile is not undone."""
    rng = np.random.default_rng(23)
    a, b, c = (_stem(f"s{i}", tmp_path, rng) for i in range(3))
    samples = {}
    _rerun(samples, n_clips=2)(a)
    once = []

    def meanwhile():
        if not once:
            once.append(refprep.forget(samples, a))
            once.append(_rerun(samples)(c))

    manifest = _prep_with_hook(samples, meanwhile, b)
    assert len(once[0]) == 3  # a's 2 clips + conditioned
    assert {e["source_path"] for e in manifest} == {
        refprep.source_identity(p) for p in (b, c)
    }
    assert all(e["key"] in samples for e in refprep.load_manifest(samples))


def test_run_ids_are_utc_sortable_and_unique():
    ids = [refprep._new_run_id() for _ in range(200)]
    assert len(set(ids)) == len(ids)
    stamps = [i.split("-")[0] for i in ids]
    assert stamps == sorted(stamps) and all(s.endswith("Z") for s in stamps)


def test_consumers_read_each_sources_latest_run(tmp_path):
    from voxy import finetune

    rng = np.random.default_rng(24)
    a, b = _stem("song-a", tmp_path, rng), _stem("song-b", tmp_path, rng)
    samples = samples_store("ada", rootdir=tmp_path / "data")
    _rerun(samples, n_clips=3)(a, b)
    manifest = _rerun(samples, n_clips=3)(a)  # a again
    assert len(manifest) == 9
    latest = refprep.latest_runs(manifest)
    a_runs = {
        e["run"] for e in latest if e["source_path"] == refprep.source_identity(a)
    }
    assert len(latest) == 6 and a_runs == {max(e["run"] for e in manifest)}

    out = tmp_path / "ft"
    report = finetune.export_dataset(manifest, samples, out, min_score=0)
    rows = (out / "metadata.csv").read_text().splitlines()
    assert report["clips"] == len(rows) == 6  # each clip of each source once
    assert len({r.split("|", 1)[1] for r in rows}) == 3  # a's and b's texts, no dupes
    assert sorted(p.name for p in (out / "wavs").iterdir()) == sorted(
        Path(e["key"]).name for e in latest
    )
    every = finetune.select_clips(manifest, min_score=0, latest_only=False)
    assert len(every) == 9

    refs = refprep.best_references(manifest, n=10, min_seconds=1)
    assert {r["key"] for r in refs} <= {e["key"] for e in latest}
    assert (
        len(refprep.best_references(manifest, n=10, min_seconds=1, latest_only=False))
        == 9
    )


def test_finetune_skips_entries_without_score_or_duration(tmp_path):
    from voxy import finetune

    a = _stem("song-a", tmp_path, np.random.default_rng(25))
    samples = {"tape1.wav": b"T"}
    _set_manifest(
        samples,
        [{"key": "tape1.wav", "text": "x"}, {"score": 0.9, "duration": 2}, {"key": 3}],
    )
    manifest = _rerun(samples)(a)
    report = finetune.export_dataset(manifest, samples, tmp_path / "ft", min_score=0)
    assert report["clips"] == 1
    refs = refprep.best_references(manifest, n=10, min_seconds=1)
    assert all(isinstance(r.get("score"), float | int) for r in refs)  # no crash


class _StoreFailingOn(dict):
    """A samples store whose write of one key fails (a full disk, mid-run)."""

    def __init__(self, failing_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failing_key = failing_key

    def __setitem__(self, key, value):
        if key == self.failing_key:
            raise OSError(f"no space left writing {key}")
        super().__setitem__(key, value)


def _two_clip_transcriber(*, fail_on_call=None):
    """Two clips per source; raises on call number ``fail_on_call`` (an ASR crash)."""
    calls = []

    def transcriber(path, language=None):
        calls.append(path)
        if len(calls) == fail_on_call:
            raise RuntimeError("asr worker died")
        words = [_word(0.2, 1.8, "Hello."), _word(2.2, 3.8, "There.")]
        return [{"start": 0, "end": 6, "text": "", "words": words}]

    return transcriber


@pytest.mark.parametrize(
    "crash, partial",
    [
        # the transcriber dies on the second source: only its conditioned file is written
        ("transcriber", ["conditioned/song-b-vocals.wav"]),
        # the store fails on the second source's second clip: one clip is already written
        (
            "store",
            ["clips/song-b-vocals-0001.wav", "conditioned/song-b-vocals.wav"],
        ),
    ],
)
def test_a_crash_mid_run_keeps_finished_sources_and_orphans_only_the_failing_one(
    tmp_path, crash, partial
):
    """#92: the manifest is written per source; #98: ``orphans`` finds the rest."""
    rng = np.random.default_rng(30)
    a, b = _stem("song-a", tmp_path, rng), _stem("song-b", tmp_path, rng)
    failing = "clips/song-b-vocals-0002.wav" if crash == "store" else None
    samples = _StoreFailingOn(failing, {"tape1.wav": b"mine"})
    with pytest.raises((RuntimeError, OSError)):
        refprep.prepare_reference(
            [a, b],
            samples=samples,
            conditioner=lambda src, dst, **kw: dst.write_bytes(open(src, "rb").read()),
            transcriber=_two_clip_transcriber(
                fail_on_call=2 if crash == "transcriber" else None
            ),
            min_seconds=1,
            log=lambda *a: None,
        )
    manifest = refprep.load_manifest(samples)
    assert {e["source_path"] for e in manifest} == {refprep.source_identity(a)}
    assert sorted(e["key"] for e in manifest) == [
        "clips/song-a-vocals-0001.wav",
        "clips/song-a-vocals-0002.wav",
    ]
    (marker,) = _keys(samples, refprep.PENDING_PREFIX)  # the crash stays visible
    assert refprep.orphans(samples) == []  # its marker claims song-b until it is stale
    assert refprep.orphans(samples, stale_after=0) == partial

    before, manifest_bytes = _snapshot(samples), samples[refprep.MANIFEST_KEY]
    stale = [*partial, marker]
    assert refprep.forget_orphans(samples, stale_after=0) == stale  # a dry run
    assert _snapshot(samples) == before
    assert refprep.forget_orphans(samples, dry_run=False, stale_after=0) == stale
    assert set(before) - set(_snapshot(samples)) == set(stale)  # those, nothing else
    assert samples[refprep.MANIFEST_KEY] == manifest_bytes
    assert samples["tape1.wav"] == b"mine"
    assert refprep.orphans(samples) == []


def test_orphans_are_only_unclaimed_keys_of_our_own_form(tmp_path):
    a = _stem("song-a", tmp_path, np.random.default_rng(31))
    samples = {}
    refprep.prepare_reference(  # a zero-clip source: its conditioned file only
        [a],
        samples=samples,
        conditioner=lambda src, dst, **kw: dst.write_bytes(open(src, "rb").read()),
        transcriber=lambda path, language=None: [],
        log=lambda *a: None,
    )
    assert refprep.load_manifest(samples) == []
    with pytest.raises(ValueError, match="missing, empty"):  # see the refusal test
        refprep.orphans(samples)
    kept = {
        "clips/x-0003.wav": b"named by an entry",
        "conditioned/x.wav": b"its slug is an entry key's",
        "conditioned/y.wav": b"its slug is an entry's slug field",
        "tape1.wav": b"a top-level sample",
        "clips/notes.txt": b"not a clip key",
        "clips/Bad-0001.wav": b"not a slug we write",
        "other/z-0001.wav": b"not our prefix",
    }
    lost = {"clips/lost-0002.wav": b"crashed run", "conditioned/lost.wav": b"crashed"}
    for key, data in {**kept, **lost}.items():
        samples[key] = data
    hand = {"key": "clips/x-0003.wav", "slug": "y", "source": "/s/x.wav"}
    _set_manifest(samples, [hand])
    expected = sorted([*lost, "conditioned/vocals.wav"])
    assert refprep.orphans(samples) == expected
    assert refprep.forget_orphans(samples, dry_run=False) == expected
    assert {k: samples[k] for k in kept} == kept
    assert refprep.load_manifest(samples) == [hand]


def test_orphans_refuses_a_manifest_it_cannot_read():
    samples = {"conditioned/a.wav": b"A", refprep.MANIFEST_KEY: b"[1, 2]"}
    with pytest.raises(ValueError, match="not JSON objects"):
        refprep.forget_orphans(samples, dry_run=False)
    assert "conditioned/a.wav" in samples


@pytest.mark.parametrize("manifest", [None, b""], ids=["missing", "zero-byte"])
def test_orphans_refuses_without_a_manifest_and_deletes_nothing(tmp_path, manifest):
    """A lost or truncated manifest must not make every clip an orphan."""
    samples = samples_store("ada", rootdir=tmp_path / "data")
    for key in ["clips/talk-0001.wav", "clips/talk-0002.wav", "conditioned/talk.wav"]:
        samples[key] = b"a clip a saved profile may point at"
    samples["tape1.wav"] = b"mine"
    if manifest is not None:
        samples[refprep.MANIFEST_KEY] = manifest
    before = _snapshot(samples)
    for call in (refprep.orphans, refprep.forget_orphans):
        with pytest.raises(ValueError, match="missing, empty"):
            call(samples)
    with pytest.raises(ValueError, match="missing, empty"):
        refprep.forget_orphans(samples, dry_run=False, stale_after=0)
    assert _snapshot(samples) == before
    del samples["clips/talk-0001.wav"], samples["clips/talk-0002.wav"]
    del samples["conditioned/talk.wav"]
    assert refprep.orphans(samples) == []  # nothing of ours: nothing to refuse


def _replace_failing_for(name, monkeypatch):
    """Make ``os.replace`` (as refprep calls it) fail when moving onto ``name``: a crash
    between writing the temporary file and moving it into place."""
    real = os.replace

    def replace(src, dst):
        if os.path.basename(dst) == name:
            raise OSError(f"killed while replacing {name}")
        real(src, dst)

    monkeypatch.setattr(refprep.os, "replace", replace)


def test_manifest_writes_are_atomic_on_a_local_store(tmp_path, monkeypatch):
    rng = np.random.default_rng(33)
    a, b = _stem("song-a", tmp_path, rng), _stem("song-b", tmp_path, rng)
    samples = samples_store("ada", rootdir=tmp_path / "data")
    _rerun(samples, n_clips=2)(a)
    folder = Path(samples.rootdir)
    assert not [p for p in folder.iterdir() if p.name.startswith(".")]  # no temp left
    manifest_bytes = samples[refprep.MANIFEST_KEY]

    _replace_failing_for(refprep.MANIFEST_KEY, monkeypatch)
    with pytest.raises(OSError, match="killed"):  # prepare_reference's per-source write
        _rerun(samples)(b)
    with pytest.raises(OSError, match="killed"):  # forget's write
        refprep.forget(samples, a)
    assert samples[refprep.MANIFEST_KEY] == manifest_bytes  # whole, never truncated
    assert not [p for p in folder.iterdir() if p.name.startswith(".")]
    assert len(refprep.load_manifest(samples)) == 2
    assert all(e["key"] in samples for e in refprep.load_manifest(samples))


def test_forget_orphans_during_a_run_spares_that_run_until_its_marker_is_stale(
    tmp_path,
):
    """A ``prepare_reference`` in flight has keys its manifest does not name yet."""
    rng = np.random.default_rng(34)
    a, b = _stem("song-a", tmp_path, rng), _stem("song-b", tmp_path, rng)
    samples = samples_store("ada", rootdir=tmp_path / "data")
    samples["tape1.wav"] = b"mine"
    _rerun(samples)(a)
    c = _stem("song-c", tmp_path, rng)
    calls, seen = [], []

    def meanwhile():  # forget_orphans from another process, mid-run
        calls.append(1)
        if len(calls) == 2:  # b is done; c's conditioned file is written, not listed
            seen.append(_keys(samples, refprep.PENDING_PREFIX))
            seen.append(refprep.forget_orphans(samples, dry_run=False))

    manifest = _prep_with_hook(samples, meanwhile, b, c)
    (markers, deleted) = seen
    assert len(markers) == 1 and deleted == []
    assert all(e["key"] in samples for e in manifest)
    assert "conditioned/song-c-vocals.wav" in samples
    assert _keys(samples, refprep.PENDING_PREFIX) == set()  # removed when it finished
    sample_keys = [k for k in samples if _is_sample_key(k)]
    assert sample_keys == ["tape1.wav"]  # what clone_from_samples would upload

    def crash():
        raise RuntimeError("asr worker died")

    with pytest.raises(RuntimeError):
        _prep_with_hook(samples, crash, _stem("song-d", tmp_path, rng))
    (marker,) = _keys(samples, refprep.PENDING_PREFIX)
    assert refprep.forget_orphans(samples, dry_run=False) == []  # still live
    (slug,) = json.loads(samples[marker])["slugs"]
    crashed = [f"conditioned/{slug}.wav", marker]
    assert refprep.forget_orphans(samples, stale_after=0) == crashed  # '/' keys, any OS
    assert refprep.forget_orphans(samples, dry_run=False, stale_after=0) == crashed
    assert _keys(samples, refprep.PENDING_PREFIX) == set()
    assert all(e["key"] in samples for e in refprep.load_manifest(samples))


def test_an_unreadable_pending_marker_refuses_rather_than_guess():
    samples = {
        "conditioned/a.wav": b"A",
        refprep.MANIFEST_KEY: b'{"key": "clips/b-0001.wav"}',
        "pending/r.json": b'{"run": "r", "slugs": ["a"]}',  # no start time
    }
    with pytest.raises(ValueError, match="pending marker"):
        refprep.forget_orphans(samples, dry_run=False, stale_after=0)
    assert "conditioned/a.wav" in samples


def test_nfc_and_nfd_spellings_of_one_file_are_one_source(tmp_path):
    import unicodedata

    nfd = unicodedata.normalize("NFD", "Café.wav")
    nfc = unicodedata.normalize("NFC", "Café.wav")
    (tmp_path / nfd).touch()
    assert refprep.source_identity(tmp_path / nfc) == refprep.source_identity(
        tmp_path / nfd
    )


def _is_case_insensitive(folder: Path) -> bool:
    probe = folder / "CaseProbe"
    probe.touch()
    try:
        return (folder / "caseprobe").exists()
    finally:
        probe.unlink()


def test_one_file_typed_in_two_cases_is_one_source(tmp_path):
    """#98: on a case-insensitive filesystem, ``Talk.wav`` and ``talk.wav`` are one file."""
    if not _is_case_insensitive(tmp_path):
        pytest.skip("tmp_path is on a case-sensitive filesystem")
    (tmp_path / "Tapes").mkdir()
    talk = tmp_path / "Tapes" / "Talk.wav"
    noise = 0.2 * np.random.default_rng(32).standard_normal(SR * 6)
    talk.write_bytes(write_wav(noise.astype(np.float32), SR))
    samples = samples_store("ada", rootdir=tmp_path / "data")
    samples["clips/old-0001.wav"] = b"an older voxy's clip"
    older = {  # recorded by an older voxy, in the case it was typed then
        "key": "clips/old-0001.wav",
        "source_path": os.path.realpath(tmp_path / "tapes" / "talk.wav"),
        "run": "20200101T000000.000000Z-aaaaaa",
    }
    _set_manifest(samples, [older])
    _rerun(samples)(talk)
    manifest = _rerun(samples)(tmp_path / "TAPES" / "talk.wav", talk)  # once, not twice
    assert len(manifest) == 3
    new = [e for e in manifest if e["key"] != older["key"]]
    assert {e["source_path"] for e in new} == {refprep.source_identity(talk)}
    assert refprep.source_identity(tmp_path / "tapes" / "TALK.WAV") == (
        os.path.normcase(os.path.realpath(talk))
    )
    latest = refprep.latest_runs(manifest)
    assert [e["run"] for e in latest] == [max(e["run"] for e in manifest)]
    deleted = refprep.forget(samples, tmp_path / "tapes" / "TALK.WAV")
    assert len(deleted) == 5  # the old clip, two runs' clip and conditioned file
    assert refprep.load_manifest(samples) == []
    assert _keys(samples, "clips/") == _keys(samples, "conditioned/") == set()


def test_plan_clips_respects_max_length():
    words = [_word(i, i + 0.9, f"w{i}") for i in range(30)]
    clips = refprep.plan_clips(
        [{"words": words, "start": 0, "end": 30, "text": ""}],
        min_seconds=3,
        max_seconds=8,
    )
    assert all(c["end"] - c["start"] <= 8.5 for c in clips)
    assert " ".join(c["text"] for c in clips) == " ".join(w["text"] for w in words)


# diagnosis --------------------------------------------------------------------


def test_render_grid_and_page(tmp_path):
    calls = []

    def factory(backend, **init):
        return LocalCloneSpeechModel(backend, generate=fake_generate(calls), **init)

    variants = diagnosis.expand_grid(
        [
            {"backend": "f5tts", "params": {"nfe_step": [16, 32]}},
            {
                "backend": "chatterbox",
                "init": {"variant": "turbo"},
                "params": {"temperature": 0.7},
            },
        ],
        references=["r1"],
    )
    assert len({v.id for v in variants}) == 3
    refs = {"r1": reference_profile("r.wav", "Hi.")}
    results = diagnosis.render_grid(
        "Hello there.",
        refs,
        variants,
        tmp_path,
        model_factory=factory,
        audio_format="wav",
        log=lambda *a: None,
    )
    assert all(not r.get("error") and (tmp_path / r["audio"]).exists() for r in results)
    page = diagnosis.build_page(results, tmp_path, title="T", text="Hello there.")
    content = page.read_text()
    assert all(v.id in content for v in variants)
    assert "<b>nfe_step=16</b>" in content and "nfe_step=32" in content
    # resume: nothing re-rendered
    n = len(calls)
    diagnosis.render_grid(
        "Hello there.",
        refs,
        variants,
        tmp_path,
        model_factory=factory,
        audio_format="wav",
        log=lambda *a: None,
    )
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

    variants = diagnosis.expand_grid(
        [{"backend": "f5tts", "params": {"nfe_step": [16, 32]}}]
    )
    results = diagnosis.render_grid(
        "Hi.",
        {"": reference_profile("r.wav", "x")},
        variants,
        tmp_path,
        model_factory=factory,
        audio_format="wav",
        log=lambda *a: None,
    )
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

    spec = importlib.util.spec_from_file_location(
        "cb_adapter", adapter_path("chatterbox")
    )
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
    kw = dict(
        text="Bonjour.",
        ref_audio="r.wav",
        ref_text="x",
        out_path=str(tmp_path / "o.wav"),
    )
    adapter.synthesize(multi, language="fr", **kw)
    assert multi.kwargs["language_id"] == "fr"
    english_only = ChatterboxTTS()
    with pytest.raises(ValueError, match="multilingual"):
        adapter.synthesize(english_only, language="fr", **kw)
    adapter.synthesize(english_only, **kw)
    assert "language_id" not in english_only.kwargs


def test_deepfilternet_is_a_known_denoise_with_no_ffmpeg_filter():
    assert refprep.DENOISE_FILTERS["deepfilternet"] == []
    assert refprep.conditioning_filters(denoise="deepfilternet", highpass_hz=70) == [
        "highpass=f=70",
        "loudnorm=I=-20:TP=-1.5:LRA=11",
    ]


def test_with_deepfilternet_runs_ffmpeg_df_ffmpeg_in_order(tmp_path):
    calls = []

    def conditioner(src, dst, *, sample_rate, filters):
        calls.append(("ffmpeg", sample_rate, list(filters)))
        dst.write_bytes(b"x")
        return dst

    def denoiser(src, dst):
        calls.append(("df",))
        dst.write_bytes(b"y")
        return dst

    chain = refprep.with_deepfilternet(conditioner, denoiser=denoiser)
    out = chain(
        "in.m4a", tmp_path / "out.wav", sample_rate=24000, filters=["highpass=f=70"]
    )
    assert calls == [
        ("ffmpeg", refprep.DF_SAMPLE_RATE, []),
        ("df",),
        ("ffmpeg", 24000, ["highpass=f=70"]),
    ]
    assert out == tmp_path / "out.wav"


def test_deepfilternet_denoise_calls_the_env_command_and_copies_the_result(tmp_path):
    seen = {}

    def ensure(spec, *, rootdir=None):
        seen["spec"] = spec
        return tmp_path / "envs" / "deepfilternet" / "bin" / "python"

    def run(cmd, check, env):
        seen["cmd"] = cmd
        out_dir = cmd[cmd.index("-o") + 1]
        (Path(out_dir) / "wide_DeepFilterNet3.wav").write_bytes(b"clean")

    wide = tmp_path / "wide.wav"
    wide.write_bytes(b"in")
    dst = refprep.deepfilternet_denoise(
        wide, tmp_path / "x" / "clean.wav", run=run, ensure=ensure
    )
    assert dst.read_bytes() == b"clean"
    assert seen["cmd"][0].endswith("deepFilter") and seen["spec"] is refprep.DF_ENV
    assert "torch==2.0.1" in refprep.DF_ENV.requirements


def test_prepare_reference_with_deepfilternet_wraps_the_conditioner(
    tmp_path, monkeypatch
):
    steps = []

    def conditioner(src, dst, *, sample_rate, filters):
        steps.append(sample_rate)
        dst.write_bytes(write_wav(np.zeros(sample_rate, np.float32), sample_rate))
        return dst

    monkeypatch.setattr(
        refprep,
        "deepfilternet_denoise",
        lambda s, d, **k: (d.write_bytes(s.read_bytes()), d)[1],
    )
    refprep.prepare_reference(
        ["a.wav"],
        samples={},
        denoise="deepfilternet",
        conditioner=conditioner,
        transcriber=lambda path, language=None: [],
        log=lambda *a: None,
    )
    assert steps == [refprep.DF_SAMPLE_RATE, refprep.DFLT_SAMPLE_RATE]


@pytest.mark.skipif(
    not is_env_ready(refprep.DF_ENV),
    reason="the deepfilternet env is not built (python -m voxy setup)",
)
def test_deepfilternet_lowers_noise_for_real(tmp_path):
    rng = np.random.default_rng(0)
    sr = refprep.DF_SAMPLE_RATE
    noisy = (0.02 * rng.standard_normal(sr * 2)).astype(np.float32)
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    src.write_bytes(write_wav(noisy, sr))
    refprep.deepfilternet_denoise(src, dst)
    cleaned, _ = read_wav(dst)
    assert np.sqrt(np.mean(cleaned**2)) < np.sqrt(np.mean(noisy**2))


# local voice design -------------------------------------------------------------


def test_qwen_designs_a_voice_then_clones_from_the_chosen_preview(tmp_path):
    designed = []

    def design(text, *, instruct, seed, **kw):
        designed.append((instruct, seed))
        return 0.1 * np.sin(np.linspace(0, 50 + seed, SR)).astype(np.float32), SR

    calls = []
    model = voxy.create_speech_model(
        "qwen3tts", generate=fake_generate(calls), design=design
    )
    previews = model.design_voice_previews("A calm deep voice", seeds=[3, 5])
    assert [p.seed for p in previews] == [3, 5]
    assert designed == [("A calm deep voice", 3), ("A calm deep voice", 5)]

    voices = voices_store(rootdir=tmp_path)
    profile = voxy.design_from_description(
        "calm",
        "A calm deep voice",
        preview=previews[1],
        model=model,
        voices=voices,
        rootdir=tmp_path,
    )
    clip = Path(profile.segment["ref_audio"])
    assert clip == tmp_path / "samples" / "calm" / "design.wav"
    assert clip.read_bytes() == previews[1].audio
    loaded = load_voice("calm", model_type="qwen3tts", voices=voices)
    speech = model.synthesize("Hello there.", loaded)  # rendering goes through cloning
    assert speech.backend == "qwen3tts" and calls[0]["ref_audio"] == str(clip)


def test_design_voice_without_preview_takes_the_first_generated(tmp_path):
    model = voxy.create_speech_model(
        "qwen3tts",
        generate=fake_generate([]),
        design=lambda text, **kw: (np.zeros(80, np.float32), SR),
    )
    profile = model.design_voice("x", name="n", rootdir=tmp_path, seeds=[9])
    assert profile.metadata["seed"] == 9 and profile.metadata["designed_from"] == "x"
    with pytest.raises(TypeError, match="design_voice_previews"):
        model.design_voice(
            "x", name="n", preview=model.design_voice_previews("x")[0], seeds=[1]
        )


def test_worker_generate_makes_one_worker_across_threads(monkeypatch):
    """Four threads calling a fresh model at once start one worker, not four (#65)."""
    import threading
    import time

    import voxy.local_clone as lc

    made = []

    class FakeWorker:
        def call(self, op, **request):
            return {}

        def close(self):
            pass

    def make_worker(self):
        made.append(1)
        time.sleep(0.05)  # widen the window two unguarded threads would both enter
        return FakeWorker()

    monkeypatch.setattr(lc.WorkerGenerate, "_make_worker", make_worker)
    monkeypatch.setattr(lc, "read_wav", lambda path: (np.zeros(1), SR))
    generate = lc.WorkerGenerate(LOCAL_CLONE_BACKENDS["f5tts"])
    threads = [threading.Thread(target=generate, args=("Hi.",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1
    generate.close()
    assert generate._worker is None


def test_one_registry_holds_every_local_backend_kokoro_included():
    """#64: setup, diagnosis and name lookup read one registry."""
    from voxy import diagnosis
    from voxy.local_clone import (
        LocalBackend,
        LocalCloneBackend,
        LocalSpeechModel,
        local_backends,
    )

    backends = local_backends()
    assert {"f5tts", "chatterbox", "xtts", "voxcpm", "qwen3tts", "kokoro"} <= set(
        backends
    )
    assert backends["kokoro"].clones is False and backends["f5tts"].clones is True
    assert "kokoro" not in LOCAL_CLONE_BACKENDS and "f5tts" in LOCAL_CLONE_BACKENDS
    assert diagnosis._defaults("kokoro")["speed"] == 1.0  # its params reach diagnosis
    assert (
        LocalSpeechModel("kokoro", generate=lambda *a, **k: None).backend.name
        == "kokoro"
    )
    assert LocalCloneBackend is LocalBackend  # the old name still works


def test_setup_builds_every_local_backends_environment(monkeypatch, capsys):
    from voxy import __main__ as cli

    built = []
    monkeypatch.setattr(
        "voxy.envs.ensure_env", lambda spec, **kw: built.append(spec.name) or "py"
    )
    cli.main(["setup"])
    assert {"kokoro", "f5tts", "asr", "speaker", "phonemes"} <= set(built)


def test_the_diagnosis_grid_does_not_cross_a_stock_voice_backend_with_references(
    tmp_path,
):
    variants = diagnosis.expand_grid(
        [
            {"backend": "kokoro", "params": {"voice": ["af_heart", "bf_emma"]}},
            {"backend": "f5tts"},
        ],
        references=["r1", "r2"],
    )
    assert [(v.backend, v.reference) for v in variants] == [
        ("kokoro", ""),
        ("kokoro", ""),
        ("f5tts", "r1"),
        ("f5tts", "r2"),
    ]
    voices = []

    class Stock:
        def synthesize(self, text, voice, **params):
            voices.append((voice, params))
            from voxy.base import Speech

            return Speech(write_wav(np.zeros(80, dtype=np.float32), SR), format="wav")

    diagnosis.render_grid(
        "Hi.",
        {},
        variants[:2],
        tmp_path,
        audio_format="wav",
        model_factory=lambda backend, **init: Stock(),
        log=lambda *a: None,
    )
    assert voices == [
        ("af_heart", {}),
        ("bf_emma", {}),
    ]  # the swept voice, as the voice


def test_a_builtin_registration_never_replaces_one_already_there():
    from voxy.local_clone import LocalBackend, local_backends, register_local_backend

    mine = LocalBackend("demo-tts", EnvSpec("demo-tts", ()), clones=False)
    builtin = LocalBackend("demo-tts", EnvSpec("demo-tts", ("pkg",)), clones=False)
    try:
        register_local_backend(mine)
        register_local_backend(builtin, replace=False)
        assert local_backends()["demo-tts"] is mine
    finally:
        local_backends().pop("demo-tts", None)


def test_latest_runs_knows_a_legacy_absolute_source_by_its_real_path(tmp_path):
    import os

    a = os.path.realpath(tmp_path / "a.wav")
    manifest = [
        {"key": "clips/a-0001.wav", "source": a},  # older voxy: absolute, no run
        {
            "key": "clips/a-2-0001.wav",
            "source": a,
            "source_path": a,
            "run": "20261102T000000.000000Z-abcdef",
        },
    ]
    assert [e["key"] for e in refprep.latest_runs(manifest)] == ["clips/a-2-0001.wav"]


def test_a_manifest_that_claims_no_clip_does_not_make_every_clip_an_orphan(tmp_path):
    samples = samples_store("ada", rootdir=tmp_path)
    samples["clips/vocals-0001.wav"] = b"RIFF"
    samples["manifest.jsonl"] = b'{"key": "tape.wav"}\n'  # entries, but none naming a clip
    with pytest.raises(ValueError, match="claims no clip"):
        refprep.forget_orphans(samples, dry_run=False)
    assert "clips/vocals-0001.wav" in samples
