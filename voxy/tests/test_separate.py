"""Tests for voxy.separate, with the Demucs worker faked."""

from pathlib import Path

import pytest

from voxy import separate as sep


def fake_run(calls):
    def run(audio, out_dir, **options):
        calls.append((Path(audio).read_bytes(), options))
        stem = options.get("stem", "vocals")
        paths = {}
        for name in (stem, "rest"):
            paths[name] = Path(out_dir) / f"{name}.wav"
            paths[name].write_bytes(name.encode())
        return paths

    return run


def test_wav_bytes_in_two_stems_out():
    calls = []
    stems = sep.separate(b"RIFF-input", run=fake_run(calls), shifts=2)
    assert stems == {"vocals": b"vocals", "rest": b"rest"}
    assert calls == [(b"RIFF-input", {"stem": "vocals", "shifts": 2})]


def test_a_path_and_another_stem(tmp_path):
    song = tmp_path / "song.wav"
    song.write_bytes(b"RIFF-song")
    calls = []
    stems = sep.separate(song, stem="drums", run=fake_run(calls))
    assert set(stems) == {"drums", "rest"} and calls[0][0] == b"RIFF-song"


def test_the_default_run_uses_the_pooled_worker_with_absolute_paths(
    tmp_path, monkeypatch
):
    requests = []

    class FakeWorker:
        def call(self, op, **request):
            requests.append((op, request))
            out = Path(request["out_dir"])
            for name in ("vocals", "rest"):
                (out / f"{name}.wav").write_bytes(name.encode())
            return {"paths": {n: str(out / f"{n}.wav") for n in ("vocals", "rest")}}

    made = []
    monkeypatch.setattr(
        sep._DEMUCS_POOL,
        "_make",
        lambda spec, adapter, **key: made.append(key) or FakeWorker(),
    )
    sep.close_separation()
    try:
        stems = sep.separate(b"RIFF", run=sep.worker_run(model="htdemucs"))
    finally:
        sep.close_separation()
    assert stems["vocals"] == b"vocals"
    ((op, request),) = requests
    assert op == "separate" and Path(request["audio"]).is_absolute()
    assert made == [{"model": "htdemucs", "threads": None, "rootdir": None}]


def test_a_missing_stem_in_the_run_result_is_an_error():
    def run(audio, out_dir, **options):
        return {}

    with pytest.raises(KeyError):
        sep.separate(b"RIFF", run=run)
