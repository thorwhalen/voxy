"""Tests for voxy.separate, with the Demucs worker faked."""

from pathlib import Path

import pytest

from voxy import separate as sep
from voxy.worker import WorkerError
from voxy.workers import demucs_adapter


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
    assert calls == [
        (b"RIFF-input", {"stem": "vocals", "sample_rate": None, "shifts": 2})
    ]


def test_the_sample_rate_reaches_the_run():
    calls = []
    sep.separate(b"RIFF", sample_rate=16_000, run=fake_run(calls))
    assert calls[0][1]["sample_rate"] == 16_000


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

    made = fake_pool(monkeypatch, FakeWorker())
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


def fake_pool(monkeypatch, worker):
    made = []
    monkeypatch.setattr(
        sep._DEMUCS_POOL,
        "_make",
        lambda spec, adapter, **key: made.append(key) or worker,
    )
    return made


def test_a_stem_a_known_model_lacks_fails_before_the_worker_starts(monkeypatch):
    made = fake_pool(monkeypatch, None)
    with pytest.raises(ValueError, match="'guitar' is not a stem of htdemucs_ft"):
        sep.separate(b"RIFF", stem="guitar")
    assert made == []


def test_a_stem_error_from_the_worker_reads_as_its_message(monkeypatch):
    message = "ValueError: 'flute' is not a stem of my_model: ['vocals', 'other']"

    class FailingWorker:
        def call(self, op, **request):
            raise WorkerError(f"{message}\nTraceback (most recent call last): ...")

    fake_pool(monkeypatch, FailingWorker())
    sep.close_separation()
    try:
        with pytest.raises(WorkerError, match=r"'flute' is not a stem of my_model"):
            sep.separate(b"RIFF", stem="flute", run=sep.worker_run(model="my_model"))
    finally:
        sep.close_separation()


class FakeBag:
    def __init__(self, weights):
        self.weights = weights
        self.models = [f"model{i}" for i in range(len(weights))]


@pytest.mark.parametrize(
    "weights, index, expected",
    [
        ([[1, 0], [0, 1]], 1, "model1"),  # htdemucs_ft: one fine-tuned model per stem
        ([[1, 1]], 0, "model0"),  # a bag of one
        ([[1, 1], [1, 1]], 0, "bag"),  # an ensemble: every model counts
    ],
)
def test_only_the_sub_model_that_makes_the_stem_runs(weights, index, expected):
    bag = FakeBag(weights)
    chosen = demucs_adapter.stem_model(bag, index)
    assert (chosen is bag and expected == "bag") or chosen == expected


def test_a_plain_model_runs_as_is():
    model = object()
    assert demucs_adapter.stem_model(model, 0) is model


def test_speech_separator_fits_mixings_separator_seam():
    """#99: (samples (n, ch), sr) -> (speech, music) at the input's shape, mono or stereo."""
    import numpy as np

    from voxy.wavutil import read_wav_channels, write_wav_channels

    seen = []

    def run(audio, out_dir, *, stem, sample_rate, **options):
        samples, sr = read_wav_channels(audio)
        seen.append((samples.shape, sr, sample_rate))
        stereo = np.repeat(samples[:, :1], 2, axis=1)[:-3]  # the model: stereo, a few frames short
        paths = {}
        for name, gain in ((stem, 0.8), ("rest", 0.2)):
            paths[name] = Path(out_dir) / f"{name}.wav"
            paths[name].write_bytes(write_wav_channels(stereo * gain, sr))
        return paths

    separator = sep.speech_separator(run=run)
    mono = np.full((100, 1), 0.5)
    speech, music = separator(mono, 16000)
    assert speech.shape == music.shape == (100, 1) and seen[-1] == ((100, 1), 16000, 16000)
    assert abs(speech[0, 0] - 0.4) < 1e-6 and speech[-1, 0] == 0  # padded back to length
    speech, music = separator(np.full((50, 2), 0.5), 44100)
    assert speech.shape == (50, 2)
    with pytest.raises(ValueError, match="mono or stereo"):
        separator(np.zeros((10, 6)), 44100)
