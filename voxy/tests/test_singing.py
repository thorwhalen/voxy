"""Tests for singing: reading scores through audiate, SoulX metadata, the adapter (fake model
libraries), the facade (fake worker), and an opt-in real run."""

import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from voxy import score as score_module
from voxy import singing, soulx
from voxy.base import VoiceProfile
from voxy.envs import GitSource, ensure_git_source, git_source_dir, is_env_ready
from voxy.score import Note, notes_from_pitches, read_score
from voxy.wavutil import read_wav, write_wav
from voxy.workers import WORKERS_DIR, adapter_path

MELODY = [(64, 1), (64, 1), (67, 1), (69, 1), (67, 1), (64, 1), (62, 1), (60, 2)]
LYRICS = "Light a can-dle in the win-dow"


# --------------------------------------------------------------------------- scores


def write_midi(path, notes, *, bpm=100, lyrics=(), tempo_change=None, gate=1.0):
    """A one-track MIDI of ``(pitch, beats)``; ``tempo_change=(beat, bpm)``."""
    mido = pytest.importorskip("mido")
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    lyrics = list(lyrics)
    pending = 0  # ticks since the last event
    beat = 0.0
    for i, (pitch, beats) in enumerate(notes):
        if tempo_change and beat == tempo_change[0]:
            track.append(
                mido.MetaMessage(
                    "set_tempo", tempo=mido.bpm2tempo(tempo_change[1]), time=pending
                )
            )
            pending = 0
        if i < len(lyrics):
            track.append(mido.MetaMessage("lyrics", text=lyrics[i], time=pending))
            pending = 0
        track.append(mido.Message("note_on", note=pitch, velocity=90, time=pending))
        on = int(beats * tpb * gate)
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=on))
        pending = int(beats * tpb) - on
        beat += beats
    mid.save(str(path))
    return path


def raw_midi(path, tracks, *, tpb=480):
    """A MIDI file from ``[[(kind, delta_ticks, fields), ...], ...]`` per track."""
    mido = pytest.importorskip("mido")
    mid = mido.MidiFile(ticks_per_beat=tpb)
    for events in tracks:
        track = mido.MidiTrack()
        mid.tracks.append(track)
        for kind, delta, fields in events:
            if kind in ("lyrics", "text"):
                track.append(mido.MetaMessage(kind, time=delta, **fields))
            else:
                track.append(mido.Message(kind, time=delta, **fields))
    mid.save(str(path))
    return path


def on(note, delta=0, channel=0):
    return ("note_on", delta, {"note": note, "velocity": 90, "channel": channel})


def off(note, delta, channel=0):
    return ("note_off", delta, {"note": note, "velocity": 0, "channel": channel})


def lyric(text, delta=0):
    return ("lyrics", delta, {"text": text})


MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <direction placement="above"><sound tempo="100"/></direction>
      {notes}
    </measure>
  </part>
</score-partwise>
"""


def xml_note(step, octave, beats, lyric=None, syllabic="single", tie=None):
    tie_xml = f'<tie type="{tie}"/>' if tie else ""
    lyric_xml = (
        f"<lyric><syllabic>{syllabic}</syllabic><text>{lyric}</text></lyric>"
        if lyric
        else ""
    )
    return (
        f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>"
        f"<duration>{beats}</duration>{tie_xml}<type>quarter</type>{lyric_xml}</note>"
    )


def test_midi_and_musicxml_of_one_melody_give_the_same_metadata(tmp_path):
    pytest.importorskip("audiate")
    pytest.importorskip("music21")
    midi = write_midi(tmp_path / "t.mid", MELODY)
    steps = {64: ("E", 4), 67: ("G", 4), 69: ("A", 4), 62: ("D", 4), 60: ("C", 4)}
    words = [
        ("Light", "single"),
        ("a", "single"),
        ("can", "begin"),
        ("dle", "end"),
        ("in", "single"),
        ("the", "single"),
        ("win", "begin"),
        ("dow", "end"),
    ]
    xml = "".join(xml_note(*steps[p], b, w, s) for (p, b), (w, s) in zip(MELODY, words))
    (tmp_path / "t.musicxml").write_text(MUSICXML.format(notes=xml))
    from_midi = soulx.soulx_metadata(read_score(midi, LYRICS))
    from_xml = soulx.soulx_metadata(read_score(tmp_path / "t.musicxml"))
    assert from_midi == from_xml
    assert (
        from_midi[0]["text"] == "<SP> Light a candle candle in the window window <SP>"
    )
    assert from_midi[0]["note_type"] == "1 2 2 2 3 2 2 2 3 1"


def test_a_score_s_own_lyrics_and_the_filler(tmp_path):
    pytest.importorskip("audiate")
    # "hey" on the first note; the second, after a rest, has no word: the filler
    path = raw_midi(
        tmp_path / "f.mid", [[lyric("hey"), on(60), off(60, 480), on(62, 480), off(62, 480)]]
    )
    assert [(n.word, n.starts_word) for n in read_score(path)] == [
        ("hey", True), ("la", True)
    ]
    assert [n.word for n in read_score(path, filler="oh")] == ["hey", "oh"]


def test_a_midi_without_lyrics_needs_them(tmp_path):
    pytest.importorskip("audiate")
    path = write_midi(tmp_path / "m.mid", [(60, 1)])
    with pytest.raises(ValueError, match="no lyrics"):
        read_score(path)
    assert [n.word for n in read_score(path, "hey")] == ["hey"]


def test_given_lyrics_replace_the_score_s_own_which_are_not_read(tmp_path):
    pytest.importorskip("audiate")
    melody = [on(60), off(60, 480)]
    late = raw_midi(tmp_path / "late.mid", [[*melody, lyric("lost", 960)]])
    with pytest.raises(ValueError, match="lost"):
        read_score(late)  # a lyric on no note: audiate refuses to guess
    assert [n.word for n in read_score(late, "found")] == ["found"]


def test_an_ambiguous_score_raises_naming_the_choices(tmp_path):
    pytest.importorskip("audiate")
    melody = [on(72), off(72, 480), on(74), off(74, 480)]
    bass = [on(36, channel=1), off(36, 960, channel=1)]
    several = raw_midi(tmp_path / "s.mid", [melody, bass])
    with pytest.raises(ValueError, match="track="):
        read_score(several, "la la")
    assert [n.pitch for n in read_score(several, "la", track=1)] == [36]


def test_notes_are_sung_as_given():
    audiate = pytest.importorskip("audiate")
    line = [audiate.SungNote(0.5, 0.5, 62, "re"), audiate.SungNote(0.0, 0.5, 60, "do")]
    assert read_score(line) == [Note(0.0, 0.5, 60, "do"), Note(0.5, 0.5, 62, "re")]
    with pytest.raises(ValueError, match="track"):
        read_score(line, track=0)
    with pytest.raises(ValueError, match="3 lyric syllables for 2 notes"):
        read_score(notes_from_pitches([(60, 1), (62, 1)]), "one two three")


ELISION = (
    "<note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration>"
    "<type>quarter</type><lyric><syllabic>single</syllabic><text>me</text><elision/>"
    "<syllabic>single</syllabic><text>and</text></lyric></note>"
)


def test_an_elision_sings_its_words_on_one_note(tmp_path):
    pytest.importorskip("audiate")
    pytest.importorskip("music21")
    path = tmp_path / "elision.musicxml"
    path.write_text(MUSICXML.format(notes=ELISION + xml_note("D", 5, 1, "you")))
    notes = read_score(path)  # audiate's own output: two words on the first note
    assert [n.word for n in notes] == ["me and", "you"]
    (segment,) = soulx.soulx_metadata(notes)
    assert segment["text"] == "<SP> me+and you <SP>"
    assert segment["note_type"] == "1 2 2 1"
    with pytest.raises(ValueError, match="joins the words"):
        soulx.soulx_metadata([Note(0, 1, 60, "a+b")])


def test_a_missing_score_file_is_named():
    with pytest.raises(FileNotFoundError, match="No score at"):
        read_score("nowhere.mid", "la")


# --------------------------------------------------------------------------- metadata


def sung(pitches, lyrics, **kw):
    """Notes with one word each (no score reading: SoulX metadata only)."""
    notes = notes_from_pitches(pitches, **kw)
    return [Note(n.start, n.dur, n.pitch, w) for n, w in zip(notes, lyrics.split(), strict=True)]


def test_metadata_rests_legato_and_overlaps():
    notes = [
        Note(1.0, 0.5, 60, "a"),
        Note(1.6, 0.5, 62, "b"),  # 0.1 s gap: joins the previous note
        Note(2.5, 0.5, 64, "c"),  # 0.4 s gap: a rest
        Note(2.9, 0.5, 65, "d"),  # overlaps: the previous note is cut
    ]
    (segment,) = soulx.soulx_metadata(notes)
    assert segment["duration"] == "0.30 0.60 0.50 0.40 0.40 0.50 0.40"
    assert segment["note_pitch"] == "0 60 62 0 64 65 0"
    assert segment["time"] == [0, 3100]


def test_long_rests_split_segments_placed_at_their_time():
    notes = [Note(0.0, 1.0, 60, "one"), Note(4.0, 1.0, 62, "two")]
    first, second = soulx.soulx_metadata(notes)
    assert first["text"] == "<SP> one <SP>" and second["text"] == "two <SP>"
    assert first["time"] == [0, 4300] and second["time"] == [4300, 5700]


def test_long_lines_split_at_the_maximum_length():
    notes = sung([(60, 1)] * 40, " ".join(["la"] * 40), bpm=60)
    segments = soulx.soulx_metadata(notes, max_segment_s=10)
    assert len(segments) == 5
    assert all(s["time"][1] - s["time"][0] <= 10_000 for s in segments)
    total = sum(float(d) for s in segments for d in s["duration"].split())
    assert round(total, 2) == 40.7  # nothing lost or added at the cuts


def test_durations_never_drift():
    notes = sung([(60, 1 / 3)] * 30, " ".join(["la"] * 30), bpm=60)
    (segment,) = soulx.soulx_metadata(notes)
    durations = [float(d) for d in segment["duration"].split()]
    assert abs(sum(durations) - (0.3 + 10 + 0.4)) < 1e-6


def test_prompt_metadata_keeps_the_clips_times():
    notes = [Note(0.3, 0.6, 67, "la"), Note(0.9, 0.6, 64, "la")]
    (segment,) = soulx.soulx_metadata(notes, lead_s=0.3)
    (prompt,) = soulx.soulx_prompt_metadata(notes)
    assert prompt["duration"] == "0.30 0.60 0.60"
    assert prompt["time"] == [0, 1500] and segment["time"] == [0, 1900]


def test_long_lines_split_at_a_rest_when_there_is_one():
    phrase = [(60, 1)] * 4 + [(0, 1)] + [(62, 1)] * 4
    notes = sung(phrase * 2, " ".join(["la"] * 16), bpm=60)
    segments = soulx.soulx_metadata(notes, max_segment_s=12)
    assert [s["note_pitch"].split()[-1] for s in segments[:-1]] == ["0"] * (len(segments) - 1)
    assert all(s["note_pitch"].split()[0] != "0" for s in segments[1:])


@pytest.mark.parametrize("word", ["la 12", "", "123"])
def test_each_sung_note_needs_one_word(word):
    with pytest.raises(ValueError, match="one word"):
        soulx.soulx_metadata([Note(0, 1, 60, word)])


def test_only_english_is_phonemized():
    with pytest.raises(NotImplementedError, match="Mandarin"):
        soulx.soulx_metadata([Note(0, 1, 60, "ni")], language="Mandarin")


# --------------------------------------------------------------------------- adapter


class FakeConfig:
    class audio:
        hop_size, sample_rate = 480, 8000

    class infer:
        n_steps, cfg = 32, 3.0


class FakeModel:
    """A SoulX model: records its calls, returns a loud ramp."""

    calls = []

    def __init__(self, config):
        self.config = config
        self.vocoder = FakeVocoder()
        self.device = None

    def load_state_dict(self, state_dict, strict):
        assert state_dict == {"w": 1} and strict

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def infer(self, meta=None, **kwargs):
        import torch

        FakeModel.calls.append((meta, kwargs))
        if meta is None:  # SVC
            return torch.linspace(-2, 2, 800)[None], -7
        return torch.linspace(-2, 2, 800)[None]


class FakeVocoder:
    def to(self, device):
        self.device = device
        return self

    def forward(self, x):
        return x * 2


class FakeProcessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def process(self, meta, wav_path):
        return {"meta": meta, "wav": wav_path}


class FakeF0:
    def __init__(self, path, device, verbose):
        self.path = path

    def process(self, audio, f0_path=None, verbose=False):
        return np.array([0.0, 220.0, 220.0, 0.0])


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    """The SoulX adapter, loaded in-process over fake upstream modules and a fake g2p."""
    pytest.importorskip("torch")
    repo = tmp_path / "repo"
    (repo / "preprocess" / "tools").mkdir(parents=True)
    (repo / "preprocess" / "tools" / "f0_extraction.py").write_text(
        "F0Extractor = None  # replaced by the test\n"
    )
    modules = {
        "soulxsinger": types.ModuleType("soulxsinger"),
        "soulxsinger.utils": types.ModuleType("soulxsinger.utils"),
        "soulxsinger.utils.file_utils": types.SimpleNamespace(
            load_config=lambda path: FakeConfig
        ),
        "soulxsinger.utils.data_processor": types.SimpleNamespace(
            DataProcessor=FakeProcessor
        ),
        "soulxsinger.utils.audio_utils": types.SimpleNamespace(
            load_wav=lambda path, sr: __import__("torch").zeros(1, 10)
        ),
        "soulxsinger.models": types.ModuleType("soulxsinger.models"),
        "soulxsinger.models.soulxsinger": types.SimpleNamespace(SoulXSinger=FakeModel),
        "soulxsinger.models.soulxsinger_svc": types.SimpleNamespace(
            SoulXSingerSVC=FakeModel
        ),
        # CI has no soundfile: write 16-bit WAV with voxy's own writer
        "soundfile": types.SimpleNamespace(
            write=lambda path, samples, sr, subtype: Path(path).write_bytes(
                write_wav(samples, sr)
            )
        ),
        "g2p_en": types.SimpleNamespace(
            G2p=lambda: lambda word: list(word.upper()) + [" "]
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(WORKERS_DIR))
    spec = importlib.util.spec_from_file_location(
        "soulx_adapter_test", adapter_path("soulx")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "_load_checkpoint", lambda path: {"state_dict": {"w": 1}}
    )
    monkeypatch.setattr(module, "_ensure_nltk_data", lambda: None)
    downloads = []
    monkeypatch.setattr(
        module,
        "_download",
        lambda repo_id, filename, revision: (
            downloads.append((repo_id, filename, revision)) or f"/hub/{filename}"
        ),
    )
    monkeypatch.setattr(
        module,
        "_f0_extractor",
        lambda state: FakeF0(
            module._weight(state, "preprocess", module.RMVPE_CKPT), "cpu", False
        ),
    )
    FakeModel.calls = []
    module.downloads = downloads
    module.repo = repo
    return module


def test_adapter_sings_filling_phonemes_and_limiting_the_peak(adapter, tmp_path):
    state = adapter.load(
        repo_dir=str(adapter.repo),
        weights={"repo_id": "org/w", "revision": "abc"},
        device="cpu",
    )
    meta = soulx.soulx_metadata([Note(0.0, 0.5, 60, "Hi!")])
    prompt = soulx.soulx_prompt_metadata([Note(0.3, 0.5, 67, "la")])
    out = tmp_path / "o.wav"
    reply = adapter.sing(
        state,
        meta=meta,
        prompt_audio="p.wav",
        prompt_meta=prompt,
        out_path=str(out),
        seed=1,
    )
    assert adapter.downloads == [("org/w", "model.pt", "abc")]
    ((sent, kwargs),) = FakeModel.calls
    assert sent["target"]["meta"]["phoneme"] == "<SP> en_H-I <SP>"
    assert sent["prompt"] == {
        "meta": {**prompt[0], "phoneme": "<SP> en_L-A"},
        "wav": "p.wav",
    }
    assert kwargs == {
        "auto_shift": False,
        "pitch_shift": 0,
        "n_steps": 32,
        "cfg": 3.0,
        "control": "score",
    }
    samples, sr = read_wav(out)
    assert sr == 8000 and len(samples) == int(1.2 * 8000)
    assert reply["peak"] == 2.0 and abs(np.abs(samples).max() - 0.95) < 1e-3


def test_adapter_places_segments_at_their_time_with_faded_joins(adapter, tmp_path):
    state = adapter.load(repo_dir=str(adapter.repo), weights_dir="/w", device="cpu")
    notes = [Note(0.0, 1.0, 60, "one"), Note(4.0, 1.0, 62, "two")]
    meta = soulx.soulx_metadata(notes)
    assert len(meta) == 2
    prompt = soulx.soulx_prompt_metadata([Note(0.3, 0.5, 67, "la")])
    out = tmp_path / "two.wav"
    adapter.sing(
        state, meta=meta, prompt_audio="p.wav", prompt_meta=prompt, out_path=str(out),
        peak=None, pitch_shift=5,
    )
    assert [kw["pitch_shift"] for _, kw in FakeModel.calls] == [5, 5]
    samples, sr = read_wav(out)  # each segment renders a ramp of 800 samples
    first, second = (int(seg["time"][0] / 1000 * sr) for seg in meta)
    assert second == int(4.3 * sr) and len(samples) == int(5.7 * sr)
    fade = int(0.02 * sr)
    assert samples[first] == pytest.approx(-1.0, abs=1e-3)  # first segment: no fade-in
    assert abs(samples[second]) < 1e-3  # second segment fades in
    assert abs(samples[first + 799]) < 0.1  # first fades out
    halfway = fade // 2  # inside the fade-in: the ramp's value times the fade's
    expected = (-2 + 4 * halfway / 799) * halfway / (fade - 1)
    assert samples[second + halfway] == pytest.approx(expected, abs=0.01)


def test_adapter_phonemizes_each_word_of_an_elision_onto_its_note(adapter):
    state = adapter.load(repo_dir=str(adapter.repo), weights_dir="/w", device="cpu")
    (segment,) = soulx.soulx_metadata([Note(0, 1, 60, "me and"), Note(1, 1, 62, "you")])
    (filled,) = adapter.fill_phonemes(state, [segment])
    assert filled["phoneme"] == "<SP> en_M-E-A-N-D en_Y-O-U <SP>"


def test_adapter_rejects_fields_of_different_lengths(adapter):
    state = adapter.load(repo_dir=str(adapter.repo), weights_dir="/w", device="cpu")
    (segment,) = soulx.soulx_metadata([Note(0, 1, 60, "hi")])
    with pytest.raises(ValueError, match="different lengths"):
        adapter.fill_phonemes(state, [{**segment, "text": "<SP> hi there <SP>"}])


def test_loading_one_singer_frees_the_other(adapter):
    state = adapter.load(repo_dir=str(adapter.repo), weights_dir="/w", device="cpu")
    adapter._model(state, "svs")
    adapter._model(state, "svc")
    assert sorted(state["models"]) == ["svc"]


def test_adapter_converts_with_rmvpe_f0_and_reports_the_shift(adapter, tmp_path):
    state = adapter.load(
        repo_dir=str(adapter.repo),
        weights_dir="/w",
        preprocess_dir="/pre",
        device="cpu",
    )
    reply = adapter.convert(
        state, audio="a.wav", prompt_audio="p.wav", out_path=str(tmp_path / "c.wav")
    )
    assert reply["pitch_shift"] == -7 and reply["duration"] == 0.1
    ((meta, kwargs),) = FakeModel.calls
    assert kwargs["auto_shift"] is True and kwargs["gt_f0"].shape == (1, 4)
    assert state["models"]["rmvpe"].path == os.path.join("/pre", "rmvpe/rmvpe.pt")
    assert adapter.unload(state, component="svc") == {"loaded": ["rmvpe"]}


def test_adapter_runs_the_vocoder_on_the_cpu_on_mps(adapter):
    import torch

    state = adapter.load(repo_dir=str(adapter.repo), weights_dir="/w", device="mps")
    model = adapter._model(state, "svs")
    assert model.device == "mps" and model.vocoder.device == "cpu"
    if not torch.backends.mps.is_available():
        with pytest.raises((RuntimeError, TypeError, ValueError)):
            model.vocoder.forward(torch.ones(2))  # moving back to mps fails here
    with pytest.raises(ValueError, match="Unknown component"):
        adapter._model(state, "nope")


def test_adapter_refuses_to_guess_other_languages(adapter):
    state = adapter.load(repo_dir=str(adapter.repo), weights_dir="/w", device="cpu")
    segment = {
        "language": "Mandarin", "text": "ni", "phoneme": "",
        "duration": "0.5", "note_pitch": "60", "note_type": "2",
    }
    with pytest.raises(NotImplementedError, match="Mandarin"):
        adapter.fill_phonemes(state, [segment])
    given = {**segment, "phoneme": "zh_n-i"}
    assert adapter.fill_phonemes(state, [given]) == [given]


# --------------------------------------------------------------------------- facade


class FakeRun:
    """The model call: records requests, returns a short ramp."""

    def __init__(self):
        self.requests = []

    def __call__(self, op, **request):
        self.requests.append((op, request))
        return np.linspace(-0.5, 0.5, 2400, dtype=np.float32), 24000


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXY_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_sing_through_the_facade_with_the_synthetic_voice(cache, tmp_path):
    run = FakeRun()
    model = soulx.SoulXSingingModel(run=run)
    melody = notes_from_pitches(MELODY)
    speech = singing.sing(
        melody,
        LYRICS,
        model=model,
        voices={},
        seed=3,
        output_path=tmp_path / "out.wav",
    )
    assert speech.backend == "soulx" and speech.sample_rate == 24000
    assert speech.text == "Light a candle in the window"
    assert (tmp_path / "out.wav").read_bytes() == speech.audio
    ((op, request),) = run.requests
    assert op == "sing" and request["seed"] == 3
    assert request["auto_shift"] is False and request["pitch_shift"] == 0
    assert request["meta"][0]["note_pitch"] == "0 64 64 67 69 67 64 62 60 0"
    prompt = Path(request["prompt_audio"])
    assert prompt.is_absolute() and prompt.parent == (cache / "singing").resolve()
    assert request["prompt_meta"][0]["text"].split() == ["<SP>"] + ["la"] * 8
    samples, sr = read_wav(prompt)
    assert len(samples) / sr == pytest.approx(
        float(request["prompt_meta"][0]["time"][1]) / 1000 + 0.4, abs=0.01
    )


def test_transpose_and_its_errors(cache):
    run = FakeRun()
    model = soulx.SoulXSingingModel(run=run)
    notes = sung([(60, 1)], "la")
    singing.sing(notes, model=model, voices={}, transpose="fit_voice")
    singing.sing(notes, model=model, voices={}, transpose=-3)
    # the synthetic phrase's median note is D4 (62): C4 moves up 2, once for the song
    assert [r["pitch_shift"] for _, r in run.requests] == [2, -3]
    assert [r["auto_shift"] for _, r in run.requests] == [False, False]
    with pytest.raises(ValueError, match="semitones"):
        singing.sing(notes, model=model, voices={}, transpose="up")
    with pytest.raises(TypeError, match="set by voxy"):
        singing.sing(notes, model=model, voices={}, auto_shift=True)


def test_synthesis_needs_the_voices_notes(tmp_path, cache):
    clip = tmp_path / "me.wav"
    clip.write_bytes(write_wav(np.zeros(2400), 24000))
    model = soulx.SoulXSingingModel(run=FakeRun())
    with pytest.raises(ValueError, match="notes and words of the voice's clip"):
        singing.sing(sung([(60, 1)], "la"), voice=str(clip), model=model, voices={})


def test_convert_singing_bytes_into_a_clip_voice(tmp_path, cache):
    run = FakeRun()
    model = soulx.SoulXSingingModel(run=run)
    target = tmp_path / "target.wav"
    target.write_bytes(write_wav(np.zeros(2400), 24000))
    take = write_wav(np.full(2400, 0.1), 24000)
    speech = singing.convert_singing(take, voice=str(target), model=model, voices={})
    ((op, request),) = run.requests
    assert op == "convert" and request["auto_shift"] is True
    assert request["prompt_audio"] == str(target.resolve())
    assert speech.voice == str(target.resolve()) and speech.format == "wav"


def test_a_library_voice_sings(tmp_path, cache):
    clip = tmp_path / "v.wav"
    clip.write_bytes(write_wav(np.zeros(2400), 24000))
    voice = singing.singing_voice(clip, sung([(60, 1)], "la"), consent="mine")
    from voxy.library import save_voice
    from voxy.stores import voices_store

    voices = voices_store(rootdir=tmp_path)
    save_voice("me", voice, voices=voices, consent="my own voice")
    run = FakeRun()
    singing.sing(
        sung([(62, 1)], "hey"),
        voice="Me",
        model=soulx.SoulXSingingModel(run=run),
        voices=voices,
    )
    ((_, request),) = run.requests
    assert request["prompt_audio"] == str(clip.resolve())
    assert request["prompt_meta"][0]["note_pitch"] == "60"
    with pytest.raises(ValueError, match="neither a library voice nor a clip"):
        singing.resolve_singing_voice("nobody", voices=voices)


def test_the_registry_and_its_model_cache(monkeypatch):
    backend = singing.singing_backends()["soulx"]
    assert backend.env is soulx.SOULX_ENV and "impersonate" in backend.consent
    assert "Apache-2.0" in backend.licence and backend.source.commit.startswith(
        "81aeb3a"
    )
    made = []

    def factory():
        made.append(1)
        return types.SimpleNamespace(name="x", close=lambda: made.append("closed"))

    singing.register_singing_backend(
        singing.SingingBackend("x", soulx.SOULX_ENV, factory)
    )
    try:
        models = {}
        assert singing.get_singing_model(
            "x", models=models
        ) is singing.get_singing_model("x", models=models)
        singing.close_singing_models(models)
        assert made == [1, "closed"] and models == {}
        with pytest.raises(KeyError, match="No singing backend 'y'"):
            singing.get_singing_model("y", models={})
    finally:
        singing._SINGING_BACKENDS.pop("x")


def test_the_worker_gets_the_fetched_source(monkeypatch, tmp_path):
    fetched, made = [], []
    monkeypatch.setattr(
        singing,
        "ensure_git_source",
        lambda source, name, rootdir=None: fetched.append(name) or tmp_path / "src",
    )

    def env_worker(spec, adapter, *, init, **kw):
        made.append((spec.name, adapter, init))
        return "worker"

    monkeypatch.setattr("voxy.local_clone.env_worker", env_worker)
    worker = singing.SingingWorker(soulx.SOULX_BACKEND, init={"device": "cpu"})
    assert worker._make_worker() == "worker"
    ((name, adapter, init),) = made
    assert (name, adapter, fetched) == ("soulx", "soulx", ["soulx"])
    assert init["repo_dir"] == str(tmp_path / "src") and init["device"] == "cpu"
    assert init["weights"] == soulx.SOULX_WEIGHTS


def test_sing_passes_the_line_choice_to_the_score(tmp_path, cache, monkeypatch):
    pytest.importorskip("audiate")
    seen = []
    real = singing.read_score
    monkeypatch.setattr(
        singing,
        "read_score",
        lambda score, lyrics, **kw: seen.append(kw) or real(score, lyrics, **kw),
    )
    melody = [on(72, channel=2), off(72, 480, channel=2)]
    bass = [on(36, channel=1), off(36, 480, channel=1)]
    path = raw_midi(tmp_path / "m.mid", [[*melody, *bass]])
    run = FakeRun()
    singing.sing(path, "hi", track=0, channel=3, model=soulx.SoulXSingingModel(run=run), voices={})
    assert seen[0] == {
        "track": 0, "channel": 3, "part": None, "verse": None,
        "score_voice": None, "chords": None,
    }
    assert run.requests[0][1]["meta"][0]["note_pitch"] == "0 72 0"
    # a singing voice's clip: its score's line is chosen the same way
    voice = singing.singing_voice(tmp_path / "clip.wav", path, "hi", track=0, channel=2)
    assert [n["pitch"] for n in voice.segment["prompt_notes"]] == [36]


def test_a_chord_names_voxy_s_keywords_and_chords_picks_one(tmp_path, cache):
    pytest.importorskip("audiate")
    pytest.importorskip("music21")
    chord = (
        xml_note("C", 5, 1, "oh")
        + "<note><chord/><pitch><step>E</step><octave>5</octave></pitch>"
        "<duration>1</duration><type>quarter</type></note>"
    )
    path = tmp_path / "chord.musicxml"
    path.write_text(MUSICXML.format(notes=chord + xml_note("D", 5, 1, "my")))
    model = soulx.SoulXSingingModel(run=FakeRun())
    with pytest.raises(ValueError, match="chords=") as raised:
        singing.sing(path, model=model, voices={})
    assert "score_voice=" in str(raised.value)
    assert not re.search(r"(?<![\w])voice=", str(raised.value))  # voice= is the singer
    assert [n.pitch for n in read_score(path, chords="top")] == [76, 74]
    assert [n.pitch for n in read_score(path, chords="bottom", score_voice=1)] == [72, 74]


def test_an_old_audiate_is_named_with_the_version_it_needs(monkeypatch):
    def old_audiate(module):
        raise ModuleNotFoundError(f"No module named {module!r}", name=module)

    monkeypatch.setattr(score_module, "optional_import", old_audiate)
    monkeypatch.setattr(score_module.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(ModuleNotFoundError, match=r"audiate>=0\.0\.4"):
        read_score(notes_from_pitches([(60, 1)]), "la")


def test_ensure_git_source_is_ps_s_in_voxy_s_models_folder(tmp_path, monkeypatch):
    seen = {}

    def fake(source, path, **kw):
        seen.update(source=source, path=path, **kw)
        return path

    monkeypatch.setattr("ps.isolated.ensure_git_source", fake)
    source = GitSource("https://example.org/r.git", "a" * 40)
    run = object()
    path = ensure_git_source(source, "b", rootdir=tmp_path, run=run, lock_timeout_s=5)
    assert path == git_source_dir(source, "b", rootdir=tmp_path)
    assert path == tmp_path / "models" / "b" / "source" / ("a" * 12)  # folders already fetched stay valid
    assert seen["run"] is run and seen["lock_timeout_s"] == 5 and seen["source"] is source


def test_a_source_needs_a_full_sha():
    with pytest.raises(ValueError, match="full commit SHA"):
        GitSource("https://example.org/r.git", "81aeb3a")


def test_voice_bytes_are_kept_in_the_samples_store(tmp_path):
    clip = write_wav(np.zeros(2400), 24000)
    voice = singing.singing_voice(clip, rootdir=tmp_path)
    path = Path(voice.segment["prompt_audio"])
    assert path.read_bytes() == clip and path.parent == (tmp_path / "samples" / "singing-voices").resolve()


def test_the_setup_command_knows_the_singing_env(monkeypatch, capsys):
    from voxy import __main__ as cli

    built = []
    monkeypatch.setattr(
        "voxy.envs.ensure_env", lambda spec: built.append(spec.name) or "py"
    )
    assert cli.main(["setup", "soulx"]) == 0
    assert built == ["soulx"]


def test_the_synthetic_voice_is_rendered_once(cache):
    first = singing.synthetic_singing_voice("low")
    path = Path(first.segment["prompt_audio"])
    stamp = path.stat().st_mtime_ns
    second = singing.synthetic_singing_voice("low")
    assert second.segment == first.segment and path.stat().st_mtime_ns == stamp
    assert [n["pitch"] for n in first.segment["prompt_notes"]][:2] == [55, 52]
    assert json.dumps(first.segment)  # plain data: the library can keep it


# --------------------------------------------------------------------------- real run

_REAL = os.environ.get("VOXY_REAL_SOULX") == "1" and is_env_ready(soulx.SOULX_ENV)


def _pitch(samples, sr, start, end):
    """Median pitch (MIDI) of ``samples[start:end]`` by autocorrelation, 25 ms frames."""
    frame, values = int(0.04 * sr), []
    for a in range(int(start * sr), int(end * sr) - frame, frame // 2):
        x = samples[a : a + frame] - samples[a : a + frame].mean()
        ac = np.correlate(x, x, "full")[frame - 1 :]
        lo, hi = int(sr / 1000), int(sr / 70)
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.5 * ac[0]:
            values.append(69 + 12 * np.log2(sr / lag / 440))
    return float(np.median(values)) if values else float("nan")


@pytest.mark.skipif(
    not _REAL,
    reason="real SoulX run: build it (python -m voxy setup soulx) and set VOXY_REAL_SOULX=1",
)
def test_real_soulx_sings_in_tune_and_converts(tmp_path):
    notes = read_score(notes_from_pitches(MELODY), LYRICS)
    sung_take = singing.sing(notes, output_path=tmp_path / "sung.wav", seed=0)
    samples, sr = read_wav(sung_take.audio)
    assert np.isfinite(samples).all() and np.abs(samples).max() <= 0.96
    assert len(samples) / sr == pytest.approx(0.3 + 9 * 0.6 + 0.4, abs=0.05)
    errors = [
        abs(_pitch(samples, sr, 0.3 + n.start + 0.1, 0.3 + n.end - 0.1) - n.pitch)
        for n in notes
    ]
    assert np.nanmedian(errors) < 0.5, errors
    converted = singing.convert_singing(
        sung_take.audio, voice=singing.synthetic_singing_voice("low"), seed=0
    )
    out, out_sr = read_wav(converted.audio)
    assert np.isfinite(out).all() and len(out) / out_sr == pytest.approx(
        len(samples) / sr, abs=0.1
    )
    singing.close_singing_models()
