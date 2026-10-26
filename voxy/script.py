"""
Render a cast script: text segments, each read by its own voice, one audio file per section.

For anything narrated by more than one voice (an audiobook with a narrator and a quoted
guide, a dialogue, a radio play). Three plain-data inputs, each a JSON-able mapping so a
person can edit them and the rest follows:

- the **script**: a list of sections ``{"id", "title"?, "segments": [{"role", "text"}]}``;
- the **casting**: ``{"roles": {role: voice name or None}, "takes": {section id: take},
  "default_role"?}``; a role mapped to ``None`` has no voice yet (its sections are listed,
  not rendered);
- the **voices**: voice name -> ``{"backend"?, "init"?, "params"?, "tempo"?, "effect"?,
  "external"?}`` plus how to load it (``"library": name``, ``"reference": {"audio", "text"}``
  or ``"voice": id-or-name`` passed to the backend as is). ``tempo`` (pitch kept, rubberband)
  and ``effect`` (an ffmpeg audio filter) are finishing steps; ``external`` marks a voice
  synthesised elsewhere (``missing_takes`` lists what it owes, ``add_take`` files the audio).

Files under ``root``:

    segments/<key>.wav|json   one take of one segment; the key hashes (voice, settings,
                              spoken text, take number), so a take is never overwritten
                              and an edit makes a new one
    sections/<id>.mp3         a section's takes joined: pauses, tempo, effect, loudness
                              (EBU R128) evened across voices, a gap where the voice changes
    index.json                per section: status, segments (key, role, voice), file, seconds

Render again after editing: only segments whose key changed are synthesised, and only
sections whose takes or finishing changed are rebuilt. A retake of a section is
``"takes": {"s1": 2}`` in the casting (a ``seed`` in the voice's params moves with the take).

>>> script = [{"id": "s1", "segments": [{"role": "guide", "text": "Come."}, {"role": "author", "text": "I come."}]}]
>>> casting = {"roles": {"guide": "g", "author": "a"}}
>>> [(s["role"], s["voice"]) for s in plan_script(script, casting)[0]["segments"]]
[('guide', 'g'), ('author', 'a')]
"""

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from voxy.wavutil import read_wav, write_wav

CASTING_FILE = "casting.json"
VOICES_FILE = "voices.json"
INDEX_FILE = "index.json"
SEGMENTS_DIR = "segments"
SECTIONS_DIR = "sections"
AWAITING = "awaiting voice"  # a role whose voice is not chosen yet
AWAITING_TAKE = (
    "awaiting take"  # an external voice has not returned this section's take
)
FINISHING_KEYS = (
    "tempo",
    "effect",
)  # voice settings applied when finishing, not synthesising
DFLT_BACKEND = "qwen3tts"
FRAME_S = 0.01
DFLT_FINISH = {
    "segment_gap_s": 1.0,  # silence where the voice changes inside a section
    "loudness_lufs": -18.0,  # integrated loudness of every segment (audiobooks: -18 to -23)
    "true_peak_db": -1.5,
    "sample_rate": 24000,
    "bitrate": "96k",
    # Breathing room: a pause (silence inside a take) at least as long as a tier's first
    # number gets that tier's second number of seconds added (the longest tier that applies).
    "pauses": [],
    "pause_floor_db": -35.0,  # a frame is silent this far below the take's loud frames
}


def plan_script(
    script: Sequence[Mapping],
    casting: Mapping,
    *,
    text_prep: Callable[[str], str] | None = None,
) -> list[dict]:
    """Each section with its segments resolved: ``{role, voice, text, tts_text, take}``.

    ``tts_text`` is ``text_prep(text)`` (what is spoken; default: the text itself), so the
    take's key follows what the voice says, not the displayed text.
    """
    roles, default = casting["roles"], casting.get("default_role")
    plan = []
    for section in script:
        take = casting.get("takes", {}).get(section["id"], 1)
        segments = []
        for seg in section["segments"]:
            role = seg.get("role") or default
            if role not in roles:
                raise ValueError(
                    f"{section['id']}: no voice for role {role!r} in casting['roles']"
                )
            text = seg["text"]
            segments.append(
                {
                    "role": role,
                    "voice": roles[role],
                    "text": text,
                    "tts_text": text_prep(text) if text_prep else text,
                    "take": take,
                }
            )
        plan.append(
            {
                "id": section["id"],
                "title": section.get("title", ""),
                "segments": segments,
            }
        )
    return plan


def segment_key(segment: Mapping, voice: Mapping) -> str:
    """A take's identity: a hash of the voice (loading and settings), spoken text and take number.

    ``tempo`` and ``effect`` are finishing steps, left out: changing them rebuilds the
    section files without new takes.

    >>> a = segment_key({"tts_text": "Hi", "take": 1}, {"params": {"seed": 1}})
    >>> a == segment_key({"tts_text": "Hi", "take": 1}, {"params": {"seed": 1}, "tempo": 0.8})
    True
    >>> a == segment_key({"tts_text": "Hi", "take": 2}, {"params": {"seed": 1}})
    False
    """
    synth = {k: v for k, v in voice.items() if k not in FINISHING_KEYS}
    blob = json.dumps(
        {"voice": synth, "tts_text": segment["tts_text"], "take": segment["take"]},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_voice_profile(spec: Mapping, *, root: str | os.PathLike | None = None):
    """The voice to hand ``synthesize`` from a voices entry (see the module docstring)."""
    backend = spec.get("backend", DFLT_BACKEND)
    if "library" in spec:
        from voxy.library import load_voice

        return load_voice(spec["library"], model_type=backend)
    if "reference" in spec:
        from voxy.local_clone import reference_profile

        ref = spec["reference"]
        audio = Path(root or ".") / ref["audio"]
        return reference_profile(audio, ref["text"], backend=backend)
    if "voice" in spec:
        return spec["voice"]
    raise ValueError(f"A voice needs 'library', 'reference' or 'voice': {dict(spec)}")


def _params(voice: Mapping, take: int) -> dict:
    params = dict(voice.get("params", {}))
    if "seed" in params and take > 1:
        params["seed"] = params["seed"] + take - 1
    return params


def lengthen_pauses(
    samples, sr: int, *, tiers: Sequence[Sequence[float]], floor_db: float = -35.0
):
    """``samples`` (mono float array) with each inner pause lengthened by its tier's extra seconds.

    A pause is a run of 10 ms frames ``floor_db`` below the loud frames (95th percentile);
    it is lengthened by repeating its own middle, so the room tone carries on. Leading and
    trailing silence is left as is.

    >>> tone = np.sin(np.arange(8000) / 3.0).astype(np.float32)
    >>> y = np.concatenate([tone, np.zeros(4000, np.float32), tone])  # 0.5 s pause at 8 kHz
    >>> len(lengthen_pauses(y, 8000, tiers=[[0.4, 0.25]])) - len(y)
    2000
    """
    hop = max(1, int(sr * FRAME_S))
    n = len(samples) // hop
    if not n or not tiers:
        return samples
    frames = samples[: n * hop].reshape(n, hop)
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    quiet = db < np.percentile(db, 95) + floor_db
    tiers = sorted(tiers)
    pieces, last, k = [], 0, 0
    while k < n:
        if not quiet[k]:
            k += 1
            continue
        j = k
        while j < n and quiet[j]:
            j += 1
        if k > 0 and j < n:  # an inner pause
            seconds = (j - k) * hop / sr
            extra = max((add for least, add in tiers if seconds >= least), default=0)
            if extra:
                a, b = k * hop, j * hop
                mid = samples[a + (b - a) // 4 : b - (b - a) // 4]
                cut = (a + b) // 2
                pieces += [samples[last:cut], np.resize(mid, int(extra * sr))]
                last = cut
        k = j
    pieces.append(samples[last:])
    return np.concatenate(pieces)


def _with_pauses(path: Path, finish: Mapping, tmp: Path) -> Path:
    if not finish.get("pauses"):
        return path
    samples, sr = read_wav(path)
    longer = lengthen_pauses(
        samples, sr, tiers=finish["pauses"], floor_db=finish["pause_floor_db"]
    )
    out = tmp / path.name
    out.write_bytes(write_wav(longer, sr))
    return out


def finish_section(
    wavs: Sequence[tuple[Path, Mapping]], out: Path, finish: Mapping
) -> None:
    """Join takes (each ``(wav path, voice spec)``) into one MP3 at ``out``.

    Per take: pauses, then tempo, effect and loudness; a silent gap follows each take but
    the last. Needs ``ffmpeg`` (with rubberband for ``tempo``).
    """
    with tempfile.TemporaryDirectory() as tmp:
        wavs = [(_with_pauses(p, finish, Path(tmp)), voice) for p, voice in wavs]
        last, chains = len(wavs) - 1, []
        for k, (_, voice) in enumerate(wavs):
            steps = [
                f"aresample={finish['sample_rate']}",
                "aformat=channel_layouts=mono",
            ]
            if (tempo := voice.get("tempo", 1)) != 1:
                steps.append(f"rubberband=tempo={tempo}")
            if effect := voice.get("effect"):
                steps.append(effect)
            steps.append(
                f"loudnorm=I={finish['loudness_lufs']}:TP={finish['true_peak_db']}:LRA=11"
            )
            steps.append(
                f"aresample={finish['sample_rate']}"
            )  # loudnorm upsamples internally
            if k < last:
                steps.append(f"apad=pad_dur={finish['segment_gap_s']}")
            chains.append(f"[{k}:a]" + ",".join(steps) + f"[a{k}]")
        graph = ";".join(chains) + ";" + "".join(f"[a{k}]" for k in range(len(wavs)))
        graph += f"concat=n={len(wavs)}:v=0:a=1[out]"
        out.parent.mkdir(parents=True, exist_ok=True)
        inputs = [x for p, _ in wavs for x in ("-i", str(p))]
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                *inputs,
                "-filter_complex",
                graph,
                "-map",
                "[out]",
                "-c:a",
                "libmp3lame",
                "-b:a",
                finish["bitrate"],
                str(out),
            ],
            check=True,
        )


def _seconds(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return round(float(out), 2)


def _take_path(root: Path, key: str) -> Path:
    return root / SEGMENTS_DIR / f"{key}.wav"


def add_take(
    root: str | os.PathLike,
    segment: Mapping,
    audio: bytes,
    *,
    generation_seconds: float | None = None,
) -> Path:
    """File one take: ``segments/<key>.wav`` (``audio``: WAV bytes) and its ``.json``.

    ``segment`` is an entry of ``missing_takes`` (or the renderer's own), with ``key`` and
    ``voice_spec``.
    """
    wav = _take_path(Path(root), segment["key"])
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(audio)
    samples, sr = read_wav(audio)
    meta = {
        **{
            k: segment[k]
            for k in (
                "section",
                "role",
                "voice",
                "text",
                "tts_text",
                "take",
                "key",
                "voice_spec",
            )
        },
        "seconds": round(len(samples) / sr, 2),
        "generation_seconds": None
        if generation_seconds is None
        else round(generation_seconds, 1),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    wav.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return wav


def _read_inputs(root: Path, casting, voices):
    casting = casting or json.loads((root / CASTING_FILE).read_text())
    voices = voices or json.loads((root / VOICES_FILE).read_text())
    return casting, voices


def missing_takes(
    script: Sequence[Mapping],
    root: str | os.PathLike,
    *,
    casting: Mapping | None = None,
    voices: Mapping | None = None,
    text_prep: Callable[[str], str] | None = None,
    external_only: bool = True,
) -> list[dict]:
    """Segments with no take yet (by default only those of external voices), in script order.

    Each is ``{section, role, voice, text, tts_text, take, key, voice_spec}``: hand it to
    whatever renders that voice, then file the audio with ``add_take(root, segment, wav)``.
    """
    root = Path(root)
    casting, voices = _read_inputs(root, casting, voices)
    out = []
    for section in plan_script(script, casting, text_prep=text_prep):
        for seg in section["segments"]:
            voice = voices.get(seg["voice"]) if seg["voice"] else None
            if voice is None or (external_only and not voice.get("external")):
                continue
            key = segment_key(seg, voice)
            if not _take_path(root, key).exists():
                out.append(
                    {**seg, "section": section["id"], "key": key, "voice_spec": voice}
                )
    return out


def render_script(
    script: Sequence[Mapping],
    root: str | os.PathLike,
    *,
    casting: Mapping | None = None,
    voices: Mapping | None = None,
    only: Iterable[str] | None = None,
    finish: Mapping | None = None,
    text_prep: Callable[[str], str] | None = None,
    model_factory: Callable | None = None,
    voice_loader: Callable = load_voice_profile,
    log: Callable[[str], None] = print,
) -> dict:
    """Render what is missing or changed, section by section in script order; return (and save) the index.

    Args:
        script: Sections ``{"id", "title"?, "segments": [{"role", "text"}]}``.
        root: The output folder; ``casting`` and ``voices`` default to its
            ``casting.json`` and ``voices.json``.
        only: Section ids to render (default all); the index still lists every section.
        finish: Overrides of ``DFLT_FINISH`` (gap between voices, loudness, bitrate, pauses).
        text_prep: ``text -> spoken text`` per segment (e.g. ``voxy.speechtext`` preparation).
        model_factory: ``(backend, **init) -> SpeechModel`` (default ``voxy.create_speech_model``).
        voice_loader: ``(voice spec, root=) -> voice`` (default ``load_voice_profile``).
    """
    root = Path(root)
    casting, voices = _read_inputs(root, casting, voices)
    finish = {**DFLT_FINISH, **(finish or {})}
    only = set(only) if only is not None else None
    plan = plan_script(script, casting, text_prep=text_prep)
    index_path = root / INDEX_FILE
    old = (
        {i["id"]: i for i in json.loads(index_path.read_text())["items"]}
        if index_path.exists()
        else {}
    )
    models, profiles = {}, {}

    def model_for(backend, init):
        name = (backend, json.dumps(init, sort_keys=True))
        if name not in models:
            factory = model_factory
            if factory is None:
                from voxy import create_speech_model as factory
            models[name] = factory(backend, **init)
        return models[name]

    def take_of(seg, voice, key):
        wav = _take_path(root, key)
        if wav.exists() and wav.with_suffix(".json").exists():
            return wav
        if voice.get("external"):
            raise LookupError(
                f"{seg['section']}: external voice {seg['voice']} has no take {key} yet"
            )
        name = seg["voice"]
        if name not in profiles:
            profiles[name] = voice_loader(voice, root=root)
        log(f"voxy.script: {seg['section']} {name} ({len(seg['tts_text'])} chars)")
        model = model_for(voice.get("backend", DFLT_BACKEND), voice.get("init", {}))
        start = time.perf_counter()
        speech = model.synthesize(
            seg["tts_text"], profiles[name], **_params(voice, seg["take"])
        )
        took = time.perf_counter() - start
        return add_take(
            root,
            {**seg, "key": key, "voice_spec": voice},
            speech.audio,
            generation_seconds=took,
        )

    def entry(section, status, segs, file=None, seconds=None, finished_with=None):
        return {
            "id": section["id"],
            "title": section["title"],
            "status": status,
            "segments": segs,
            "file": file,
            "seconds": seconds,
            "finished_with": finished_with,
        }

    def save(entries):
        root.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps({"items": entries}, indent=1, ensure_ascii=False)
        )

    entries = []
    for section in plan:  # what exists already, so the index is whole from the start
        if unknown := {s["voice"] for s in section["segments"]} - set(voices) - {None}:
            raise ValueError(
                f"{section['id']}: voice(s) {sorted(unknown)} not in the voices"
            )
        segs = [
            {
                "role": s["role"],
                "voice": s["voice"],
                "take": s["take"],
                "key": segment_key(s, voices[s["voice"]]) if s["voice"] else None,
            }
            for s in section["segments"]
        ]
        prev = old.get(section["id"], {})
        done = prev.get("status") == "done" and [
            s["key"] for s in prev.get("segments", [])
        ] == [s["key"] for s in segs]
        waiting = any(s["voice"] is None for s in segs)
        owed = any(
            voices[s["voice"]].get("external")
            and not _take_path(root, s["key"]).exists()
            for s in segs
            if s["voice"]
        )
        entries.append(
            prev
            if done
            else entry(
                section,
                AWAITING if waiting else AWAITING_TAKE if owed else "pending",
                segs,
            )
        )
    save(entries)
    try:
        for n, section in enumerate(plan):
            if only is not None and section["id"] not in only:
                continue
            segs = entries[n]["segments"]
            if entries[n]["status"] in (AWAITING, AWAITING_TAKE):
                continue
            stamp = {
                "finish": finish,
                "finishing": [
                    {k: voices[s["voice"]].get(k) for k in FINISHING_KEYS} for s in segs
                ],
            }
            file = Path(SECTIONS_DIR) / f"{section['id']}.mp3"
            if (
                entries[n]["status"] == "done"
                and entries[n].get("finished_with") == stamp
                and (root / file).exists()
            ):
                continue
            wavs = [
                take_of({**s, "section": section["id"]}, voices[s["voice"]], k["key"])
                for s, k in zip(section["segments"], segs)
            ]
            finish_section(
                [(w, voices[s["voice"]]) for w, s in zip(wavs, segs)],
                root / file,
                finish,
            )
            entries[n] = entry(
                section, "done", segs, str(file), _seconds(root / file), stamp
            )
            save(entries)
            log(f"voxy.script: {section['id']} done ({entries[n]['seconds']} s)")
    finally:
        for m in models.values():
            if hasattr(m, "close"):
                m.close()
    save(entries)
    return {"items": entries}


def status(root: str | os.PathLike) -> dict:
    """Progress from ``index.json``: counts per status, total ``seconds``, ``segments_by_voice``."""
    items = json.loads((Path(root) / INDEX_FILE).read_text())["items"]
    out = {
        "done": 0,
        "pending": 0,
        AWAITING: 0,
        AWAITING_TAKE: 0,
        "seconds": 0.0,
        "segments_by_voice": {},
    }
    for i in items:
        out[i["status"]] = out.get(i["status"], 0) + 1
        out["seconds"] += i["seconds"] or 0
        for s in i["segments"]:
            out["segments_by_voice"][s["voice"]] = (
                out["segments_by_voice"].get(s["voice"], 0) + 1
            )
    out["seconds"] = round(out["seconds"], 1)
    return out
