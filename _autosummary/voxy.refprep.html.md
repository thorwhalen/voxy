# voxy.refprep

Prepare a speaker’s recordings for cloning and fine-tuning: clean, cut, transcribe, score.

```pycon
>>> from voxy import refprep
>>> clips = refprep.prepare_reference(["tape1.m4a", "tape2.mp3"], voice="ada")
>>> ref = refprep.best_references(clips)[0]
>>> ref["path"], ref["text"], ref["score"]
```

The steps, each a function you can call alone or replace:

1. `condition` (ffmpeg): any format to mono PCM WAV at `sample_rate`, with a
   high-pass, optional denoising (`denoise=`: ‘afftdn’, ‘anlmdn’, ‘deepfilternet’, or None) and
   loudness normalisation (EBU R128, `loudness_lufs`).
2. `transcribe`: timed segments and words from a local Whisper (scribed’s
   faster-whisper backend, in its own environment, see `voxy.envs`).
3. `speaker` (optional, `voxy.speakers.speaker_filter`): marks words, and
   untranscribed stretches, spoken by someone other than the target voice.
4. `plan_clips`: cut points at sentence/segment boundaries, so each clip is
   `min_seconds`-`max_seconds` long, carries its exact text, and never
   contains a marked word.
5. `score_clip`: duration, signal-to-noise estimate, clipping, level, ASR
   confidence, speaking rate and (with a speaker step) the speaker match,
   folded into a `score` in [0, 1].

`prepare_reference` runs them all and saves the clips plus `manifest.jsonl` in
the voice’s samples store (`voxy.stores.samples_store`), so any backend (a
reference clip) and fine-tuning (`voxy.finetune`, the whole set) read the same data.
It only ever adds to the store, keeping every run; `latest_runs` gives each source’s
latest run. `forget` (a source’s clips) and `forget_orphans` (keys no manifest entry
claims, listed by `orphans`) are the only functions that delete clips; edit the
manifest by hand and the clips of the entries you removed become orphans.

### Module Attributes

| [`SPEAKER_SCORE_RANGE`](#voxy.refprep.SPEAKER_SCORE_RANGE)   | Speaker similarity mapped to the score's `speaker` part (0 at the low end, 1 at the high).                                                                                                                                |
|------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`SLUG_PATTERN`](#voxy.refprep.SLUG_PATTERN)          | The slugs this module gives sources (see `_slug`), and the clip keys made from them.                                                                                                                                      |
| [`RUN_ID_TIME_FORMAT`](#voxy.refprep.RUN_ID_TIME_FORMAT)    | its UTC start time to the microsecond, plus random hex characters, so ids sort by time and two runs never share one.                                                                                                      |
| [`LEGACY_RUN`](#voxy.refprep.LEGACY_RUN)            | `forget(..., run=LEGACY_RUN)` selects entries written before runs were recorded (no `run` field), matching even a relative `source` exactly as it was typed.                                                              |
| [`PENDING_PREFIX`](#voxy.refprep.PENDING_PREFIX)        | A running `prepare_reference` keeps a marker at `pending/<run>.json` (its run id, UTC start time and the slugs it reserved) until it finishes, so `orphans` does not take its keys, not yet in the manifest, for orphans. |
| [`UNICODE_FORM`](#voxy.refprep.UNICODE_FORM)          | Names are compared in this Unicode normal form (APFS stores either; users type NFC).                                                                                                                                      |
| [`ATOMIC_TMP_SUFFIX`](#voxy.refprep.ATOMIC_TMP_SUFFIX)     | Suffix of the temporary file an atomic write goes through (its name starts with `.`).                                                                                                                                     |
| [`ATOMIC_FILE_MODE`](#voxy.refprep.ATOMIC_FILE_MODE)      | File mode of a file written atomically (`mkstemp` would leave it 0600).                                                                                                                                                   |
| [`DENOISE_FILTERS`](#voxy.refprep.DENOISE_FILTERS)       | no denoising).                                                                                                                                                                                                            |
| [`DF_SAMPLE_RATE`](#voxy.refprep.DF_SAMPLE_RATE)        | DeepFilterNet works at 48 kHz; it needs an old torch/torchaudio pair, hence its own env.                                                                                                                                  |

### Functions

| [`best_references`](#voxy.refprep.best_references)(manifest, \*[, n, ...])            | The `n` best clips for zero-shot reference (length in range, best score first).                                                                                                                                                                                                                                                                                                                                       |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`close_asr`](#voxy.refprep.close_asr)()                                        | Stop the transcription workers (free their memory).                                                                                                                                                                                                                                                                                                                                                                   |
| [`condition`](#voxy.refprep.condition)(src, dst, \*[, sample_rate, start, ...]) | Convert `src` (any format ffmpeg reads) to clean mono 16-bit WAV at `dst`.                                                                                                                                                                                                                                                                                                                                            |
| [`conditioning_filters`](#voxy.refprep.conditioning_filters)(\*[, denoise, ...])           | The ffmpeg audio filter chain for `condition`.                                                                                                                                                                                                                                                                                                                                                                        |
| [`deepfilternet_denoise`](#voxy.refprep.deepfilternet_denoise)(src, dst, \*[, ...])         | Denoise the 48 kHz WAV `src` into `dst` with DeepFilterNet 3.                                                                                                                                                                                                                                                                                                                                                         |
| [`forget`](#voxy.refprep.forget)(samples, source, \*[, run, dry_run, log])   | Remove `source`'s clips (only those of one `run`, if given) from a samples store: its manifest entries, their clip keys and the source's conditioned key.                                                                                                                                                                                                                                                             |
| [`forget_orphans`](#voxy.refprep.forget_orphans)(samples, \*[, dry_run, ...])        | Delete the `orphans` of a samples store, and the stale pending markers (those older than `stale_after` seconds); return their keys.                                                                                                                                                                                                                                                                                   |
| [`latest_runs`](#voxy.refprep.latest_runs)(manifest)                              | `manifest` with, per source, only the entries of its most recent run.                                                                                                                                                                                                                                                                                                                                                 |
| [`load_manifest`](#voxy.refprep.load_manifest)(samples)                             | The manifest saved by `prepare_reference` in a samples store.                                                                                                                                                                                                                                                                                                                                                         |
| [`orphans`](#voxy.refprep.orphans)(samples, \*[, stale_after])                | The keys of this module's own form (`clips/<slug>-NNNN.wav`, `conditioned/<slug>.wav`) whose slug no manifest entry and no live pending marker uses, sorted.                                                                                                                                                                                                                                                          |
| [`partial_transcribe`](#voxy.refprep.partial_transcribe)([rootdir])                      | The default transcriber, bound to a data root.                                                                                                                                                                                                                                                                                                                                                                        |
| [`plan_clips`](#voxy.refprep.plan_clips)(segments, \*[, min_seconds, ...])       | Clips `[{start, end, text, words}]` cut between words, ending sentences where possible.                                                                                                                                                                                                                                                                                                                               |
| [`prepare_reference`](#voxy.refprep.prepare_reference)(sources, \*[, voice, ...])       | Condition, transcribe, cut and score `sources`; save clips and manifest; return it.                                                                                                                                                                                                                                                                                                                                   |
| [`score_clip`](#voxy.refprep.score_clip)(samples, sample_rate, \*[, words, ...]) | Quality measures of a clip, and their combination `score` in [0, 1].                                                                                                                                                                                                                                                                                                                                                  |
| [`source_identity`](#voxy.refprep.source_identity)(source)                            | What tells sources apart: the real path (so a relative and an absolute path, or a symlink, to one file are one source), spelled as the filesystem stores it when the file exists (so `Talk.wav` and `talk.wav` typed for one file on a case-insensitive filesystem are one source), then in Unicode normal form `UNICODE_FORM` (so an NFC and an NFD spelling of `Café.wav` are one source), then `os.path.normcase`. |
| [`summarize`](#voxy.refprep.summarize)(manifest)                                | Totals for a manifest: clips, minutes, minutes above score thresholds.                                                                                                                                                                                                                                                                                                                                                |
| [`transcribe`](#voxy.refprep.transcribe)(audio, \*[, language, model, ...])      | Timed segments `[{start, end, text, confidence, words: [...]}, ...]` of `audio`.                                                                                                                                                                                                                                                                                                                                      |
| [`transcribe_text`](#voxy.refprep.transcribe_text)(audio, \*\*kwargs)                 | The transcript of `audio` as one string.                                                                                                                                                                                                                                                                                                                                                                              |
| [`unique_slugs`](#voxy.refprep.unique_slugs)(sources, \*[, reserved])              | One distinct slug per source: the file's stem, plus parent folders only within a group of sources sharing a stem (stem tracks: every one is `vocals.wav`), numbered where folders cannot tell them apart.                                                                                                                                                                                                             |
| [`with_deepfilternet`](#voxy.refprep.with_deepfilternet)([conditioner, denoiser, ...])   | A `conditioner` that denoises with DeepFilterNet between two ffmpeg passes.                                                                                                                                                                                                                                                                                                                                           |

### voxy.refprep.ATOMIC_FILE_MODE *= 420*

File mode of a file written atomically (`mkstemp` would leave it 0600).

### voxy.refprep.ATOMIC_TMP_SUFFIX *= '.tmp'*

Suffix of the temporary file an atomic write goes through (its name starts with `.`).

### voxy.refprep.DENOISE_FILTERS *= {'afftdn': ['afftdn=nr=12:nf=-50:tn=1'], 'anlmdn': ['anlmdn=s=0.0005'], 'deepfilternet': [], None: []}*

no denoising).

* **Type:**
  ffmpeg filters for each `denoise` strategy (None

### voxy.refprep.DF_SAMPLE_RATE *= 48000*

DeepFilterNet works at 48 kHz; it needs an old torch/torchaudio pair, hence its own env.

### voxy.refprep.LEGACY_RUN *= 'legacy'*

`forget(..., run=LEGACY_RUN)` selects entries written before runs were recorded
(no `run` field), matching even a relative `source` exactly as it was typed.
Older voxy rewrote the whole manifest on every call, so such entries all come from
one call, in one folder: a relative string among them names one file.

### voxy.refprep.PENDING_PREFIX *= 'pending/'*

A running `prepare_reference` keeps a marker at `pending/<run>.json` (its run id,
UTC start time and the slugs it reserved) until it finishes, so `orphans` does not
take its keys, not yet in the manifest, for orphans. A crashed run leaves its marker;
one older than `PENDING_STALE_S` seconds is stale and claims nothing.

### voxy.refprep.RUN_ID_TIME_FORMAT *= '%Y%m%dT%H%M%S.%fZ'*

its UTC start time to the microsecond, plus
random hex characters, so ids sort by time and two runs never share one.

* **Type:**
  A run id (`run` in each manifest entry)

### voxy.refprep.SLUG_PATTERN *= re.compile('[a-z0-9-]+')*

The slugs this module gives sources (see `_slug`), and the clip keys made from them.

### voxy.refprep.SPEAKER_SCORE_RANGE *= (0.3, 0.7)*

Speaker similarity mapped to the score’s `speaker` part (0 at the low end, 1 at the high).

### voxy.refprep.UNICODE_FORM *= 'NFC'*

Names are compared in this Unicode normal form (APFS stores either; users type NFC).

### voxy.refprep.best_references(manifest, , n=3, min_seconds=5.0, max_seconds=12.0, min_score=0.0, latest_only=True)

The `n` best clips for zero-shot reference (length in range, best score first).

With `latest_only` (the default) it picks from `latest_runs(manifest)`: an older
run of a source is the same audio cut again, so its clips would come back as
near-duplicates of the newer ones, and the newer run is the one made with the
current settings. A reference saved from an older run stays valid either way (its
key and bytes are kept); pass `latest_only=False` to choose among every run.

```pycon
>>> m = [{"duration": 8, "score": 0.7}, {"duration": 2, "score": 0.9},
...      {"duration": 10, "score": 0.8}]
>>> [c["score"] for c in best_references(m, n=2)]
[0.8, 0.7]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.refprep.close_asr()

Stop the transcription workers (free their memory).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.refprep.condition(src, dst, \*, sample_rate=24000, start=None, duration=None, filters=None, run=<function run>)

Convert `src` (any format ffmpeg reads) to clean mono 16-bit WAV at `dst`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.refprep.conditioning_filters(, denoise='afftdn', loudness_lufs=-20.0, highpass_hz=70)

The ffmpeg audio filter chain for `condition`.

```pycon
>>> conditioning_filters(denoise=None, loudness_lufs=-20, highpass_hz=70)
['highpass=f=70', 'loudnorm=I=-20:TP=-1.5:LRA=11']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.refprep.deepfilternet_denoise(src, dst, \*, rootdir=None, threads=None, run=<function run>, ensure=<function ensure_env>)

Denoise the 48 kHz WAV `src` into `dst` with DeepFilterNet 3.

Runs the `deepFilter` command of the isolated `DF_ENV` (built once with
`uv` under the data root: see `voxy.envs`). One short call per file, so it needs
no resident worker.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.refprep.forget(samples, source, \*, run=None, dry_run=False, log=<built-in function print>)

Remove `source`’s clips (only those of one `run`, if given) from a samples
store: its manifest entries, their clip keys and the source’s conditioned key.
Returns the deleted keys (with `dry_run=True`: the keys it would delete, changing
nothing).

With `forget_orphans`, one of the only two functions in voxy that delete voice
samples (`prepare_reference` never does), so it is careful about what it deletes:

- entries match `source` by `source_identity` (real path, in the case the
  filesystem stores it): of their `source_path`, or, for entries an older voxy
  wrote without one, of their `source` only if it is absolute. A relative
  `source` names whatever file had that name where that run started, so such
  entries never match; they are logged, and `run=LEGACY_RUN` removes them
  deliberately (matching `source` exactly as recorded, as in
  `forget(samples, "vocals.wav", run=LEGACY_RUN)`);
- only keys of the form this module writes (`clips/<slug>-NNNN.wav`,
  `conditioned/<slug>.wav`), with a slug taken from a matched entry’s clip key, are
  deleted, never a top-level sample;
- a key a remaining entry names, or whose slug a remaining entry uses, is kept;
- one writer at a time: two processes changing one store at once can lose a
  manifest update (the store has no lock);
- the manifest is read just before it is written, and written first, so an
  interruption leaves unlisted files, never a manifest naming deleted ones.

```pycon
>>> store = {"clips/a-0001.wav": b"A", "conditioned/a.wav": b"C",
...          "clips/b-0001.wav": b"B", "tape1.wav": b"T"}
>>> store[MANIFEST_KEY] = _dump_manifest([
...     {"key": "clips/a-0001.wav", "slug": "a", "source": "/s/a.wav"},
...     {"key": "clips/b-0001.wav", "slug": "b", "source": "/s/b.wav"},
...     {"key": "tape1.wav", "source": "/s/a.wav"}])
>>> forget(store, "/s/a.wav", dry_run=True)
['clips/a-0001.wav', 'conditioned/a.wav']
>>> len(load_manifest(store)), len(store)
(3, 5)
>>> forget(store, "/s/a.wav")
['clips/a-0001.wav', 'conditioned/a.wav']
>>> [e["key"] for e in load_manifest(store)], sorted(store)
(['clips/b-0001.wav'], ['clips/b-0001.wav', 'manifest.jsonl', 'tape1.wav'])
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.refprep.forget_orphans(samples, , dry_run=True, stale_after=21600)

Delete the `orphans` of a samples store, and the stale pending markers (those
older than `stale_after` seconds); return their keys. A dry run by default: it
returns what it would delete and changes nothing, so look first, then pass
`dry_run=False`.

**Removing entries from the manifest by hand makes their clips orphans**, which
`forget_orphans` then deletes, even if a saved voice profile’s `ref_audio` still
points at one. To drop a source’s clips, use `forget` (it lists, with
`dry_run=True`, exactly what it removes), never a hand edit of the manifest.

With `forget`, one of the only two functions in voxy that delete voice samples.
It refuses (raises, deleting nothing) where `orphans` does: a missing or empty
manifest beside unclaimed clip or conditioned keys, a manifest or pending marker it
cannot read. Keys a running `prepare_reference` reserved (its live marker) are
kept, but a run that is still going after `stale_after` seconds loses that claim.

```pycon
>>> store = {"conditioned/z.wav": b"", "clips/a-0001.wav": b"", "tape1.wav": b""}
>>> store[MANIFEST_KEY] = _dump_manifest([{"key": "clips/a-0001.wav"}])
>>> forget_orphans(store), sorted(store)
(['conditioned/z.wav'], ['clips/a-0001.wav', 'conditioned/z.wav', 'manifest.jsonl', 'tape1.wav'])
>>> forget_orphans(store, dry_run=False), sorted(store)
(['conditioned/z.wav'], ['clips/a-0001.wav', 'manifest.jsonl', 'tape1.wav'])
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.refprep.latest_runs(manifest)

`manifest` with, per source, only the entries of its most recent run.

`prepare_reference` keeps every run, so a source processed twice is in the
manifest twice (the same audio, cut again); consumers that want one copy of each
recording (training sets, reference picks, summaries) read this. Sources are told
apart by `source_identity` (of `source_path`, or an absolute `source`), so one
file recorded in two spellings on a case-insensitive filesystem is one source;
entries with no `run` count as one run older than any recorded one; entries whose
source cannot be known (none, or a relative path from an older voxy) are all kept.
Order is kept. A source whose newest run gave no clips keeps its older run’s
entries.

```pycon
>>> m = [{"key": "a1", "source_path": "/a", "run": "20260101T000000.000000Z-x"},
...      {"key": "a2", "source_path": "/a", "run": "20261101T000000.000000Z-y"},
...      {"key": "a0", "source_path": "/a"},
...      {"key": "b1", "source_path": "/b"},
...      {"key": "t", "source": "tape.wav"}]
>>> [e["key"] for e in latest_runs(m)]
['a2', 'b1', 't']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]

### voxy.refprep.load_manifest(samples)

The manifest saved by `prepare_reference` in a samples store.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.refprep.orphans(samples, , stale_after=21600)

The keys of this module’s own form (`clips/<slug>-NNNN.wav`,
`conditioned/<slug>.wav`) whose slug no manifest entry and no live pending marker
uses, sorted.

**Removing entries from the manifest by hand makes their clips orphans**, which
`forget_orphans` then deletes, even if a saved voice profile’s `ref_audio` still
points at one. To drop a source’s clips, use `forget` (it lists, with
`dry_run=True`, exactly what it removes), never a hand edit of the manifest.

Where they come from: a run that failed part-way (the failing source’s conditioned
file and any clips it had cut), a source that gave no clips (its conditioned file),
a manifest edited by hand. `prepare_reference` never reuses their slugs and
`forget` never deletes them, since no entry names them. A key an entry names is
never an orphan (its slug is that entry’s), nor is any other key (a top-level
sample, the manifest). A running `prepare_reference` claims its slugs with a
pending marker (`PENDING_PREFIX`), so its keys not in the manifest yet are not
orphans; a marker older than `stale_after` seconds (a crashed run’s) claims
nothing.

It raises, rather than guess, when the manifest has a line that is not an object,
when a pending marker cannot be read, and when the manifest is missing or empty
while keys of this module’s form are unclaimed (a lost or truncated manifest would
otherwise make every clip an orphan).

```pycon
>>> store = {"clips/a-0001.wav": b"", "clips/x-0001.wav": b"", "conditioned/a.wav": b"",
...          "conditioned/z.wav": b"", "tape1.wav": b""}
>>> store[MANIFEST_KEY] = _dump_manifest([{"key": "clips/a-0001.wav", "slug": "a"}])
>>> orphans(store)
['clips/x-0001.wav', 'conditioned/z.wav']
>>> store["pending/r.json"] = _pending_marker("r", ["x"])  # a run is cutting x
>>> orphans(store), orphans(store, stale_after=0)
(['conditioned/z.wav'], ['clips/x-0001.wav', 'conditioned/z.wav'])
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.refprep.partial_transcribe(rootdir=None)

The default transcriber, bound to a data root.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### voxy.refprep.plan_clips(segments, , min_seconds=3.0, max_seconds=12.0, pad_seconds=0.12, total_seconds=None)

Clips `[{start, end, text, words}]` cut between words, ending sentences where possible.

Words accumulate until the clip passes `min_seconds` and a word ends a
sentence (or the next word would push it past `max_seconds`). Cuts fall in
the pause between two words, padded by up to `pad_seconds`. A word marked
`keep=False` (someone else’s, see `voxy.speakers`) ends the clip before it,
and the next clip starts after it.

```pycon
>>> w = lambda s, e, t: {"start": s, "end": e, "text": t, "confidence": 0.9}
>>> segs = [{"words": [w(0, 1, "One"), w(1.1, 2, "two."), w(3, 4, "Three"),
...                    w(4.1, 5, "four.")], "start": 0, "end": 5, "text": ""}]
>>> [c["text"] for c in plan_clips(segs, min_seconds=1.5, max_seconds=3)]
['One two.', 'Three four.']
>>> segs[0]["words"].insert(2, {**w(2.2, 2.6, " yes"), "keep": False})
>>> [(c["text"], c["start"]) for c in plan_clips(segs, min_seconds=1.5, max_seconds=3)]
[('One two.', 0.0), ('Three four.', 2.88)]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.refprep.prepare_reference(sources, \*, voice=None, samples=None, sample_rate=24000, denoise='afftdn', loudness_lufs=-20.0, min_seconds=3.0, max_seconds=12.0, language=None, transcriber=None, conditioner=None, speaker=None, rootdir=None, log=<built-in function print>)

Condition, transcribe, cut and score `sources`; save clips and manifest; return it.

* **Parameters:**
  * **sources** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike)]) – Recordings, any format ffmpeg reads.
  * **voice** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice name: clips go to `samples_store(voice)`.
  * **samples** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A store to write to instead (`filename -> bytes`).
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Conditioning (see `condition`).
  * **denoise** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Conditioning (see `condition`).
  * **loudness_lufs** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Conditioning (see `condition`).
  * **min_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Clip length bounds.
  * **max_seconds** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Clip length bounds.
  * **language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Force the ASR language (default: detected).
  * **transcriber** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(wav_path, language=) -> segments` (default `transcribe`).
  * **conditioner** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(src, dst, sample_rate=, filters=) -> path` (default `condition`).
  * **speaker** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `(wav_path, segments) -> segments` marking other voices’ words
    (`voxy.speakers.speaker_filter`); None keeps every voice.
* **Returns:**
  one dict per clip (`key`, `path` if on disk, `text`,
  `source` as given, `source_path` (its `source_identity`), `slug`,
  `run` (one id per call), `start`, `end`, and the `score_clip`
  measures), best first, earlier runs’ entries included.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

It never deletes or overwrites anything in the store. Run again into the same store
(another recording, or the same file again), it gives this run’s sources fresh slugs
that no manifest entry and no clip or conditioned key in the store already uses, so
every earlier clip keeps its key and its bytes, and saved reference profiles
(`ref_audio` + `ref_text`) stay valid. The manifest keeps every earlier entry and
gets each source’s new ones as soon as that source is done (re-read just before
each write, so a `forget` or another run that finished meanwhile is not undone):
a run that fails part-way leaves its finished sources recorded, and only the
failing source’s keys (its conditioned file, any clips already cut) in no manifest,
which `orphans` lists and `forget_orphans` removes. Re-processing a source
therefore adds clips beside its earlier ones; consumers read
`latest_runs(manifest)` to see one run per source, and
`forget(samples, source, run=...)` removes the others explicitly. A file given
twice in one call (by `source_identity`) is processed once. While it runs, a
pending marker (`pending/<run>.json`: run id, start time, reserved slugs) keeps
`forget_orphans` off its keys; it is removed when the run finishes, and a crashed
run’s marker stays (showing the crash) until it is stale (`PENDING_STALE_S`).

### voxy.refprep.score_clip(samples, sample_rate, , words=(), text='', min_seconds=3.0, max_seconds=12.0)

Quality measures of a clip, and their combination `score` in [0, 1].

- `snr_db`: loud-frame level minus quiet-frame level (speech vs floor).
- `clipped`: fraction of samples at full scale.
- `level_db`: level of the loud frames.
- `asr_confidence`: mean Whisper word probability (low: mumbling, noise, music).
- `chars_per_second`: speaking rate (outliers: bad alignment).
- `speaker_similarity`: lowest match of a word to the target voice (words
  annotated by `voxy.speakers`; None otherwise).

```pycon
>>> rng = np.random.default_rng(0)
>>> speechy = np.concatenate([0.3 * rng.standard_normal(40000), 0.001 * rng.standard_normal(8000)])
>>> s = score_clip(speechy, 8000, text="x" * 70)
>>> s["snr_db"] > 30, 0 < s["score"] <= 1
(True, True)
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.refprep.source_identity(source)

What tells sources apart: the real path (so a relative and an absolute path, or
a symlink, to one file are one source), spelled as the filesystem stores it when
the file exists (so `Talk.wav` and `talk.wav` typed for one file on a
case-insensitive filesystem are one source), then in Unicode normal form
`UNICODE_FORM` (so an NFC and an NFD spelling of `Café.wav` are one source), then
`os.path.normcase`. A file that does not exist (any more) falls back to its real
path as typed.

```pycon
>>> source_identity("/no/such/Dir/a.wav") == os.path.normcase(
...     os.path.realpath("/no/such/Dir/a.wav"))
True
>>> source_identity("/no/Cafe\u0301.wav") == source_identity("/no/Caf\u00e9.wav")
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.refprep.summarize(manifest)

Totals for a manifest: clips, minutes, minutes above score thresholds.

```pycon
>>> summarize([{"duration": 60, "score": 0.9}, {"duration": 30, "score": 0.4}])
{'clips': 2, 'minutes': 1.5, 'minutes_score_0.5+': 1.0, 'minutes_score_0.7+': 1.0}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.refprep.transcribe(audio, , language=None, model='large-v3-turbo', threads=None, rootdir=None)

Timed segments `[{start, end, text, confidence, words: [...]}, ...]` of `audio`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.refprep.transcribe_text(audio, \*\*kwargs)

The transcript of `audio` as one string.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.refprep.unique_slugs(sources, , reserved=())

One distinct slug per source: the file’s stem, plus parent folders only within a
group of sources sharing a stem (stem tracks: every one is `vocals.wav`), numbered
where folders cannot tell them apart. Slugs never climb past what separates a group,
and a source whose stem is unique keeps it.

```pycon
>>> unique_slugs(["/s/one/vocals.wav", "/s/two/vocals.wav", "/s/talk.m4a"])
['one-vocals', 'two-vocals', 'talk']
>>> unique_slugs(["/Users/x/audio/talk.m4a", "/Users/x/audio/talk.wav"])
['talk', 'talk-2']
>>> unique_slugs(["/s/a.wav", "/s/a.wav", "/s/a-2.wav"])
['a', 'a-3', 'a-2']
```

`reserved` slugs (already used in the store) are never given out:

```pycon
>>> unique_slugs(["/t/vocals.wav"], reserved=["vocals"])
['vocals-2']
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.refprep.with_deepfilternet(conditioner=None, , denoiser=None, rootdir=None)

A `conditioner` that denoises with DeepFilterNet between two ffmpeg passes.

Pass 1 converts the source to a 48 kHz mono WAV (no filters); DeepFilterNet cleans
it; pass 2 applies the usual `filters` (high-pass, loudness) and resamples to
`sample_rate`. `conditioner` (default [`condition()`](#voxy.refprep.condition)) does both ffmpeg passes.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]
