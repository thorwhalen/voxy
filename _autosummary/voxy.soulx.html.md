# voxy.soulx

SoulX-Singer (Soul AI Lab, 2026; code and weights Apache-2.0): zero-shot singing
voice synthesis and conversion, as the `soulx` singing backend.

```pycon
>>> import voxy
>>> voxy.sing("melody.mid", "Light a can-dle in the win-dow", backend="soulx")
```

How it runs: in its own environment (`python -m voxy setup soulx`: Python 3.11,
torch 2.2, transformers 4.41, the upstream pins). SoulX-Singer has no
`pyproject.toml`, so pip can’t install it: voxy fetches its repository at a pinned
commit (`SOULX_SOURCE`) into `models/soulx/source/<commit>` on first use, and the
adapter puts that folder on `sys.path`. Weights (two 2.8 GB checkpoints, SVS and
SVC, plus RMVPE for F0) come from the Hugging Face Hub at pinned revisions; SVC also
loads `openai/whisper-base` (Apache-2.0) on first use. On Apple Silicon it runs on
MPS, with the vocoder on the CPU; elsewhere on CUDA or the CPU (about 8x real time).

What it reads: SoulX metadata, a list of *segments*, one entry per note in every
field, made here from `voxy.score` notes by `soulx_metadata`:

> {“index”: “target_0”, “language”: “English”, “time”: [0, 6080],
> : “duration”: “0.30 0.60 …”, “text”: “<SP> Light a candle candle …”,
>   “phoneme”: “”, “note_pitch”: “0 64 64 67 …”, “note_type”: “1 2 2 3 …”, “f0”: “”}

`note_type` is 1 for a rest (`<SP>`), 2 where a word starts, 3 where it
continues; a word sung over several notes is written on each. Phonemes are left
empty and filled in the worker (English, g2p_en), so the host needs no phonemizer.
A synthesis prompt needs the same metadata, aligned to its clip
(`soulx_prompt_metadata`).

Usage disclaimer (upstream README; not a licence term): do not use the model to
impersonate anyone without authorisation or to create deceptive audio; respect
intellectual property, privacy and personal consent.

```pycon
>>> from voxy.score import Note
>>> meta = soulx_metadata([Note(0.0, 0.5, 64, "hel"), Note(0.5, 0.5, 67, "hel", False)])
>>> meta[0]["text"], meta[0]["note_type"], meta[0]["duration"]
('<SP> hel hel <SP>', '1 2 3 1', '0.30 0.50 0.50 0.40')
```

### Module Attributes

| [`WORD_JOINER`](#voxy.soulx.WORD_JOINER)        | Joins the words of an elision (several words sung on one note) into the one `text` token of that note; the adapter phonemizes each word and joins their phones (the same constant is in `workers/soulx_adapter.py`).   |
|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`DFLT_LEAD_S`](#voxy.soulx.DFLT_LEAD_S)        | Rest before the first note and after the last (upstream's examples start with <SP>).                                                                                                                                   |
| [`DFLT_MIN_REST_S`](#voxy.soulx.DFLT_MIN_REST_S)    | A gap shorter than this joins the previous note (legato); a longer one is a rest.                                                                                                                                      |
| [`DFLT_SPLIT_REST_S`](#voxy.soulx.DFLT_SPLIT_REST_S)  | A rest longer than this ends a segment.                                                                                                                                                                                |
| [`DFLT_MAX_SEGMENT_S`](#voxy.soulx.DFLT_MAX_SEGMENT_S) | No segment longer than this (upstream allows 60 s; shorter is safer on MPS).                                                                                                                                           |
| [`SOULX_ENV`](#voxy.soulx.SOULX_ENV)          | Upstream's inference pins (its requirements.txt minus the preprocessing, web UI and training stacks).                                                                                                                  |

### Functions

| [`fit_voice_shift`](#voxy.soulx.fit_voice_shift)(target, prompt)                   | Semitones that move `target` into the register of the voice singing `prompt`: median prompt pitch minus median target pitch, over notes.    |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| [`soulx_metadata`](#voxy.soulx.soulx_metadata)(notes, \*[, name, language, ...])  | SoulX-Singer metadata (segments, phonemes left empty) for sung `notes`.                                                                     |
| [`soulx_prompt_metadata`](#voxy.soulx.soulx_prompt_metadata)(notes, \*[, language, ...]) | The one-segment metadata of a prompt clip whose notes are `notes`, at the clip's own times (SoulX crops the clip to the metadata's length). |

### Classes

| [`SoulXSingingModel`](#voxy.soulx.SoulXSingingModel)(\*[, run, init, threads, ...])   | SoulX-Singer as a singing model (`voxy.sing(..., backend="soulx")`).   |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|

### voxy.soulx.DFLT_LEAD_S *= 0.3*

Rest before the first note and after the last (upstream’s examples start with <SP>).

### voxy.soulx.DFLT_MAX_SEGMENT_S *= 30.0*

No segment longer than this (upstream allows 60 s; shorter is safer on MPS).

### voxy.soulx.DFLT_MIN_REST_S *= 0.2*

A gap shorter than this joins the previous note (legato); a longer one is a rest.

### voxy.soulx.DFLT_SPLIT_REST_S *= 2.0*

A rest longer than this ends a segment.

### voxy.soulx.SOULX_ENV *= EnvSpec(name='soulx', requirements=('torch==2.2.0', 'torchaudio==2.2.0', 'numpy<2', 'transformers==4.41.2', 'accelerate==1.11.0', 'librosa==0.11.0', 'soundfile', 'omegaconf==2.3.0', 'scipy', 'tqdm', 'huggingface_hub>=0.20,<1', 'g2p_en==2.1.0', 'nltk==3.9.2'), python='3.11', torch_backend='cpu', env_vars={'PYTORCH_ENABLE_MPS_FALLBACK': '1', 'NLTK_DATA': '{models}/nltk_data'})*

Upstream’s inference pins (its requirements.txt minus the preprocessing, web UI
and training stacks).

### *class* voxy.soulx.SoulXSingingModel(, run=None, init=None, threads=None, rootdir=None, auto_setup=True, env_vars=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

SoulX-Singer as a singing model (`voxy.sing(..., backend="soulx")`).

* **Parameters:**
  * **run** – `(op, **request) -> (samples, sample_rate)`, where `op` is the
    adapter’s ‘sing’ or ‘convert’; default: a `SingingWorker` running
    `workers/soulx_adapter.py` in the `soulx` environment (built, and the
    source fetched, on first use). Tests inject a fake.
  * **init** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Keyword arguments of the adapter’s `load` over the backend’s
    (`device='cpu'`, `weights_dir=` local weights, `preload=`).
  * **threads** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – As `voxy.local_clone.WorkerGenerate`.
  * **rootdir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – As `voxy.local_clone.WorkerGenerate`.
  * **auto_setup** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – As `voxy.local_clone.WorkerGenerate`.
  * **env_vars** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – As `voxy.local_clone.WorkerGenerate`.

#### close()

Stop the worker (free the weights).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### convert(audio, voice, , transpose='fit_voice', \*\*params)

Re-voice the sung clip at `audio` with `voice`’s timbre.

* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

#### sing(notes, voice, , transpose=0, language='English', \*\*params)

Sing `notes` (with words) in `voice` (a clip and the notes it sings).

* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

### voxy.soulx.WORD_JOINER *= '+'*

Joins the words of an elision (several words sung on one note) into the one
`text` token of that note; the adapter phonemizes each word and joins their
phones (the same constant is in `workers/soulx_adapter.py`).

### voxy.soulx.fit_voice_shift(target, prompt)

Semitones that move `target` into the register of the voice singing
`prompt`: median prompt pitch minus median target pitch, over notes.

Upstream computes this per segment (and weights notes by their phoneme count),
so a long song would change key between segments; voxy computes it once.

```pycon
>>> fit_voice_shift([Note(0, 1, 60), Note(1, 1, 64)], [Note(0, 1, 55)])
-5
```

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### voxy.soulx.soulx_metadata(notes, , name='target', language='English', lead_s=0.3, tail_s=0.4, min_rest_s=0.2, split_rest_s=2.0, max_segment_s=30.0)

SoulX-Singer metadata (segments, phonemes left empty) for sung `notes`.

The line starts `lead_s` after a rest, gaps of `min_rest_s` or more become
rests (shorter ones join the previous note), and a segment ends at a rest
longer than `split_rest_s` or before it would pass `max_segment_s`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### voxy.soulx.soulx_prompt_metadata(notes, , language='English', tail_s=0.0, min_rest_s=0.2)

The one-segment metadata of a prompt clip whose notes are `notes`, at the
clip’s own times (SoulX crops the clip to the metadata’s length).

```pycon
>>> from voxy.score import Note
>>> soulx_prompt_metadata([Note(0.3, 0.6, 67, "la")])[0]["duration"]
'0.30 0.60'
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
