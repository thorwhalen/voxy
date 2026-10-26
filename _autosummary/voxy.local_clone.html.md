# voxy.local_clone

Local voice cloning: speak any text in a voice learned from a short reference clip.

Each backend here (F5-TTS, Chatterbox, XTTS-v2, …) is a model that runs on your
own machine, CPU included. A *voice* is a reference: a clean clip of the speaker
plus its transcript, as a `VoiceProfile` whose segment is plain data
(`{"ref_audio": path, "ref_text": text}`), so it can be saved in the voice
library and reused by every local backend.

```pycon
>>> import voxy
>>> model = voxy.create_speech_model("f5tts")
>>> voice = model.clone_voice("ref.wav", "What the clip says.")
>>> model.synthesize(chapter_text, voice, nfe_step=32).save("ch1.wav")
```

Long text is split into paragraphs and sentence-packed chunks (`voxy.textsplit`),
each chunk is synthesized, trimmed of edge silence, and joined with `gap_ms` of
pause (`paragraph_gap_ms` between paragraphs).

Seams (keyword arguments of every model):

- `generate(text, *, ref_audio, ref_text, **params) -> (samples, sample_rate)`: the
  model call. Default: a `voxy.worker.Worker` running the backend’s adapter in its
  own environment (built on first use under `$VOXY_DATA_DIR/envs`). Tests inject a fake.
- `chunker(text) -> [[chunk, ...], ...]`: default `split_text` at the backend’s
  `max_chars`.

```pycon
>>> import numpy as np
>>> fake = lambda text, **kw: (np.full(800, 0.1, dtype=np.float32), 8000)
>>> model = LocalCloneSpeechModel("f5tts", generate=fake)
>>> voice = reference_profile("ref.wav", "Hello there.", backend="f5tts")
>>> speech = model.synthesize("One. Two.\n\nThree.", voice)
>>> speech.format, speech.sample_rate, speech.backend
('wav', 8000, 'f5tts')
```

### Functions

| [`reference_profile`](#voxy.local_clone.reference_profile)(ref_audio, ref_text, \*[, ...])   | A voice for local backends: a reference clip (or clips) and what it says.         |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`register_local_clone_backend`](#voxy.local_clone.register_local_clone_backend)(backend)               | Add (or replace) a local cloning backend description.                             |
| [`voxy_cache_dir`](#voxy.local_clone.voxy_cache_dir)(\*parts)                             | Intermediates (worker scratch audio, logs): `$VOXY_CACHE_DIR` or `~/.cache/voxy`. |

### Classes

| [`ChatterboxSpeechModel`](#voxy.local_clone.ChatterboxSpeechModel)([backend, generate, ...])   | Chatterbox: expressive zero-shot cloning (`variant='original'|'turbo'`).                  |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [`DesignPreview`](#voxy.local_clone.DesignPreview)(seed, audio, text, ...)             | One candidate voice from `design_voice_previews`: listen, then pick.                      |
| [`F5TTSSpeechModel`](#voxy.local_clone.F5TTSSpeechModel)([backend, generate, ...])        | F5-TTS: flow-matching zero-shot cloning (`params`: nfe_step, cfg_strength, speed...).     |
| [`LocalCloneBackend`](#voxy.local_clone.LocalCloneBackend)(name, env[, init, params, ...]) | What voxy knows about one local cloning model.                                            |
| [`LocalCloneSpeechModel`](#voxy.local_clone.LocalCloneSpeechModel)([backend, generate, ...])   | A local voice-cloning model as a voxy backend.                                            |
| [`Qwen3TTSSpeechModel`](#voxy.local_clone.Qwen3TTSSpeechModel)(\*args[, design])             | Qwen3-TTS: stable cloning (`init`: model_id 1.7B or 0.6B) and local voice design.         |
| [`VoxCPMSpeechModel`](#voxy.local_clone.VoxCPMSpeechModel)([backend, generate, ...])       | VoxCPM2: high-similarity cloning (`params`: cfg_value, inference_timesteps, prompt_mode). |
| [`WorkerGenerate`](#voxy.local_clone.WorkerGenerate)(backend, \*[, init, threads, ...]) | The default `generate`: the backend's adapter in its own environment.                     |
| [`XTTSSpeechModel`](#voxy.local_clone.XTTSSpeechModel)(\*args[, accept_licence])         | XTTS-v2 through coqui-tts (weights under the non-commercial Coqui licence).               |

### *class* voxy.local_clone.ChatterboxSpeechModel(backend=None, , generate=None, chunker=None, text_prep=None, gap_ms=250, paragraph_gap_ms=700, trim=True, threads=None, rootdir=None, auto_setup=True, env_vars=None, device='cpu', init=None, \*\*init_kwargs)

Bases: [`LocalCloneSpeechModel`](#voxy.local_clone.LocalCloneSpeechModel)

Chatterbox: expressive zero-shot cloning (`variant='original'|'turbo'`).

### *class* voxy.local_clone.DesignPreview(seed, audio, text, sample_rate, description)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One candidate voice from `design_voice_previews`: listen, then pick.

#### save(path)

Write the clip to `path` (to audition it).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### *class* voxy.local_clone.F5TTSSpeechModel(backend=None, , generate=None, chunker=None, text_prep=None, gap_ms=250, paragraph_gap_ms=700, trim=True, threads=None, rootdir=None, auto_setup=True, env_vars=None, device='cpu', init=None, \*\*init_kwargs)

Bases: [`LocalCloneSpeechModel`](#voxy.local_clone.LocalCloneSpeechModel)

F5-TTS: flow-matching zero-shot cloning (`params`: nfe_step, cfg_strength, speed…).

### *class* voxy.local_clone.LocalCloneBackend(name, env, init=dict(), params=dict(), max_chars=300, licence='', notes='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What voxy knows about one local cloning model.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Registry name (also the adapter’s name, `workers/{name}_adapter.py`).
  * **env** ([`EnvSpec`](voxy.envs.html.md#voxy.envs.EnvSpec)) – The environment it runs in.
  * **init** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Default keyword arguments of the adapter’s `load` (model variant,
    checkpoint, device); override per model with `init=` or `**init`.
  * **params** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Documented synthesis parameters and their defaults (pass any of
    them to `synthesize`; the diagnosis grid sweeps them).
  * **max_chars** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Chunk size for long text.
  * **licence** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Code and weights licences (flag non-commercial ones).
  * **notes** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – One line on strengths and caveats.

### *class* voxy.local_clone.LocalCloneSpeechModel(backend=None, , generate=None, chunker=None, text_prep=None, gap_ms=250, paragraph_gap_ms=700, trim=True, threads=None, rootdir=None, auto_setup=True, env_vars=None, device='cpu', init=None, \*\*init_kwargs)

Bases: [`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel)

A local voice-cloning model as a voxy backend.

* **Parameters:**
  * **backend** ([`LocalCloneBackend`](#voxy.local_clone.LocalCloneBackend) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – A `LocalCloneBackend` or its name (subclasses fix it).
  * **generate** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The model call (see the module docstring); default: worker.
  * **chunker** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `text -> [[chunk, ...], ...]`; default `split_text`.
  * **text_prep** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `text -> text` applied before chunking (e.g.
    `lambda t: prepare_speech_text(t, ...).speech`, see `voxy.speechtext`).
  * **gap_ms** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Pause between chunks of a paragraph.
  * **paragraph_gap_ms** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Pause between paragraphs.
  * **trim** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Trim each chunk’s leading and trailing silence before joining.
  * **threads** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Torch threads of the worker (default `$VOXY_NUM_THREADS` or 8).
  * **rootdir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Data root for environments and weights (default `$VOXY_DATA_DIR`).
  * **auto_setup** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Build the environment on first use (else raise if missing).
  * **env_vars** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Extra environment variables for the worker.
  * **init** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Keyword arguments of the model’s loader (e.g. `variant='turbo'`).

#### clone_voice(audio_input, transcript=None, speaker_id=0, \*\*metadata)

A reference voice from a clip (path, or WAV bytes) and its transcript.

Without `transcript` the clip is transcribed locally (`voxy.refprep`).

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

#### close()

Free the model’s memory (the next call reloads it).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### synthesize(text, voice=None, \*\*params)

Speak `text` (any length) in the reference `voice`.

`params` go to the model (see `self.backend.params` for the knobs).

* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

### *class* voxy.local_clone.Qwen3TTSSpeechModel(\*args, design=None, \*\*kwargs)

Bases: [`LocalCloneSpeechModel`](#voxy.local_clone.LocalCloneSpeechModel)

Qwen3-TTS: stable cloning (`init`: model_id 1.7B or 0.6B) and local voice design.

Designing follows Qwen’s design-then-clone recipe: `design_voice_previews`
speaks a sample text in voices made from a description (one per seed, no real
person involved); `design_voice` keeps the chosen sample as a reference
profile, so long texts render through Base cloning with one stable voice.

Args (beyond `LocalCloneSpeechModel`):
: design: `(text, *, instruct, **params) -> (samples, sample_rate)`, the
  : VoiceDesign model call. Default: the worker’s `design` operation.

#### design_voice(description, , name, preview=None, rootdir=None, \*\*preview_kwargs)

A reference profile from the chosen `preview` (default: the first generated).

The clip is kept at `samples/{name}/design.wav` under the data root, and
the profile points at it, so the voice survives like a cloned one.

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

#### design_voice_previews(description, , text='The morning light crossed the water, and for a moment nothing else moved.', seeds=(0, 1, 2), language='English', \*\*params)

Candidate voices for `description`, each speaking `text` (one per seed).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`DesignPreview`](#voxy.local_clone.DesignPreview)]

### *class* voxy.local_clone.VoxCPMSpeechModel(backend=None, , generate=None, chunker=None, text_prep=None, gap_ms=250, paragraph_gap_ms=700, trim=True, threads=None, rootdir=None, auto_setup=True, env_vars=None, device='cpu', init=None, \*\*init_kwargs)

Bases: [`LocalCloneSpeechModel`](#voxy.local_clone.LocalCloneSpeechModel)

VoxCPM2: high-similarity cloning (`params`: cfg_value, inference_timesteps, prompt_mode).

### *class* voxy.local_clone.WorkerGenerate(backend, , init=None, threads=None, rootdir=None, env_vars=None, auto_setup=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The default `generate`: the backend’s adapter in its own environment.

Builds the environment on first call (`auto_setup`), starts one worker,
keeps it (and the weights) alive until `close()`.

#### close()

Stop the worker and free its memory.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### design(text, , instruct, \*\*params)

The adapter’s `design` operation (backends that can design voices).

### *class* voxy.local_clone.XTTSSpeechModel(\*args, accept_licence=None, \*\*kwargs)

Bases: [`LocalCloneSpeechModel`](#voxy.local_clone.LocalCloneSpeechModel)

XTTS-v2 through coqui-tts (weights under the non-commercial Coqui licence).

The weights’ licence must be accepted once: `accept_licence=True` or
`$VOXY_ACCEPT_COQUI_CPML=1` (else loading would stop to ask on a terminal).

#### synthesize(text, voice=None, \*\*params)

As `LocalCloneSpeechModel.synthesize`, once the licence is accepted.

* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

### voxy.local_clone.reference_profile(ref_audio, ref_text, , backend='f5tts', sample_rate=24000, \*\*metadata)

A voice for local backends: a reference clip (or clips) and what it says.

```pycon
>>> reference_profile("ref.wav", "Hi.", backend="xtts").segment
{'ref_audio': 'ref.wav', 'ref_text': 'Hi.'}
```

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

### voxy.local_clone.register_local_clone_backend(backend)

Add (or replace) a local cloning backend description.

* **Return type:**
  [`LocalCloneBackend`](#voxy.local_clone.LocalCloneBackend)

### voxy.local_clone.voxy_cache_dir(\*parts)

Intermediates (worker scratch audio, logs): `$VOXY_CACHE_DIR` or `~/.cache/voxy`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
