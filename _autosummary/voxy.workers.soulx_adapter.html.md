# voxy.workers.soulx_adapter

SoulX-Singer (Soul AI Lab; code and weights Apache-2.0): zero-shot singing voice
synthesis (notes and lyrics to a sung vocal) and singing voice conversion.

Runs in the `soulx` environment, with no voxy imports. SoulX-Singer is not
pip-installable, so `load` puts its checkout (`repo_dir`, fetched by voxy at a
pinned commit) on `sys.path`. Weights come from the Hugging Face Hub at pinned
revisions (into `HF_HOME`), unless `weights_dir` / `preprocess_dir` point at
local copies.

Operations:

- `sing(state, *, meta, prompt_audio, prompt_meta, out_path, ...)`: SVS. `meta`
  and `prompt_meta` are SoulX metadata (lists of segments); `prompt_audio` is a
  sung clip whose notes and words `prompt_meta` describes. Empty `phoneme`
  fields are filled here (English, with g2p_en).
- `convert(state, *, audio, prompt_audio, out_path, ...)`: SVC, audio to audio;
  F0 is extracted with RMVPE.
- `f0(state, *, audio, out_path=None)`: the RMVPE F0 of a clip (50 frames a second).
- `unload(state, *, component)`: free one model (‘svs’, ‘svc’ or ‘rmvpe’).

The SVS and SVC checkpoints are 2.8 GB each; each loads on first use (`preload`
loads them at start). On Apple Silicon the models run on MPS except the vocoder,
which runs on the CPU: its ISTFT head multiplies complex tensors, which torch 2.2’s
MPS backend rejects (an internal assert in BinaryKernel.mm). Output is written as
16-bit WAV, scaled down if it peaks above `peak`.

Usage disclaimer (upstream README): do not impersonate anyone without authorisation
or make deceptive audio; respect intellectual property, privacy and consent.

### Module Attributes

| [`DFLT_FADE_S`](#voxy.workers.soulx_adapter.DFLT_FADE_S)    | Fade-out and fade-in length where segments meet (they are rendered separately and added, not overlapped).   |
|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [`SINGERS`](#voxy.workers.soulx_adapter.SINGERS)        | The two singers; loading one frees the other (2.8 GB each, about 6 GB in use).                              |
| [`NOTE_FIELDS`](#voxy.workers.soulx_adapter.NOTE_FIELDS)    | Per-note fields of a segment, which must all have one token per note.                                       |
| [`NLTK_RESOURCES`](#voxy.workers.soulx_adapter.NLTK_RESOURCES) | g2p_en's NLTK resources (nltk 3.9 tags with the `_eng` tagger).                                             |

### Functions

| [`convert`](#voxy.workers.soulx_adapter.convert)(state, \*, audio, prompt_audio, out_path)   | SVC: re-voice the sung `audio` with the timbre of `prompt_audio`.              |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`f0`](#voxy.workers.soulx_adapter.f0)(state, \*, audio[, out_path])                    | RMVPE F0 of `audio` on SoulX's 20 ms grid (Hz, 0 when unvoiced).               |
| [`fill_phonemes`](#voxy.workers.soulx_adapter.fill_phonemes)(state, meta)                          | `meta` with each empty `phoneme` field filled from its `text` (English).       |
| [`load`](#voxy.workers.soulx_adapter.load)(\*, repo_dir[, weights, preprocess, ...])      | Get ready to sing (models load on first use, or now for those in `preload`).   |
| [`phonemize_en`](#voxy.workers.soulx_adapter.phonemize_en)(word, g2p)                             | SoulX's token for an English word: `en_` and its ARPAbet phones joined by '-'. |
| [`sing`](#voxy.workers.soulx_adapter.sing)(state, \*, meta, prompt_audio, ...[, ...])     | SVS: sing `meta` in the voice of `prompt_audio`, into `out_path`.              |
| [`unload`](#voxy.workers.soulx_adapter.unload)(state, \*, component)                        | Free one component's weights; returns what stays loaded.                       |

### voxy.workers.soulx_adapter.DFLT_FADE_S *= 0.02*

Fade-out and fade-in length where segments meet (they are rendered separately
and added, not overlapped).

### voxy.workers.soulx_adapter.NLTK_RESOURCES *= {'corpora/cmudict': 'cmudict', 'taggers/averaged_perceptron_tagger_eng': 'averaged_perceptron_tagger_eng'}*

g2p_en’s NLTK resources (nltk 3.9 tags with the `_eng` tagger).

### voxy.workers.soulx_adapter.NOTE_FIELDS *= ('duration', 'text', 'phoneme', 'note_pitch', 'note_type')*

Per-note fields of a segment, which must all have one token per note.

### voxy.workers.soulx_adapter.SINGERS *= ('svs', 'svc')*

The two singers; loading one frees the other (2.8 GB each, about 6 GB in use).

### voxy.workers.soulx_adapter.convert(state, , audio, prompt_audio, out_path, auto_shift=True, pitch_shift=0, n_steps=None, cfg=None, seed=None, peak=0.95)

SVC: re-voice the sung `audio` with the timbre of `prompt_audio`.

`auto_shift` moves the melody into the prompt voice’s register (the shift
used is returned); `pitch_shift` sets it in semitones instead.

### voxy.workers.soulx_adapter.f0(state, , audio, out_path=None)

RMVPE F0 of `audio` on SoulX’s 20 ms grid (Hz, 0 when unvoiced).

### voxy.workers.soulx_adapter.fill_phonemes(state, meta)

`meta` with each empty `phoneme` field filled from its `text` (English).

### voxy.workers.soulx_adapter.load(, repo_dir, weights=None, preprocess=None, weights_dir=None, preprocess_dir=None, device='auto', preload=())

Get ready to sing (models load on first use, or now for those in `preload`).

* **Parameters:**
  * **repo_dir** – The SoulX-Singer checkout.
  * **weights** – `{"repo_id", "revision"}` of the SVS/SVC weights on the Hub.
  * **preprocess** – `{"repo_id", "revision"}` of the repo holding RMVPE.
  * **weights_dir** – Local folders with the same files, instead.
  * **preprocess_dir** – Local folders with the same files, instead.
  * **device** – ‘auto’ (MPS when available, else CPU), ‘cpu’, ‘mps’ or ‘cuda’.
  * **preload** – Components to load now (‘svs’, ‘svc’, ‘rmvpe’).

### voxy.workers.soulx_adapter.phonemize_en(word, g2p)

SoulX’s token for an English word: `en_` and its ARPAbet phones joined by ‘-‘.

### voxy.workers.soulx_adapter.sing(state, , meta, prompt_audio, prompt_meta, out_path, control='score', auto_shift=False, pitch_shift=0, n_steps=None, cfg=None, seed=None, peak=0.95, fade_s=0.02)

SVS: sing `meta` in the voice of `prompt_audio`, into `out_path`.

`control='score'` follows the notes; `'melody'` follows each segment’s
`f0`. `pitch_shift` transposes by that many semitones; `auto_shift`
(upstream’s median rule, recomputed for each segment) is best left off for a
score of several segments. Each segment is placed at its `time[0]`, with
`fade_s` fades where segments meet.

### voxy.workers.soulx_adapter.unload(state, , component)

Free one component’s weights; returns what stays loaded.
