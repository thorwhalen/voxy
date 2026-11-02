# voxy.kokoro_model

Kokoro: a neutral local voice that needs no reference clip, no key and no GPU.

Kokoro-82M (weights Apache-2.0) through `kokoro-onnx` (MIT) speaks in 54 stock
voices (`af_heart`, `am_michael`, `bf_emma`, …) and runs faster than
realtime on a CPU. It runs in its own small environment, like the local cloners
(`voxy.local_clone`): `python -m voxy setup kokoro` builds it, and the model
files (about 354 MB) are fetched from the Hugging Face hub on first use into
`$VOXY_DATA_DIR/models/hf`, like every other local backend’s weights.

```pycon
>>> import voxy
>>> voxy.text_to_speech("Hello.", "bf_emma", backend="kokoro").save("hi.wav")
```

A voice’s first letter is its language (`a` American English, `b` British,
`f` French, …) and its second its gender; `lang` follows the voice unless
you pass one. The voice list is static (`KOKORO_VOICES`), so listing voices
needs no model.

On macOS the espeak-ng bundled with kokoro-onnx (espeakng-loader) has its data
folder compiled in and cannot find it, so the first synthesis fails; use `say`
there (thorwhalen/voxy#61).

```pycon
>>> import numpy as np
>>> fake = lambda text, **kw: (np.full(800, 0.1, dtype=np.float32), 24000)
>>> speech = KokoroSpeechModel(generate=fake).synthesize("One. Two.", "bm_george")
>>> speech.backend, speech.voice, speech.sample_rate
('kokoro', 'bm_george', 24000)
```

### Module Attributes

| [`KOKORO_VOICES`](#voxy.kokoro_model.KOKORO_VOICES)    | The voices in `voices-v1.0.bin`.                                       |
|-------------------------------------------------------------------|------------------------------------------------------------------------|
| [`KOKORO_LANGUAGES`](#voxy.kokoro_model.KOKORO_LANGUAGES) | A voice name's first letter -> the language kokoro-onnx phonemizes in. |
| [`KOKORO_GENDERS`](#voxy.kokoro_model.KOKORO_GENDERS)   | A voice name's second letter -> the voice's gender.                    |

### Functions

| [`parse_voice`](#voxy.kokoro_model.parse_voice)(voice)   | `(language, gender)` of a stock voice, from its name's first two letters.   |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`voice_info`](#voxy.kokoro_model.voice_info)(voice)    | What a stock voice's name says about it.                                    |

### Classes

| [`KokoroSpeechModel`](#voxy.kokoro_model.KokoroSpeechModel)(\*[, voice, voices, ...])   | Kokoro-82M as a voxy backend (`params`: speed; lang, else the voice's).   |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|

### voxy.kokoro_model.KOKORO_GENDERS *= {'f': 'female', 'm': 'male'}*

A voice name’s second letter -> the voice’s gender.

### voxy.kokoro_model.KOKORO_LANGUAGES *= {'a': 'en-us', 'b': 'en-gb', 'e': 'es', 'f': 'fr-fr', 'h': 'hi', 'i': 'it', 'j': 'ja', 'p': 'pt-br', 'z': 'cmn'}*

A voice name’s first letter -> the language kokoro-onnx phonemizes in.

### voxy.kokoro_model.KOKORO_VOICES *= ('af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora', 'em_alex', 'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara', 'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'pf_dora', 'pm_alex', 'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang')*

The voices in `voices-v1.0.bin`.

### *class* voxy.kokoro_model.KokoroSpeechModel(, voice=None, voices=('af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora', 'em_alex', 'em_santa', 'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara', 'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo', 'pf_dora', 'pm_alex', 'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang'), generate=None, device='cpu', \*\*kwargs)

Bases: [`LocalSpeechModel`](voxy.local_clone.html.md#voxy.local_clone.LocalSpeechModel)

Kokoro-82M as a voxy backend (`params`: speed; lang, else the voice’s).

* **Parameters:**
  * **voice** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Default stock voice (else `$VOXY_KOKORO_VOICE`, else ‘af_heart’).
  * **voices** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The voices the model accepts (default `KOKORO_VOICES`). For a
    newer voices file, also pass `init={'voices_file': ..., 'revision': ...}`;
    names must start with a known language letter (`KOKORO_LANGUAGES`).
  * **generate** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – The model call (default: the worker, see `LocalSpeechModel`).
  * **device** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Only ‘cpu’ (kokoro-onnx runs on onnxruntime’s CPU provider).
  * **\*\*kwargs** – The other `LocalSpeechModel` options (threads, rootdir, …).

#### list_voices()

The stock voices (no model needed).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`VoiceInfo`](voxy.base.html.md#voxy.base.VoiceInfo)]

#### synthesize(text, voice=None, , lang=None, \*\*params)

Speak `text` in a stock `voice`, in `lang` (default: the voice’s).

* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

### voxy.kokoro_model.parse_voice(voice)

`(language, gender)` of a stock voice, from its name’s first two letters.

```pycon
>>> parse_voice("bf_emma")
('en-gb', 'female')
```

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.kokoro_model.voice_info(voice)

What a stock voice’s name says about it.

```pycon
>>> voice_info("am_michael").labels
{'language': 'en-us', 'gender': 'male'}
```

* **Return type:**
  [`VoiceInfo`](voxy.base.html.md#voxy.base.VoiceInfo)
