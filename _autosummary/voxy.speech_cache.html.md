# voxy.speech_cache

A content-addressed cache of synthesized speech, so re-rendering unchanged text is free.

The key covers everything that reaches the audio: the backend, its model settings, the
voice, the text and the synthesis options. A store is any `MutableMapping` from key to
[`Speech`](voxy.base.html.md#voxy.base.Speech); [`SpeechCache`](#voxy.speech_cache.SpeechCache) is the default, a folder of audio files
with a small JSON record each.

```pycon
>>> import tempfile
>>> cache = SpeechCache(tempfile.mkdtemp())
>>> calls = []
>>> class Fake:
...     name, model_id = "fake", "m1"
...     def synthesize(self, text, voice=None, **kw):
...         from voxy.base import Speech
...         calls.append(text)
...         return Speech(text.encode(), format="wav", backend="fake", voice=voice, text=text)
>>> synth = cached_synthesizer(Fake(), cache)
>>> synth("hi", "v").cached, synth("hi", "v").cached, len(calls)
(False, True, 1)
```

### Module Attributes

| [`KEY_VERSION`](#voxy.speech_cache.KEY_VERSION)   | Bump when the meaning of a key changes, so old entries stop matching.   |
|----------------------------------------------------------------|-------------------------------------------------------------------------|

### Functions

| [`cached_synthesizer`](#voxy.speech_cache.cached_synthesizer)(model, cache)        | `model.synthesize`, served from `cache` when the same request was made before.                    |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [`default_speech_cache_dir`](#voxy.speech_cache.default_speech_cache_dir)()              | `$VOXY_SPEECH_CACHE_DIR`, else `$XDG_CACHE_HOME/voxy/speech`, else `~/.cache/voxy/speech`.        |
| [`resolve_cache`](#voxy.speech_cache.resolve_cache)(cache)                    | `None`/`False`: no cache; `True`: the default folder; a path: that folder; a mapping: used as is. |
| [`speech_key`](#voxy.speech_cache.speech_key)(model, text, voice, options) | The cache key: a SHA-256 over backend, model settings, voice, text and options.                   |

### Classes

| [`SpeechCache`](#voxy.speech_cache.SpeechCache)([root])   | Speech on disk: `<key>.<format>` audio plus `<key>.json` (everything else).   |
|------------------------------------------------------------------------|-------------------------------------------------------------------------------|

### voxy.speech_cache.KEY_VERSION *= 1*

Bump when the meaning of a key changes, so old entries stop matching.

### *class* voxy.speech_cache.SpeechCache(root=None)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

Speech on disk: `<key>.<format>` audio plus `<key>.json` (everything else).

### voxy.speech_cache.cached_synthesizer(model, cache)

`model.synthesize`, served from `cache` when the same request was made before.

A served result has `cached=True`, so callers can tell spend from reuse.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Speech`](voxy.base.html.md#voxy.base.Speech)]

### voxy.speech_cache.default_speech_cache_dir()

`$VOXY_SPEECH_CACHE_DIR`, else `$XDG_CACHE_HOME/voxy/speech`, else `~/.cache/voxy/speech`.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.speech_cache.resolve_cache(cache)

`None`/`False`: no cache; `True`: the default folder; a path: that folder;
a mapping: used as is.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.speech_cache.speech_key(model, text, voice, options)

The cache key: a SHA-256 over backend, model settings, voice, text and options.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
