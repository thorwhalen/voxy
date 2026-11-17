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

| [`KEY_VERSION`](#voxy.speech_cache.KEY_VERSION)       | Bump when the meaning of a key changes, so old entries stop matching.   |
|--------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`MIN_CHARS_PER_S`](#voxy.speech_cache.MIN_CHARS_PER_S)   | Speaking-rate bounds (characters per second) for a take of a long text. |
| [`MAX_CHARS_PER_S`](#voxy.speech_cache.MAX_CHARS_PER_S)   | Speaking-rate bounds (characters per second) for a take of a long text. |
| [`MIN_CHECKED_CHARS`](#voxy.speech_cache.MIN_CHECKED_CHARS) | Speaking-rate bounds (characters per second) for a take of a long text. |

### Functions

| [`cached_synthesizer`](#voxy.speech_cache.cached_synthesizer)(model, cache, \*[, ...])    | `model.synthesize`, served from `cache` when the same request was made before.                                                                                                 |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`default_speech_cache_dir`](#voxy.speech_cache.default_speech_cache_dir)()                     | `$VOXY_SPEECH_CACHE_DIR`, else `$XDG_CACHE_HOME/voxy/speech`, else `~/.cache/voxy/speech`.                                                                                     |
| [`resolve_cache`](#voxy.speech_cache.resolve_cache)(cache)                           | `None`/`False`: no cache; `True`: the default folder; a path: that folder; a mapping: used as is.                                                                              |
| [`speaking_rate_check`](#voxy.speech_cache.speaking_rate_check)(speech, text, \*[, ...])   | A problem description if `speech` cannot be `text` spoken, else `None`.                                                                                                        |
| [`speech_key`](#voxy.speech_cache.speech_key)(model, text, voice, options)        | The cache key: a SHA-256 over backend, model settings, voice, text and options.                                                                                                |
| [`validated`](#voxy.speech_cache.validated)(synthesize, validate, \*[, retries]) | `synthesize` that re-asks (up to `retries` times) while `validate(speech, text)` reports a problem, then raises [`TakeRejected`](#voxy.speech_cache.TakeRejected). |

### Classes

| [`SpeechCache`](#voxy.speech_cache.SpeechCache)([root])   | Speech on disk: `<key>.<format>` audio plus `<key>.json` (everything else).   |
|------------------------------------------------------------------------|-------------------------------------------------------------------------------|

### Exceptions

| [`TakeRejected`](#voxy.speech_cache.TakeRejected)   | An engine kept returning takes that failed validation (never cached).   |
|-----------------------------------------------------------------|-------------------------------------------------------------------------|

### voxy.speech_cache.KEY_VERSION *= 1*

Bump when the meaning of a key changes, so old entries stop matching.

### voxy.speech_cache.MAX_CHARS_PER_S *= 30.0*

Speaking-rate bounds (characters per second) for a take of a long text. Speech
runs about 12-17 chars/s; a take far slower is noise, breath or silence (seen: a
177-character request answered with 29 s of breathing), far faster is a truncation.

### voxy.speech_cache.MIN_CHARS_PER_S *= 7.0*

Speaking-rate bounds (characters per second) for a take of a long text. Speech
runs about 12-17 chars/s; a take far slower is noise, breath or silence (seen: a
177-character request answered with 29 s of breathing), far faster is a truncation.

### voxy.speech_cache.MIN_CHECKED_CHARS *= 60*

Speaking-rate bounds (characters per second) for a take of a long text. Speech
runs about 12-17 chars/s; a take far slower is noise, breath or silence (seen: a
177-character request answered with 29 s of breathing), far faster is a truncation.

### *class* voxy.speech_cache.SpeechCache(root=None)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

Speech on disk: `<key>.<format>` audio plus `<key>.json` (everything else).

### *exception* voxy.speech_cache.TakeRejected

Bases: [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

An engine kept returning takes that failed validation (never cached).

### voxy.speech_cache.cached_synthesizer(model, cache, , validate=None, retries=2)

`model.synthesize`, served from `cache` when the same request was made before.

A served result has `cached=True`, so callers can tell spend from reuse.
`validate(speech, text) -> problem | None` (e.g. [`speaking_rate_check()`](#voxy.speech_cache.speaking_rate_check))
runs on every new take before it is cached: a bad take is re-asked up to
`retries` times and never stored, so a cache cannot freeze a broken take.

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

### voxy.speech_cache.speaking_rate_check(speech, text, , min_chars_per_s=7.0, max_chars_per_s=30.0, min_chars=60)

A problem description if `speech` cannot be `text` spoken, else `None`.

Only texts of `min_chars` or more are judged (a short line’s lead-in and
tail dominate its duration). Needs no model: the duration comes from the audio.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### voxy.speech_cache.speech_key(model, text, voice, options)

The cache key: a SHA-256 over backend, model settings, voice, text and options.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.speech_cache.validated(synthesize, validate, , retries=2)

`synthesize` that re-asks (up to `retries` times) while `validate(speech, text)`
reports a problem, then raises [`TakeRejected`](#voxy.speech_cache.TakeRejected).

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Speech`](voxy.base.html.md#voxy.base.Speech)]
