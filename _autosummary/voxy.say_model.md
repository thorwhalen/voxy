# voxy.say_model

Local speech with the macOS `say` command: free, offline, no API key, no deps.

Good for previews and drafts before spending credits on a cloud voice. Voices are
the system’s (`say -v '?'`): e.g. “Samantha”, “Daniel”, “Karen”.

```pycon
>>> SaySpeechModel(run=lambda *a, **k: None).dflt_voice
'Samantha'
```

### Functions

| [`parse_say_voices`](#voxy.say_model.parse_say_voices)(listing)   | Parse `say -v '?'` output.   |
|------------------------------------------------------------------------------|------------------------------|

### Classes

| [`SaySpeechModel`](#voxy.say_model.SaySpeechModel)(\*[, voice, sample_rate, run])   | macOS `say` as a voxy backend.   |
|--------------------------------------------------------------------------------------------------|----------------------------------|

### *class* voxy.say_model.SaySpeechModel(, voice='Samantha', sample_rate=22050, run=None)

Bases: [`SpeechModel`](voxy.base.md#voxy.base.SpeechModel)

macOS `say` as a voxy backend.

* **Parameters:**
  * **voice** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Default system voice.
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Output WAV sample rate.
  * **run** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `subprocess.run`-like callable (tests inject a fake).

#### list_voices()

The voices this backend offers.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`VoiceInfo`](voxy.base.md#voxy.base.VoiceInfo)]

#### name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)* *= 'say'*

Registry name of the backend (also each profile’s `model_type`).

#### synthesize(text, voice=None, \*\*kwargs)

WAV speech from `say` (`voice`: a system voice name or profile).

* **Return type:**
  [`Speech`](voxy.base.md#voxy.base.Speech)

### voxy.say_model.parse_say_voices(listing)

Parse `say -v '?'` output.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`VoiceInfo`](voxy.base.md#voxy.base.VoiceInfo)]

```pycon
>>> [v.name for v in parse_say_voices("Albert              en_US    # Hello!\n"
...                                     "Eddy (English (US)) en_US    # Hi!\n")]
['Albert', 'Eddy (English (US))']
```
