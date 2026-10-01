# voxy.elevenlabs_model

ElevenLabs backend for voxy: instant voice cloning and synthesis through the API.

Select it with `create_speech_model("elevenlabs")`. Cloning uploads one or more
audio samples to ElevenLabs’ Instant Voice Cloning (IVC) endpoint and returns a
`VoiceProfile` whose `segment` is the new `voice_id`; synthesis renders text
in that voice. ElevenLabs keeps the voice, so a later session can rebuild the
profile from the id alone with `ElevenLabsSpeechModel.voice_profile(voice_id)`.

The key is read from `api_key=` or, failing that, the `ELEVEN_API_KEY` /
`ELEVENLABS_API_KEY` environment variables (the same order the fleet’s other
ElevenLabs clients use). The `elevenlabs` SDK is imported lazily, so it is only
needed when a request is actually made (`pip install voxy[elevenlabs]`).

ElevenLabs requires that whoever clones a voice has the right and the consent of
the voice’s owner (or their guardian) to do so.

```pycon
>>> model = ElevenLabsSpeechModel(api_key="unused", client_factory=lambda key: None)
>>> model.sample_rate
24000
```

### Functions

| [`default_elevenlabs_client_factory`](#voxy.elevenlabs_model.default_elevenlabs_client_factory)(api_key)     | Build the official SDK client (imported lazily).                        |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`resolve_elevenlabs_api_key`](#voxy.elevenlabs_model.resolve_elevenlabs_api_key)([api_key, envvars]) | The explicit `api_key` if given, else the first set env var, else None. |

### Classes

| [`ElevenLabsSpeechModel`](#voxy.elevenlabs_model.ElevenLabsSpeechModel)(\*[, api_key, ...])   | Speech model backed by the ElevenLabs API (Instant Voice Cloning + TTS).   |
|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|

### *class* voxy.elevenlabs_model.ElevenLabsSpeechModel(, api_key=None, model_id='eleven_multilingual_v2', output_format='pcm_24000', client_factory=None, device='cpu')

Bases: [`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel)

Speech model backed by the ElevenLabs API (Instant Voice Cloning + TTS).

* **Parameters:**
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – ElevenLabs key; defaults to `ELEVEN_API_KEY` / `ELEVENLABS_API_KEY`.
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS model used by `generate_speech`.
  * **output_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ElevenLabs output format. `pcm_*` and `wav_*` formats
    decode to a tensor with no extra dependency.
  * **client_factory** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – `api_key -> client`. Defaults to the official SDK; tests
    inject a fake so nothing reaches the API.
  * **device** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Device the returned tensors are placed on.

#### *property* client

The ElevenLabs client, built on first use.

#### clone_voice(audio_input, transcript=None, speaker_id=999, , cleanup_audio_fn=None, name=None, description=None, labels=None, remove_background_noise=None, assumed_sample_rate=16000, max_files=25)

Create an ElevenLabs instant voice clone from one or more audio samples.

ElevenLabs recommends about 1-2 minutes of clean, single-speaker audio in
total (at least 1, at most about 3); how it is split across files does
not matter.

* **Parameters:**
  * **audio_input** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO) | `Tensor` | `ndarray` | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO) | `Tensor` | `ndarray`] | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO) | `Tensor` | `ndarray`, [`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis)]) – One audio input, or an iterable of them (paths, bytes,
    file-likes, tensors, arrays).
  * **transcript** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ignored: IVC needs no transcript. Kept so every voxy
    backend shares one `clone_voice` signature.
  * **speaker_id** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Carried on the profile, and used in the default name.
  * **cleanup_audio_fn** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional `(audio, sample_rate) -> audio` applied
    before upload. Off by default: ElevenLabs does its own processing,
    and voxy’s silence trimmer can chop soft speech.
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice name shown in ElevenLabs (default `voxy-<speaker_id>`).
  * **description** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice description.
  * **labels** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Voice labels (keys such as language, accent, gender, age).
  * **remove_background_noise** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Ask ElevenLabs to isolate the voice. Can
    make clean samples worse.
  * **assumed_sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Sample rate of raw tensor/array inputs.
  * **max_files** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Refuse, before uploading anything, more samples than
    ElevenLabs accepts per voice (`None` to skip the check).
    Concatenate short clips to stay under it.
* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)
* **Returns:**
  A `VoiceProfile` whose `segment` (and `metadata['voice_id']`)
  is the new ElevenLabs voice id.

#### delete_voice(voice)

Delete a voice from the ElevenLabs account.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### generate_speech(text, voice_profile=None, output_path=None, max_length_ms=10000, \*\*kwargs)

Synthesize `text` in a cloned (or stock) ElevenLabs voice.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes) | [`TextIOBase`](https://docs.python.org/3/library/io.html#io.TextIOBase)) – Text to synthesize.
  * **voice_profile** ([`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – From `clone_voice` or `voice_profile`, or a bare
    voice id. Required: ElevenLabs has no voiceless default.
  * **output_path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – If given, save the audio there (WAV for pcm/wav formats,
    the raw payload otherwise).
  * **max_length_ms** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Ignored; ElevenLabs sizes the audio to the text.
  * **\*\*kwargs** – Passed to `synthesize_bytes`.
* **Return type:**
  `Tensor`
* **Returns:**
  A mono float tensor at the output format’s sample rate (that is
  `self.sample_rate` unless `output_format=` is passed here).

#### *property* sample_rate *: [int](https://docs.python.org/3/builtins/functions.html#int)*

Sample rate of the audio `generate_speech` returns.

#### synthesize_bytes(text, voice_profile, , model_id=None, output_format=None, \*\*convert_kwargs)

The raw audio ElevenLabs returns, in `output_format` (any format).

Extra keyword arguments (`voice_settings`, `seed`, `language_code`…)
go straight to `client.text_to_speech.convert`.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)

#### voice_profile(voice_id, , speaker_id=999, metadata=None)

A profile for an existing ElevenLabs voice (a past clone, or a stock voice).

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

```pycon
>>> model = ElevenLabsSpeechModel(api_key="unused")
>>> model.voice_profile("abc123").segment
'abc123'
```

### voxy.elevenlabs_model.default_elevenlabs_client_factory(api_key)

Build the official SDK client (imported lazily).

### voxy.elevenlabs_model.resolve_elevenlabs_api_key(api_key=None, , envvars=('ELEVEN_API_KEY', 'ELEVENLABS_API_KEY'))

The explicit `api_key` if given, else the first set env var, else None.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> resolve_elevenlabs_api_key("explicit")
'explicit'
>>> resolve_elevenlabs_api_key(envvars=["VOXY_SURELY_UNSET_VAR"]) is None
True
```
