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

| [`ElevenLabsSpeechModel`](#voxy.elevenlabs_model.ElevenLabsSpeechModel)(\*[, api_key, ...])     | Speech model backed by the ElevenLabs API (Instant Voice Cloning + TTS).   |
|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`VoiceDesignPreview`](#voxy.elevenlabs_model.VoiceDesignPreview)(generated_voice_id, audio) | One candidate voice from `design_voice_previews`: listen, then pick.       |

### *class* voxy.elevenlabs_model.ElevenLabsSpeechModel(, api_key=None, model_id='eleven_multilingual_v2', output_format='pcm_24000', client_factory=None, device='cpu')

Bases: [`SpeechModel`](voxy.base.html.md#voxy.base.SpeechModel)

Speech model backed by the ElevenLabs API (Instant Voice Cloning + TTS).

* **Parameters:**
  * **api_key** (`str` | `None`) – ElevenLabs key; defaults to `ELEVEN_API_KEY` / `ELEVENLABS_API_KEY`.
  * **model_id** (`str`) – TTS model used by `generate_speech`.
  * **output_format** (`str`) – ElevenLabs output format. `pcm_*` and `wav_*` formats
    decode to a tensor with no extra dependency.
  * **client_factory** (`Callable`[[`str`], `Any`] | `None`) – `api_key -> client`. Defaults to the official SDK; tests
    inject a fake so nothing reaches the API.
  * **device** (`str`) – Device the returned tensors are placed on.

#### *property* client

The ElevenLabs client, built on first use.

#### clone_voice(audio_input, transcript=None, speaker_id=999, , cleanup_audio_fn=None, name=None, description=None, labels=None, remove_background_noise=None, assumed_sample_rate=16000, max_files=25)

Create an ElevenLabs instant voice clone from one or more audio samples.

ElevenLabs recommends about 1-2 minutes of clean, single-speaker audio in
total (at least 1, at most about 3); how it is split across files does
not matter.

* **Parameters:**
  * **audio_input** (`str` | `PathLike` | `bytes` | `BinaryIO` | `Tensor` | `ndarray` | `list`[`str` | `PathLike` | `bytes` | `BinaryIO` | `Tensor` | `ndarray`] | `tuple`[`str` | `PathLike` | `bytes` | `BinaryIO` | `Tensor` | `ndarray`, `...`]) – One audio input, or an iterable of them (paths, bytes,
    file-likes, tensors, arrays).
  * **transcript** (`str` | `None`) – Ignored: IVC needs no transcript. Kept so every voxy
    backend shares one `clone_voice` signature.
  * **speaker_id** (`int`) – Carried on the profile, and used in the default name.
  * **cleanup_audio_fn** (`Callable` | `None`) – Optional `(audio, sample_rate) -> audio` applied
    before upload. Off by default: ElevenLabs does its own processing,
    and voxy’s silence trimmer can chop soft speech.
  * **name** (`str` | `None`) – Voice name shown in ElevenLabs (default `voxy-<speaker_id>`).
  * **description** (`str` | `None`) – Voice description.
  * **labels** (`Mapping`[`str`, `str`] | `None`) – Voice labels (keys such as language, accent, gender, age).
  * **remove_background_noise** (`bool` | `None`) – Ask ElevenLabs to isolate the voice. Can
    make clean samples worse.
  * **assumed_sample_rate** (`int`) – Sample rate of raw tensor/array inputs.
  * **max_files** (`int` | `None`) – Refuse, before uploading anything, more samples than
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
  `None`

#### design_voice(description, , preview=None, name=None, labels=None, speaker_id=999, \*\*preview_kwargs)

Create a voice from a text `description` and return its profile.

* **Parameters:**
  * **description** (`str`) – What the voice sounds like (age, accent, tone, pace…).
  * **preview** ([`VoiceDesignPreview`](#voxy.elevenlabs_model.VoiceDesignPreview) | `str` | `None`) – The chosen preview (or its `generated_voice_id`) from
    `design_voice_previews`. If omitted, previews are generated
    and the first is used.
  * **name** (`str` | `None`) – Voice name in ElevenLabs (default `voxy-<speaker_id>`).
  * **labels** (`Mapping`[`str`, `str`] | `None`) – Voice labels (language, accent, gender, age…).
  * **\*\*preview_kwargs** – Passed to `design_voice_previews` when
    `preview` is omitted.
* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

#### design_voice_previews(description, , text=None, model_id='eleven_multilingual_ttv_v2', seed=None, guidance_scale=None, loudness=None, \*\*design_kwargs)

Candidate voices for a text `description` (ElevenLabs Voice Design).

`text` (100-1000 characters) is what the previews say; without it
ElevenLabs writes a fitting line. Nothing is added to the account until
one preview is passed to `design_voice`.

* **Return type:**
  `list`[[`VoiceDesignPreview`](#voxy.elevenlabs_model.VoiceDesignPreview)]

#### generate_speech(text, voice_profile=None, output_path=None, max_length_ms=10000, \*\*kwargs)

Synthesize `text` in a cloned (or stock) ElevenLabs voice.

* **Parameters:**
  * **text** (`str` | `bytes` | `TextIOBase`) – Text to synthesize.
  * **voice_profile** ([`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile) | `str` | `None`) – From `clone_voice` or `voice_profile`, or a bare
    voice id. Required: ElevenLabs has no voiceless default.
  * **output_path** (`str` | `None`) – If given, save the audio there (WAV for pcm/wav formats,
    the raw payload otherwise).
  * **max_length_ms** (`int`) – Ignored; ElevenLabs sizes the audio to the text.
  * **\*\*kwargs** – Passed to `synthesize_bytes`.
* **Return type:**
  `Tensor`
* **Returns:**
  A mono float tensor at the output format’s sample rate (that is
  `self.sample_rate` unless `output_format=` is passed here).

#### list_voices(, search=None, voice_type=None, page_size=100)

Voices in the account and its library (`voice_type`: e.g. ‘default’,
‘personal’, ‘cloned’, ‘generated’).

* **Return type:**
  `list`[[`VoiceInfo`](voxy.base.html.md#voxy.base.VoiceInfo)]

#### name *: str* *= 'elevenlabs'*

Registry name of the backend (also each profile’s `model_type`).

#### *property* sample_rate *: int*

Sample rate of the audio `generate_speech` returns.

#### synthesize(text, voice=None, , output_format='mp3_44100_128', \*\*kwargs)

Encoded speech (mp3 by default) in `voice` (a profile or voice id).

* **Return type:**
  [`Speech`](voxy.base.html.md#voxy.base.Speech)

#### synthesize_bytes(text, voice_profile, , model_id=None, output_format=None, \*\*convert_kwargs)

The raw audio ElevenLabs returns, in `output_format` (any format).

Extra keyword arguments (`voice_settings`, `seed`, `language_code`…)
go straight to `client.text_to_speech.convert`.

* **Return type:**
  `bytes`

#### voice_profile(voice_id, , speaker_id=999, metadata=None)

A profile for an existing ElevenLabs voice (a past clone, or a stock voice).

* **Return type:**
  [`VoiceProfile`](voxy.base.html.md#voxy.base.VoiceProfile)

```pycon
>>> model = ElevenLabsSpeechModel(api_key="unused")
>>> model.voice_profile("abc123").segment
'abc123'
```

### *class* voxy.elevenlabs_model.VoiceDesignPreview(generated_voice_id, audio, text='', duration_s=None)

Bases: `object`

One candidate voice from `design_voice_previews`: listen, then pick.

### voxy.elevenlabs_model.default_elevenlabs_client_factory(api_key)

Build the official SDK client (imported lazily).

### voxy.elevenlabs_model.resolve_elevenlabs_api_key(api_key=None, , envvars=('ELEVEN_API_KEY', 'ELEVENLABS_API_KEY'))

The explicit `api_key` if given, else the first set env var, else None.

* **Return type:**
  `str` | `None`

```pycon
>>> resolve_elevenlabs_api_key("explicit")
'explicit'
>>> resolve_elevenlabs_api_key(envvars=["VOXY_SURELY_UNSET_VAR"]) is None
True
```
