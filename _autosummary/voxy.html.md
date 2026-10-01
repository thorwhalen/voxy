# voxy

Facade for voice generation: speech synthesis in any voice, cloning and designing voices.

```pycon
>>> import voxy
>>> {'elevenlabs', 'say', 'aix', 'fal', 'csm'} <= set(voxy.speech_model_factories)
True
```

### Modules

| [`aix_model`](voxy.aix_model.html.md#module-voxy.aix_model)               | Speech through aix (LiteLLM): OpenAI's TTS voices and the other providers LiteLLM routes.   |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`base`](voxy.base.html.md#module-voxy.base)                         | Voxy: A flexible speech synthesis and voice cloning module.                                 |
| [`elevenlabs_model`](voxy.elevenlabs_model.html.md#module-voxy.elevenlabs_model) | ElevenLabs backend for voxy: instant voice cloning and synthesis through the API.           |
| [`facade`](voxy.facade.html.md#module-voxy.facade)                     | Voice generation in one call, whatever the service behind it.                               |
| [`fal_model`](voxy.fal_model.html.md#module-voxy.fal_model)               | Speech through fal.ai, via falaw: many hosted TTS models, chosen by quality tier.           |
| [`library`](voxy.library.html.md#module-voxy.library)                   | A persistent voice library: clone a voice once, then use it by name.                        |
| [`say_model`](voxy.say_model.html.md#module-voxy.say_model)               | Local speech with the macOS `say` command: free, offline, no API key, no deps.              |
| [`stores`](voxy.stores.html.md#module-voxy.stores)                     | Where voxy keeps its data: voice records and audio, as `MutableMapping` stores.             |
