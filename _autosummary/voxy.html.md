# voxy

Facade for voice generation: speech synthesis in any voice, cloning and designing voices.

```pycon
>>> import voxy
>>> {'elevenlabs', 'say', 'aix', 'fal', 'voicestudio', 'csm'} <= set(voxy.speech_model_factories)
True
```

### Modules

| [`aix_model`](voxy.aix_model.html.md#module-voxy.aix_model)                 | Speech through aix (LiteLLM): OpenAI's TTS voices and the other providers LiteLLM routes.     |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| [`base`](voxy.base.html.md#module-voxy.base)                           | Voxy: A flexible speech synthesis and voice cloning module.                                   |
| [`benchmarks`](voxy.benchmarks.html.md#module-voxy.benchmarks)               | Published benchmarks of the cloning backends, kept as data, rendered as tables and charts.    |
| [`diagnosis`](voxy.diagnosis.html.md#module-voxy.diagnosis)                 | Diagnosis pages: render one text through a grid of (model x parameters), listen side by side. |
| [`elevenlabs_model`](voxy.elevenlabs_model.html.md#module-voxy.elevenlabs_model)   | ElevenLabs backend for voxy: instant voice cloning and synthesis through the API.             |
| [`envs`](voxy.envs.html.md#module-voxy.envs)                           | Isolated Python environments for backends whose dependencies clash with yours.                |
| [`facade`](voxy.facade.html.md#module-voxy.facade)                       | Voice generation in one call, whatever the service behind it.                                 |
| [`fal_model`](voxy.fal_model.html.md#module-voxy.fal_model)                 | Speech through fal.ai, via falaw: many hosted TTS models, chosen by quality tier.             |
| [`finetune`](voxy.finetune.html.md#module-voxy.finetune)                   | Fine-tuning datasets from prepared clips: one export per training recipe.                     |
| [`library`](voxy.library.html.md#module-voxy.library)                     | A persistent voice library: clone a voice once, then use it by name.                          |
| [`local_clone`](voxy.local_clone.html.md#module-voxy.local_clone)             | Local voice cloning: speak any text in a voice learned from a short reference clip.           |
| [`refprep`](voxy.refprep.html.md#module-voxy.refprep)                     | Prepare a speaker's recordings for cloning and fine-tuning: clean, cut, transcribe, score.    |
| [`say_model`](voxy.say_model.html.md#module-voxy.say_model)                 | Local speech with the macOS `say` command: free, offline, no API key, no deps.                |
| [`stores`](voxy.stores.html.md#module-voxy.stores)                       | Where voxy keeps its data: voice records and audio, as `MutableMapping` stores.               |
| [`textsplit`](voxy.textsplit.html.md#module-voxy.textsplit)                 | Split long text (a chapter) into chunks a speech model can say in one breath.                 |
| [`voicestudio_model`](voxy.voicestudio_model.html.md#module-voxy.voicestudio_model) | Local description-prompted speech through VoiceStudio (Parler-TTS and friends).               |
| [`wavutil`](voxy.wavutil.html.md#module-voxy.wavutil)                     | Mono PCM WAV in and out as float arrays, plus the few edits narration needs.                  |
| [`worker`](voxy.worker.html.md#module-voxy.worker)                       | Talk to a model running in another Python environment, as if it were local.                   |
| [`workers`](voxy.workers.html.md#module-voxy.workers)                     | Backend adapters run by `voxy.worker` inside each backend's own environment.                  |
