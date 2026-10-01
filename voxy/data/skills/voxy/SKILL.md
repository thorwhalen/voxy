---
name: voxy
description: Read first whenever spoken voice must be generated - "use voice X to...", "use a cloned voice for...", "have X narrate...", "say/read this in X's voice", voice-over, dubbing, giving characters voices, text-to-speech, which voices we have, cloning or designing a voice - and when making a video, animation, podcast or demo where a character or narrator speaks. voxy is the fleet's voice-generation facade; never call a TTS SDK directly.
---

# voxy: the fleet's voice-generation facade

One call speaks text in any voice, on any registered service. Cloning and designing voices are features on top, and named voices live in a library.

```python
import voxy

# a library voice (aliases work: "Coco")
voxy.text_to_speech("Hello!", voice="cora").save("hi.mp3")
# free local preview (macOS)
voxy.text_to_speech("Draft line", voice="Daniel", backend="say")
# our named voices: name, aliases, backends
voxy.list_voices()
# a service's own voices
voxy.list_voices("elevenlabs")
# provider id, for code that must call a provider itself
voxy.voice_id("cora")
```

CLI: `python -m voxy voices [--backend X]`, `python -m voxy speak "text" --voice cora -o out.mp3`.

## 1. Settle the voice first

- A person or one of "our" voices (Cora, Maman, ov...): `voxy.list_voices()`; use the name. Never paste a provider voice id into code.
- A service's stock voice: `voxy.list_voices("elevenlabs")` (or `say`, `aix`), then pass its id or name with `backend=`.
- Characters in a scene: give each character one voice and keep it. In `an`, a character's `voice_ref` points into the project's voices store; for a library voice, use `voxy.voice_id(name)` as its `voice_id`.
- Nothing specified: ask, or use the project's default narrator.
- A backend voice whose name is also a library alias: pass `use_library=False`.

## 2. Pick the backend (`voxy.speech_model_factories`)

| backend | use for | needs |
|---|---|---|
| `elevenlabs` | best quality; cloned and designed voices | `ELEVENLABS_API_KEY`, `voxy[elevenlabs]` |
| `say` | free, offline drafts and previews (macOS voices) | macOS |
| `aix` | OpenAI voices (nova, onyx...) and other LiteLLM providers | `voxy[aix]` + provider key |
| `fal` | fal.ai TTS models picked by quality tier | `voxy[fal]` + `FAL_KEY` |
| `csm` | local Sesame CSM-1B (GPU) | torch, the CSM repo |

Add a service with `voxy.register_speech_model("name", factory)`: a `SpeechModel` subclass implementing `synthesize(text, voice) -> Speech` (and optionally `list_voices`, `clone_voice`, `design_voice`). Synthesis costs credits: draft with `say`, render the final with the real voice.

## 3. New voices

- **Clone** a real person, only with their consent (a parent for a child), never a public figure or creator who hasn't agreed: put 1-2 min of clean, single-speaker audio in `voxy.samples_store(name)`, then `voxy.clone_from_samples(name, record_fields={"aliases": [...], "consent": "..."})`. Prepare samples first: separate the voice from music (Demucs), keep only the target speaker (VAD + speaker embeddings), at most 25 files. Show the user the sample summary before uploading.
- **Design** a voice from a description (no real person): `model = voxy.get_speech_model("elevenlabs")`, `previews = model.design_voice_previews(description, text=...)` (text: 100-1000 characters, or omit it), save each `preview.save(path)` for the user to listen to, then `voxy.design_from_description(name, description, preview=chosen)`.

## Where the data lives

`voxy.stores` (MutableMappings under `$VOXY_DATA_DIR`, default `~/.local/share/voxy`): `voices/{name}.json` records (aliases, consent, one saved profile per backend), `samples/{name}/`, `sources/{name}/`. Intermediates go to `~/.cache/voxy/`. Voice data is personal: never commit it.

## Not this skill

Music and singing: arioso, audiate. Sound effects: foley. Speech-to-text: scribed. Mixing narration with video or music: mixing, braidio.
