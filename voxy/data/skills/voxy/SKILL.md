---
name: voxy
description: Clone a voice and synthesize speech with the voxy facade, and keep the voice and its audio in voxy's data store. Use when asked to clone someone's voice, make a voice model, speak text in a cloned voice, reuse a voice made earlier, or prepare audio samples for a clone; and whenever deciding where voice samples, source media or voice ids should be saved. Triggers on "clone a voice", "voice clone", "ElevenLabs voice", "make a voice model of", "say this in X's voice", "voxy".
---

# voxy: voice cloning through one facade

Every clone goes through voxy, never a provider SDK called directly. A backend is one keyword: `create_speech_model("elevenlabs")` (instant voice cloning, API) or `"csm"` (local Sesame CSM-1B). New backends are entries in `voxy.speech_model_factories`.

## Where the data lives (fleet rule: a package's data goes in its store)

Nothing about a voice is saved by building paths. Use the stores in `voxy.stores`, all `MutableMapping`s rooted at `$VOXY_DATA_DIR` (default `~/.local/share/voxy`):

| store | layout | holds |
|---|---|---|
| `voices_store()` | `voices/{name}.json` | the voice record: aliases, consent, one saved profile per backend (the ElevenLabs `voice_id`) |
| `samples_store(name)` | `samples/{name}/` | the prepared audio the clone is made from |
| `sources_store(name)` | `sources/{name}/` | raw source media (downloads, videos) the samples were cut from |

Recomputable intermediates (resampled audio, embeddings, separated stems) are cache: `~/.cache/voxy/`. Session notes about the work go to `$PP/_agent_work/voxy/`. None of it is ever committed: voice data is personal.

## Clone, save, reuse

```python
from voxy import clone_from_samples, load_voice, create_speech_model

profile = clone_from_samples(
    "ada", record_fields={"aliases": ["Ada"], "consent": "..."}
)
# uploads every file in samples_store("ada"), saves the profile in voices_store()["ada"]

model = create_speech_model("elevenlabs")
model.generate_speech("Hello!", load_voice("ada"), output_path="hello.wav")
```

## Preparing samples (ElevenLabs instant cloning)

- Aim for 1-2 minutes of clean, single-speaker audio in total (at least ~30 s, at most ~3 min). How it is split across files does not matter; at most 25 files are accepted.
- Keep only the target speaker: no overlapping voices, music, or laughter-only stretches. Separate vocals from music first (e.g. Demucs), then use voice activity detection plus speaker embeddings to keep the stretches that match the target.
- Prefer consistent, natural speech over volume. `remove_background_noise=True` asks ElevenLabs to clean samples but can make clean ones worse.
- Cloning requires the right and the consent of the voice's owner (or their guardian). Record who consented in the voice record.
- Show the user the sample summary (count, total duration, a few clips to listen to) before uploading anything.
