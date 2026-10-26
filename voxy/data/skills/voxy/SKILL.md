---
name: voxy
description: Read first whenever spoken voice must be generated - "use voice X to...", "use a cloned voice for...", "have X narrate...", "say/read this in X's voice", voice-over, dubbing, giving characters voices, text-to-speech, which voices we have, cloning or designing a voice - and when making a video, animation, podcast or demo where a character or narrator speaks. voxy is the fleet's voice-generation facade; never call a TTS SDK directly.
---

# voxy: the fleet's voice-generation facade

One call speaks text in any voice, on any registered service. Cloning and designing voices are features on top, and named voices live in a library.

```python
import voxy

# a library voice (aliases work: "Addie")
voxy.text_to_speech("Hello!", voice="ada").save("hi.mp3")
# free local preview (macOS)
voxy.text_to_speech("Draft line", voice="Daniel", backend="say")
# our named voices: name, aliases, backends
voxy.list_voices()
# a service's own voices
voxy.list_voices("elevenlabs")
# provider id, for code that must call a provider itself
voxy.voice_id("ada")
```

CLI: `python -m voxy voices [--backend X]`, `python -m voxy speak "text" --voice ada -o out.mp3`.

## 1. Settle the voice first

- A person or one of "our" voices (Ada, Gran, ov...): `voxy.list_voices()`; use the name. Never paste a provider voice id into code.
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
| `voicestudio` | local; a voice is a text description (Parler-TTS), no cloning | `voxy[voicestudio]` + model weights (GBs) |
| `voxcpm`, `qwen3tts`, `chatterbox`, `f5tts`, `xtts` | local cloning from one reference clip + transcript, CPU OK (slow: RTF > 1); long text chunked | `uv`, `ffmpeg`; env + weights built on first use under `$VOXY_DATA_DIR` |
| `csm` | local Sesame CSM-1B (GPU) | torch, the CSM repo |

To choose between cloning backends on evidence (French and English accuracy, voice likeness judged by people, arenas, licence, price, watermark), read `docs/benchmarks.md` in the voxy repo; its data is `voxy/data/benchmarks.json` (`voxy.benchmarks.load_benchmarks()`), every number with its source. When you find a newer published result, add a row and run `python -m voxy benchmarks docs/benchmarks.md`.

To put music under a narration (an audiobook bed, ducked under speech and swelling between sections, or synced to phrases), make the music with `arioso` and read its note on scoring a narration (https://github.com/thorwhalen/arioso/blob/main/misc/docs/research/general/scoring-narration.md). Narrate one file per section with known durations so the gaps can be chosen to fit the music.

Add a service with `voxy.register_speech_model("name", factory)`: a `SpeechModel` subclass implementing `synthesize(text, voice) -> Speech` (and optionally `list_voices`, `clone_voice`, `design_voice`). Synthesis costs credits: draft with `say`, render the final with the real voice.

## 3. Prepare the text, then check what was said (heteronyms)

Speech models misread words that are spelled one way and said two ways, and an ASR (WER) check never catches it: Whisper writes "est" whether the model said /ɛ/ (verb) or /ɛst/ ("l'Est"). Measured on a Qwen3-TTS clone: a verse line starting "Est moins cité..." was read as the noun in 4 of 4 takes, and lowercased in 4 of 4 correctly; the capital, not the line break, is the trigger. ElevenLabs multilingual v2 takes no phonemes either (aliases only), so for every engine the text sent is the lever. Before rendering any verse, devotional text or long French text:

1. **Speech text, not display text:** `voxy.speechtext.prepare_speech_text(display, keep={...})` lowercases verse-initial capitals that do not start a sentence (keeping deliberate capitals such as a deference "Vous"), applies respellings and name aliases, and lists its changes. Send `.speech`; show `.display`.
2. **Flag what is left:** `voxy.speechtext.flag_risks(speech, heteronyms=voxy.pronounce.heteronym_readings("fr"), extra=[pos_detector()])` (the POS detector needs `voxy[pos]`; `llm_detector()` proposes readings for the rest). Words whose readings share a part of speech (*fils*, *plus*) need a rule or a person.
3. **Check the audio, not the transcript:** `voxy.pronounce.audit(speech, audio, expected=...)` finds each flagged word in the render and scores its readings with a phoneme recogniser; re-render the segments with `f.ok is False`. While rendering, `voxy.pronounce.render_verified` does this per line and retries. A respelling fixes a word that keeps failing (French *-ent* verbs written singular: *ils conte*).

Cross-lingual clones have the same problem: a French-reference clone speaking English said "son" as /sɔ̃/; a respelling (`respellings={"son": "sun"}`) fixed it.

## 4. New voices

- **Clone** a real person, only with their consent (a parent for a child), never a public figure or creator who hasn't agreed: put 1-2 min of clean, single-speaker audio in `voxy.samples_store(name)`, then `voxy.clone_from_samples(name, record_fields={"aliases": [...], "consent": "..."})`. Prepare samples first: separate the voice from music (Demucs), keep only the target speaker (VAD + speaker embeddings), at most 25 files. Show the user the sample summary before uploading.
- **Design** a voice from a description (no real person): `model = voxy.get_speech_model("elevenlabs")`, `previews = model.design_voice_previews(description, text=...)` (text: 100-1000 characters, or omit it), save each `preview.save(path)` for the user to listen to, then `voxy.design_from_description(name, description, preview=chosen)`.

- **Change a recording's voice** (speech to speech, ElevenLabs): `voxy.speech_to_speech(audio, voice=name)` keeps the recording's words, timing and intonation and gives it the voice. Use it to correct a word in a narration: someone says the corrected phrase with the narrator's delivery, it is converted to the narrator's voice, and spliced in. Replace the whole coarticulated phrase, not one word: a surname spliced alone was heard with the wrong initial consonant because the preceding first name anticipated the old surname's lips; replacing first name plus surname worked. Check the splice with a transcription and a voice print (`voxy.speakers`). A `replace_words` helper is tracked in thorwhalen/voxy#43.
- **Local clone** (no service, e.g. a voice that can't be enrolled with ElevenLabs): `voxy.refprep.prepare_reference(recordings, voice=name)` cleans, transcribes and scores clips (add `speaker=voxy.speakers.speaker_filter()` when other voices are in the recording); `voxy.local_clone.reference_profile(clip_path, clip_text)` is the voice; compare models and parameters with a diagnosis page (`python -m voxy diagnose spec.json --out DIR --publish`) before rendering long text. F5-TTS and XTTS weights are non-commercial.

## Where the data lives

`voxy.stores` (MutableMappings under `$VOXY_DATA_DIR`, default `~/.local/share/voxy`): `voices/{name}.json` records (aliases, consent, one saved profile per backend), `samples/{name}/`, `sources/{name}/`. Intermediates go to `~/.cache/voxy/`. Voice data is personal: never commit it.

## Not this skill

Music and singing: arioso, audiate. Sound effects: foley. Speech-to-text: scribed. Mixing narration with video or music: mixing, braidio.
