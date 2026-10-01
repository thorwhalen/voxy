# voxy (dev map)

The fleet's voice-generation facade. `voxy/facade.py` (`text_to_speech`, `list_voices`, `voice_id`, voice resolution) over backends registered in `voxy/base.py` (`SpeechModel`, `Speech`, `speech_model_factories`, `register_speech_model`): `elevenlabs_model.py`, `say_model.py`, `aix_model.py`, `fal_model.py`, and CSM in `base.py`. Every backend takes an injectable client/command so tests never reach a service.

- **Data never lives in this repo.** Voices, samples and sources go through `voxy/stores.py` (`MutableMapping`s under `$VOXY_DATA_DIR`, default `~/.local/share/voxy/{voices,samples,sources}/`); intermediates go to `~/.cache/voxy/`. Voice data is personal: never commit it, never add it as a fixture.
- The voice library (`voxy/library.py`) saves one profile per backend in each voice record; use `clone_from_samples` / `load_voice` rather than re-cloning.
- Usage skill (shipped): `voxy/data/skills/voxy/SKILL.md`, bridged into `.claude/skills/voxy`.
- Tests fake the ElevenLabs API via `client_factory=`; live tests are opt-in (`VOXY_LIVE_ELEVENLABS=1`). Tests that touch stores pass `rootdir=tmp_path`.
