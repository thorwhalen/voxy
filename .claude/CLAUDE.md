# voxy (dev map)

Facade for voice cloning and speech synthesis. Backends: `voxy/base.py` (`SpeechModel`, `VoiceProfile`, CSM, the `speech_model_factories` registry) and `voxy/elevenlabs_model.py`.

- **Data never lives in this repo.** Voices, samples and sources go through `voxy/stores.py` (`MutableMapping`s under `$VOXY_DATA_DIR`, default `~/.local/share/voxy/{voices,samples,sources}/`); intermediates go to `~/.cache/voxy/`. Voice data is personal: never commit it, never add it as a fixture.
- The voice library (`voxy/library.py`) saves one profile per backend in each voice record; use `clone_from_samples` / `load_voice` rather than re-cloning.
- Usage skill (shipped): `voxy/data/skills/voxy/SKILL.md`, bridged into `.claude/skills/voxy`.
- Tests fake the ElevenLabs API via `client_factory=`; live tests are opt-in (`VOXY_LIVE_ELEVENLABS=1`). Tests that touch stores pass `rootdir=tmp_path`.
