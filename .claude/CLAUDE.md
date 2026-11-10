# voxy (dev map)

The fleet's voice-generation facade. `voxy/facade.py` (`text_to_speech`, `list_voices`, `voice_id`, voice resolution) over backends registered in `voxy/base.py` (`SpeechModel`, `Speech`, `speech_model_factories`, `register_speech_model`): `elevenlabs_model.py`, `say_model.py`, `aix_model.py`, `fal_model.py`, and CSM in `base.py`. Every backend takes an injectable client/command so tests never reach a service.

- **Data never lives in this repo.** Voices, samples and sources go through `voxy/stores.py` (`MutableMapping`s under `$VOXY_DATA_DIR`, default `~/.local/share/voxy/{voices,samples,sources}/`); intermediates go to `~/.cache/voxy/`. Voice data is personal: never commit it, never add it as a fixture.
- The voice library (`voxy/library.py`) saves one profile per backend in each voice record; use `clone_from_samples` / `load_voice` rather than re-cloning.
- Pronunciation: `voxy/speechtext.py` (display text → speech text, `flag_risks`, POS and LLM detectors) and `voxy/pronounce.py` (phoneme recogniser in its own env, `audit` of a finished render, `render_verified` while rendering); the shipped heteronym tables are `voxy/data/heteronyms/<lang>.json`. ASR checks cannot catch a heteronym misread; these do.
- Singing: `voxy/singing.py` (facade `sing`, `convert_singing`, singing voices, the `SingingBackend` registry), `voxy/score.py` (MIDI/MusicXML + lyrics to `Note`s, backend-agnostic), `voxy/soulx.py` (SoulX-Singer metadata + model; its code is fetched at a pinned commit by `envs.ensure_git_source`, since it has no pyproject), adapter `voxy/workers/soulx_adapter.py`. Test voices are synthetic (`voxy/formant_singer.py`), never a real singer. Real run: `VOXY_REAL_SOULX=1`.
- Usage skill (shipped): `voxy/data/skills/voxy/SKILL.md`, bridged into `.claude/skills/voxy`.
- Tests fake the ElevenLabs API via `client_factory=`; live tests are opt-in (`VOXY_LIVE_ELEVENLABS=1`). Tests that touch stores pass `rootdir=tmp_path`.
