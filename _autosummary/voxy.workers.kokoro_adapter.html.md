# voxy.workers.kokoro_adapter

Kokoro-82M through kokoro-onnx: stock voices on CPU, no torch, no reference clip.

The model files come from the Hugging Face hub (`hf_hub_download`, cached under
`$HF_HOME`, which voxy sets for every worker); which repo, revision and files is
the caller’s `init` (`voxy.kokoro_model.KOKORO`). onnxruntime ignores
`OMP_NUM_THREADS`, so its thread pool is capped here from `$VOXY_NUM_THREADS`.

### Functions

| `load`(\*, repo_id, revision, model_file, ...)       |    |
|------------------------------------------------------|----|
| `synthesize`(state, \*, text, voice, out_path, lang) |    |
