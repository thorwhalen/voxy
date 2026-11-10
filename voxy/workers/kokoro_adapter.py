"""Kokoro-82M through kokoro-onnx: stock voices on CPU, no torch, no reference clip.

The model files come from the Hugging Face hub (``hf_hub_download``, cached under
``$HF_HOME``, which voxy sets for every worker); which repo, revision and files is
the caller's ``init`` (``voxy.kokoro_model.KOKORO``). onnxruntime ignores
``OMP_NUM_THREADS``, so its thread pool is capped here from ``$VOXY_NUM_THREADS``.
"""


def load(*, repo_id, revision, model_file, voices_file):
    import onnxruntime
    from huggingface_hub import hf_hub_download
    from kokoro_onnx import Kokoro

    model_path, voices_path = (
        hf_hub_download(repo_id, filename, revision=revision)
        for filename in (model_file, voices_file)
    )
    from _worker_threads import thread_cap

    options = onnxruntime.SessionOptions()
    threads = thread_cap()
    if threads:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        model_path, options, providers=["CPUExecutionProvider"]
    )
    return Kokoro.from_session(session, voices_path)


def synthesize(state, *, text, voice, out_path, lang, speed=1.0):
    import soundfile

    samples, sample_rate = state.create(text, voice=voice, speed=speed, lang=lang)
    soundfile.write(out_path, samples, sample_rate, subtype="PCM_16")
    return {"sample_rate": sample_rate}
