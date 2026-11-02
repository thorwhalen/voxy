"""Kokoro-82M through kokoro-onnx: stock voices on CPU, no torch, no reference clip.

The model files are in ``$VOXY_MODELS_DIR`` (fetched there by ``voxy.envs.ensure_env``).
onnxruntime ignores ``OMP_NUM_THREADS``, so its thread pool is capped here from
``$VOXY_NUM_THREADS``.
"""

import os


def load(*, model_file, voices_file):
    import onnxruntime
    from kokoro_onnx import Kokoro

    folder = os.environ["VOXY_MODELS_DIR"]
    options = onnxruntime.SessionOptions()
    threads = os.environ.get("VOXY_NUM_THREADS")
    if threads:
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        os.path.join(folder, model_file),
        options,
        providers=["CPUExecutionProvider"],
    )
    return Kokoro.from_session(session, os.path.join(folder, voices_file))


def synthesize(state, *, text, voice, out_path, lang, speed=1.0):
    import soundfile

    samples, sample_rate = state.create(text, voice=voice, speed=speed, lang=lang)
    soundfile.write(out_path, samples, sample_rate, subtype="PCM_16")
    return {"sample_rate": sample_rate}


def voices(state):
    return {"voices": sorted(state.get_voices())}
