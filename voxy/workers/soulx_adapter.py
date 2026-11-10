"""SoulX-Singer (Soul AI Lab; code and weights Apache-2.0): zero-shot singing voice
synthesis (notes and lyrics to a sung vocal) and singing voice conversion.

Runs in the ``soulx`` environment, with no voxy imports. SoulX-Singer is not
pip-installable, so ``load`` puts its checkout (``repo_dir``, fetched by voxy at a
pinned commit) on ``sys.path``. Weights come from the Hugging Face Hub at pinned
revisions (into ``HF_HOME``), unless ``weights_dir`` / ``preprocess_dir`` point at
local copies.

Operations:

- ``sing(state, *, meta, prompt_audio, prompt_meta, out_path, ...)``: SVS. ``meta``
  and ``prompt_meta`` are SoulX metadata (lists of segments); ``prompt_audio`` is a
  sung clip whose notes and words ``prompt_meta`` describes. Empty ``phoneme``
  fields are filled here (English, with g2p_en).
- ``convert(state, *, audio, prompt_audio, out_path, ...)``: SVC, audio to audio;
  F0 is extracted with RMVPE.
- ``f0(state, *, audio, out_path=None)``: the RMVPE F0 of a clip (50 frames a second).
- ``unload(state, *, component)``: free one model ('svs', 'svc' or 'rmvpe').

The SVS and SVC checkpoints are 2.8 GB each; each loads on first use (``preload``
loads them at start). On Apple Silicon the models run on MPS except the vocoder,
which runs on the CPU: its ISTFT head multiplies complex tensors, which torch 2.2's
MPS backend rejects (an internal assert in BinaryKernel.mm). Output is written as
16-bit WAV, scaled down if it peaks above ``peak``.

Usage disclaimer (upstream README): do not impersonate anyone without authorisation
or make deceptive audio; respect intellectual property, privacy and consent.
"""

import os
import sys
import time

SVS_CKPT, SVC_CKPT = "model.pt", "model-svc.pt"
RMVPE_CKPT = "rmvpe/rmvpe.pt"
CONFIG = os.path.join("soulxsinger", "config", "soulxsinger.yaml")
PHONESET = os.path.join("soulxsinger", "utils", "phoneme", "phone_set.json")
F0_EXTRACTION = os.path.join("preprocess", "tools", "f0_extraction.py")
SP = "<SP>"
#: Joins the words sung on one note (an elision) in a ``text`` token; voxy.soulx
#: writes it (its ``WORD_JOINER``).
WORD_JOINER = "+"
DFLT_PEAK = 0.95
#: Fade-out and fade-in length where segments meet (they are rendered separately
#: and added, not overlapped).
DFLT_FADE_S = 0.02
#: The two singers; loading one frees the other (2.8 GB each, about 6 GB in use).
SINGERS = ("svs", "svc")
#: Per-note fields of a segment, which must all have one token per note.
NOTE_FIELDS = ("duration", "text", "phoneme", "note_pitch", "note_type")
#: g2p_en's NLTK resources (nltk 3.9 tags with the ``_eng`` tagger).
NLTK_RESOURCES = {
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    "corpora/cmudict": "cmudict",
}


def _device(device):
    import torch

    if device in (None, "auto"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return device


def load(
    *,
    repo_dir,
    weights=None,
    preprocess=None,
    weights_dir=None,
    preprocess_dir=None,
    device="auto",
    preload=(),
):
    """Get ready to sing (models load on first use, or now for those in ``preload``).

    Args:
        repo_dir: The SoulX-Singer checkout.
        weights: ``{"repo_id", "revision"}`` of the SVS/SVC weights on the Hub.
        preprocess: ``{"repo_id", "revision"}`` of the repo holding RMVPE.
        weights_dir, preprocess_dir: Local folders with the same files, instead.
        device: 'auto' (MPS when available, else CPU), 'cpu', 'mps' or 'cuda'.
        preload: Components to load now ('svs', 'svc', 'rmvpe').
    """
    from _worker_threads import cap_torch_threads

    cap_torch_threads()
    if repo_dir not in sys.path:
        sys.path.append(repo_dir)  # after the environment's packages: no shadowing
    from soulxsinger.utils.file_utils import load_config

    state = {
        "repo_dir": repo_dir,
        "sources": {
            "weights": (weights_dir, weights),
            "preprocess": (preprocess_dir, preprocess),
        },
        "device": _device(device),
        "config": load_config(os.path.join(repo_dir, CONFIG)),
        "models": {},
    }
    for component in preload:
        _model(state, component)
    return state


def _weight(state, source, filename):
    """The local path of ``filename`` from ``source`` ('weights' or 'preprocess')."""
    folder, hub = state["sources"][source]
    if folder:
        return os.path.join(folder, filename)
    if not hub:
        raise ValueError(f"No {source} folder or Hub repo given for {filename}")
    return _download(hub["repo_id"], filename, hub.get("revision"))


def _download(repo_id, filename, revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, revision=revision)


def _load_checkpoint(path):
    import torch

    return torch.load(path, weights_only=False, map_location="cpu")


def _model(state, component):
    """The ``component`` model, loaded on first use."""
    if component in state["models"]:
        return state["models"][component]
    if component == "rmvpe":
        model = _f0_extractor(state)
    elif component in ("svs", "svc"):
        model = _singer(state, component)
    else:
        raise ValueError(f"Unknown component {component!r}: 'svs', 'svc' or 'rmvpe'")
    state["models"][component] = model
    return model


def _singer(state, component):
    for other in SINGERS:
        if other != component and other in state["models"]:
            unload(state, component=other)
    if component == "svs":
        from soulxsinger.models.soulxsinger import SoulXSinger as cls

        ckpt = SVS_CKPT
    else:
        from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC as cls

        ckpt = SVC_CKPT
    model = cls(state["config"])
    checkpoint = _load_checkpoint(_weight(state, "weights", ckpt))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.eval().to(state["device"])
    if state["device"] == "mps":
        _vocoder_on_cpu(model, state["device"])
    return model


def _f0_extractor(state):
    """Upstream's RMVPE ``F0Extractor``, loaded from its file: the ``preprocess.tools``
    package's ``__init__`` imports the whole (heavy, CUDA-minded) preprocessing stack."""
    import importlib.util

    path = os.path.join(state["repo_dir"], F0_EXTRACTION)
    spec = importlib.util.spec_from_file_location("soulx_f0_extraction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.F0Extractor(
        _weight(state, "preprocess", RMVPE_CKPT), device=state["device"], verbose=False
    )


def _vocoder_on_cpu(model, device):
    """Run ``model``'s vocoder on the CPU, handing its output back on ``device``."""
    vocoder = model.vocoder.to("cpu")
    forward = vocoder.forward
    vocoder.forward = lambda x: forward(x.float().to("cpu")).to(device)


def unload(state, *, component):
    """Free one component's weights; returns what stays loaded."""
    import gc

    if state["models"].pop(component, None) is not None:
        gc.collect()
        _empty_device_cache(state["device"])
    return {"loaded": sorted(state["models"])}


def _empty_device_cache(device):
    try:
        import torch

        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - freeing a cache is best effort
        pass


# --------------------------------------------------------------------------- phonemes


def _ensure_nltk_data():
    """Fetch g2p_en's NLTK resources once, into ``$NLTK_DATA`` when it is set."""
    import nltk

    target = os.environ.get("NLTK_DATA")
    if target:
        os.makedirs(target, exist_ok=True)
    for resource, package in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, download_dir=target, quiet=True)


def _g2p(state):
    """g2p_en's ``G2p``, made once."""
    if "g2p" not in state:
        _ensure_nltk_data()
        from g2p_en import G2p

        state["g2p"] = G2p()
    return state["g2p"]


def phonemize_en(word, g2p):
    """SoulX's token for an English word: ``en_`` and its ARPAbet phones joined by '-'.

    The words of an elision (``me+and``, sung on one note) are phonemized one by
    one and their phones joined into the one token.
    """
    phones = []
    for part in word.split(WORD_JOINER):
        letters = "".join(c for c in part if c.isalpha() or c == "'").lower()
        if letters:
            phones += [p for p in g2p(letters) if p.strip() and p.isalnum()]
    return "en_" + "-".join(phones) if phones else SP


def fill_phonemes(state, meta):
    """``meta`` with each empty ``phoneme`` field filled from its ``text`` (English)."""
    out = []
    for segment in meta:
        segment = dict(segment)
        if not segment.get("phoneme", "").strip():
            if segment.get("language", "English") != "English":
                raise NotImplementedError(
                    f"Only English words are phonemized here, not {segment['language']}:"
                    " give the segment's phonemes"
                )
            g2p = _g2p(state)
            segment["phoneme"] = " ".join(
                SP if w == SP else phonemize_en(w, g2p) for w in segment["text"].split()
            )
        _check_lengths(segment)
        out.append(segment)
    return out


def _check_lengths(segment):
    """Every per-note field has one token per note, else the words slide off the notes."""
    counts = {f: len(str(segment.get(f, "")).split()) for f in NOTE_FIELDS}
    if len(set(counts.values())) != 1:
        raise ValueError(
            f"Segment {segment.get('index')!r} has fields of different lengths: {counts}"
        )


# --------------------------------------------------------------------------- operations


def _seed(seed):
    if seed is not None:
        import torch

        torch.manual_seed(seed)


def _write(out_path, samples, sample_rate, peak):
    """Write ``samples`` as 16-bit WAV, scaled down if they peak above ``peak``."""
    import numpy as np
    import soundfile

    samples = np.nan_to_num(np.asarray(samples, dtype=np.float32).reshape(-1))
    top = float(np.abs(samples).max()) if samples.size else 0.0
    gain = peak / top if peak and top > peak else 1.0
    soundfile.write(out_path, samples * gain, sample_rate, subtype="PCM_16")
    return {"peak": round(top, 4), "gain": round(gain, 4)}


def sing(
    state,
    *,
    meta,
    prompt_audio,
    prompt_meta,
    out_path,
    control="score",
    auto_shift=False,
    pitch_shift=0,
    n_steps=None,
    cfg=None,
    seed=None,
    peak=DFLT_PEAK,
    fade_s=DFLT_FADE_S,
):
    """SVS: sing ``meta`` in the voice of ``prompt_audio``, into ``out_path``.

    ``control='score'`` follows the notes; ``'melody'`` follows each segment's
    ``f0``. ``pitch_shift`` transposes by that many semitones; ``auto_shift``
    (upstream's median rule, recomputed for each segment) is best left off for a
    score of several segments. Each segment is placed at its ``time[0]``, with
    ``fade_s`` fades where segments meet.
    """
    import numpy as np
    import torch
    from soulxsinger.utils.data_processor import DataProcessor

    config, device = state["config"], state["device"]
    model = _model(state, "svs")
    _seed(seed)
    processor = DataProcessor(
        hop_size=config.audio.hop_size,
        sample_rate=config.audio.sample_rate,
        phoneset_path=os.path.join(state["repo_dir"], PHONESET),
        device=device,
    )
    prompt = processor.process(fill_phonemes(state, prompt_meta)[0], prompt_audio)
    segments = fill_phonemes(state, meta)
    sr = config.audio.sample_rate
    total = np.zeros(int(segments[-1]["time"][1] / 1000 * sr), dtype=np.float32)
    fade = int(fade_s * sr)
    t0 = time.perf_counter()
    for i, segment in enumerate(segments):
        start = int(segment["time"][0] / 1000 * sr)
        with torch.no_grad():
            audio = model.infer(
                {"prompt": prompt, "target": processor.process(segment, None)},
                auto_shift=auto_shift,
                pitch_shift=pitch_shift,
                n_steps=n_steps or config.infer.n_steps,
                cfg=cfg or config.infer.cfg,
                control=control,
            )
        audio = audio.squeeze().float().cpu().numpy()
        n = min(len(audio), len(total) - start)
        audio = _faded(audio[:n], fade, fade_in=i > 0, fade_out=i < len(segments) - 1)
        total[start : start + n] += audio
    elapsed = time.perf_counter() - t0
    return {
        "sample_rate": sr,
        "duration": len(total) / sr,
        "infer_seconds": round(elapsed, 3),
        "device": device,
        "segments": len(segments),
        **_write(out_path, total, sr, peak),
    }


def _faded(audio, n, *, fade_in, fade_out):
    """``audio`` with linear fades of ``n`` samples at the ends asked for."""
    import numpy as np

    audio = audio.copy()
    n = min(n, len(audio) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=audio.dtype)
        if fade_in:
            audio[:n] *= ramp
        if fade_out:
            audio[-n:] *= ramp[::-1]
    return audio


def f0(state, *, audio, out_path=None):
    """RMVPE F0 of ``audio`` on SoulX's 20 ms grid (Hz, 0 when unvoiced)."""
    values = _model(state, "rmvpe").process(audio, f0_path=out_path, verbose=False)
    return {
        "frames": len(values),
        "path": out_path,
        "voiced": float((values > 0).mean()),
    }


def convert(
    state,
    *,
    audio,
    prompt_audio,
    out_path,
    auto_shift=True,
    pitch_shift=0,
    n_steps=None,
    cfg=None,
    seed=None,
    peak=DFLT_PEAK,
):
    """SVC: re-voice the sung ``audio`` with the timbre of ``prompt_audio``.

    ``auto_shift`` moves the melody into the prompt voice's register (the shift
    used is returned); ``pitch_shift`` sets it in semitones instead.
    """
    import numpy as np
    import torch
    from soulxsinger.utils.audio_utils import load_wav

    config, device = state["config"], state["device"]
    rmvpe = _model(state, "rmvpe")
    source_f0 = rmvpe.process(audio, verbose=False)
    prompt_f0 = rmvpe.process(prompt_audio, verbose=False)
    model = _model(state, "svc")
    _seed(seed)

    def tensor(values):
        return torch.from_numpy(np.asarray(values, dtype=np.float32))[None].to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        generated, shift = model.infer(
            pt_wav=load_wav(prompt_audio, config.audio.sample_rate).to(device),
            gt_wav=load_wav(audio, config.audio.sample_rate).to(device),
            pt_f0=tensor(prompt_f0),
            gt_f0=tensor(source_f0),
            auto_shift=auto_shift,
            pitch_shift=pitch_shift,
            n_steps=n_steps or config.infer.n_steps,
            cfg=cfg or config.infer.cfg,
        )
    elapsed = time.perf_counter() - t0
    out = generated.squeeze().float().cpu().numpy()
    sr = config.audio.sample_rate
    return {
        "sample_rate": sr,
        "duration": len(out) / sr,
        "infer_seconds": round(elapsed, 3),
        "pitch_shift": int(shift),
        "device": device,
        **_write(out_path, out, sr, peak),
    }
