"""Phoneme recognition (wav2vec2 espeak CTC): frame-wise log-probabilities of a crop, and espeak phonemization."""

DFLT_SAMPLE_RATE = 16000
#: The CTC blank: the recogniser's padding symbol.
BLANK_SYMBOL = "<pad>"


def _vocab_file(model):
    """The model's ``vocab.json``, from a local folder or the Hugging Face hub.

    Read directly rather than through ``AutoTokenizer``: the phoneme tokenizer class
    imports ``phonemizer`` (and espeak) just to load, and only the symbol table is used.
    """
    import os

    if os.path.isdir(model):
        return os.path.join(model, "vocab.json")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model, "vocab.json")


def load(*, model="facebook/wav2vec2-lv-60-espeak-cv-ft", device="cpu"):
    import json

    import torch
    from transformers import AutoModelForCTC

    with open(_vocab_file(model), encoding="utf-8") as file:
        ids = json.load(file)
    vocab = [None] * (max(ids.values()) + 1)
    for symbol, index in ids.items():
        vocab[index] = symbol
    return {
        "torch": torch,
        "model": AutoModelForCTC.from_pretrained(model).to(device).eval(),
        "vocab": vocab,
        "blank": ids.get(BLANK_SYMBOL, 0),
        "device": device,
    }


def emissions(state, *, audio, start=0.0, end=None, sample_rate=DFLT_SAMPLE_RATE):
    import librosa

    duration = None if end is None else max(0.0, end - start)
    samples, _ = librosa.load(audio, sr=sample_rate, offset=start, duration=duration)
    torch = state["torch"]
    inputs = torch.tensor(samples, dtype=torch.float32)[None].to(state["device"])
    inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-7)  # the processor's norm
    with torch.no_grad():
        logits = state["model"](inputs).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1).cpu().tolist()
    return {"log_probs": log_probs, "vocab": state["vocab"], "blank": state["blank"]}


def phonemize(state, *, texts, language="fr-fr"):
    """espeak IPA of each text (the symbols the recogniser was trained on), stress marks kept."""
    if "phonemize" not in state:
        import espeakng_loader
        from phonemizer import phonemize as run
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        state["phonemize"] = run
    ipa = state["phonemize"](
        list(texts), language=language, backend="espeak", strip=True
    )
    return {"ipa": list(ipa)}
