"""Phoneme recognition (wav2vec2 espeak CTC): frame-wise log-probabilities of a crop."""

DFLT_SAMPLE_RATE = 16000


def load(*, model="facebook/wav2vec2-lv-60-espeak-cv-ft", device="cpu"):
    import torch
    from transformers import AutoModelForCTC, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    vocab = [None] * len(tokenizer.get_vocab())
    for symbol, index in tokenizer.get_vocab().items():
        vocab[index] = symbol
    return {
        "torch": torch,
        "model": AutoModelForCTC.from_pretrained(model).to(device).eval(),
        "vocab": vocab,
        "blank": tokenizer.pad_token_id,
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
