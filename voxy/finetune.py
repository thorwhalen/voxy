"""
Fine-tuning datasets from prepared clips: one export per training recipe.

``voxy.refprep.prepare_reference`` leaves scored, transcribed clips and a
manifest; ``export_dataset`` writes the subset worth training on in the layout a
recipe expects:

- ``ljspeech``: ``wavs/{id}.wav`` + ``metadata.csv`` (``id|text|text``, no header).
  Read by Coqui's XTTS-v2 GPT fine-tuning (formatter ``ljspeech``), StyleTTS2 and
  most community recipes (Chatterbox fine-tuning included).
- ``f5tts``: ``wavs/`` + ``metadata.csv`` with an ``audio_file|text`` header and
  absolute paths, the input of ``f5-tts_prepare_csv_wavs`` then ``f5-tts_finetune-cli``.

Plus ``train.txt`` / ``eval.txt`` (a deterministic split) for recipes that want one.
Commands to run each recipe on a rented GPU are in ``finetune_commands``.

>>> rows = [{"key": "clips/a.wav", "text": "Hi there.", "score": 0.9, "duration": 4.0},
...         {"key": "clips/b.wav", "text": "Noisy.", "score": 0.2, "duration": 3.0}]
>>> [r["key"] for r in select_clips(rows, min_score=0.5)]
['clips/a.wav']
"""

import os
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path

DFLT_MIN_SCORE = 0.5
DFLT_MIN_SECONDS = 1.5
DFLT_MAX_SECONDS = 15.0
DFLT_EVAL_FRACTION = 0.05
FORMATS = ("ljspeech", "f5tts")


def select_clips(
    manifest: Iterable[Mapping],
    *,
    min_score: float = DFLT_MIN_SCORE,
    min_seconds: float = DFLT_MIN_SECONDS,
    max_seconds: float = DFLT_MAX_SECONDS,
) -> list[dict]:
    """Clips good enough to train on (score and length within bounds)."""
    return [
        dict(c)
        for c in manifest
        if c["score"] >= min_score
        and min_seconds <= c["duration"] <= max_seconds
        and c["text"].strip()
    ]


def _is_eval(key: str, fraction: float) -> bool:
    """Deterministic split by key hash (stable across exports)."""
    return (zlib.crc32(key.encode()) % 1000) < fraction * 1000


def export_dataset(
    manifest: Iterable[Mapping],
    samples: Mapping,
    out_dir: str | os.PathLike,
    *,
    format: str = "ljspeech",
    min_score: float = DFLT_MIN_SCORE,
    min_seconds: float = DFLT_MIN_SECONDS,
    max_seconds: float = DFLT_MAX_SECONDS,
    eval_fraction: float = DFLT_EVAL_FRACTION,
) -> dict:
    """Write the training set to ``out_dir`` and return a summary.

    Args:
        manifest: From ``prepare_reference`` (or ``refprep.load_manifest``).
        samples: The store the clips are in (``key -> wav bytes``).
        format: One of ``FORMATS``.
    """
    if format not in FORMATS:
        raise ValueError(f"Unknown format {format!r} (have: {', '.join(FORMATS)})")
    out = Path(out_dir)
    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    clips = select_clips(
        manifest, min_score=min_score, min_seconds=min_seconds, max_seconds=max_seconds
    )
    lines, train, evals = [], [], []
    for clip in clips:
        clip_id = Path(clip["key"]).stem
        path = wavs / f"{clip_id}.wav"
        path.write_bytes(samples[clip["key"]])
        text = " ".join(clip["text"].split()).replace("|", ",")
        if format == "f5tts":
            lines.append(f"{path.resolve()}|{text}")
        else:
            lines.append(f"{clip_id}|{text}|{text}")
        (evals if _is_eval(clip["key"], eval_fraction) else train).append(lines[-1])
    header = ["audio_file|text"] if format == "f5tts" else []
    (out / "metadata.csv").write_text("\n".join(header + lines) + "\n")
    (out / "train.txt").write_text("\n".join(train) + "\n")
    (out / "eval.txt").write_text("\n".join(evals) + "\n")
    minutes = sum(c["duration"] for c in clips) / 60
    return {
        "format": format,
        "dir": str(out),
        "clips": len(clips),
        "minutes": round(minutes, 1),
        "eval_clips": len(evals),
    }


def finetune_commands(backend: str, dataset_dir: str, *, name: str = "voice") -> str:
    """Shell commands to fine-tune ``backend`` on an exported dataset (run on a GPU box).

    >>> "f5-tts_finetune-cli" in finetune_commands("f5tts", "/data/ds")
    True
    """
    recipes = {
        "f5tts": f"""\
# F5-TTS: export with format='f5tts'. Fine-tunes the v1 base model.
pip install f5-tts
f5-tts_prepare_csv_wavs {dataset_dir}/metadata.csv data/{name}_char --workers 8
f5-tts_finetune-cli --exp_name F5TTS_v1_Base --dataset_name {name} \\
  --finetune --pretrain "" --tokenizer char \\
  --learning_rate 1e-5 --batch_size_per_gpu 3200 --batch_size_type frame \\
  --max_samples 64 --grad_accumulation_steps 1 --max_grad_norm 1 \\
  --epochs 100 --num_warmup_updates 200 --save_per_updates 400 \\
  --keep_last_n_checkpoints 3 --last_per_updates 200
# Checkpoints: ckpts/{name}/model_*.pt -> f5tts backend init: ckpt_file=..., vocab_file=...
""",
        "xtts": f"""\
# XTTS-v2 GPT fine-tuning (coqui-tts recipe). Export with format='ljspeech'.
pip install coqui-tts
git clone --depth 1 https://github.com/idiap/coqui-ai-TTS && cd coqui-ai-TTS
# edit recipes/ljspeech/xtts_v2/train_gpt_xtts.py: dataset path {dataset_dir},
# formatter "ljspeech", language "en", speaker reference = a few best clips
COQUI_TOS_AGREED=1 python recipes/ljspeech/xtts_v2/train_gpt_xtts.py
# Output: run/training/*/best_model.pth + config.json -> xtts backend init:
#   model_path=<run dir>, config_path=<run dir>/config.json
""",
    }
    if backend not in recipes:
        raise ValueError(
            f"No fine-tuning recipe for {backend!r} yet (have: {', '.join(recipes)})"
        )
    return recipes[backend]
