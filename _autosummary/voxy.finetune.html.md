# voxy.finetune

Fine-tuning datasets from prepared clips: one export per training recipe.

`voxy.refprep.prepare_reference` leaves scored, transcribed clips and a
manifest; `export_dataset` writes the subset worth training on in the layout a
recipe expects:

- `ljspeech`: `wavs/{id}.wav` + `metadata.csv` (`id|text|text`, no header).
  Read by Coqui’s XTTS-v2 GPT fine-tuning (formatter `ljspeech`), StyleTTS2 and
  most community recipes (Chatterbox fine-tuning included).
- `f5tts`: `wavs/` + `metadata.csv` with an `audio_file|text` header and
  absolute paths, the input of `f5-tts_prepare_csv_wavs` then `f5-tts_finetune-cli`.

Plus `train.txt` / `eval.txt` (a deterministic split) for recipes that want one.
Commands to run each recipe on a rented GPU are in `finetune_commands`.

```pycon
>>> rows = [{"key": "clips/a.wav", "text": "Hi there.", "score": 0.9, "duration": 4.0},
...         {"key": "clips/b.wav", "text": "Noisy.", "score": 0.2, "duration": 3.0}]
>>> [r["key"] for r in select_clips(rows, min_score=0.5)]
['clips/a.wav']
```

### Functions

| [`export_dataset`](#voxy.finetune.export_dataset)(manifest, samples, out_dir, \*)     | Write the training set to `out_dir` and return a summary.                        |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`finetune_commands`](#voxy.finetune.finetune_commands)(backend, dataset_dir, \*[, ...]) | Shell commands to fine-tune `backend` on an exported dataset (run on a GPU box). |
| [`select_clips`](#voxy.finetune.select_clips)(manifest, \*[, min_score, ...])       | Clips good enough to train on (score and length within bounds).                  |

### voxy.finetune.export_dataset(manifest, samples, out_dir, , format='ljspeech', min_score=0.5, min_seconds=1.5, max_seconds=15.0, eval_fraction=0.05)

Write the training set to `out_dir` and return a summary.

* **Parameters:**
  * **manifest** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)]) – From `prepare_reference` (or `refprep.load_manifest`).
  * **samples** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – The store the clips are in (`key -> wav bytes`).
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – One of `FORMATS`.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.finetune.finetune_commands(backend, dataset_dir, , name='voice')

Shell commands to fine-tune `backend` on an exported dataset (run on a GPU box).

```pycon
>>> "f5-tts_finetune-cli" in finetune_commands("f5tts", "/data/ds")
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.finetune.select_clips(manifest, , min_score=0.5, min_seconds=1.5, max_seconds=15.0)

Clips good enough to train on (score and length within bounds).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
