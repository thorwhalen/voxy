# voxy.benchmarks

Published benchmarks of the cloning backends, kept as data, rendered as tables and charts.

The data (`voxy/data/benchmarks.json`) has two parts: `models` (licence, size, languages,
price, watermark per backend id such as `"qwen3tts"` or `"elevenlabs/eleven_v3"`) and
`results`, one row per published number:

```default
{model, benchmark, metric, value, higher_is_better, source_url, self_reported, retrieved, note?}
```

`docs/benchmarks.md` holds the prose; its tables and charts are generated between markers
(`<!-- voxy-benchmarks:NAME -->` … `<!-- /voxy-benchmarks -->`) by `render_docs`, from the
`TABLES` and `CHARTS` specs below. To update: edit the JSON (add rows, keep every source URL),
then `python -m voxy benchmarks docs/benchmarks.md`.

```pycon
>>> rows = [{"model": "a", "benchmark": "B", "metric": "WER %", "value": 2.0, "self_reported": True,
...          "source_url": "https://x.org/p"}]
>>> print(results_table(rows, [("B", "WER %")], models={"a": {"name": "Model A"}}))
| Model | B (WER %) |
|---|---|
| Model A | [2.0](https://x.org/p) (S) |
```

### Functions

| [`chart`](#voxy.benchmarks.chart)(rows, spec, path, \*[, models])             | Draw one `CHARTS` spec to an image file (SVG or PNG by extension; needs matplotlib).    |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`facts_table`](#voxy.benchmarks.facts_table)(models, \*[, columns])                | A Markdown table of each backend's practical facts (licence, French, price, ...).       |
| [`load_benchmarks`](#voxy.benchmarks.load_benchmarks)([path])                           | The benchmark data: `{"retrieved", "models", "results"}` (default: the shipped file).   |
| [`render_docs`](#voxy.benchmarks.render_docs)(doc_path, \*[, data, tables, ...])    | Regenerate the generated blocks of a benchmarks doc; return the names updated.          |
| [`results_table`](#voxy.benchmarks.results_table)(rows, columns, \*[, models])        | A Markdown table: one row per model with any value, one column per (benchmark, metric). |
| [`select`](#voxy.benchmarks.select)(rows, \*[, benchmark, metric, model, ...]) | The rows matching every filter given (`source`: a substring of `source_url`).           |

### voxy.benchmarks.chart(rows, spec, path, , models=None)

Draw one `CHARTS` spec to an image file (SVG or PNG by extension; needs matplotlib).

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### voxy.benchmarks.facts_table(models, , columns=[('licence', 'Licence'), ('french', 'French'), ('languages', 'Languages'), ('params', 'Size'), ('runs', 'Runs'), ('price', 'Price'), ('watermark', 'Watermark')])

A Markdown table of each backend’s practical facts (licence, French, price, …).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.benchmarks.load_benchmarks(path=None)

The benchmark data: `{"retrieved", "models", "results"}` (default: the shipped file).

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### voxy.benchmarks.render_docs(doc_path, , data=None, tables={'french': [('CV3-eval: French', 'WER %'), ('CV3-eval: French', 'SIM (cosine, 0-1)'), ('MiniMax multilingual test: French', 'WER %'), ('MiniMax multilingual test: French', 'SIM (cosine, 0-1)')], 'human': [('Hume Voice Replication Leaderboard (overall, 13 models)', 'same-speaker rating (1-5)'), ('Hume Voice Replication Leaderboard (overall, 13 models)', 'different-person %'), ('Artificial Analysis Controlled Voice arena (8 cloned EN voices, 44 models)', 'Elo'), ('TTS Arena V2 (TTS-AGI, 39 listed)', 'Elo')], 'seed': [('Seed-TTS-eval test-en', 'WER %'), ('Seed-TTS-eval test-en', 'SIM (cosine, 0-1)'), ('Seed-TTS-eval test-zh', 'CER %'), ('Seed-TTS-eval test-zh', 'SIM (cosine, 0-1)')]}, charts={'cloned-voice-arena': {'kind': 'bar', 'title': 'Blind preference votes on cloned voices (Artificial Analysis, Elo)', 'y': ('Artificial Analysis Controlled Voice arena (8 cloned EN voices, 44 models)', 'Elo')}, 'french-tradeoff': {'kind': 'scatter', 'source': 'open_tts_leaderboard', 'title': 'French cloning: intelligibility vs likeness (CV3-eval, HF Open TTS Leaderboard)', 'x': ('CV3-eval: French', 'WER %'), 'y': ('CV3-eval: French', 'SIM (cosine, 0-1)')}, 'human-likeness': {'kind': 'bar', 'title': 'Human raters: does it sound like the reference person? (Hume, 1-5)', 'y': ('Hume Voice Replication Leaderboard (overall, 13 models)', 'same-speaker rating (1-5)')}}, charts_dir='benchmarks', draw=True)

Regenerate the generated blocks of a benchmarks doc; return the names updated.

A block `<!-- voxy-benchmarks:NAME -->...<!-- /voxy-benchmarks -->` gets the table `NAME`
(`facts` is the practical-facts table), or the chart `NAME` drawn to
`<charts_dir>/NAME.svg` next to the doc and embedded as an image.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### voxy.benchmarks.results_table(rows, columns, , models=None)

A Markdown table: one row per model with any value, one column per (benchmark, metric).

A cell lists every source’s number, linked, with `(S)` when the developer measured its own model.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### voxy.benchmarks.select(rows, , benchmark=None, metric=None, model=None, source=None, independent=None)

The rows matching every filter given (`source`: a substring of `source_url`).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]
