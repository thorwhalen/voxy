"""
Published benchmarks of the cloning backends, kept as data, rendered as tables and charts.

The data (``voxy/data/benchmarks.json``) has two parts: ``models`` (licence, size, languages,
price, watermark per backend id such as ``"qwen3tts"`` or ``"elevenlabs/eleven_v3"``) and
``results``, one row per published number::

    {model, benchmark, metric, value, higher_is_better, source_url, self_reported, retrieved, note?}

``misc/docs/benchmarks.md`` holds the prose; its tables and charts are generated between markers
(``<!-- voxy-benchmarks:NAME -->`` ... ``<!-- /voxy-benchmarks -->``) by ``render_docs``, from the
``TABLES`` and ``CHARTS`` specs below. To update: edit the JSON (add rows, keep every source URL),
then ``python -m voxy benchmarks misc/docs/benchmarks.md``.

>>> rows = [{"model": "a", "benchmark": "B", "metric": "WER %", "value": 2.0, "self_reported": True,
...          "source_url": "https://x.org/p"}]
>>> print(results_table(rows, [("B", "WER %")], models={"a": {"name": "Model A"}}))
| Model | B (WER %) |
|---|---|
| Model A | [2.0](https://x.org/p) (S) |
"""

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from importlib.resources import files
from pathlib import Path

BENCHMARKS_FILE = "benchmarks.json"
DOC_MARKER = "voxy-benchmarks"
SELF_MARK = "(S)"
MISSING = "—"

# Doc tables: name -> columns of (benchmark, metric); rows are the models that have any value.
TABLES = {
    "french": [
        ("CV3-eval: French", "WER %"),
        ("CV3-eval: French", "SIM (cosine, 0-1)"),
        ("MiniMax multilingual test: French", "WER %"),
        ("MiniMax multilingual test: French", "SIM (cosine, 0-1)"),
    ],
    "seed": [
        ("Seed-TTS-eval test-en", "WER %"),
        ("Seed-TTS-eval test-en", "SIM (cosine, 0-1)"),
        ("Seed-TTS-eval test-zh", "CER %"),
        ("Seed-TTS-eval test-zh", "SIM (cosine, 0-1)"),
    ],
    "human": [
        ("Hume Voice Replication Leaderboard (overall, 13 models)", "same-speaker rating (1-5)"),
        ("Hume Voice Replication Leaderboard (overall, 13 models)", "different-person %"),
        ("Artificial Analysis Controlled Voice arena (8 cloned EN voices, 44 models)", "Elo"),
        ("TTS Arena V2 (TTS-AGI, 39 listed)", "Elo"),
    ],
}
# Doc charts: name -> spec. "bar": one (benchmark, metric); "scatter": x and y, one point per model.
# ``source`` keeps only rows whose source_url contains it (one source = comparable numbers).
CHARTS = {
    "french-tradeoff": {
        "kind": "scatter",
        "title": "French cloning: intelligibility vs likeness (CV3-eval, HF Open TTS Leaderboard)",
        "x": ("CV3-eval: French", "WER %"),
        "y": ("CV3-eval: French", "SIM (cosine, 0-1)"),
        "source": "open_tts_leaderboard",
    },
    "human-likeness": {
        "kind": "bar",
        "title": "Human raters: does it sound like the reference person? (Hume, 1-5)",
        "y": ("Hume Voice Replication Leaderboard (overall, 13 models)", "same-speaker rating (1-5)"),
    },
    "cloned-voice-arena": {
        "kind": "bar",
        "title": "Blind preference votes on cloned voices (Artificial Analysis, Elo)",
        "y": ("Artificial Analysis Controlled Voice arena (8 cloned EN voices, 44 models)", "Elo"),
    },
}
FACT_COLUMNS = [
    ("licence", "Licence"),
    ("french", "French"),
    ("languages", "Languages"),
    ("params", "Size"),
    ("runs", "Runs"),
    ("price", "Price"),
    ("watermark", "Watermark"),
]


def load_benchmarks(path: str | os.PathLike | None = None) -> dict:
    """The benchmark data: ``{"retrieved", "models", "results"}`` (default: the shipped file)."""
    source = Path(path) if path else files("voxy") / "data" / BENCHMARKS_FILE
    return json.loads(source.read_text())


def select(
    rows: Iterable[Mapping],
    *,
    benchmark: str | None = None,
    metric: str | None = None,
    model: str | None = None,
    source: str | None = None,
    independent: bool | None = None,
) -> list[dict]:
    """The rows matching every filter given (``source``: a substring of ``source_url``)."""

    def keep(r):
        return (
            (benchmark is None or r["benchmark"] == benchmark)
            and (metric is None or r["metric"] == metric)
            and (model is None or r["model"] == model)
            and (source is None or source in r.get("source_url", ""))
            and (independent is None or independent != bool(r.get("self_reported")))
        )

    return [dict(r) for r in rows if keep(r)]


def _name(model: str, models: Mapping) -> str:
    return models.get(model, {}).get("name", model)


def _cell(rows: Sequence[Mapping]) -> str:
    if not rows:
        return MISSING
    return " · ".join(
        f"[{r['value']}]({r['source_url']})" + (f" {SELF_MARK}" if r.get("self_reported") else "")
        for r in rows
    )


def _markdown(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in body])


def results_table(
    rows: Iterable[Mapping],
    columns: Sequence[tuple[str, str]],
    *,
    models: Mapping | None = None,
) -> str:
    """A Markdown table: one row per model with any value, one column per (benchmark, metric).

    A cell lists every source's number, linked, with ``(S)`` when the developer measured its own model.
    """
    models = models or {}
    rows = [r for r in rows if (r["benchmark"], r["metric"]) in set(columns)]
    order = [m for m in models if any(r["model"] == m for r in rows)]
    order += sorted({r["model"] for r in rows} - set(order))
    header = ["Model"] + [f"{b} ({m})" for b, m in columns]
    body = [
        [_name(model, models)]
        + [_cell(select(rows, benchmark=b, metric=m, model=model)) for b, m in columns]
        for model in order
    ]
    return _markdown(header, body)


def facts_table(models: Mapping, *, columns: Sequence[tuple[str, str]] = FACT_COLUMNS) -> str:
    """A Markdown table of each backend's practical facts (licence, French, price, ...)."""

    def fmt(v):
        return MISSING if v is None else ("yes" if v is True else "no" if v is False else str(v))

    body = [[m["name"]] + [fmt(m.get(k)) for k, _ in columns] for m in models.values()]
    return _markdown(["Model"] + [label for _, label in columns], body)


def chart(rows: Iterable[Mapping], spec: Mapping, path: str | os.PathLike, *, models=None) -> Path:
    """Draw one ``CHARTS`` spec to an image file (SVG or PNG by extension; needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = models or {}
    rows = select(rows, source=spec.get("source"))

    def values(benchmark, metric):
        found = {}
        for r in select(rows, benchmark=benchmark, metric=metric):
            found.setdefault(r["model"], r["value"])  # first source listed wins
        return found

    fig, ax = plt.subplots(figsize=(7, 4.2))
    if spec["kind"] == "bar":
        ys = values(*spec["y"])
        names = sorted(ys, key=ys.get, reverse=True)
        ax.barh([_name(m, models) for m in names], [ys[m] for m in names], color="#6a8caf")
        ax.invert_yaxis()
        ax.set_xlabel(spec["y"][1])
        lo = min(ys.values())
        ax.set_xlim(left=lo * 0.9 if lo > 10 else 0)
        for i, m in enumerate(names):
            ax.text(ys[m], i, f" {ys[m]}", va="center", fontsize=9)
    elif spec["kind"] == "scatter":
        xs, ys = values(*spec["x"]), values(*spec["y"])
        for m in sorted(set(xs) & set(ys)):
            ax.scatter(xs[m], ys[m], s=60, color="#6a8caf")
            ax.annotate(_name(m, models), (xs[m], ys[m]), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel(spec["x"][1] + " (lower is better)")
        ax.set_ylabel(spec["y"][1] + " (higher is better)")
        ax.margins(0.25)
    else:
        raise ValueError(f"Unknown chart kind {spec['kind']!r}: use 'bar' or 'scatter'")
    ax.set_title(spec["title"], fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def render_docs(
    doc_path: str | os.PathLike,
    *,
    data: Mapping | None = None,
    tables: Mapping = TABLES,
    charts: Mapping = CHARTS,
    charts_dir: str = "benchmarks",
    draw: bool = True,
) -> list[str]:
    """Regenerate the generated blocks of a benchmarks doc; return the names updated.

    A block ``<!-- voxy-benchmarks:NAME -->...<!-- /voxy-benchmarks -->`` gets the table ``NAME``
    (``facts`` is the practical-facts table), or the chart ``NAME`` drawn to
    ``<charts_dir>/NAME.svg`` next to the doc and embedded as an image.
    """
    data = data or load_benchmarks()
    doc_path = Path(doc_path)
    rows, models = data["results"], data["models"]
    updated = []

    def block(match):
        name = match.group(1)
        if name == "facts":
            body = facts_table(models)
        elif name in tables:
            body = results_table(rows, tables[name], models=models)
        elif name in charts:
            rel = f"{charts_dir}/{name}.svg"
            if draw:
                chart(rows, charts[name], doc_path.parent / rel, models=models)
            body = f"![{charts[name]['title']}]({rel})"
        else:
            raise ValueError(f"No table or chart named {name!r} for {match.group(0)[:60]!r}")
        updated.append(name)
        return f"<!-- {DOC_MARKER}:{name} -->\n{body}\n<!-- /{DOC_MARKER} -->"

    pattern = re.compile(rf"<!-- {DOC_MARKER}:([\w-]+) -->.*?<!-- /{DOC_MARKER} -->", re.S)
    doc_path.write_text(pattern.sub(block, doc_path.read_text()))
    return updated
