"""The shipped benchmark data and the doc renderer."""

import pytest

from voxy import benchmarks as bm


def test_shipped_data_is_sourced_and_charts_resolve():
    data = bm.load_benchmarks()
    rows = data["results"]
    assert rows and all(r["source_url"].startswith("https://") and "retrieved" in r for r in rows)
    for spec in bm.CHARTS.values():
        for axis in ("x", "y"):
            if axis in spec:
                assert bm.select(rows, benchmark=spec[axis][0], metric=spec[axis][1]), spec[axis]
    for columns in bm.TABLES.values():
        assert all(bm.select(rows, benchmark=b, metric=m) for b, m in columns)


def test_render_docs_fills_blocks(tmp_path):
    doc = tmp_path / "b.md"
    doc.write_text("intro\n<!-- voxy-benchmarks:facts -->old<!-- /voxy-benchmarks -->\n"
                   "<!-- voxy-benchmarks:french -->\n<!-- /voxy-benchmarks -->\nend\n")
    assert bm.render_docs(doc) == ["facts", "french"]
    text = doc.read_text()
    assert "old" not in text and "| Qwen3-TTS 1.7B | Apache-2.0 |" in text and text.endswith("end\n")
    assert "CV3-eval: French (WER %)" in text
    bm.render_docs(doc)
    assert doc.read_text() == text  # idempotent


def test_render_docs_rejects_unknown_block(tmp_path):
    doc = tmp_path / "b.md"
    doc.write_text("<!-- voxy-benchmarks:nope --><!-- /voxy-benchmarks -->")
    with pytest.raises(ValueError, match="nope"):
        bm.render_docs(doc)


def test_chart_draws(tmp_path):
    pytest.importorskip("matplotlib")
    data = bm.load_benchmarks()
    for name, spec in bm.CHARTS.items():
        assert bm.chart(data["results"], spec, tmp_path / f"{name}.svg", models=data["models"]).stat().st_size
