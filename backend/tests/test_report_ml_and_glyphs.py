"""
tests/test_report_ml_and_glyphs.py — the report says what was asked for,
in characters the reader can see.

The Reports page offered "ML results (if trained)" and the PDF ignored it.
And after the switch to an embedded font, symbols the font lacks (⚠ ★ ◆ ✓)
printed as empty boxes.
"""
from __future__ import annotations

import functools
import io

import pypdf
from fontTools.ttLib import TTFont

from app.engines.pdf_fonts import FONT_DIR


def _text(pdf: bytes) -> str:
    return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)


@functools.lru_cache(maxsize=1)
def _cmap():
    import os
    return TTFont(os.path.join(FONT_DIR, "Carlito-Regular.ttf")).getBestCmap()


def _missing_glyphs(text: str) -> set:
    return {ch for ch in text if not ch.isspace() and ord(ch) not in _cmap()}


def _report(client, ds, **kw):
    r = client.post(f"/api/reports/{ds}/pdf", json={"max_charts": 1, **kw})
    assert r.status_code == 200, r.text
    return _text(r.content)


def test_the_ml_box_adds_the_trained_model_to_the_report(client, uploaded_dataset_id):
    ds = uploaded_dataset_id
    assert client.post(f"/api/ml/{ds}/train", json={"target": "salary"}).status_code == 200
    with_ml = _report(client, ds, include_ml=True)
    without = _report(client, ds, include_ml=False)
    assert "Predictive Model" in with_ml and "salary" in with_ml
    assert "Predictive Model" not in without


def test_a_model_with_no_signal_reports_no_ranking(client):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n),
                       "group": rng.choice(list("abc"), n),
                       "outcome": rng.normal(100, 15, n)})
    r = client.post("/api/datasets/upload", files={
        "file": ("noise.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    body = client.post(f"/api/ml/{ds}/train", json={"target": "outcome"}).json()
    assert "barely better than always guessing its average" in body["no_signal"]
    assert not any("Most important predictor" in i for i in body["insights"])
    assert all("not evidence of an effect" in f["explanation"]
               for f in body["feature_importance"])
    text = " ".join(_report(client, ds, include_ml=True).split())
    assert "barely better than always guessing" in text
    assert "What the model relies on" not in text


def _string_literals(path):
    """String constants in a module, docstrings excluded."""
    import ast
    tree = ast.parse(open(path, encoding="utf-8").read())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant):
                docs.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docs:
            yield node.lineno, node.value


# Shown only in the web app, which has emoji fonts: insight_engine's icon
# field and chart_engine's interactive chart titles (the PDF's charts come
# from chart_exporter, which is checked). Nothing else is exempt.
_WEB_ONLY = ({("insight_engine.py", ch) for ch in "📋📊📈📉🏆🔗"}
             | {("chart_engine.py", ch) for ch in "📈🔗📊📉🔵"})


def test_no_report_text_uses_a_character_the_font_lacks():
    # A missing glyph cannot be caught from the PDF: text extraction drops
    # it along with the glyph. So check the strings the reports are built from.
    import glob
    import os
    files = (glob.glob("app/engines/*.py") + glob.glob("app/engines/domains/*.py")
             + ["app/rag/service.py"])
    bad = []
    for path in files:
        for line, text in _string_literals(path):
            for ch in _missing_glyphs(text):
                if ord(ch) < 0x2500 or ord(ch) > 0x257F:      # box-drawing: code comments
                    if (os.path.basename(path), ch) not in _WEB_ONLY:
                        bad.append(f"{path}:{line} {ch!r}")
    assert not bad, bad
