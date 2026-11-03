"""
tests/test_multilingual_pdf.py — reports on Hindi, Arabic and Chinese data
build, and draw that text in fonts that have it.

Before: Carlito has no Devanagari, Arabic or CJK glyphs, so every such
value — a column name, a category, the client's name — printed as boxes.
Devanagari also needs shaping (conjuncts, vowel signs before the
consonant) and Arabic needs joining and right-to-left order.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


def _fonts_in(pdf: bytes) -> set:
    import pypdf
    names = set()
    for page in pypdf.PdfReader(io.BytesIO(pdf)).pages:
        fonts = (page.get("/Resources") or {}).get("/Font") or {}
        for ref in fonts.values():
            names.add(str(ref.get_object().get("/BaseFont")))
    return names


DATA = {
    "hindi": lambda rng, n: pd.DataFrame({
        "कर्मचारी_आईडी": range(n),
        "विभाग": rng.choice(["बिक्री", "मानव संसाधन", "विपणन"], n),
        "वेतन": rng.normal(50000, 9000, n).round(),
        "attrition": rng.choice(["Yes", "No"], n, p=[.2, .8])}),
    "arabic": lambda rng, n: pd.DataFrame({
        "order_id": range(n),
        "المنطقة": rng.choice(["الرياض", "جدة", "الدمام"], n),
        "amount": rng.gamma(2, 80, n).round(2),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D")}),
    "cjk": lambda rng, n: pd.DataFrame({
        "product_id": [f"P{i}" for i in range(n)],
        "类别": rng.choice(["电子产品", "服装", "東京データ"], n),
        "price": rng.gamma(2, 20, n).round(2),
        "rating": rng.uniform(1, 5, n).round(1)}),
}
FONT = {"hindi": "NotoSansDevanagari", "arabic": "NotoSansArabic", "cjk": "NotoSansSC"}


@pytest.mark.parametrize("script", list(DATA))
def test_reports_build_and_embed_the_script_font(client, script):
    rng = np.random.default_rng(1)
    df = DATA[script](rng, 300)
    ds = client.post("/api/datasets/upload", files={
        "file": (f"{script}_डेटा_数据.csv", io.BytesIO(df.to_csv(index=False).encode()),
                 "text/csv")}).json()["meta"]["dataset_id"]
    main = client.post(f"/api/reports/{ds}/pdf",
                       json={"title": "रिपोर्ट 报告", "client_name": "ग्राहक شركة 公司"})
    health = client.post(f"/api/reports/{ds}/health-pdf",
                         json={"agency_name": "विश्लेषण 分析"})
    assert main.status_code == 200, main.text[:300]
    assert health.status_code == 200, health.text[:300]
    for pdf in (main.content, health.content):
        fonts = " ".join(_fonts_in(pdf))
        assert FONT[script] in fonts, fonts
        assert "NotoSansDevanagari" in fonts      # the Hindi title / agency name


def test_script_runs_get_their_own_font():
    from app.engines.pdf_fonts import script_markup
    out = script_markup("Sales विभाग प्रबंधक and قسم المبيعات and 销售部")
    assert '<font name="ReportDeva">विभाग प्रबंधक</font>' in out
    assert '<font name="ReportCJK">销售部</font>' in out
    assert '<font name="ReportArab">' in out
    assert script_markup("plain &amp; english") == "plain &amp; english"


def test_mixed_font_words_shape_without_failing():
    # One word that changes font part-way used to fail the whole PDF with
    # "'TTFont' object has no attribute 'hbAddPrivate'".
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate
    from app.engines.pdf_fonts import FONT_REG, script_markup
    style = ParagraphStyle("t", fontName=FONT_REG, fontSize=11)
    doc = SimpleDocTemplate(io.BytesIO())
    doc.build([Paragraph(script_markup(t), style) for t in (
        "कर्मचारी_आईडी", "Source: arabic_مبيعات.csv by विश्लेषण 分析", "x<b>विभाग</b>_y")])


def test_identifiers_are_not_attrition_drivers():
    from app.engines.domains.hr import _run_attrition
    rng = np.random.default_rng(3)
    n = 600
    left = rng.random(n) < 0.25
    df = pd.DataFrame({"कर्मचारी_आईडी": np.arange(n),
                       "tenure": rng.integers(0, 20, n),
                       "attrition": np.where(left, "Yes", "No")})
    df = df.sort_values("attrition").reset_index(drop=True)   # id now tracks leaving
    df["कर्मचारी_आईडी"] = np.arange(n)
    rep = _run_attrition(df)
    assert all(d["factor"] != "कर्मचारी_आईडी" for d in rep.top_drivers)


def test_no_satisfaction_card_without_a_satisfaction_column(hr_df):
    import pypdf
    from app.engines.domains.hr import _run_attrition
    from app.engines.pdf_builder import build_pdf
    from app.engines.data_profiler import profile_dataset
    df = hr_df.drop(columns=["satisfaction"])
    rep = _run_attrition(df)
    assert rep.satisfaction_col is None
    pdf = build_pdf(df=df, config={"title": "T", "client_name": "C"},
                    profile=profile_dataset(df), cleaning_summary=None, stats_report=None,
                    bi_report=None, ml_report=None, chart_data=[], executive_summary="",
                    findings=[], risks=[], opportunities=[], recommendations=[],
                    top_insights=[], attrition=rep, domain="hr")
    text = "".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)
    assert "LOWEST-QUARTER SATISFACTION" not in text.upper()
