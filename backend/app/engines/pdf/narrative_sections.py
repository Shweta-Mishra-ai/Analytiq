"""
engines/pdf/narrative_sections.py — the sections that make an argument.

Executive summary, top insights, data-quality note, readiness, benchmark
context and the attrition page. These read as prose; the tabular
counterparts live in data_sections.py.
"""
import logging
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether,
)
from reportlab.pdfgen import canvas as CV

logger = logging.getLogger(__name__)

from app.engines.pdf.theme import _c, W, H, CW_DEFAULT
from app.engines.pdf.primitives import (
    _sec, _kpi_row, _narrative_box, _gtable, _insight_card, _clean,
)
from app.engines.report_blueprints import blueprint_for
from app.services.dtypes import is_text_dtype, text_columns


# ══════════════════════════════════════════════════════════
#  EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════

def _exec_summary(story, s, T, summary, findings, risks, opps, CW):
    _sec(story, s, T, "Executive Summary",
         "Key findings and strategic priorities")
    if summary:
        _narrative_box(story, s, T, summary)
    if findings:
        story.append(Paragraph("Key Findings", s["h3"]))
        for f in findings[:5]:
            story.append(Paragraph("+ " + str(f), s["bl"]))
        story.append(Spacer(1, 2*mm))
    if risks:
        story.append(Paragraph("Business Risks", s["h3"]))
        for r in risks[:4]:
            story.append(Paragraph("! " + str(r),
                ParagraphStyle("risk_p", fontName="Helvetica", fontSize=9,
                               textColor=_c(T["negative"]), leading=13,
                               leftIndent=10, firstLineIndent=-10,
                               spaceAfter=3)))
        story.append(Spacer(1, 2*mm))
    if opps:
        story.append(Paragraph("Opportunities", s["h3"]))
        for o in opps[:3]:
            story.append(Paragraph("* " + str(o),
                ParagraphStyle("opp_p", fontName="Helvetica", fontSize=9,
                               textColor=_c(T["positive"]), leading=13,
                               leftIndent=10, firstLineIndent=-10,
                               spaceAfter=3)))


# ══════════════════════════════════════════════════════════
#  TOP INSIGHTS
# ══════════════════════════════════════════════════════════

def _top_insights(story, s, T, insights, CW, domain: str = "general"):
    """Findings under the headings the domain's reader expects.

    A flat "Top Insights" list is a tool's default output. A finance
    director reads position, then cost structure, then variance; an HR
    director reads workforce profile, then attrition. Grouping the same
    findings under those headings is the difference between a report that
    was written for them and one that was generated.
    """
    from app.engines.insight_guard import guard_insights, withheld_note
    from app.engines.report_blueprints import blueprint_for, group_insights

    bp = blueprint_for(domain)
    # Last check before anything is printed: a finding with no figure, a
    # future asserted as fact, or a cause the data cannot support does
    # more damage than the finding is worth.
    guarded = guard_insights(list(insights or []))
    insights = guarded.kept
    if not insights:
        _sec(story, s, T, "Findings",
             "Each finding: Problem → Cause → Evidence → Action → Impact")
        story.append(Paragraph(
            "No finding in this dataset met the evidence threshold for "
            "inclusion. That is a result, not an omission: the analysis "
            "ran and found nothing it could support.", s["body"]))
        note = withheld_note(guarded)
        if note:
            story.append(Paragraph(note, s["note"]))
        return

    grouped = group_insights(bp, list(insights))
    _sec(story, s, T, "{} — Findings".format(bp.label),
         "Each finding: Problem → Cause → Evidence → Action → Impact")

    num = 0
    for section, items in grouped:
        # The heading travels with its first finding. Left to flow, a
        # section title lands at the foot of a page with its content
        # overleaf, which is the kind of thing a reader registers as
        # sloppy without being able to say why.
        head = [Spacer(1, 2 * mm),
                Paragraph(section.title, s["h3"]),
                Paragraph(section.purpose, s["sm"])]
        first: list = []
        _insight_card(first, s, T, items[0], CW, num=num + 1)
        story.append(KeepTogether(head + first))
        num += 1

        for ins in items[1:6]:
            num += 1
            _insight_card(story, s, T, ins, CW, num=num)
        if num >= 10:
            break

    note = withheld_note(guarded)
    if note:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(note, s["note"]))


# ══════════════════════════════════════════════════════════
#  DATA QUALITY NOTE
# ══════════════════════════════════════════════════════════

def _dq_note(story, s, T, df: pd.DataFrame, profile, CW):
    _sec(story, s, T, "Data Quality & Transparency Note",
         "Must read before interpreting any finding")

    raw_rows  = getattr(profile, "total_rows",     len(df))
    dupes     = getattr(profile, "duplicate_rows", 0)
    miss_pct  = getattr(profile, "missing_pct",
                        df.isna().mean().mean() * 100)
    qual      = getattr(profile, "overall_quality_score", "—")
    grade     = getattr(profile, "data_quality_grade", "")

    if dupes > 0:
        dup_pct = dupes / max(raw_rows, 1) * 100
        warn_t = Table([[Paragraph(
            "<b>Deduplication Alert:</b> "
            "{:,} exact duplicate rows ({:.1f}%) were detected in the raw data. "
            "All analysis in this report uses the deduplicated dataset "
            "({:,} clean rows). "
            "If comparing to any prior report on the raw file, numbers will differ — "
            "this is expected and correct.".format(dupes, dup_pct, len(df)),
            s["warn"])]],
            colWidths=["100%"])
        warn_t.setStyle(TableStyle([
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING",   (0,0), (-1,-1), 8),
            ("BOTTOMPADDING",(0,0), (-1,-1), 8),
            ("BOX",          (0,0), (-1,-1), 1, _c(T["warning"])),
        ]))
        story.append(warn_t)
        story.append(Spacer(1, 3*mm))

    # KPI strip
    _kpi_row(story, s, T, [
        {"label": "TOTAL ROWS",    "value": "{:,}".format(len(df)),
         "sub": "After deduplication", "color": T["accent"]},
        {"label": "COLUMNS",       "value": str(df.shape[1]),
         "sub": "{} num · {} cat".format(
             len(df.select_dtypes(include="number").columns),
             len(text_columns(df))),
         "color": T["accent"]},
        {"label": "MISSING DATA",  "value": "{:.1f}%".format(miss_pct),
         "sub": "0% = perfect",
         "color": T["positive"] if miss_pct == 0 else T["warning"]},
        {"label": "QUALITY SCORE", "value": str(qual),
         "sub": "Grade {}".format(grade) if grade else "/ 100",
         "color": T["positive"]},
    ], CW)

    _readiness_block(story, s, T, df, CW)

    # DQ table from profile
    recs = getattr(profile, "recommendations", [])
    if recs:
        story.append(Paragraph("Data Quality Recommendations", s["h3"]))
        for rec in recs[:6]:
            sty = "bl"
            story.append(Paragraph("• " + str(rec), s[sty]))


def _readiness_block(story, s, T, df: pd.DataFrame, CW):
    """State whether the data was fit to analyse before showing findings.

    A reader is entitled to know that the revenue column was text and was
    therefore in none of the numbers above it. Putting this on the same
    page as the quality note, rather than in an appendix, is the point:
    it is a precondition for the report, not a footnote to it.
    """
    try:
        from app.engines.readiness import assess_readiness
        rep = assess_readiness(df)
    except Exception:
        logger.warning("readiness assessment failed", exc_info=True)
        return

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Fitness for Analysis", s["h3"]))
    story.append(Paragraph(_clean(rep.summary), s["body"]))

    if rep.blockers:
        rows = [[b.column, b.issue, b.consequence] for b in rep.blockers[:6]]
        _gtable(story, T, ["Column", "Issue", "Effect on the analysis"],
                rows, [CW * x for x in [0.20, 0.25, 0.55]])
        story.append(Paragraph(
            "Findings in this report were produced despite the above. Treat "
            "any figure that depends on an affected column as provisional "
            "until it is resolved.", s["note"]))
    if rep.personal_data_columns:
        story.append(Paragraph(
            "Personal data present: {}. This report and the underlying "
            "extract should be handled and retained accordingly.".format(
                ", ".join(rep.personal_data_columns[:8])),
            s["note"]))


# ══════════════════════════════════════════════════════════
#  INDUSTRY BENCHMARKS (HR domain)
# ══════════════════════════════════════════════════════════

# "Published hr ranges" reads as machine output. Acronym domains keep
# their capitalisation; the rest are title-cased.
def _domain_label(domain: str) -> str:
    """Prose label for a domain. Acronym domains keep their capitalisation
    ("HR", "SaaS"); "Published hr ranges" reads as machine output."""
    from app.engines.domains.registry import label_for
    return label_for(domain)


def _has_reference_ranges(domain, df) -> bool:
    """Whether this dataset has any metric with a published range.

    Checked before the section is added to the contents page, so the
    contents never promises a section the report does not contain.
    """
    from app.engines.industry_benchmarks import (DOMAIN_BENCHMARKS,
                                                 lookup_benchmark)
    if df is None or str(domain or "").lower() not in DOMAIN_BENCHMARKS:
        return False
    return any(lookup_benchmark(str(domain).lower(), str(c)) is not None
               for c in df.columns)


def _benchmark_section(story, s, T, domain, CW, df=None):
    """Reference ranges for this domain, and only this domain.

    What was here before was three hardcoded tables citing "Salesforce
    2024", "Gartner 2024", "Klaviyo 2024" and similar against specific
    figures — precise-looking attributions to reports whose contents
    cannot be checked from inside this app. That is the single most
    damaging thing a report can contain: a reader who looks one up and
    cannot find the number stops believing the rest of the document.
    Most rows also read "—" for the client's own value, so the table was
    largely a list of other people's numbers.

    Every range now comes from services/industry_benchmarks with a named
    source, is looked up strictly within the report's own domain, and is
    only shown for metrics this dataset actually contains. An HR report
    therefore cites HR sources and nothing else.
    """
    from app.engines.industry_benchmarks import (DOMAIN_BENCHMARKS,
                                                 lookup_benchmark)

    domain = str(domain or "").lower()
    if domain not in DOMAIN_BENCHMARKS or df is None:
        return

    def _num_mean(series):
        """Mean robust to Yes/No/True/False string encodings."""
        col = series
        if is_text_dtype(col):
            mapped = col.astype(str).str.strip().str.lower().map(
                {"yes": 1, "no": 0, "true": 1, "false": 0,
                 "y": 1, "n": 0, "1": 1, "0": 0})
            col = mapped if mapped.notna().any() else pd.to_numeric(
                col, errors="coerce")
        else:
            col = pd.to_numeric(col, errors="coerce")
        col = col.dropna()
        return float(col.mean()) if len(col) else None

    rows = []
    for c in df.columns:
        bm = lookup_benchmark(domain, str(c))
        if bm is None:
            continue
        value = _num_mean(df[c])
        if value is None:
            continue
        # A rate stored as 0-1 is the same measure as one stored as 0-100.
        shown = value
        if bm.unit == "%" and 0 <= value <= 1 and bm.high > 1:
            shown = value * 100
        if bm.unit in ("%", "x") or bm.unit.startswith("/"):
            value_str = "{:,.1f}{}".format(shown, bm.unit)
            range_str = "{:g}–{:g}{}".format(bm.low, bm.high, bm.unit)
        else:
            value_str = "{:,.1f} {}".format(shown, bm.unit).strip()
            range_str = "{:g}–{:g} {}".format(bm.low, bm.high, bm.unit).strip()

        if shown < bm.low:
            position = "below the range"
        elif shown > bm.high:
            position = "above the range"
        else:
            position = "within the range"
        rows.append([str(c), value_str, range_str, position, bm.source])

    if not rows:
        return

    _sec(story, s, T, "Performance Against Published Ranges",
         "Published {} ranges, against the figures in this dataset".format(
             _domain_label(domain)))
    story.append(Paragraph(
        "These are general, publicly-cited ranges, not a licensed benchmark "
        "set, and they move with sector, company size and region. They "
        "answer one question — is this figure in a plausible place — and "
        "nothing more. Where a figure sits outside a range, that is a "
        "prompt to look, not a finding in itself; the comparisons against "
        "this organisation's own distribution elsewhere in this report are "
        "the stronger evidence.", s["body"]))
    story.append(Spacer(1, 3 * mm))

    _gtable(story, T,
            ["Metric", "This dataset", "Published range", "Position", "Source"],
            rows,
            [CW * x for x in [0.22, 0.14, 0.17, 0.15, 0.32]])
    story.append(Paragraph(
        "Sources are named per row so each can be checked. No range here is "
        "attributed to a report this analysis has not drawn it from.",
        s["note"]))


# ══════════════════════════════════════════════════════════
#  ATTRITION PAGE
# ══════════════════════════════════════════════════════════

def _attrition_page(story, s, T, attrition, CW):
    if attrition is None: return
    _sec(story, s, T, "Attrition Deep Dive",
         "Employee turnover analysis — drivers, segments, cost")

    _kpi_row(story, s, T, [
        {"label": "ATTRITION RATE", "value": "{:.1f}%".format(attrition.rate),
         "sub": "{:,} left".format(attrition.n_left),
         "color": T["negative"] if attrition.rate > 15 else T["warning"]},
        {"label": "SEVERITY", "value": attrition.severity.upper(),
         "sub": "Benchmark: 10–15%", "color": T["negative"]},
        {"label": "FLIGHT RISK", "value": "{:,}".format(attrition.n_flight_risk),
         "sub": "{:.0f}% of remaining".format(attrition.flight_risk_pct),
         "color": T["warning"]},
        {"label": "COST RISK",
         "value": "HIGH" if attrition.n_left > 50 else "MED",
         "sub": "50–200% salary/hire", "color": T["negative"]},
    ], CW)

    _narrative_box(story, s, T,
                   getattr(attrition, "interpretation", ""))
    _narrative_box(story, s, T,
                   getattr(attrition, "cost_estimate", ""))

    # Drivers
    drivers = getattr(attrition, "top_drivers", [])
    if drivers:
        story.append(Paragraph("Attrition Drivers", s["h3"]))
        rows = [[d.get("factor","")[:18], d.get("type","").title(),
                 "{:.0f}%".format(d.get("impact",0)), d.get("detail","")[:65]]
                for d in drivers[:6]]
        _gtable(story, T, ["Factor","Type","Impact","Finding"],
                rows, [CW*x for x in [0.22,0.13,0.14,0.51]])

    # Dept breakdown
    dept_atr = getattr(attrition, "dept_attrition", {})
    if dept_atr:
        story.append(Paragraph("Attrition by Department", s["h3"]))
        sorted_d = sorted(dept_atr.items(), key=lambda x: x[1], reverse=True)
        rows = [[str(dept), "{:.1f}%".format(rate),
                 "CRITICAL" if rate > 25 else "HIGH" if rate > 18 else "OK"]
                for dept, rate in sorted_d]
        _gtable(story, T, ["Department","Rate","Status"],
                rows, [CW*0.50, CW*0.25, CW*0.25], severity_col=2)
