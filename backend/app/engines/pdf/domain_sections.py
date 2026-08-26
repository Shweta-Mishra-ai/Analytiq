"""
engines/pdf/domain_sections.py — sections that only apply to some domains.

Today: the appendix and the prepared-by line. Domain deep pages (finance
P&L, and the per-domain equivalents) belong here as they land.
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
    _sec, _kpi_row, _narrative_box, _gtable, _clean,
)
from app.engines.report_blueprints import blueprint_for
from app.services.dtypes import is_text_dtype, text_columns


# ══════════════════════════════════════════════════════════
#  APPENDIX
# ══════════════════════════════════════════════════════════

def _prepared_by_line(config: dict) -> str:
    """Who prepared the report, for the basis of preparation.

    A review deliverable is signed by the person or firm accountable for
    it. What produced the document is not the reader's concern and is
    named nowhere — the analyst's or consultancy's name is what belongs
    here, and it is the client's or freelancer's to set.
    """
    who = str(config.get("prepared_by") or "").strip()
    if not who:
        return ""
    return " Prepared by {}, who is responsible for the analysis and the " \
           "conclusions drawn from it.".format(_clean(who))


def _appendix(story, s, T, config, CW, domain: str = "general"):
    _sec(story, s, T, "Appendix — Methodology & Sources")

    # This section is what a reviewing analyst reads to decide whether to
    # trust the rest. It states the tests actually applied and why each was
    # chosen — it does not describe the tooling that rendered the document,
    # which tells the reader nothing about validity.
    story.append(Paragraph("A. Analytical Method", s["h3"]))
    story.append(Paragraph(
        "Every figure in this report is computed directly from the supplied "
        "dataset. No values are estimated, imputed into the findings, or "
        "carried over from other engagements.",
        s["body"]))
    story.append(Paragraph(
        "<b>Distributional testing.</b> Normality is assessed with Shapiro-Wilk "
        "(n≤5,000) and D'Agostino-Pearson, rather than assumed. The outcome "
        "determines which downstream test is used, so a non-normal column is "
        "never summarised with a statistic that presumes normality.",
        s["body"]))
    story.append(Paragraph(
        "<b>Association.</b> Pearson's r is used where both variables are "
        "approximately normal; Spearman's rank correlation otherwise. "
        "Correlations are reported with their p-value and sample size. "
        "Pairs that are mechanically related (a rate against its own "
        "numerator, a duplicated column) are excluded rather than presented "
        "as findings.",
        s["body"]))
    story.append(Paragraph(
        "<b>Group differences.</b> Two-group comparisons use Welch's t-test "
        "where the normality condition holds and Mann-Whitney U where it does "
        "not; comparisons across three or more groups use one-way ANOVA or "
        "Kruskal-Wallis on the same basis. Categorical association uses "
        "Chi-square with an expected-frequency check.",
        s["body"]))
    story.append(Paragraph(
        "<b>Outliers.</b> Flagged by the 1.5×IQR rule and cross-checked with "
        "the modified Z-score (Iglewicz &amp; Hoaglin), which is robust to "
        "skew. Outliers are reported, never silently removed — an extreme "
        "value is frequently the finding rather than an error.",
        s["body"]))
    story.append(Paragraph(
        "<b>Missing and duplicate records.</b> Completeness is measured per "
        "column and reported before any analysis. Records are not dropped "
        "to improve a result; where a test required complete cases, the "
        "excluded count is stated alongside it.",
        s["body"]))
    story.append(Paragraph(
        "<b>Judgement applied.</b> Where more than one treatment was "
        "defensible, the more conservative was taken: findings that did not "
        "survive correction for multiple testing were dropped rather than "
        "reported with a caveat, effect sizes below the level that would "
        "change a decision were left out, and no figure was carried into a "
        "conclusion that the underlying column could not support. Candidate "
        "findings withheld on that basis are counted in the findings "
        "section rather than removed silently.",
        s["body"]))
    story.append(Paragraph(
        "<b>Limitations.</b> Findings describe association within this "
        "dataset and the period it covers. They do not establish causation, "
        "and do not extrapolate beyond the observed range of each variable. "
        "Segment-level results with small denominators are marked as "
        "directional. Where a question could not be answered from the data "
        "supplied, this report says so rather than answering it from "
        "general expectation.",
        s["body"]))

    story.append(Paragraph("B. Quality Score Formula", s["h3"]))
    _gtable(story, T,
            ["Component", "Weight", "Description"],
            [["Completeness",  "60%", "% of non-missing cells"],
             ["Deduplication", "30%", "% of unique rows"],
             ["Column Health", "10%", "Avg per-column quality score"]],
            [CW*0.25, CW*0.15, CW*0.60])

    # Reference ranges are listed per detected domain. This list was
    # previously hardcoded to HR sources, so a finance or e-commerce report
    # cited SHRM attrition benchmarks and Gallup engagement data — an
    # immediate credibility failure for any reader who checks.
    # Sources come from the domain blueprint so a finance report cites
    # finance conventions and an HR report cites HR bodies — the previous
    # single list put SHRM and Gallup in the footer of every report
    # regardless of what it was about.
    _bp = blueprint_for(domain)
    _sources = list(_bp.references) or [
        "No external benchmark set applies to this dataset's domain. All "
        "comparisons in this report are internal — each metric is measured "
        "against its own distribution within the supplied data."]
    story.append(Paragraph("C. Reference Ranges & Sources", s["h3"]))
    for src in _sources:
        story.append(Paragraph("• " + src, s["bl"]))
    if _bp.reference_note:
        story.append(Paragraph(_bp.reference_note, s["note"]))

    story.append(Spacer(1, 4*mm))
    disc = Table([[Paragraph(
        "<b>BASIS OF PREPARATION</b><br/>"
        "Prepared for {} on {}. All figures derive solely from the dataset "
        "supplied for this engagement and describe the period it covers. "
        "Statistical association is reported where present; it does not "
        "establish causation. Any external reference range cited is "
        "indicative and should be validated against the organisation's own "
        "sector and prior periods before it informs a decision. "
        "Recommendations assume the data is complete and accurate as "
        "supplied.{}".format(
            config.get("client_name", "Client"),
            datetime.now().strftime("%B %d, %Y"),
            _prepared_by_line(config)),
        s["wh"])]],
        colWidths=["100%"])
    disc.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["header_bg"])),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("BOX",           (0,0), (-1,-1), 1.5, _c(T["accent"])),
    ]))
    story.append(disc)
