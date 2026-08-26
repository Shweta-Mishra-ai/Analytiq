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

from app.engines.pdf.theme import (
    _c, W, H, CW_DEFAULT, FONT_BODY, FONT_BOLD, FONT_ITALIC,
)
from app.engines.pdf.primitives import (
    _sec, _kpi_row, _narrative_box, _gtable, _clean, truncate_label,
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


# ══════════════════════════════════════════════════════════
#  DOMAIN DEEP PAGES
#
#  A domain that can say something specific gets a page for it. Registered
#  on DomainSpec.deep_page so adding one for a new domain stays a
#  one-file change; domains without one simply do not get the section.
# ══════════════════════════════════════════════════════════

def _finance_page(story, s, T, df, config, CW, profile=None):
    """
    Finance-domain PDF section.
    Generates P&L summary, margin analysis, budget vs actual,
    cost concentration, and period trend.
    All values computed from dataset — no external benchmarks hardcoded.
    """
    def _find(keywords, exclude=None):
        excl = exclude or []
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in keywords) and not any(e in cl for e in excl):
                return c
        return None

    _sec(story, s, T, "Finance Analysis",
         "P&amp;L summary · Margin · Budget vs Actual · Cost breakdown "
         "— all from dataset")

    story.append(Paragraph(
        "All figures below are computed directly from the submitted dataset. "
        "No external benchmarks are embedded. Any general guidance references "
        "are clearly labelled and must be verified against sector-specific data.",
        s["body"]))
    story.append(Spacer(1, 3*mm))

    rev_col    = _find(["revenue","total_revenue","income","turnover","sales_amount"])
    cost_col   = _find(["cost","cogs","cost_of_goods","direct_cost"])
    profit_col = _find(["net_profit","profit","net_income"])
    gross_col  = _find(["gross_profit","gross_income"])
    budget_col = _find(["budget","plan","target","forecast"])
    actual_col = _find(["actual","actuals"], exclude=["target","budget"])
    if budget_col and not actual_col:
        actual_col = rev_col
    period_col = _find(["month","quarter","period","year","date"])
    cat_col    = _find(["category","department","cost_center","account","segment"])
    opex_col   = _find(["opex","operating_expense","overhead"])
    expense_col= _find(["expense","spend","expenditure"])
    val_col    = cost_col or expense_col or opex_col

    # ── P&L Summary Table ─────────────────────────────────────────────────
    # ReportLab parses Paragraph text as markup, so a bare "&" is read as
    # the start of an entity and "P&L" renders as "P&L;" on the page.
    story.append(Paragraph("P&amp;L Summary", s["h3"]))

    pl_rows = []
    total_rev, total_cost, gross_profit, gross_margin = 0, 0, 0, 0
    total_opex, ebitda_proxy, total_profit = 0, 0, 0

    if rev_col:
        total_rev = float(df[rev_col].sum())
        pl_rows.append(["Total Revenue", f"{total_rev:,.0f}", "100.0%", "—"])

    if cost_col:
        total_cost  = float(df[cost_col].sum())
        gross_profit = total_rev - total_cost
        gross_margin = gross_profit / total_rev * 100 if total_rev else 0
        pl_rows.append(["Cost of Goods / Direct Cost", f"({total_cost:,.0f})",
                         f"({total_cost/total_rev*100:.1f}%)" if total_rev else "—",
                         "Dataset computed"])
        pl_rows.append(["Gross Profit", f"{gross_profit:,.0f}",
                         f"{gross_margin:.1f}%",
                         "Revenue minus direct cost"])
    elif gross_col:
        gross_profit = float(df[gross_col].sum())
        gross_margin = gross_profit / total_rev * 100 if total_rev else 0
        pl_rows.append(["Gross Profit", f"{gross_profit:,.0f}",
                         f"{gross_margin:.1f}%", "Dataset computed"])

    if opex_col:
        total_opex   = float(df[opex_col].sum())
        ebitda_proxy = gross_profit - total_opex
        opex_ratio   = total_opex / total_rev * 100 if total_rev else 0
        pl_rows.append(["Operating Expenses (OpEx)", f"({total_opex:,.0f})",
                         f"({opex_ratio:.1f}%)", "Dataset computed"])
        pl_rows.append(["Operating Profit (proxy)", f"{ebitda_proxy:,.0f}",
                         f"{ebitda_proxy/total_rev*100:.1f}%" if total_rev else "—",
                         "Gross profit minus OpEx"])

    if profit_col:
        total_profit = float(df[profit_col].sum())
        net_margin   = total_profit / total_rev * 100 if total_rev else 0
        pl_rows.append(["Net Profit / Income", f"{total_profit:,.0f}",
                         f"{net_margin:.1f}%", "Dataset computed"])

    if pl_rows:
        header = ["Line Item", "Amount", "% Revenue", "Source"]
        all_rows = [header] + pl_rows
        col_w = [CW*0.38, CW*0.22, CW*0.18, CW*0.22]
        t = Table([[Paragraph(str(c), s["h3"] if ri == 0 else s["body"])
                    for c in row]
                   for ri, row in enumerate(all_rows)],
                  colWidths=col_w)
        pl_style = [
            ("BACKGROUND",    (0,0), (-1,0), _c(T["header_bg"])),
            ("TEXTCOLOR",     (0,0), (-1,0), _c("#FFFFFF")),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("GRID",          (0,0), (-1,-1), 0.3, _c("#E2E8F0")),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [_c("#FFFFFF"), _c("#F8FAFC")]),
        ]
        # Highlight the gross profit row, but only when there is one.
        # These were written as `(...) if cost_col else ("",)` inline, and
        # a bare ("",) is not a style command — ReportLab raised
        # "not enough values to unpack" and the whole finance page was
        # dropped for any dataset without a cost column.
        if cost_col and len(all_rows) > 2:
            pl_style.append(("BACKGROUND", (0,2), (-1,2), _c("#F0FDF4")))
            pl_style.append(("FONTNAME",   (0,2), (-1,2), FONT_BOLD))
        t.setStyle(TableStyle(pl_style))
        story.append(t)
        story.append(Spacer(1, 3*mm))

        # Margin summary chips
        if gross_margin:
            margin_color = T["positive"] if gross_margin > 40 else T["warning"] if gross_margin > 20 else T["negative"]
            margin_box = Table([[Paragraph(
                f"<b>Gross Margin: {gross_margin:.1f}%</b> | "
                f"{'Healthy — focus on protecting it.' if gross_margin > 40 else 'Moderate — review cost drivers.' if gross_margin > 20 else 'Low — immediate cost review required.'}"
                f" (Computed from dataset — compare to your prior periods, not generic norms.)",
                s["note"])]],
                colWidths=[CW])
            margin_box.setStyle(TableStyle([
                ("LEFTPADDING",  (0,0),(0,0), 10),
                ("TOPPADDING",   (0,0),(0,0), 8),
                ("BOTTOMPADDING",(0,0),(0,0), 8),
                ("BOX",          (0,0),(0,0), 1, _c(margin_color)),
            ]))
            story.append(margin_box)
            story.append(Spacer(1, 3*mm))

    # ── Budget vs Actual Table ────────────────────────────────────────────
    if budget_col and actual_col and budget_col != actual_col:
        story.append(Paragraph("Budget vs Actual Variance", s["h3"]))
        try:
            comp_col = period_col or cat_col
            if comp_col:
                bva = df.groupby(comp_col)[[budget_col, actual_col]].sum().reset_index()
                bva.columns = ["Period/Category", "Budget", "Actual"]
                bva["Variance"]     = bva["Actual"] - bva["Budget"]
                bva["Variance %"]   = ((bva["Actual"] - bva["Budget"]) /
                                        bva["Budget"].replace(0, np.nan) * 100).round(1)
                bva = bva.sort_values("Variance %", key=abs, ascending=False).head(12)

                bva_header = ["Period / Category", "Budget", "Actual", "Variance", "Variance %"]
                bva_data   = [[truncate_label(str(row["Period/Category"]), 28),
                               f"{row['Budget']:,.0f}",
                               f"{row['Actual']:,.0f}",
                               f"{row['Variance']:+,.0f}",
                               f"{row['Variance %']:+.1f}%"]
                              for _, row in bva.iterrows()]

                all_rows_bva = [bva_header] + bva_data
                col_w_bva    = [CW*0.30, CW*0.17, CW*0.17, CW*0.18, CW*0.18]

                t_bva = Table([[Paragraph(str(c), s["h3"] if ri == 0 else s["body"])
                                for c in row]
                               for ri, row in enumerate(all_rows_bva)],
                              colWidths=col_w_bva)
                t_bva.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0), _c(T["header_bg"])),
                    ("TEXTCOLOR",     (0,0), (-1,0), _c("#FFFFFF")),
                    ("TOPPADDING",    (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                    ("LEFTPADDING",   (0,0), (-1,-1), 5),
                    ("GRID",          (0,0), (-1,-1), 0.3, _c("#E2E8F0")),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1), [_c("#FFFFFF"), _c("#F8FAFC")]),
                ]))
                story.append(t_bva)
                story.append(Spacer(1, 3*mm))

                # Variance summary
                over_n  = int((bva["Variance %"] > 10).sum())
                under_n = int((bva["Variance %"] < -10).sum())
                if over_n + under_n > 0:
                    story.append(Paragraph(
                        f"⚠ {over_n} items exceed budget by >10% | "
                        f"{under_n} items under budget by >10%. "
                        f"A variance trigger of ±10% is commonly used as a review threshold — "
                        f"adjust to your organisation's planning standards.",
                        s["note"]))
        except Exception as e:
            story.append(Paragraph(f"Budget vs actual table unavailable: {e}", s["note"]))

    # ── Cost by Category ──────────────────────────────────────────────────
    if cat_col and val_col:
        story.append(Paragraph("Cost / Expense by Category", s["h3"]))
        try:
            cat_cost = df.groupby(cat_col)[val_col].sum().sort_values(ascending=False).head(10)
            total_c  = float(cat_cost.sum())
            cat_rows = [[str(idx)[:32], f"{val:,.0f}", f"{val/total_c*100:.1f}%"]
                        for idx, val in cat_cost.items()]
            cat_header = ["Category / Segment", "Total Amount", "% of Total"]
            all_cat    = [cat_header] + cat_rows
            col_w_cat  = [CW*0.50, CW*0.28, CW*0.22]
            t_cat = Table([[Paragraph(str(c), s["h3"] if ri == 0 else s["body"])
                            for c in row]
                           for ri, row in enumerate(all_cat)],
                          colWidths=col_w_cat)
            t_cat.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), _c(T["header_bg"])),
                ("TEXTCOLOR",     (0,0), (-1,0), _c("#FFFFFF")),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("LEFTPADDING",   (0,0), (-1,-1), 5),
                ("GRID",          (0,0), (-1,-1), 0.3, _c("#E2E8F0")),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [_c("#FFFFFF"), _c("#F8FAFC")]),
            ]))
            story.append(t_cat)

            top_cat     = str(cat_cost.index[0])
            top_pct     = float(cat_cost.iloc[0] / total_c * 100)
            top3_pct    = float(cat_cost.iloc[:3].sum() / total_c * 100)
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph(
                f"'{top_cat}' = {top_pct:.1f}% of total | "
                f"Top 3 combined = {top3_pct:.1f}%. "
                f"{'High concentration — assess dependency risk.' if top_pct > 50 else 'Moderate concentration — monitor for shifts.'} "
                f"All values from dataset.", s["note"]))
        except Exception as e:
            story.append(Paragraph(f"Cost breakdown unavailable: {e}", s["note"]))

    story.append(Spacer(1, 3*mm))

    # ── Period Trend Summary ──────────────────────────────────────────────
    if period_col and rev_col:
        story.append(Paragraph("Period-over-Period Revenue Summary", s["h3"]))
        try:
            period_rev = df.groupby(period_col)[rev_col].sum()
            try:
                period_rev = period_rev.sort_index()
            except Exception:
                logger.warning("%s unexpected failure", exc_info=True)

            if len(period_rev) >= 2:
                period_rows = [[truncate_label(str(idx), 20), f"{val:,.0f}",
                                f"{(val - period_rev.iloc[max(0,i-1)]) / period_rev.iloc[max(0,i-1)] * 100:+.1f}%"
                                if i > 0 else "—"]
                               for i, (idx, val) in enumerate(period_rev.items())]
                period_header = ["Period", "Revenue", "Change vs Prior"]
                all_period    = [period_header] + period_rows
                col_w_p       = [CW*0.38, CW*0.35, CW*0.27]
                t_p = Table([[Paragraph(str(c), s["h3"] if ri == 0 else s["body"])
                              for c in row]
                             for ri, row in enumerate(all_period)],
                            colWidths=col_w_p)
                t_p.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0), _c(T["header_bg"])),
                    ("TEXTCOLOR",     (0,0), (-1,0), _c("#FFFFFF")),
                    ("TOPPADDING",    (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                    ("LEFTPADDING",   (0,0), (-1,-1), 5),
                    ("GRID",          (0,0), (-1,-1), 0.3, _c("#E2E8F0")),
                    ("ROWBACKGROUNDS",(0,1), (-1,-1), [_c("#FFFFFF"), _c("#F8FAFC")]),
                ]))
                story.append(t_p)
        except Exception as e:
            story.append(Paragraph(f"Period trend unavailable: {e}", s["note"]))

    # Disclaimer
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "All financial metrics above are computed from the submitted dataset only. "
        "No external financial benchmarks are embedded. "
        "Verify all figures with your accounting team before using in board materials.",
        s["note"]))



def _domain_deep_page(story, s, T, df, config, CW, domain, profile=None):
    """Render the domain's deep page, if it has one.

    Looked up through the registry rather than an if/elif on domain name,
    so a new domain's page is wired by registering it.
    """
    try:
        from app.engines.domains.registry import spec_for
        page_fn = spec_for(domain).deep_page
    except Exception:
        logger.warning("deep-page lookup failed for domain %r", domain,
                       exc_info=True)
        return False
    if page_fn is None:
        return False
    try:
        page_fn(story, s, T, df, config, CW, profile=profile)
        return True
    except Exception:
        logger.warning("%s deep page failed — section omitted", domain,
                       exc_info=True)
        return False


def has_deep_page(domain: str) -> bool:
    """Whether this domain contributes a deep page. Checked before the
    contents page is written, so the contents never promises a section the
    report does not contain."""
    try:
        from app.engines.domains.registry import spec_for
        return spec_for(domain).deep_page is not None
    except Exception:
        logger.warning("deep-page check failed for domain %r", domain,
                       exc_info=True)
        return False


# Wire the finance page onto its domain. Done here rather than in the
# registry because this module imports the registry; attaching from the
# other direction would be a cycle.
try:
    from app.engines.domains.registry import attach_deep_page
    attach_deep_page("finance", _finance_page)
except Exception:
    logger.warning("could not attach the finance deep page", exc_info=True)
