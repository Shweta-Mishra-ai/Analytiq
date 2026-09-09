"""
engines/pdf/domain_sections.py — sections that only apply to some domains.

Today: the appendix and the prepared-by line. Domain deep pages (finance
P&L, and the per-domain equivalents) belong here as they land.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
)

logger = logging.getLogger(__name__)

from app.engines.present import label as _PL, truncate as _fit

# Shared with the contents, which locates a section by matching
# this exact string. When the two drifted apart the appendix
# simply lost its page number.
APPENDIX_TITLE = "Appendix — Methodology, Sources & Glossary"

from app.engines.pdf.theme import (
    _c, FONT_BOLD,
)
from app.engines.pdf.primitives import (
    _sec, _gtable, _clean, truncate_label,
)
from app.engines.report_blueprints import blueprint_for


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


def _appendix(story, s, T, config, CW, domain: str = "general",
              used_terms=None, cleaning_summary=None,
              source_table="source_table"):
    _sec(story, s, T, APPENDIX_TITLE)

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

    # D. The SQL lineage. It sat at pages five and six of the body, ahead
    # of every finding; a reader who wants it will look for it here, and
    # a reader who does not is no longer made to scroll past it.
    if cleaning_summary:
        story.append(Spacer(1, 4*mm))
        from app.engines.pdf.lineage import _sql_lineage_block
        _sql_lineage_block(story, s, T, cleaning_summary, CW,
                           table=source_table)

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


    story.append(Spacer(1, 4 * mm))
    _glossary(story, s, T, CW, used_terms)


# ══════════════════════════════════════════════════════════
#  DOMAIN DEEP PAGES
#
#  A domain that can say something specific gets a page for it. Registered
#  on DomainSpec.deep_page so adding one for a new domain stays a
#  one-file change; domains without one simply do not get the section.
# ══════════════════════════════════════════════════════════


# How many rows of a period table a reader will actually use. Beyond
# this it stops being a summary: an unbounded one turned an eleven-page
# finance report into thirty-three, twenty-two of them a daily revenue
# extract with day-over-day changes of +1,886%.
PERIOD_ROWS = 14

# Above this many distinct periods, daily rows are rolled up.
PERIOD_ROLLUP_AT = 24


def _revenue_by_period(df, period_col, rev_col):
    """(revenue per period, the grain it was rolled up to or "").

    A date column holding two years of daily rows is not a set of
    reporting periods. Summed per day it produces hundreds of lines
    whose period-over-period change is noise — a business does not read
    "Tuesday was 1,886% up on Monday" as a finding.
    """
    series = df.groupby(period_col)[rev_col].sum()
    try:
        series = series.sort_index()
    except Exception:
        logger.debug("period index would not sort", exc_info=True)

    if len(series) <= PERIOD_ROLLUP_AT:
        return series, ""

    as_dates = pd.to_datetime(series.index, errors="coerce")
    if as_dates.isna().mean() > 0.2:
        # Not dates — a long list of labels. Nothing to roll up to, so
        # the caller's row cap is the only defence.
        return series, ""

    dated = pd.Series(series.values, index=as_dates).dropna()
    for rule, grain in (("MS", "calendar months"), ("QS", "quarters"),
                        ("YS", "years")):
        rolled = dated.resample(rule).sum()
        if len(rolled) <= PERIOD_ROLLUP_AT:
            fmt = {"MS": "%b %Y", "QS": "%b %Y", "YS": "%Y"}[rule]
            rolled.index = [d.strftime(fmt) for d in rolled.index]
            return rolled, grain
    rolled = dated.resample("YS").sum()
    rolled.index = [d.strftime("%Y") for d in rolled.index]
    return rolled, "years"


def _finance_page(story, s, T, df, config, CW, profile=None):
    """
    Finance-domain PDF section.
    Generates P&L summary, margin analysis, budget vs actual,
    cost concentration, and period trend.
    All values computed from dataset — no external benchmarks hardcoded.
    """
    def _find(keywords, exclude=None, numeric=True):
        """A column matching one of these words.

        `numeric` is not optional decoration: every value column found
        here is summed or averaged a few lines later, and matching on
        the name alone picked `cost_center` — a text column of "CC-100",
        "CC-200" — for `cost`. Summing it raised ValueError, the whole
        Finance Analysis section was dropped from the report, and the
        reader got a finance report with no finance page in it.
        """
        excl = exclude or []
        for c in df.columns:
            cl = str(c).lower()
            if not any(k in cl for k in keywords):
                continue
            if any(e in cl for e in excl):
                continue
            if numeric and not pd.api.types.is_numeric_dtype(df[c]):
                continue
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
    # These two are read as labels, not measured, so they are the
    # exception to the numeric rule above.
    period_col = _find(["month","quarter","period","year","date"],
                       numeric=False)
    cat_col    = _find(["category","department","cost_center","account",
                        "segment"], numeric=False)
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
            cat_rows = [[_fit(idx, 32), f"{val:,.0f}", f"{val/total_c*100:.1f}%"]
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
            period_rev, grain = _revenue_by_period(df, period_col, rev_col)
            if grain:
                story.append(Paragraph(
                    "Rolled up to {} — the raw column records {:,} distinct "
                    "periods, and a table of those is a data extract rather "
                    "than a summary.".format(grain, df[period_col].nunique()),
                    s["note"]))
            shown = period_rev.tail(PERIOD_ROWS)
            if len(period_rev) > len(shown):
                story.append(Paragraph(
                    "Most recent {} of {:,} periods shown.".format(
                        len(shown), len(period_rev)), s["note"]))
                period_rev = shown

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



# ══════════════════════════════════════════════════════════
#  GLOSSARY
# ══════════════════════════════════════════════════════════

# Terms this report uses that carry a precise meaning a general business
# reader is not obliged to already know. Defined in the language of what
# the number lets you do, not the language of the method that produced it.
GLOSSARY = (
    ("AUC",
     "How well the model orders records by risk, from 0.5 (no better than "
     "shuffling them) to 1.0 (perfect order). It says nothing about how "
     "many records the model flags — that is the threshold."),
    ("Base rate",
     "How often the outcome occurs across the whole dataset. Every claim "
     "about a segment is only meaningful against it: a 20% churn rate in "
     "one region is good news if the base rate is 30%."),
    ("Baseline",
     "The score achieved by the most obvious guess — always predicting the "
     "commonest answer, or always predicting the average. A model that "
     "cannot beat it has found nothing, whatever its accuracy."),
    ("Calibration",
     "Whether a predicted probability matches observed reality: of the "
     "records scored at 30%, do roughly 30% record the outcome. An "
     "uncalibrated score still ranks correctly but cannot be quoted as a "
     "likelihood."),
    ("Confidence interval",
     "The range in which the true value plausibly sits, given that this is "
     "a sample rather than the whole population. A difference smaller than "
     "the interval is not evidence of a change."),
    ("Cross-validation",
     "Testing the model on data it was not trained on, repeatedly, so the "
     "reported accuracy reflects how it will behave on new records rather "
     "than how well it memorised the old ones."),
    ("Effect size",
     "How large a difference is, independent of how certain we are that it "
     "exists. Significance says a difference is real; effect size says "
     "whether it is worth acting on."),
    ("Hit rate (precision)",
     "Of the records the model flags, the share that actually record the "
     "outcome. This is what determines whether an intervention is worth "
     "its cost."),
    ("Lift",
     "How much better targeting by the model is than choosing at random. "
     "A lift of 2.5x means the same effort reaches two and a half times as "
     "many cases."),
    ("Missing not at random",
     "Data that is absent for a reason connected to what is being "
     "measured — a satisfaction score missing precisely for the people who "
     "left. Filling these gaps with an average erases the pattern."),
    ("p-value",
     "The chance of seeing a difference this large if there were really no "
     "difference at all. Below 0.05 by convention, though on a large "
     "sample almost everything clears it."),
    ("Recall",
     "Of all the records that record the outcome, the share the model "
     "finds. Raising it means casting a wider net, which lowers the hit "
     "rate — the trade-off is a budget decision."),
    ("Target leakage",
     "A field that is only filled in once the outcome is already known. It "
     "makes a model look excellent in testing and useless in practice, "
     "because the field is empty when a real prediction is needed."),
    ("Threshold",
     "The score above which a record is treated as at risk. Moving it "
     "trades hit rate against coverage; 0.5 is a convention, not a "
     "recommendation."),
)


def _glossary(story, s, T, CW, used_terms=None):
    """Define the terms the report actually used.

    Restricted to terms that appear in this report — a glossary listing
    concepts the reader never encountered is padding, and padding is how a
    document loses the reader's trust in the parts that matter.
    """
    entries = list(GLOSSARY)
    if used_terms:
        lowered = {t.lower() for t in used_terms}
        entries = [e for e in entries if e[0].lower() in lowered] or entries
    if not entries:
        return
    story.append(Paragraph("Glossary", s["h3"]))
    _gtable(story, T, ["Term", "What it means"],
            [[term, definition] for term, definition in entries],
            [CW * 0.24, CW * 0.76])


# ══════════════════════════════════════════════════════════
#  THE PERFORMANCE PAGE EVERY DOMAIN GETS
# ══════════════════════════════════════════════════════════
# Twelve of the thirteen domains had no deep page at all: finance had a
# bespoke one and the rest went straight from the findings to the
# descriptive statistics. So a healthcare report and a logistics report
# were the same report with different words in the narrative, and
# neither carried a page a practitioner in that field would recognise as
# their own.
#
# This page is generic in code and specific in output, because the
# registry already knows what each domain measures: its KPIs, the
# outcome it cares about, which end of that outcome is the bad one, and
# its published benchmark ranges. Everything below is read from the
# spec, so registering a domain gets it this page.

# How many rows of a breakdown a reader will use before it becomes an
# extract. The same discipline as the period table above.
BREAKDOWN_ROWS = 8

# A dimension with more levels than this is an identifier in disguise;
# below two there is nothing to compare.
MAX_DIMENSION_LEVELS = 12
MIN_DIMENSION_LEVELS = 2

# A group smaller than this cannot carry a rate anybody should act on.
MIN_GROUP_ROWS = 25


def _verdict_against(card, domain: str = "general") -> str:
    """How this KPI reads against its published range, in words.

    The KPI carries a benchmark *key* ("attrition_rate"), which is what
    indexes the published table. Passing it to the column-name lookup —
    which is what an earlier version did — matched nothing, so every row
    of every scorecard read "No published range" even where a sourced
    range existed.
    """
    try:
        from app.engines.industry_benchmarks import DOMAIN_BENCHMARKS
        from app.engines.domains.registry import spec_for
        if card.value is None:
            return "No published range — read against your own history."
        # The card carries the range already rendered as prose. The
        # numeric bounds needed for a verdict live in the published
        # table, indexed by the key on the KPI's own spec.
        key = next((k.benchmark for k in spec_for(domain).kpis
                    if k.label == card.label and getattr(k, "benchmark", "")),
                   "")
        if not key:
            return "No published range — read against your own history."
        bm = (DOMAIN_BENCHMARKS.get(str(domain).lower(), {}) or {}).get(key)
        if bm is None:
            return "No published range — read against your own history."
        low = float(getattr(bm, "low", 0) or 0)
        high = float(getattr(bm, "high", 0) or 0)
        if high <= 0:
            return "No published range — read against your own history."
        value = float(card.value)
        if value < low:
            side = "below"
        elif value > high:
            side = "above"
        else:
            return "Within the typical range of {:g}-{:g}.".format(low, high)
        good = card.higher_is_better
        if good is None:
            reading = "outside"
        elif (side == "above") == bool(good):
            reading = "better than"
        else:
            reading = "worse than"
        return "{} the typical {:g}-{:g} range — {} typical.".format(
            side.capitalize(), low, high, reading)
    except Exception:
        logger.debug("benchmark verdict failed", exc_info=True)
        return "No published range — read against your own history."


def _dimensions(df):
    """Categorical columns worth breaking a rate down by."""
    from app.engines.domains.base import is_id_column
    out = []
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            if is_id_column(col, df[col]):
                continue
            levels = int(df[col].nunique(dropna=True))
            if MIN_DIMENSION_LEVELS <= levels <= MAX_DIMENSION_LEVELS:
                out.append((col, levels))
        except Exception:
            logger.debug("dimension check failed on %s", col, exc_info=True)
    out.sort(key=lambda pair: pair[1])
    return [c for c, _ in out]


def _outcome_column(df, spec):
    """The column this domain's outcome lives in, however it is named."""
    from app.engines.domains._common import binary_mask, find_col
    try:
        if spec.outcome_keywords:
            col = find_col(df, spec.outcome_keywords)
            if col is not None and binary_mask(df[col]) is not None:
                return col, (spec.outcome_noun or "the outcome"), spec.outcome_good
    except Exception:
        logger.debug("declared outcome lookup failed", exc_info=True)
    try:
        from app.engines.domains.outcome_discovery import discover_outcomes
        found = discover_outcomes(df, limit=1)
        if found:
            return found[0].column, found[0].noun, bool(found[0].good)
    except Exception:
        logger.debug("outcome discovery failed", exc_info=True)
    return None, "", False


def _domain_performance_page(story, s, T, df, config, CW, profile=None,
                             domain="general"):
    """KPI scorecard, outcome breakdown, and segment ranking."""
    from app.engines.domains.registry import label_for, spec_for
    from app.engines.kpi_engine import compute_kpis

    spec = spec_for(domain)
    label = label_for(domain) or "Business"
    _sec(story, s, T, "{} Performance Analysis".format(label.title()),
         "Headline measures, who the outcome happens to, and how "
         "segments rank — computed from the submitted data")

    wrote_something = False

    # ── 1. KPI scorecard ──────────────────────────────────
    try:
        cards = compute_kpis(df, domain, limit=6)
    except Exception:
        logger.warning("KPI computation failed", exc_info=True)
        cards = []
    if cards:
        wrote_something = True
        story.append(Paragraph("Headline Measures", s["h3"]))
        rows = []
        for card in cards:
            if card.value is None:
                continue
            if card.unit == "%":
                shown = "{:,.1f}%".format(float(card.value))
            elif abs(float(card.value)) >= 1000:
                shown = "{:,.0f}".format(float(card.value))
            else:
                shown = "{:,.2f}".format(float(card.value))
            rows.append([card.label, shown,
                         _fit(_PL(card.source_column or "—"), 24),
                         _verdict_against(card, domain)])
        if rows:
            _gtable(story, T, ["Measure", "Value", "From column", "Reading"],
                    rows, [CW * 0.22, CW * 0.14, CW * 0.22, CW * 0.42])
            story.append(Spacer(1, 3 * mm))

    # ── 2. Who the outcome happens to ─────────────────────
    outcome_col, noun, good = _outcome_column(df, spec)
    if outcome_col is not None:
        try:
            from app.engines.domains._common import binary_mask
            mask = binary_mask(df[outcome_col])
        except Exception:
            logger.debug("outcome mask failed", exc_info=True)
            mask = None
        if mask is not None:
            overall = float(mask.mean() * 100)
            dims = [c for c in _dimensions(df) if c != outcome_col][:2]
            for dim in dims:
                try:
                    frame = pd.DataFrame({dim: df.loc[mask.index, dim],
                                          "_hit": mask.values})
                    grouped = frame.groupby(dim, observed=True)["_hit"]
                    table = grouped.agg(["size", "mean"])
                    table = table[table["size"] >= MIN_GROUP_ROWS]
                    if len(table) < 2:
                        continue
                    table["rate"] = table["mean"] * 100
                    table = table.sort_values("rate", ascending=bool(good))
                    table = table.head(BREAKDOWN_ROWS)
                except Exception:
                    logger.debug("outcome breakdown failed on %s", dim,
                                 exc_info=True)
                    continue
                wrote_something = True
                story.append(Paragraph(
                    "{} by {}".format(noun.title(), _PL(dim)), s["h3"]))
                story.append(Paragraph(
                    "Across the whole file the rate is {:.1f}%. Groups are "
                    "ordered {}-first, and any holding fewer than {} records "
                    "is left out — a rate over a handful of rows moves on one "
                    "case.".format(overall,
                                   "lowest" if good else "highest",
                                   MIN_GROUP_ROWS),
                    s["note"]))
                rows = []
                for name, row in table.iterrows():
                    delta = float(row["rate"]) - overall
                    rows.append([
                        _fit(str(name), 28),
                        "{:,.0f}".format(float(row["size"])),
                        "{:.1f}%".format(float(row["rate"])),
                        "{:+.1f} pp".format(delta),
                    ])
                _gtable(story, T,
                        [_PL(dim), "Records", noun.title() + " rate",
                         "vs overall"],
                        rows, [CW * 0.34, CW * 0.16, CW * 0.25, CW * 0.25])
                story.append(Spacer(1, 3 * mm))

    # ── 3. Where the value sits ───────────────────────────
    try:
        measure = _primary_measure(df, spec)
    except Exception:
        logger.debug("primary measure lookup failed", exc_info=True)
        measure = None
    if measure is not None:
        for dim in _dimensions(df)[:1]:
            try:
                grouped = df.groupby(dim, observed=True)[measure]
                table = grouped.agg(["size", "sum", "median"])
                table = table[table["size"] >= MIN_GROUP_ROWS]
                if len(table) < 2:
                    continue
                total = float(table["sum"].sum())
                if total <= 0:
                    continue
                table = table.sort_values("sum", ascending=False).head(
                    BREAKDOWN_ROWS)
            except Exception:
                logger.debug("segment ranking failed on %s", dim, exc_info=True)
                continue
            wrote_something = True
            story.append(Paragraph(
                "{} by {}".format(_PL(measure), _PL(dim)), s["h3"]))
            rows = []
            running = 0.0
            for name, row in table.iterrows():
                share = float(row["sum"]) / total * 100
                running += share
                rows.append([
                    _fit(str(name), 26),
                    "{:,.0f}".format(float(row["size"])),
                    "{:,.0f}".format(float(row["sum"])),
                    "{:.1f}%".format(share),
                    "{:.1f}%".format(running),
                ])
            _gtable(story, T,
                    [_PL(dim), "Records", "Total " + _PL(measure),
                     "Share", "Cumulative"],
                    rows, [CW * 0.28, CW * 0.14, CW * 0.22, CW * 0.16,
                           CW * 0.20])
            story.append(Paragraph(
                "Cumulative share is the concentration check: when the top "
                "two lines carry most of the column, an average across all "
                "of them describes none of them.", s["note"]))
            story.append(Spacer(1, 3 * mm))

    if not wrote_something:
        # Never leave the heading standing over nothing.
        story.append(Paragraph(
            "This dataset carries no measure, outcome flag or grouping "
            "column that this section can compute from. The findings and "
            "statistics sections cover what it does support.", s["body"]))


def _primary_measure(df, spec):
    """The number this domain is mostly about."""
    from app.engines.domains._common import binary_mask, find_measure
    from app.engines.domains.base import is_id_column

    def _usable(col):
        """A column that can be summed into a meaningful total."""
        if col is None or col not in df.columns:
            return False
        if is_id_column(col, df[col]):
            return False
        # The domain's chart metrics include its outcome — `on_time` for
        # logistics — and that route reached this function before the
        # numeric filter below could reject it.
        return binary_mask(df[col]) is None

    if spec.chart_metrics:
        col = find_measure(df, list(spec.chart_metrics))
        if _usable(col):
            return col
    # A 0/1 flag summed by group is a count wearing the word "Total":
    # a logistics page reported "Total On Time 1,355 — 54.1% share",
    # which is the number of on-time shipments, not an amount of
    # anything. Flags are covered by the rate tables above.
    numeric = [c for c in df.select_dtypes(include="number").columns
               if not is_id_column(c, df[c])
               and df[c].nunique(dropna=True) > 2]
    if not numeric:
        return None
    # The widest-spread column carries the most information.
    best, best_spread = None, -1.0
    for col in numeric:
        try:
            series = df[col].dropna()
            if series.empty or float(series.sum()) <= 0:
                continue
            spread = float(series.std()) / max(abs(float(series.mean())), 1e-9)
            if spread > best_spread:
                best, best_spread = col, spread
        except Exception:
            logger.debug("spread check failed on %s", col, exc_info=True)
    return best


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
    try:
        if page_fn is not None:
            page_fn(story, s, T, df, config, CW, profile=profile)
        else:
            _domain_performance_page(story, s, T, df, config, CW,
                                     profile=profile, domain=domain)
        return True
    except Exception:
        logger.warning("%s deep page failed — section omitted", domain,
                       exc_info=True)
        return False


def has_deep_page(domain: str) -> bool:
    """Whether this domain contributes a deep page. Checked before the
    contents page is written, so the contents never promises a section the
    report does not contain."""
    # Always true now: a domain without a bespoke page gets the generic
    # performance page, which is built from its own registered KPIs and
    # outcome. Kept as a function because the contents page asks.
    return True


# Wire the finance page onto its domain. Done here rather than in the
# registry because this module imports the registry; attaching from the
# other direction would be a cycle.
try:
    from app.engines.domains.registry import attach_deep_page
    attach_deep_page("finance", _finance_page)
except Exception:
    logger.warning("could not attach the finance deep page", exc_info=True)
