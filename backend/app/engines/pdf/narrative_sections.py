"""
engines/pdf/narrative_sections.py — the sections that make an argument.

Executive summary, top insights, data-quality note, readiness, benchmark
context and the attrition page. These read as prose; the tabular
counterparts live in data_sections.py.
"""
import logging

import pandas as pd

from reportlab.lib.units import mm
from reportlab.lib.colors import white
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
    KeepTogether,
)

logger = logging.getLogger(__name__)

from app.engines.pdf.theme import (
    _c, FONT_BODY, FONT_BOLD,
)
from app.engines.pdf.primitives import (
    _sec, _sec_flowables, _kpi_row, _narrative_box, _gtable, _insight_card,
    _clean, _exhibit, _exhibit_source,
)
from app.engines.industry_benchmarks import REPLACEMENT_COST_RANGE
from app.engines.present import (label as _PL, num as _PN,
                                 truncate as _PT)
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
                ParagraphStyle("risk_p", fontName=FONT_BODY, fontSize=9,
                               textColor=_c(T["negative"]), leading=13,
                               leftIndent=10, firstLineIndent=-10,
                               spaceAfter=3)))
        story.append(Spacer(1, 2*mm))
    if opps:
        story.append(Paragraph("Opportunities", s["h3"]))
        for o in opps[:3]:
            story.append(Paragraph("* " + str(o),
                ParagraphStyle("opp_p", fontName=FONT_BODY, fontSize=9,
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
    # The report heading travels with the first finding too, not only the
    # group heading. Kept apart, the section title claimed a page of its
    # own and the reader turned to a sheet carrying nothing but the words
    # "Findings" — the section that is meant to be the report.
    banner = _sec_flowables(s, T, "{} — Findings".format(bp.label),
                            "Each finding: Problem → Cause → Evidence → "
                            "Action → Impact")

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
        story.append(KeepTogether(banner + head + first))
        banner = []
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
         "sub": REPLACEMENT_COST_RANGE.replace("% of annual salary",
                                               "% of salary"),
         "color": T["negative"]},
    ], CW)

    _narrative_box(story, s, T,
                   getattr(attrition, "interpretation", ""))
    _narrative_box(story, s, T,
                   getattr(attrition, "cost_estimate", ""))

    # Drivers
    drivers = getattr(attrition, "top_drivers", [])
    if drivers:
        story.append(Paragraph("What separates leavers from stayers", s["h3"]))
        story.append(Paragraph(
            "Ranked by how far apart the two groups actually sit, not by "
            "how large the percentage difference looks. Every row below "
            "survived a false-discovery correction across all factors "
            "tested; factors that were statistically significant but "
            "separated almost nobody are not shown.", s["sm"]))
        rows = [[_PL(d.get("factor", "")),
                 "{} ({})".format(_PN(d.get("effect", 0), decimals=2),
                                  d.get("effect_label", "—")),
                 _PT(d.get("detail", ""), 78)]
                for d in drivers[:6]]
        _gtable(story, T, ["Factor", "Separation", "What the data shows"],
                rows, [CW*x for x in [0.24, 0.20, 0.56]])
        story.append(Paragraph(
            "Separation is Cliff's delta for a measured factor and "
            "Cramér's V for a grouped one; both run 0 to 1 and both answer "
            "the same question, so the rows can be compared with each "
            "other. A driver is an association, not a proven cause.",
            s["note"]))

    # Dept breakdown
    dept_atr = getattr(attrition, "dept_attrition", {})
    if dept_atr:
        sizes = getattr(attrition, "dept_sizes", {}) or {}
        story.append(Paragraph("Attrition by Department", s["h3"]))
        sorted_d = sorted(dept_atr.items(), key=lambda x: x[1], reverse=True)
        rows = []
        thin = False
        for dept, rate in sorted_d:
            n = sizes.get(str(dept))
            # A rate over a small head count moves several points if one
            # person leaves, so it is marked rather than graded against
            # the same threshold as a department twenty times its size.
            small = bool(n is not None and n < 50)
            thin = thin or small
            status = ("Directional" if small else
                      "CRITICAL" if rate > 25 else
                      "HIGH" if rate > 18 else "OK")
            rows.append([_PL(dept),
                         "{:,}".format(n) if n is not None else "—",
                         "{:.1f}%".format(rate), status])
        _gtable(story, T, ["Department", "People", "Rate", "Status"],
                rows, [CW*0.40, CW*0.16, CW*0.18, CW*0.26], severity_col=3)
        if thin:
            story.append(Paragraph(
                "Departments under 50 people are marked directional: at "
                "that size a single leaver moves the rate by more than the "
                "difference being graded.", s["note"]))


# ══════════════════════════════════════════════════════════
#  REPORT AT A GLANCE
#  The one page an executive reads before deciding whether to
#  read any of the others.
# ══════════════════════════════════════════════════════════

def _exec_dashboard(story, s, T, df, profile, top_insights,
                    executive_summary, CW, top_cluster=None,
                    driver_result=None, avg_salary_k: float = 0.0):
    """One-page summary: scale, quality, the verdict, the sharpest pocket of
    risk, and the findings that need action — with the evidence for each in
    the sections that follow."""
    _sec(story, s, T, "Report at a Glance",
         "One-page summary — full evidence in the sections that follow")
    story.append(Spacer(1, 3 * mm))

    # Run the same guard the Top Insights section runs. This page renders
    # insight titles and actions directly, so without it an unsupported
    # claim ("revenue will collapse next quarter") is withheld from the
    # findings section and then printed on the summary page anyway — the
    # one page most likely to be read on its own.
    from app.engines.insight_guard import guard_insights
    top_insights = guard_insights(list(top_insights or [])).kept

    qual = getattr(profile, "overall_quality_score", None)
    miss = float(df.isna().mean().mean() * 100)
    n_crit = sum(1 for i in (top_insights or [])
                 if getattr(i, "severity", "") == "critical")
    _kpi_row(story, s, T, [
        {"label": "RECORDS ANALYSED", "value": "{:,}".format(len(df)),
         "sub": "{} columns".format(len(df.columns)), "color": T["accent"]},
        {"label": "DATA QUALITY",
         "value": str(qual) if qual is not None else "—",
         "sub": "/ 100", "color": T["positive"]},
        {"label": "MISSING DATA", "value": "{:.1f}%".format(miss),
         "sub": "0% = complete",
         "color": T["positive"] if miss < 1 else T["warning"]},
        {"label": "CRITICAL FINDINGS", "value": str(n_crit),
         "sub": "require action",
         "color": T["negative"] if n_crit else T["positive"]},
    ], CW)
    story.append(Spacer(1, 4 * mm))

    if executive_summary:
        verdict = ". ".join(str(executive_summary).split(". ")[:2]).strip()
        if verdict and not verdict.endswith("."):
            verdict += "."
        if verdict:
            _narrative_box(story, s, T, "<b>Verdict:</b> " + verdict)
            story.append(Spacer(1, 3 * mm))

    # The headline decision: the specific segment, and what it is worth.
    if (top_cluster is not None and driver_result is not None
            and getattr(top_cluster, "n_events", 0) >= 10
            and driver_result.top_drivers):
        tc, dr = top_cluster, driver_result
        lift = tc.rate / tc.base_rate if tc.base_rate else 0
        avoidable = (int(round(dr.high_risk_n *
                               max(dr.high_risk_rate - dr.base_rate, 0) / 100.0))
                     if dr.high_risk_n else 0)
        money = ""
        if avoidable > 0 and avg_salary_k and avg_salary_k > 0:
            lo = avoidable * avg_salary_k * 0.5
            hi = avoidable * avg_salary_k * 2.0
            money = (" On the {:,.0f}K replacement cost supplied for this "
                     "report and the published 50-200% band, avoiding those "
                     "{:,} events is worth <b>{:,.0f}K-{:,.0f}K</b>."
                     .format(avg_salary_k, avoidable, lo, hi))
        elif avoidable > 0:
            # No cost was supplied, so no cost is asserted. The headcount
            # is what the analysis actually establishes; the reader can
            # multiply by a figure they trust.
            money = (" Closing the gap between this group and the base rate "
                     "is worth <b>{:,} events</b> a cycle; multiply by your "
                     "own replacement cost to value it."
                     .format(avoidable))
        _narrative_box(
            story, s, T,
            "<b>Decision headline:</b> the sharpest pocket of risk is "
            "<b>{}</b> — {:,} records at a {:.0f}% rate ({:.1f}x base), "
            "driving {:.0f}% of all events. The factor the model relies on "
            "most is <b>{}</b>.{}".format(
                tc.description, tc.n, tc.rate, lift, tc.share_of_events,
                _PL(dr.top_drivers[0][0]), money))
        story.append(Spacer(1, 3 * mm))

    if top_insights:
        story.append(Paragraph("Top Findings and Required Actions", s["h3"]))
        sev_color = {"critical": T["negative"], "high": T["warning"],
                     "warning": T["warning"], "info": T["info"],
                     "positive": T["positive"]}
        rows = [[Paragraph("<b>Priority</b>", s["sm"]),
                 Paragraph("<b>Finding</b>", s["sm"]),
                 Paragraph("<b>First action</b>", s["sm"])]]
        for ins in top_insights[:4]:
            sev = getattr(ins, "severity", "info")
            first_action = str(getattr(ins, "action", "")).split("2.")[0]
            first_action = first_action.lstrip("1.").strip()
            rows.append([
                Paragraph('<font color="{}"><b>{}</b></font>'.format(
                    sev_color.get(sev, T["info"]), sev.upper()), s["sm"]),
                Paragraph(_PT(getattr(ins, "title", ""), 110), s["sm"]),
                Paragraph(_PT(first_action, 130), s["sm"]),
            ])
        tbl = Table(rows, colWidths=[CW * 0.14, CW * 0.46, CW * 0.40])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  _c(T["header_bg"])),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, _c(T["bg_light"])]),
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",     (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
            ("LEFTPADDING",    (0, 0), (-1, -1), 7),
            ("GRID",           (0, 0), (-1, -1), 0.3, _c(T["border"])),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 5 * mm))

    # Severity mix as a proportional bar — readable in one glance.
    sev_counts = {}
    for ins in (top_insights or []):
        sv = getattr(ins, "severity", "info")
        sev_counts[sv] = sev_counts.get(sv, 0) + 1
    total_f = sum(sev_counts.values())
    if total_f:
        story.append(Paragraph("Findings by Severity", s["h3"]))
        seg_defs = [("critical", "Critical", T["negative"]),
                    ("high", "High", T["warning"]),
                    ("warning", "Medium", T["warning"]),
                    ("info", "Low", T["info"]),
                    ("positive", "Strength", T["positive"])]
        cells, widths, styles = [], [], []
        col = 0
        for key, lbl, color in seg_defs:
            c = sev_counts.get(key, 0)
            if c == 0:
                continue
            cells.append(Paragraph(
                '<font color="#FFFFFF"><b>{} {}</b></font>'.format(c, lbl),
                ParagraphStyle("sgv", fontName=FONT_BOLD, fontSize=8.5,
                               alignment=TA_CENTER, textColor=white)))
            widths.append(CW * c / total_f)
            styles.append(("BACKGROUND", (col, 0), (col, 0), _c(color)))
            col += 1
        if cells:
            segbar = Table([cells], colWidths=widths, rowHeights=[9 * mm])
            segbar.setStyle(TableStyle(styles + [
                ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN",     (0, 0), (-1, -1), "CENTER"),
                ("LINEAFTER", (0, 0), (-2, 0), 1.5, white),
            ]))
            story.append(segbar)
            story.append(Spacer(1, 5 * mm))

    num_n = len(df.select_dtypes(include="number").columns)
    cat_n = len(text_columns(df))
    dt_n = len(df.select_dtypes(include="datetime").columns)
    story.append(Paragraph("Scope of Analysis", s["h3"]))
    _gtable(story, T,
            ["Dimension", "Detail"],
            [["Records examined",
              "{:,} rows across {} fields".format(len(df), len(df.columns))],
             ["Field types",
              "{} numeric · {} categorical · {} date/time".format(
                  num_n, cat_n, dt_n)],
             ["Methods applied",
              "Descriptive statistics · distribution and normality tests · "
              "correlation analysis · segment significance testing"],
             ["Findings surfaced",
              "{} total ({} require action)".format(
                  total_f, sev_counts.get("critical", 0)
                  + sev_counts.get("high", 0))]],
            [CW * 0.26, CW * 0.74])


# ══════════════════════════════════════════════════════════
#  PREDICTIVE RISK
#  Model-based: what the data predicts, not only what it records.
# ══════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════
#  OUTLOOK
# ══════════════════════════════════════════════════════════

def _forecast_section(story, s, T, fc, CW):
    """What the series is projected to do, or why it is not projected.

    Both outcomes are reported. "This measure is a level with noise around
    it, and the planning figure is the current level" is a useful answer —
    more useful than a projection that no test supported.
    """
    if fc is None:
        return

    measure = str(fc.column).replace("_", " ")
    _sec(story, s, T, "Outlook",
         "Projection for {} — and what it was tested against".format(measure))
    story.append(Spacer(1, 3 * mm))

    if not fc.usable:
        _narrative_box(story, s, T,
                       "<b>No forecast is shown.</b> " + fc.verdict)
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            "A projection is only worth printing when it beats the naive "
            "alternative — carrying the last value forward — on periods the "
            "method never saw. Most business series are a level with noise "
            "around them, and for those the current level is the honest "
            "planning number.", s["body"]))
        return

    _kpi_row(story, s, T, [
        {"label": "METHOD", "value": fc.method.split()[0],
         "sub": fc.method, "color": T["accent"]},
        {"label": "HORIZON", "value": "{} {}".format(fc.horizon, fc.freq),
         "sub": "from {:,} periods".format(fc.n_periods),
         "color": T["accent"]},
        {"label": "BEATS NAIVE BY", "value": "{:.0f}%".format(fc.skill * 100),
         "sub": "on held-out periods", "color": T["positive"]},
        {"label": "TYPICAL ERROR", "value": "{:,.1f}".format(fc.model_error),
         "sub": "vs {:,.1f} naive".format(fc.naive_error),
         "color": T["text_muted"]},
    ], CW)
    story.append(Spacer(1, 3 * mm))

    _exhibit(story, s, T,
             "Projected {} with 95% interval".format(measure))
    _gtable(story, T, ["Period", "Projected", "Low", "High", "Width"],
            [[p.period, "{:,.1f}".format(p.value), "{:,.1f}".format(p.lower),
              "{:,.1f}".format(p.upper),
              "±{:,.1f}".format((p.upper - p.lower) / 2)]
             for p in fc.points],
            [CW * 0.22, CW * 0.2, CW * 0.19, CW * 0.19, CW * 0.2])
    _exhibit_source(story, s, T,
                    "Computed from the submitted dataset; interval from "
                    "held-out forecast error.")

    _narrative_box(story, s, T, fc.verdict)
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "The interval widens with distance because uncertainty compounds: "
        "a projection four periods out carries the error of every step "
        "before it. Plan against the range, not the line.", s["note"]))

    if fc.candidates:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("Methods Considered", s["h3"]))
        _exhibit(story, s, T, "Forecast methods and their held-out error")
        _gtable(story, T, ["Method", "Average error on held-out periods"],
                [[n + ("  (selected)" if n == fc.method else ""),
                  "{:,.2f}".format(e)] for n, e in fc.candidates],
                [CW * 0.6, CW * 0.4])
        _exhibit_source(story, s, T,
                        "Each method fitted without the final {} periods, "
                        "then scored on them.".format(
                            max(2, min(4, fc.n_periods // 4))))
