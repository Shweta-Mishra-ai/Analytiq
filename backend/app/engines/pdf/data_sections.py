"""
engines/pdf/data_sections.py — the sections that show the numbers.

Data preparation (including the SQL lineage block), dataset overview,
statistics, BI, chart pages and recommendations.
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

from app.engines import present as _P
from app.engines.present import (label as _PL, num as _PN,
                                 truncate as _fit, value as _PV)

from app.engines.pdf.theme import (
    _c, W, H, CW_DEFAULT, FONT_BODY, FONT_BOLD, FONT_ITALIC,
    FONT_SERIF, FONT_SERIF_BOLD,
)
from app.engines.pdf.primitives import (
    _sec, _kpi_row, _narrative_box, _gtable, _insight_card, _clean,
    _exhibit, _exhibit_source, is_id_col, truncate_label,
)
from app.services.dtypes import is_text_dtype, text_columns


# ══════════════════════════════════════════════════════════
#  DATA PREPARATION  (what was changed, and the SQL for it)
# ══════════════════════════════════════════════════════════

def _data_prep_section(story, s, T, cleaning_summary, CW, table="source_table"):
    """Every transformation applied before analysis, with its SQL.

    A reader who cannot see what was changed between the file they sent
    and the numbers they are reading has to take the whole report on
    trust. Listing each step — and the statement that reproduces it in
    their own warehouse — is what makes the rest checkable.
    """
    if not cleaning_summary:
        return
    actions = cleaning_summary.get("actions")
    if not actions:
        # Older cached summaries carry only the display groups. Flattening
        # them loses execution order, so the SQL block is suppressed rather
        # than printed in an order that would not reproduce the table.
        groups = cleaning_summary.get("groups") or {}
        actions = [a for g in groups.values() for a in g]
        ordered = False
    else:
        ordered = True
    if not actions:
        return

    _sec(story, s, T, "Data Preparation",
         "Every change made to the source data before any figure was computed")

    story.append(Paragraph(
        "The source file contained {:,} rows across {} columns. After the "
        "steps below, {:,} rows and {} columns were carried into the "
        "analysis. Nothing else was altered.".format(
            cleaning_summary.get("original_rows", 0),
            cleaning_summary.get("original_cols", 0),
            cleaning_summary.get("cleaned_rows", 0),
            cleaning_summary.get("cleaned_cols", 0)),
        s["body"]))
    story.append(Spacer(1, 2*mm))

    rows = []
    for a in actions[:18]:
        rows.append([
            getattr(a, "column", ""),
            getattr(a, "issue", ""),
            getattr(a, "action", ""),
            "{:,}".format(getattr(a, "rows_affected", 0) or 0),
        ])
    _gtable(story, T, ["Column", "Observed", "Treatment", "Rows"],
            rows, [CW*x for x in [0.18, 0.30, 0.42, 0.10]])
    if len(actions) > 18:
        story.append(Paragraph(
            "{} further steps of the same kinds are listed in the "
            "accompanying SQL script.".format(len(actions) - 18), s["note"]))

    # ── The same steps as SQL ──────────────────────────────
    sql_actions = [a for a in actions if getattr(a, "sql", "")] if ordered else []
    if not sql_actions:
        return
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Equivalent SQL", s["h3"]))
    story.append(Paragraph(
        "The analysis itself was performed in pandas. These statements "
        "express the same treatment against <b>{}</b> so your data team can "
        "verify each step, or apply it upstream and stop the issue "
        "recurring. Order matters: deduplicating after imputing does not "
        "give the same table. None of it has been executed.".format(table),
        s["body"]))
    story.append(Spacer(1, 1.5*mm))

    mono = ParagraphStyle(
        "sqlmono", fontName="Courier", fontSize=7.2, leading=9.6,
        textColor=_c(T["text"]), leftIndent=4, spaceAfter=0)
    lines = []
    for a in sql_actions[:14]:
        stmt = a.sql.replace("{table}", '"{}"'.format(table))
        for raw in stmt.splitlines():
            lines.extend(_wrap_sql_line(raw))
        lines.append("")
    block = [[Paragraph(_sql_escape(ln) or "&nbsp;", mono)] for ln in lines[:70]]
    tbl = Table(block, colWidths=[CW])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), _c(T["bg_light"])),
        ("BOX",          (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    story.append(tbl)
    if len(lines) > 70:
        story.append(Paragraph(
            "Truncated for length — the full script is available from the "
            "Data Quality screen.", s["note"]))


_SQL_COLS = 96   # fits Courier 7.2pt across the content frame


def _wrap_sql_line(line: str) -> list:
    """Break one SQL line to the page width, at a space where possible.

    A `GROUP BY` over thirty columns is a single line in the script. Left
    to itself the Paragraph cannot break it (every space is
    non-breaking), so it runs off the page and the reader loses the tail
    of the statement without any sign that it happened. Continuations are
    indented so the break is visibly a wrap, not a new statement.
    """
    line = line.rstrip()
    if len(line) <= _SQL_COLS:
        return [line]
    indent = len(line) - len(line.lstrip())
    out, rest = [], line
    while len(rest) > _SQL_COLS:
        cut = rest.rfind(" ", indent + 1, _SQL_COLS)
        if cut <= indent:
            cut = _SQL_COLS
        out.append(rest[:cut].rstrip())
        rest = " " * (indent + 4) + rest[cut:].lstrip()
    out.append(rest)
    return out


def _sql_escape(text: str) -> str:
    """Escape SQL for ReportLab's mini-HTML parser.

    `WHERE "x" < 108 OR "x" > 877` is a legal comparison and an illegal
    tag; unescaped, ReportLab swallows the rest of the line. Spaces become
    non-breaking so indentation survives — the wrapping is done by
    `_wrap_sql_line` beforehand, where the break points can be chosen.
    """
    return (str(text).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;")
            .replace(" ", "&nbsp;") if text else "")


# ══════════════════════════════════════════════════════════
#  DATASET OVERVIEW
# ══════════════════════════════════════════════════════════

def _dataset_overview(story, s, T, df, profile, CW):
    _sec(story, s, T, "Dataset Overview & Descriptive Statistics",
         "Column breakdown and statistical summary")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = text_columns(df)
    dt_cols  = df.select_dtypes(include="datetime").columns.tolist()

    _gtable(story, T,
            ["Type", "Count", "Columns (sample)"],
            [["Numeric",     len(num_cols), ", ".join(num_cols[:6])],
             ["Categorical", len(cat_cols), ", ".join(cat_cols[:6])],
             ["DateTime",    len(dt_cols),  ", ".join(dt_cols[:4]) or "None"]],
            [CW*0.20, CW*0.12, CW*0.68])

    if num_cols:
        story.append(Paragraph("Descriptive Statistics", s["h3"]))
        show  = num_cols[:5]
        desc  = df[show].describe().round(3)
        hrow  = ["Stat"] + [c[:10] for c in show]
        rows  = [hrow] + [
            [stat] + [str(desc.loc[stat, c]) for c in show]
            for stat in ["mean","std","min","25%","50%","75%","max"]
            if stat in desc.index
        ]
        cw_s = CW / (len(show) + 1)
        tbl  = Table(rows, colWidths=[cw_s] * (len(show)+1), repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONTNAME",     (0,0), (-1,0),  FONT_BOLD),
            ("FONTNAME",     (0,1), (-1,-1), FONT_BODY),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("TEXTCOLOR",    (0,0), (-1,0),  HexColor("#FFFFFF")),
            ("BACKGROUND",   (0,0), (-1,0),  _c(T["header_bg"])),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [HexColor("#FFFFFF"), _c(T["bg_light"])]),
            ("GRID",         (0,0), (-1,-1), 0.3, _c(T["border"])),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ]))
        story.append(KeepTogether([tbl]))
        story.append(Spacer(1, 2*mm))

        # Skew warning
        for col in num_cols[:6]:
            try:
                sk = float(df[col].skew())
                if abs(sk) > 1.0:
                    story.append(Paragraph(
                        "★ {} is heavily skewed (skew={:.2f}) — "
                        "use median not mean for reporting.".format(col, sk),
                        s["note"]))
            except Exception:
                logger.debug("_dataset_overview: suppressed exception", exc_info=True)


# ══════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════

def _estimates_block(story, s, T, df, CW):
    """Headline averages with the uncertainty that belongs to them.

    "Average order value is 412" and "412, and on this sample it could
    reasonably be anywhere from 388 to 436" support different decisions.
    A point estimate printed alone invites the reader to treat sampling
    noise as a change worth acting on.
    """
    try:
        from app.engines.eda_depth import key_estimates
        estimates = key_estimates(df, max_results=5)
    except Exception:
        logger.warning("estimates block failed", exc_info=True)
        return
    if not estimates:
        return

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Headline Measures, with Uncertainty", s["h3"]))
    _exhibit(story, s, T, "Headline measures and their 95% confidence "
                          "intervals")
    rows = []
    for e in estimates:
        rows.append([
            str(e.column).replace("_", " "),
            "{:,.2f}".format(e.value),
            "{:,.2f} to {:,.2f}".format(e.ci_low, e.ci_high),
            "±{:,.2f}".format(e.margin),
            "{:,}".format(e.n),
        ])
    _gtable(story, T, ["Measure", "Mean", "95% confidence interval",
                       "Margin", "Records"], rows,
            [CW * 0.28, CW * 0.16, CW * 0.26, CW * 0.15, CW * 0.15])
    _exhibit_source(story, s, T,
                    "Computed from the submitted dataset; intervals from "
                    "the t distribution.")
    story.append(Paragraph(
        "The interval is the range in which the true average plausibly "
        "sits, given this sample. A difference smaller than the margin is "
        "not evidence of a change.", s["note"]))
    story.append(Spacer(1, 2 * mm))


def _stats_section(story, s, T, stats_report, CW):
    if stats_report is None: return
    _sec(story, s, T, "Statistical Analysis",
         "Distribution, normality, correlations")

    # Correlation honest-warning box
    warn_t = Table([[Paragraph(
        "<b>⚠ Analyst Note:</b> "
        "Correlation r does NOT mean 'Variable A changes Variable B by r%.' "
        "That is a dangerous misread. r = -0.35 means the two variables share "
        "12.3% of their variance (r² = 0.123). Association only — "
        "NOT causation, NOT magnitude of effect.",
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

    # Distribution summary
    col_stats = getattr(stats_report, "column_stats", {})
    if col_stats:
        story.append(Paragraph("Distribution Summary", s["h3"]))
        for col, cs in list(col_stats.items())[:8]:
            if getattr(cs, "mean", None) is None: continue
            normal = "Normal" if getattr(cs, "is_normal", False) else "Non-normal"
            sk_lbl = getattr(cs, "skew_label", "") or ""
            outs   = getattr(cs, "outlier_count_iqr", 0)
            story.append(Paragraph(
                "• '{}': {} | {} | Outliers: {}".format(col, normal, sk_lbl, outs),
                s["bl"]))

    # Correlations
    corrs = getattr(stats_report, "correlations", [])
    sig   = [c for c in corrs
             if getattr(c, "is_significant", False)
             and abs(getattr(c, "pearson_r", 0)) >= 0.15]
    if sig:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Significant Correlations (Correct Interpretation)", s["h3"]))
        rows = [[c.col_a, c.col_b,
                 str(round(c.pearson_r, 4)),
                 str(round(getattr(c, "p_value", 0), 4)),
                 c.strength.title(),
                 "r²={:.3f} — {:.1f}% variance shared. Association only.".format(
                     c.pearson_r**2, c.pearson_r**2 * 100)]
                for c in sig[:6]]
        _gtable(story, T,
                ["Col A", "Col B", "r", "p", "Strength", "Interpretation"],
                rows, [CW*x for x in [0.17, 0.17, 0.08, 0.08, 0.12, 0.38]])


# ══════════════════════════════════════════════════════════
#  BUSINESS INTELLIGENCE
# ══════════════════════════════════════════════════════════

def _bi_section(story, s, T, bi_report, CW):
    if bi_report is None: return
    _sec(story, s, T, "Business Intelligence",
         "Benchmarking, cohort analysis, segment performance")

    brief = getattr(bi_report, "executive_brief", "")
    if brief:
        _narrative_box(story, s, T, brief)

    # Benchmarks
    bms = getattr(bi_report, "benchmarks", [])
    if bms:
        story.append(Paragraph("Benchmarking Summary", s["h3"]))
        rows = [[bm.column, str(bm.mean), str(bm.median),
                 str(bm.top_10_pct), str(bm.bottom_10_pct),
                 bm.benchmark_label.split("—")[0].strip()[:15]]
                for bm in bms[:4]]
        _gtable(story, T,
                ["Metric","Mean","Median","Top 10%","Bottom 10%","Variation"],
                rows, [CW*x for x in [0.22,0.12,0.12,0.12,0.13,0.29]])

    # Cohorts
    sig_c = [c for c in getattr(bi_report, "cohorts", [])
             if c.is_significant]
    if sig_c:
        story.append(Paragraph("Significant Cohort Differences", s["h3"]))
        for c in sig_c[:3]:
            story.append(Paragraph("• " + c.interpretation, s["bl"]))

    # Key insights
    ki = getattr(bi_report, "key_insights", [])
    if ki:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Key Business Insights", s["h3"]))
        for ins in ki[:5]:
            story.append(Paragraph("• " + str(ins), s["bl"]))


# ══════════════════════════════════════════════════════════
#  CHART PAGE
# ══════════════════════════════════════════════════════════

def _chart_page(story, s, T, img_bytes, title, narrative, num, CW,
                source: str = "", df=None, spec=None):
    _sec(story, s, T, title)
    _exhibit(story, s, T, title)
    if img_bytes:
        try:
            img = Image(io.BytesIO(img_bytes),
                        width=CW, height=CW * 0.48)
            story.append(KeepTogether([img, Spacer(1, 2*mm)]))
        except Exception:
            logger.debug("_chart_page: suppressed exception", exc_info=True)
    _exhibit_source(story, s, T,
                    source or "Computed from the submitted dataset.")
    if narrative:
        story.append(Paragraph("Analysis", s["h3"]))
        _narrative_box(story, s, T, narrative)
    if df is not None and spec is not None:
        _chart_figures(story, s, T, df, spec, CW)


def _chart_figures(story, s, T, df, spec, CW) -> None:
    """The numbers the chart was drawn from.

    A bar reaching a little past another bar is a claim the reader has to
    take on trust; the figures beside it are what let them check it, and
    what they will paste into their own deck. It also gives each group's
    row count, which the chart cannot show — a bar built on eleven
    records looks exactly like one built on four hundred.
    """
    try:
        if spec.kind != "bar" or not spec.metric or not spec.dimension:
            return
        if spec.metric not in df.columns or spec.dimension not in df.columns:
            return
        grouped = df.groupby(spec.dimension, observed=True)[spec.metric]
        table = grouped.agg(["mean", "median", "count"]).dropna()
        if table.empty or len(table) > 15:
            return
        table = table.sort_values("mean", ascending=False)
        overall = float(df[spec.metric].mean())

        rows = [[Paragraph("<b>{}</b>".format(_clean(_PL(spec.dimension))),
                           s["sm"])]
                + [Paragraph("<b>{}</b>".format(h), s["sm"])
                   for h in ("Mean", "Median", "Records", "vs overall")]]
        for name, row in table.iterrows():
            delta = row["mean"] - overall
            rows.append([
                Paragraph(_clean(_fit(_PV(name), 34)), s["sm"]),
                Paragraph(_PN(row["mean"]), s["sm"]),
                Paragraph(_PN(row["median"]), s["sm"]),
                Paragraph("{:,}".format(int(row["count"])), s["sm"]),
                Paragraph("{}{}".format("+" if delta >= 0 else "−",
                                        _PN(abs(delta))), s["sm"]),
            ])
        widths = [CW * x for x in (0.36, 0.16, 0.16, 0.16, 0.16)]
        figures = Table(rows, colWidths=widths, repeatRows=1)
        figures.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), _c(T["header_bg"])),
            ("TEXTCOLOR",     (0, 0), (-1, 0), white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, _c(T["bg_light"])]),
            ("ALIGN",         (1, 0), (-1, -1), "RIGHT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("GRID",          (0, 0), (-1, -1), 0.3, _c(T["border"])),
        ]))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("The figures behind this exhibit", s["h3"]))
        story.append(figures)
        story.append(Paragraph(
            "Overall mean {}. A group's mean is only as steady as its "
            "record count — read the small ones as directional."
            .format(_PN(overall)), s["note"]))
    except Exception:
        logger.warning("could not tabulate the figures behind %r",
                       getattr(spec, "metric", "?"), exc_info=True)


# ══════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════

def _recommendations(story, s, T, actions, CW, insights=None):
    """The action plan, as something a client can run a meeting from.

    This page used to be a list of sentences: what to do, and nothing
    else. An action with no evidence beside it is an opinion, and one
    with nowhere to write an owner and a date is a suggestion nobody
    picks up. Each row now carries the finding it answers, and leaves the
    two columns only the client can fill.
    """
    _sec(story, s, T, "Recommendations & Action Plan",
         "Each action, the finding behind it, and space to assign it")

    pri_map = {
        "CRITICAL":   (T["negative"],  T["critical_bg"]),
        "SHORT TERM": (T["warning"],   T["warning_bg"]),
        "LONG TERM":  (T["info"],      T["info_bg"]),
    }

    # Rows come from the findings themselves, so the action and the
    # evidence beside it are the pair the engine actually produced. The
    # first attempt matched them on shared words and cited an income
    # finding next to an overtime action — the same guessing that
    # captioned one chart with another chart's narrative.
    ordered, seen = [], set()
    rank = {"critical": 0, "high": 1, "warning": 2}
    for ins in sorted(insights or [],
                      key=lambda i: rank.get(
                          str(getattr(i, "severity", "")).lower(), 3)):
        step = str(getattr(ins, "action", "")).split("2.")[0]
        step = step.lstrip("1.").strip()
        if not step or step.lower() in seen:
            continue
        seen.add(step.lower())
        severity = str(getattr(ins, "severity", "")).lower()
        ordered.append((
            "CRITICAL" if severity == "critical"
            else "SHORT TERM" if severity in ("high", "warning")
            else "LONG TERM",
            step,
            str(getattr(ins, "problem", "")) or str(getattr(ins, "title", "")),
        ))

    # Then the standalone recommendations, which answer no single finding
    # and so cite none.
    for action in (actions or []):
        text = str(action)
        priority = "LONG TERM"
        for candidate in ("CRITICAL", "SHORT TERM"):
            if candidate in text.upper():
                priority = candidate
                break
        text = (text.replace("[CRITICAL] ", "")
                    .replace("[SHORT TERM] ", "")
                    .replace("[LONG TERM] ", "").strip())
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        ordered.append((priority, text, ""))

    head = [Paragraph("<b>{}</b>".format(h), s["sm"])
            for h in ("Priority", "Action", "Because", "Owner", "By when")]
    rows = [head]
    tones = []
    for priority, text, because in ordered[:9]:
        col, bg = pri_map.get(priority, (T["accent"], T["bg_light"]))
        tones.append((col, bg))
        rows.append([
            Paragraph(priority, ParagraphStyle(
                "pri", fontName=FONT_BOLD, fontSize=6.5,
                textColor=HexColor(col), alignment=TA_CENTER)),
            Paragraph(_clean(text), s["sm"]),
            Paragraph(_clean(_fit(because, 150)) if because
                      else '<font color="{}">—</font>'.format(T["text_muted"]),
                      s["sm"]),
            Paragraph("", s["sm"]),
            Paragraph("", s["sm"]),
        ])

    if len(rows) == 1:
        story.append(Paragraph(
            "No action met the evidence threshold for inclusion. That is a "
            "result rather than an omission: the analysis ran and found "
            "nothing it could recommend on this data alone.", s["body"]))
        return

    table = Table(rows, colWidths=[CW * x for x in
                                   (0.11, 0.34, 0.33, 0.11, 0.11)],
                  repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0), _c(T["header_bg"])),
        ("TEXTCOLOR",     (0, 0), (-1, 0), white),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("GRID",          (0, 0), (-1, -1), 0.3, _c(T["border"])),
        # The two blank columns are meant to be written in, on paper or
        # on screen, so they are tinted rather than left to read as an
        # empty cell someone forgot to fill.
        ("BACKGROUND",    (3, 1), (-1, -1), _c(T["bg_light"])),
    ]
    for i, (col, bg) in enumerate(tones, start=1):
        style.append(("LINEBEFORE", (0, i), (0, i), 3, HexColor(col)))
        style.append(("BACKGROUND", (0, i), (0, i), HexColor(bg)))
    table.setStyle(TableStyle(style))
    story.append(table)

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "Owner and date are left for you to assign — the analysis can say "
        "what to do and why, not who should do it. Every recommendation "
        "rests solely on the dataset provided; confirm with the people who "
        "know the process before acting.", s["sm"]))

# ══════════════════════════════════════════════════════════
#  DATA GOVERNANCE
# ══════════════════════════════════════════════════════════

_SENSITIVITY_ORDER = {"special": 0, "direct": 1, "quasi-identifier": 2,
                      "none": 3}


def _governance_section(story, s, T, record, CW, integrity=None):
    """What the data is, where it came from, and who it could identify.

    The half of a deliverable a client's data owner asks for, usually
    after the analysis has already been circulated.
    """
    _sec(story, s, T, "Data Governance",
         "Provenance, classification, and re-identification risk")

    facts = [("Source", record.source_file or "Supplied dataset"),
             ("Received", record.ingested_at or "—"),
             ("Scope", "{} rows x {} columns".format(_P.count(record.rows),
                                                     record.columns))]
    if record.retention_days:
        facts.append(("Retention", "{} days".format(record.retention_days)))
    _gtable(story, T, ["", ""],
            [[k, v] for k, v in facts], [CW * 0.25, CW * 0.75])
    story.append(Spacer(1, 3*mm))

    # What this classification requires of whoever holds the file.
    if record.obligations:
        story.append(Paragraph("What this means for handling", s["h3"]))
        for item in record.obligations:
            _narrative_box(story, s, T, _clean(item))
        story.append(Spacer(1, 2*mm))

    risk = record.reidentification
    if risk is not None:
        story.append(Paragraph("Re-identification risk", s["h3"]))
        _kpi_row(story, s, T, [
            {"label": "SMALLEST GROUP", "value": "{:,}".format(risk.k_min),
             "sub": "records sharing a profile",
             "color": T["negative"] if risk.k_min < 5 else T["positive"]},
            {"label": "UNIQUE RECORDS", "value": "{:,}".format(risk.unique_rows),
             "sub": "{}% of the file".format(risk.unique_pct),
             "color": T["negative"] if risk.unique_pct >= 5
                      else T["positive"]},
            {"label": "RISK", "value": risk.verdict, "sub": "on these fields",
             "color": {"High": T["negative"], "Moderate": T["warning"]}
                      .get(risk.verdict, T["positive"])},
        ], CW)
        story.append(Paragraph(
            "Measured as k-anonymity over {}: how many records share each "
            "combination of those fields. A group of one is a person who "
            "can be picked out by anyone holding the same facts about them, "
            "whether or not a name column is present.".format(
                _P.join_and([_PL(c) for c in risk.quasi_identifiers],
                            limit=4)),
            s["note"]))
        story.append(Spacer(1, 3*mm))

    # The data dictionary. Sensitive columns first, because they are what
    # the page is read for.
    if record.dictionary:
        story.append(Paragraph("Data dictionary", s["h3"]))
        rows = [[Paragraph("<b>{}</b>".format(h), s["sm"])
                 for h in ("Field", "Type", "Role", "Sensitivity",
                           "Complete", "Distinct", "Example")]]
        ordered = sorted(record.dictionary,
                         key=lambda c: (_SENSITIVITY_ORDER.get(c.sensitivity, 3),
                                        c.name))
        for col in ordered[:40]:
            rows.append([
                Paragraph(_clean(col.label), s["sm"]),
                Paragraph(_clean(col.dtype), s["sm"]),
                Paragraph(col.role, s["sm"]),
                Paragraph(
                    "—" if col.sensitivity == "none" else
                    '<font color="{}"><b>{}</b></font>'.format(
                        T["negative"] if col.sensitivity in
                        ("direct", "special") else T["warning"],
                        col.sensitivity), s["sm"]),
                Paragraph("{:.0f}%".format(col.completeness_pct), s["sm"]),
                Paragraph("{:,}".format(col.distinct), s["sm"]),
                Paragraph(_clean(_fit(col.example, 22)), s["sm"]),
            ])
        table = Table(rows, colWidths=[CW * x for x in
                                       (0.22, 0.11, 0.13, 0.17, 0.10,
                                        0.11, 0.16)], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), _c(T["header_bg"])),
            ("TEXTCOLOR",      (0, 0), (-1, 0), white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, _c(T["bg_light"])]),
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 5),
            ("GRID",           (0, 0), (-1, -1), 0.3, _c(T["border"])),
        ]))
        story.append(table)
        if len(record.dictionary) > 40:
            story.append(Paragraph(
                "{} further columns are not listed here; the full "
                "dictionary is available from the API."
                .format(len(record.dictionary) - 40), s["note"]))

    if record.lineage:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("What was done to the data", s["h3"]))
        for step in record.lineage[:12]:
            story.append(Paragraph("\u2022 " + _clean(str(step)), s["body"]))

    if record.retention_note:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(_clean(record.retention_note), s["note"]))

    if integrity:
        _integrity_block(story, s, T, integrity, CW)


_VERDICT_WORDING = {
    "intact": ("Verified",
               "The figures in this report were computed from the data as "
               "received. Every change made to it between upload and "
               "analysis is recorded below."),
    "unaccounted": ("Unverified",
                    "The working copy of the data does not match any "
                    "recorded change, so at least one modification was made "
                    "outside the audited path. Treat the figures in this "
                    "report as unconfirmed until the data is re-uploaded."),
    "tampered": ("Audit trail broken",
                 "The record of changes has been edited or truncated since "
                 "it was written. The data itself still matches what was "
                 "received, but its history can no longer be relied on."),
    "compromised": ("Failed",
                    "The stored copy of the uploaded data no longer matches "
                    "the digest taken when it was received. Do not rely on "
                    "any figure in this report."),
    "unverifiable": ("Not tracked",
                     "This dataset predates integrity tracking, so its "
                     "history cannot be checked. Re-upload it to bring it "
                     "under the audit trail."),
}


def _integrity_block(story, s, T, integrity, CW):
    """Where the numbers came from and whether anything moved them.

    A report is an assertion about someone's business, and the reader has
    no way to check it against the source. This is the page that makes
    the assertion falsifiable: the digest of the file as received, the
    digest of the data the figures were computed from, and the list of
    every step in between. Anyone holding the original file can recompute
    the first hash themselves and see whether it matches.
    """
    record = integrity.get("record") or {}
    verdict = integrity.get("verdict") or {}
    audit = integrity.get("audit") or []
    manifest = integrity.get("manifest") or {}

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Data integrity", s["h3"]))

    label, wording = _VERDICT_WORDING.get(
        verdict.get("verdict", ""), ("Unknown", ""))
    good = bool(verdict.get("intact"))
    story.append(Paragraph(
        '<font color="{}"><b>{}</b></font> — {}'.format(
            T["positive"] if good else T["negative"], label, _clean(wording)),
        s["body"]))
    story.append(Spacer(1, 2*mm))

    facts = []
    if record.get("source_sha256"):
        facts.append(("Source file digest (SHA-256)",
                      _grouped(record["source_sha256"])))
    if record.get("source_bytes"):
        facts.append(("Source file size",
                      "{} bytes".format(_P.count(record["source_bytes"]))))
    if record.get("raw_digest"):
        facts.append(("Data as received", _grouped(record["raw_digest"])))
    if record.get("active_digest"):
        facts.append(("Data analysed", _grouped(record["active_digest"])))
    if facts:
        # Markup rather than a Paragraph object: _gtable wraps every cell
        # in a Paragraph of its own, so a Paragraph passed in renders as
        # its repr.
        _gtable(story, T, ["", ""],
                [[k, '<font face="Courier">{}</font>'.format(v)]
                 for k, v in facts],
                [CW * 0.34, CW * 0.66])
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            "Digests are shown in groups of eight for readability; the "
            "spaces are not part of the value. The source digest is taken "
            "from the uploaded bytes before parsing, so anyone holding the "
            "original file can recompute it and confirm this report was "
            "built from that file and no other.", s["note"]))

    if audit:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("Chain of custody", s["h3"]))
        rows = [[Paragraph("<b>{}</b>".format(h), s["sm"])
                 for h in ("#", "When", "Event", "Detail")]]
        for entry in audit[-15:]:
            rows.append([
                Paragraph(str(entry.get("seq", "")), s["sm"]),
                Paragraph(_fmt_ts(entry.get("at")), s["sm"]),
                Paragraph(_clean(str(entry.get("event", ""))).title(), s["sm"]),
                Paragraph(_clean(_audit_detail(entry)), s["sm"]),
            ])
        table = Table(rows, colWidths=[CW * x for x in (0.06, 0.24, 0.16, 0.54)],
                      repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), _c(T["header_bg"])),
            ("TEXTCOLOR",      (0, 0), (-1, 0), white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, _c(T["bg_light"])]),
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 5),
            ("GRID",           (0, 0), (-1, -1), 0.3, _c(T["border"])),
        ]))
        story.append(table)
        if len(audit) > 15:
            story.append(Paragraph(
                "The {} earlier entries are omitted here; the full trail is "
                "available from the API.".format(len(audit) - 15), s["note"]))
        story.append(Paragraph(
            "Each entry carries the hash of the one before it, so an entry "
            "that was removed or altered after the fact breaks the chain and "
            "is reported rather than hidden.", s["note"]))

    if manifest:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph("Computed with", s["h3"]))
        parts = [f"{k.replace('_', '-')} {v}" for k, v in manifest.items()
                 if v and v != "unavailable"]
        story.append(Paragraph(
            _clean(", ".join(parts)) + ". Version numbers are recorded "
            "because they move results: a quantile, a solver default or a "
            "tie-break rule can change between releases, and a figure that "
            "cannot be reproduced on the same versions has not been "
            "reproduced.", s["note"]))


def _grouped(digest: str, size: int = 8) -> str:
    """A 64-character hex string has no spaces, so a table cell cannot
    wrap it — it overflows the column silently. Grouping gives the
    renderer break points and gives the reader something checkable by
    eye."""
    digest = str(digest or "")
    return " ".join(digest[i:i + size] for i in range(0, len(digest), size))


def _fmt_ts(value) -> str:
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(float(value), timezone.utc).strftime(
            "%d %b %Y %H:%M UTC")
    except (TypeError, ValueError):
        return "—"


def _audit_detail(entry: dict) -> str:
    detail = entry.get("detail") or {}
    if not isinstance(detail, dict) or not detail:
        return "—"
    parts = []
    for key, value in list(detail.items())[:4]:
        if isinstance(value, (list, tuple)):
            value = "{} item(s)".format(len(value))
        parts.append("{}: {}".format(_PL(str(key)), value))
    return "; ".join(parts)
