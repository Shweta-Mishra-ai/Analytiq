"""
engines/pdf/lineage.py — what changed to the data, and the SQL for it.

Two audiences, two places. The table of changes belongs in the body: a
reader who cannot see what happened between the file they sent and the
figures they are reading has to take the whole report on trust. The
script that reproduces those changes belongs in the appendix, where a
data team can find it — it ran to two full pages ahead of the first
finding, which is not what a document prepared for a client should open
with.
"""
import logging

from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

logger = logging.getLogger(__name__)

from app.engines.pdf.theme import _c
from app.engines.pdf.primitives import _sec, _gtable


# ══════════════════════════════════════════════════════════
#  DATA PREPARATION  (what was changed, and the SQL for it)
# ══════════════════════════════════════════════════════════

def _data_prep_section(story, s, T, cleaning_summary, CW, table="source_table"):
    """Every transformation applied between the file supplied and the
    figures reported.

    A reader who cannot see what changed has to take the whole report on
    trust. This is the table of changes; the SQL that reproduces them is
    in the appendix, where a data team can find it and everyone else can
    walk past it.
    """
    if not cleaning_summary:
        return
    actions = cleaning_summary.get("actions")
    if not actions:
        # Older cached summaries carry only the display groups, which have
        # lost execution order. They can still show what changed; only the
        # SQL depends on the order, and `_sql_actions` declines to build a
        # script it cannot order correctly.
        groups = cleaning_summary.get("groups") or {}
        actions = [a for g in groups.values() for a in g]
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
            "{} further steps of the same kinds were applied; all of them "
            "appear in the script in Appendix D.".format(len(actions) - 18),
            s["note"]))

    # The SQL that reproduces these steps used to print here, and ran to
    # two full pages before the reader had reached a single finding — a
    # `DELETE ... USING (SELECT MIN(ctid) ...)` on page six of a document
    # prepared for a client. The script is worth keeping and worth
    # nothing in the middle of the argument, so it moved to the appendix
    # and this says where.
    if _sql_actions(cleaning_summary):
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            "The statements that reproduce every step above against "
            "<b>{}</b>, in the order they must run, are set out in "
            "Appendix D.".format(table), s["note"]))


def _sql_actions(cleaning_summary) -> list:
    """The cleaning steps that carry SQL, in execution order.

    Order is the whole point: deduplicating after imputing does not give
    the same table. A summary that only kept the display groups has lost
    it, so those produce no script rather than one that would not
    reproduce the result.
    """
    if not cleaning_summary:
        return []
    actions = cleaning_summary.get("actions")
    if not actions:
        return []
    return [a for a in actions if getattr(a, "sql", "")]


def _sql_lineage_block(story, s, T, cleaning_summary, CW,
                       table="source_table"):
    """The cleaning steps as SQL — for the appendix, not the argument.

    A data team needs this to verify each step or to apply it upstream
    and stop the issue recurring. A chief executive reading the same
    document needs it not to be between them and the findings.
    """
    sql_actions = _sql_actions(cleaning_summary)
    if not sql_actions:
        return

    story.append(Paragraph("D. Data Preparation — Equivalent SQL", s["h3"]))
    story.append(Paragraph(
        "The analysis itself was performed in pandas. These statements "
        "express the same treatment against <b>{}</b>, in the order they "
        "must run — deduplicating after imputing does not give the same "
        "table. None of it has been executed against your systems.".format(
            table),
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
            "Truncated for length — the full script, for every SQL dialect, "
            "is on the Data Quality screen.", s["note"]))


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


