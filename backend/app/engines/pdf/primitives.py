"""
engines/pdf/primitives.py — the building blocks every section is made of.

Section headings, KPI strips, narrative boxes, tables, insight cards and
the contents page. A section module should compose these rather than
emit raw flowables, so spacing and colour stay consistent across the
report.
"""
import logging


from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

logger = logging.getLogger(__name__)

from app.engines.pdf.theme import (
    _c, FONT_BODY, FONT_BOLD, FONT_ITALIC,
)

# Shared with the health report builder so both describe the same columns.
from app.engines.pdf_primitives import is_id_col, truncate_label  # noqa: F401


# ══════════════════════════════════════════════════════════
#  REUSABLE COMPONENT HELPERS
# ══════════════════════════════════════════════════════════

def _sec_flowables(s: dict, T: dict, title: str, sub: str = "") -> list:
    """The section header as a list, for callers that need to keep it with
    the content beneath it."""
    head = Paragraph(title, s["h2"])
    # Tagged so the document can record which page it landed on. Without
    # this the contents could only number the sections 1..n, which tells
    # the reader the order of a document they are already holding and not
    # the one thing a contents page exists to tell them.
    head._toc_title = str(title)
    out = [Spacer(1, 3*mm),
           HRFlowable(width="100%", thickness=3,
                      color=_c(T["accent"]), spaceAfter=3),
           head]
    if sub:
        out.append(Paragraph(sub, s["sm"]))
    return out


def _sec(story: list, s: dict, T: dict, title: str, sub: str = ""):
    story.extend(_sec_flowables(s, T, title, sub))


def _kpi_row(story: list, s: dict, T: dict, kpis: list, CW: float):
    cols = min(4, len(kpis))
    cw   = CW / cols
    vals = [Paragraph(
                "<b>{}</b>".format(k.get("value", "")),
                ParagraphStyle("kv", fontName=FONT_BOLD, fontSize=18,
                               textColor=HexColor(k.get("color", T["accent"])),
                               alignment=TA_CENTER))
            for k in kpis[:cols]]
    lbls = [Paragraph(
                k.get("label", ""),
                ParagraphStyle("kl", fontName=FONT_BOLD, fontSize=7.5,
                               textColor=_c(T["text"]), alignment=TA_CENTER))
            for k in kpis[:cols]]
    subs = [Paragraph(
                k.get("sub", ""),
                ParagraphStyle("ks", fontName=FONT_BODY, fontSize=7,
                               textColor=_c(T["text_muted"]), alignment=TA_CENTER))
            for k in kpis[:cols]]
    t = Table([vals, lbls, subs], colWidths=[cw]*cols)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["bg_light"])),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("BOX",           (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]))
    story.append(t)
    story.append(Spacer(1, 3*mm))


def _narrative_box(story: list, s: dict, T: dict, text: str):
    if not text: return
    t = Table([[Paragraph(text, s["body"])]], colWidths=["100%"])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["bg_light"])),
        ("LINEBEFORE",    (0,0), (0,-1),  4, _c(T["accent"])),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 2*mm))


def next_exhibit(s: dict) -> int:
    """The next exhibit number for this report."""
    counter = s.setdefault("_exhibit", {"n": 0})
    counter["n"] += 1
    return counter["n"]


def _exhibit(story: list, s: dict, T: dict, title: str,
             source: str = "") -> int:
    """Label the next table or figure, and say where its numbers came from.

    Both halves matter. The number lets the narrative refer to a specific
    exhibit rather than "the table above", which stops being true as soon
    as a page break moves it. The source line is what separates a figure
    computed from the client's own data from one carried in from
    somewhere else — a distinction a reader is entitled to make without
    asking.
    """
    n = next_exhibit(s)
    story.append(Paragraph(
        '<font color="{}"><b>Exhibit {}</b></font>&nbsp;&nbsp;{}'.format(
            T["accent"], n, title),
        ParagraphStyle("exh", fontName=FONT_BOLD, fontSize=8.5,
                       textColor=_c(T["text"]), spaceAfter=2,
                       spaceBefore=1)))
    if source:
        story.append(Paragraph(
            "Source: {}".format(source),
            ParagraphStyle("exhsrc", fontName=FONT_ITALIC, fontSize=6.8,
                           textColor=_c(T["text_muted"]), spaceAfter=3)))
    return n


def _exhibit_source(story: list, s: dict, T: dict, source: str):
    """A source line under an exhibit that was numbered earlier."""
    story.append(Paragraph(
        "Source: {}".format(source),
        ParagraphStyle("exhsrc2", fontName=FONT_ITALIC, fontSize=6.8,
                       textColor=_c(T["text_muted"]), spaceBefore=1,
                       spaceAfter=3)))


def _gtable(story: list, T: dict, headers: list,
            rows_data: list, col_widths: list,
            severity_col: int = -1):
    """Generic styled table."""
    hrow = [Paragraph(h, ParagraphStyle(
                "th", fontName=FONT_BOLD, fontSize=8,
                textColor=HexColor("#FFFFFF"), alignment=TA_CENTER))
            for h in headers]
    body = []
    for row in rows_data:
        body.append([Paragraph(str(c), ParagraphStyle(
                "td", fontName=FONT_BODY, fontSize=8,
                textColor=_c(T["text"]), leading=12))
                     for c in row])
    tbl = Table([hrow] + body, colWidths=col_widths)
    sty = [
        # ReportLab's TableStyle default is Helvetica, which it declares in
        # the page's font resources even when every cell is a Paragraph
        # carrying its own face. Setting it keeps the document to one
        # family.
        ("FONTNAME",      (0,0), (-1,-1), FONT_BODY),
        ("BACKGROUND",    (0,0), (-1,0),  _c(T["header_bg"])),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#FFFFFF"), _c(T["bg_light"])]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("BOX",           (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]
    if severity_col >= 0:
        _sev = {"CRITICAL": "#EF4444", "HIGH": "#F59E0B",
                "MEDIUM":   "#06B6D4", "LOW":  "#10B981",
                "IMMEDIATE":"#EF4444", "SHORT TERM": "#F59E0B",
                "LONG TERM":"#3B82F6", "WARNING": "#3B82F6",
                "NONE":     "#10B981"}
        for i, row in enumerate(rows_data, 1):
            val = str(row[severity_col]).upper()
            for k, col in _sev.items():
                if k in val:
                    sty += [
                        ("BACKGROUND", (severity_col, i), (severity_col, i), HexColor(col)),
                        ("TEXTCOLOR",  (severity_col, i), (severity_col, i), HexColor("#FFFFFF")),
                        ("FONTNAME",   (severity_col, i), (severity_col, i), FONT_BOLD),
                        ("FONTSIZE",   (severity_col, i), (severity_col, i), 7),
                    ]
                    break
    tbl.setStyle(TableStyle(sty))
    story.append(tbl)
    story.append(Spacer(1, 2*mm))


class KeepWholeIfItFits(KeepTogether):
    """Keep a block whole — unless it is taller than the page.

    `KeepTogether` is unconditional: when its content cannot fit in the
    space that is left it moves the whole block to a fresh page, and when
    the content cannot fit on a *page* it still moves it, then overflows.
    That is what stranded a section heading. `Workforce Analytics Review
    — Findings` was kept together with the first finding card; the card
    ran longer than one page, so it jumped to the next page as an
    indivisible unit while the heading and its subtitle — small enough to
    fit — stayed behind. The reader turned to a page carrying a title and
    nothing else, and the report looked like it had failed to generate.

    A block that cannot fit on any page has no business being atomic, so
    this dissolves it and lets the tables inside split by row. Everything
    that does fit keeps the original behaviour exactly.
    """

    def split(self, aW, aH):
        if getattr(self, "_wrapInfo", None) != (aW, aH):
            self.wrap(aW, aH)
        frame = getattr(self, "_frame", None)
        page = getattr(frame, "_height", None)
        if page and self._H > page:
            return self._content[:]
        return super().split(aW, aH)


def _insight_card_flowables(s: dict, T: dict, ins, CW: float, num=None) -> list:
    """The card as a plain list of flowables.

    Callers that need the card glued to a heading above it must be able
    to put the pieces into *their* block. Handing back a pre-wrapped
    `KeepTogether` made that impossible: an atomic card nested inside an
    outer keep-together is the one thing the outer block cannot place
    next to its heading, which is how a section title ended up alone on
    a page.

    Works with both:
      - Dataclass objects (has .severity, .title, .problem …)
      - Plain dicts (keys: severity, title, problem …)
    """
    def _get(obj, key, default=""):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    sev    = _get(ins, "severity", "info").lower()
    sev_c  = {"critical": T["negative"], "high": T["warning"],
              "warning":  T["info"],     "info": T["text_muted"]}
    sev_bg = {"critical": T["critical_bg"], "high": T["warning_bg"],
              "warning":  T["info_bg"],     "info": T["bg_card"]}
    col = sev_c.get(sev, T["accent"])
    bg  = sev_bg.get(sev, T["bg_card"])

    bs = ParagraphStyle("bi_badge", fontName=FONT_BOLD, fontSize=7.5,
                        textColor=HexColor("#FFFFFF"), alignment=TA_CENTER)
    ts = ParagraphStyle("bi_title", fontName=FONT_BOLD, fontSize=9.5,
                        textColor=_c(T["text"]))
    rl = ParagraphStyle("bi_lbl",   fontName=FONT_BOLD, fontSize=8,
                        textColor=HexColor("#FFFFFF"), alignment=TA_CENTER)
    rv = ParagraphStyle("bi_val",   fontName=FONT_BODY,      fontSize=8.5,
                        textColor=_c(T["text"]),  leading=12.5)

    num_str = "{}. ".format(num) if num else ""
    hdr = Table([[
        Paragraph(_get(ins, "severity", "INFO").upper(), bs),
        Paragraph("{}{}".format(num_str, _get(ins, "title", "")), ts),
    ]], colWidths=[20*mm, CW - 20*mm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), HexColor(col)),
        ("BACKGROUND", (1,0), (1,0), _c(T["bg_light"])),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",      (0,0), (0,0),  "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0),(-1,-1), 8),
        ("BOX",        (0,0), (-1,-1), 0.5, _c(T["border"])),
    ]))

    lw, vw = 26*mm, CW - 26*mm
    rows = [
        [Paragraph(k, rl), Paragraph(_get(ins, fk, ""), rv)]
        for k, fk in [("PROBLEM", "problem"), ("CAUSE",  "cause"),
                      ("EVIDENCE","evidence"),("ACTION", "action"),
                      ("IMPACT",  "impact")]
        if _get(ins, fk, "").strip()
    ]
    if not rows:
        rows = [[Paragraph("DETAIL", rl),
                 Paragraph(str(ins)[:200], rv)]]

    body = Table(rows, colWidths=[lw, vw])
    body.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), _c(T["header_bg"])),
        ("ROWBACKGROUNDS",(1,0),(1,-1),
         [HexColor("#FFFFFF"), _c(T["bg_light"]),
          HexColor("#FFFFFF"), _c(T["bg_light"]), HexColor("#FFFFFF")]),
        ("VALIGN",  (0,0), (-1,-1), "TOP"),
        ("ALIGN",   (0,0), (0,-1),  "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("BOX",       (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("INNERGRID", (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]))
    return [hdr, body, Spacer(1, 4*mm)]


def _insight_card(story: list, s: dict, T: dict, ins, CW: float, num=None):
    """Append one finding card, whole unless it is taller than a page."""
    story.append(KeepWholeIfItFits(
        _insight_card_flowables(s, T, ins, CW, num=num)))


# ══════════════════════════════════════════════════════════
#  TOC PAGE
# ══════════════════════════════════════════════════════════

def _toc(story, s, T, entries, CW, pages=None):
    """Contents with page numbers, and sub-entries indented under their
    parent so a chart does not sit in the list as a peer of the executive
    summary."""
    pages = pages or {}
    _sec(story, s, T, "Contents")
    num_style = ParagraphStyle(
        "tn", fontName=FONT_BOLD, fontSize=10,
        textColor=_c(T["accent"]), alignment=TA_CENTER)
    page_style = ParagraphStyle(
        "tp", fontName=FONT_BODY, fontSize=9,
        textColor=_c(T["text_muted"]), alignment=TA_RIGHT)
    sub_style = ParagraphStyle(
        "ts", parent=s["toc"], leftIndent=6*mm, fontSize=8.5,
        textColor=_c(T["text_muted"]))

    for entry in entries:
        num, title = entry[0], entry[1]
        level = entry[2] if len(entry) > 2 else 0
        page = pages.get(str(title))
        row = Table([[
            Paragraph("" if level else str(num), num_style),
            Paragraph(_clean(title), sub_style if level else s["toc"]),
            Paragraph(str(page) if page else "", page_style),
        ]], colWidths=[9*mm, CW - 23*mm, 14*mm])
        row.setStyle(TableStyle([
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("LINEBELOW",   (0,0), (-1,-1), 0.3, _c(T["border"])),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        story.append(row)


def _clean(text: str) -> str:
    """Escape for ReportLab's mini-HTML parser."""
    return (str(text).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))
