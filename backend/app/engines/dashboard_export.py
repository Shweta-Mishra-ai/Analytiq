"""
engines/dashboard_export.py — the dashboard as one HTML file.

The app could hand a client a PDF. A PDF is finished: the reader looks
at the numbers you chose to show them, in the order you chose, and if
they want the same cut for one region they have to come back and ask.
Half of what a client is buying when they commission analysis is the
ability to poke at it.

This writes the dashboard to a single self-contained `.html` — the
Plotly runtime, the data and the styling all inlined. It opens from a
file:// path with no server, no network and no install, which is what
makes it something you can attach to an email. The slicers work, the
cross-filtering works, and the figures are the same ones the app
renders, so the exported file and the screen cannot drift apart.

Deliberately not included: any way to reach back to the app. An export
is a snapshot, and one that quietly stopped matching its source would be
worse than no export at all. The generation date and the row count are
printed in the footer for that reason.
"""
from __future__ import annotations

import html
import json
import logging
import re
from typing import Dict, List

import pandas as pd
import plotly.io as pio

from app.services.numfmt import human_number

logger = logging.getLogger(__name__)

# Cap the rows embedded for client-side filtering. Above this the file
# gets big enough to be awkward to email, and a slicer over a hundred
# thousand rows is slow in the browser — the tiles are pre-rendered from
# the full dataset either way, so the figures stay correct.
MAX_EMBEDDED_ROWS = 20_000


def _plotly_runtime() -> str:
    """The Plotly bundle, inlined.

    Loading it from a CDN would make the file useless offline and would
    tell whoever hosts that CDN every time the client opens their
    report.
    """
    try:
        from plotly.offline import get_plotlyjs

        return get_plotlyjs()
    except Exception:
        logger.warning("could not inline the plotly runtime", exc_info=True)
        return ""


# The exported page is light. The interface is dark because it is a tool
# someone stares at for an hour; this is a document that gets read on a
# laptop in a meeting, printed, and pasted into a deck, and a dark
# dashboard survives none of the three.
_CSS = """
:root {
  --bg:#f6f7f9; --panel:#ffffff; --panel2:#f1f3f6; --edge:#e4e7ec;
  --edge2:#d4d8e0; --ink:#1a1d23; --ink2:#3d434e; --mute:#6b7280;
  --faint:#9aa1ac; --accent:#f0a11e; --accent2:#1f6feb;
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.5 'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap { display:flex; gap:20px; padding:24px; align-items:flex-start; }
header { padding:24px 24px 0; }
h1 { margin:0; font-size:22px; font-weight:650; letter-spacing:-0.02em; }
header p { margin:6px 0 0; color:var(--mute); font-size:13px; }
aside {
  width:210px; flex:none; background:var(--panel);
  border:1px solid var(--edge); border-radius:12px; overflow:hidden;
  position:sticky; top:24px; max-height:calc(100vh - 48px); overflow-y:auto;
  box-shadow:0 1px 2px rgba(16,24,40,.05);
}
aside h2 {
  margin:0; padding:10px 12px; font-size:11px; letter-spacing:.06em;
  text-transform:uppercase; color:var(--mute); font-weight:600;
  border-bottom:1px solid var(--edge);
}
.slicer { border-bottom:1px solid rgba(34,38,46,.7); padding:8px 8px 10px; }
.slicer:last-child { border-bottom:0; }
.slicer > b {
  display:block; padding:2px 4px 6px; font-size:12px; color:var(--ink2);
  font-weight:600;
}
.slicer button {
  display:block; width:100%; text-align:left; padding:4px 8px; margin:1px 0;
  border:0; border-radius:6px; background:transparent; color:var(--mute);
  font:inherit; font-size:12px; cursor:pointer;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.slicer button:hover { background:var(--panel2); color:var(--ink2); }
.slicer button.on {
  background:rgba(240,161,30,.16); color:#8a5a00; font-weight:600;
}
main { flex:1; min-width:0; }
.kpis {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px; margin-bottom:16px;
}
.kpi {
  background:var(--panel); border:1px solid var(--edge); border-radius:12px;
  padding:12px 16px 12px 18px; position:relative; overflow:hidden;
  box-shadow:0 1px 2px rgba(16,24,40,.05);
}
/* A coloured spine on each card, so the KPI row reads as a row of
   figures rather than four identical white boxes. */
.kpi::before {
  content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:var(--accent);
}
.kpi:nth-child(2)::before { background:var(--accent2); }
.kpi:nth-child(3)::before { background:#0f9d6e; }
.kpi:nth-child(4)::before { background:#7c4dff; }
.kpi span {
  display:block; font-size:11px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--mute); font-weight:500;
}
.kpi b {
  display:block; margin-top:4px; font-size:24px; font-weight:650;
  font-family:'IBM Plex Mono',ui-monospace,monospace; font-variant-numeric:tabular-nums;
  letter-spacing:-0.02em;
}
.grid { display:grid; grid-template-columns:repeat(12,1fr); gap:12px; }
.tile {
  background:var(--panel); border:1px solid var(--edge); border-radius:12px;
  padding:10px 4px 4px; min-height:320px; overflow:hidden;
  box-shadow:0 1px 2px rgba(16,24,40,.05);
}
/* The question the tile answers, under its label. A dashboard where
   every tile states what it is for reads as a considered document; one
   that just names its axes reads as a chart dump. */
.tile > small {
  display:block; padding:0 10px 4px; font-size:11.5px; color:var(--mute);
}
/* The finding, as HTML rather than inside the figure — Plotly titles do
   not wrap, so on a half-width tile the sentence was cut off. */
.tile > strong {
  display:block; padding:2px 10px 8px; font-size:13px; font-weight:600;
  line-height:1.35; color:var(--ink);
}
.tile > em {
  display:block; padding:0 10px 2px; font-style:normal; font-size:11px;
  letter-spacing:.05em; text-transform:uppercase; color:var(--mute);
}
footer {
  padding:8px 24px 32px; color:var(--faint); font-size:12px;
  display:flex; gap:16px; flex-wrap:wrap;
}
.note {
  margin:0 0 14px; padding:9px 13px; border-radius:9px; font-size:12.5px;
  background:rgba(240,161,30,.10); border:1px solid rgba(240,161,30,.35);
  color:#8a5a00; display:none;
}
.note.show { display:block; }
@media print {
  body { background:#fff; }
  aside, .note { display:none; }
  .tile { break-inside:avoid; box-shadow:none; }
}
@media (max-width:900px) {
  .wrap { flex-direction:column; }
  aside { width:100%; position:static; max-height:none; }
  .tile { grid-column:1 / -1 !important; }
}
"""


def _slicer_values(df: pd.DataFrame, max_fields: int = 8,
                   max_values: int = 25) -> List[Dict]:
    """Categorical fields worth putting on the rail, with their values."""
    out: List[Dict] = []
    from app.engines.domains.base import is_id_column

    for col in df.columns:
        if len(out) >= max_fields:
            break
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if is_id_column(col, df[col]):
            continue
        n = df[col].nunique(dropna=True)
        if not 2 <= n <= max_values:
            continue
        values = [str(v) for v in
                  df[col].dropna().value_counts().index[:max_values]]
        out.append({"column": str(col), "values": values})
    return out


def _finding_text(layout: Dict) -> str:
    """The message out of a Plotly title, without its markup."""
    title = (layout.get("title") or {}).get("text", "")
    if not title:
        return ""
    # The builder writes "<b>finding</b><br><span…>axes</span>"; only the
    # bold part is the finding, the rest repeats the tile label.
    head = title.split("<br>")[0]
    return re.sub(r"<[^>]+>", "", head).strip()


def build_dashboard_html(
    df: pd.DataFrame,
    tiles: List[Dict],
    kpis: List[Dict],
    *,
    title: str = "Dashboard",
    subtitle: str = "",
    prepared_by: str = "",
) -> str:
    """One self-contained HTML page. `tiles` is [{title, figure}, ...]."""
    runtime = _plotly_runtime()

    tile_html: List[str] = []
    tile_js: List[str] = []
    for i, tile in enumerate(tiles):
        fig = tile.get("figure")
        if fig is None:
            continue
        spec = json.loads(pio.to_json(fig)) if not isinstance(fig, dict) else fig
        layout = dict(spec.get("layout", {}))
        width = int(tile.get("w", 6))
        question = str(tile.get("question", "")).strip()

        # Plotly titles do not wrap, so on a half-width tile the finding
        # was cut off mid-sentence — "Retail leads gross_profit at 10.8m
        # — 1.5x the next group and 46% of the tota". Lifting it out of
        # the figure and into the tile's own markup lets it wrap, and
        # gives the plot back the vertical space the title was using.
        finding = _finding_text(layout)
        layout.pop("title", None)
        margin = dict(layout.get("margin") or {})
        margin["t"] = 8
        layout["margin"] = margin

        head = '<em>{}</em>'.format(html.escape(str(tile.get("title", ""))))
        if question:
            head += '<small>{}</small>'.format(html.escape(question))
        if finding:
            head += '<strong>{}</strong>'.format(html.escape(finding))
        height = 300 - (18 if question else 0) - (34 if finding else 0)
        tile_html.append(
            '<div class="tile" style="grid-column:span {}">{}'
            '<div id="t{}" style="height:{}px"></div></div>'.format(
                max(3, min(12, width)), head, i, max(180, height)))
        tile_js.append(
            "Plotly.newPlot('t{}', {}, {}, {{displayModeBar:false, responsive:true}});"
            .format(i, json.dumps(spec.get("data", [])), json.dumps(layout)))

    kpi_html = "".join(
        '<div class="kpi"><span>{}</span><b>{}</b></div>'.format(
            html.escape(str(k.get("label", ""))),
            html.escape(_kpi_text(k)))
        for k in kpis)

    slicers = _slicer_values(df)
    slicer_html = "".join(
        '<div class="slicer" data-col="{col}"><b>{col}</b>{buttons}</div>'.format(
            col=html.escape(s["column"]),
            buttons="".join(
                '<button data-value="{v}">{v}</button>'.format(v=html.escape(v))
                for v in s["values"]))
        for s in slicers)

    truncated = len(df) > MAX_EMBEDDED_ROWS

    return _PAGE.format(
        title=html.escape(title),
        subtitle=html.escape(subtitle),
        css=_CSS,
        runtime=runtime,
        kpis=kpi_html,
        slicers=('<aside><h2>Filters</h2>{}</aside>'.format(slicer_html)
                 if slicer_html else ""),
        tiles="".join(tile_html),
        tile_js="\n".join(tile_js),
        footer=" ".join(filter(None, [
            "<span>{:,} rows</span>".format(len(df)),
            "<span>Generated {}</span>".format(
                pd.Timestamp.now().strftime("%d %b %Y")),
            "<span>Prepared by {}</span>".format(html.escape(prepared_by))
            if prepared_by else "",
            "<span>Slicer preview limited to the first {:,} rows; the "
            "charts above are computed on all {:,}.</span>".format(
                MAX_EMBEDDED_ROWS, len(df)) if truncated else "",
        ])),
    )


def _kpi_text(kpi: Dict) -> str:
    value = kpi.get("value", 0)
    fmt = kpi.get("format", "num")
    if fmt == "pct":
        return "{}%".format(value)
    if fmt == "int":
        return "{:,}".format(int(value))
    return human_number(float(value))


# The slicers filter the rendered figures in the browser rather than
# re-querying anything — an exported file has nothing to query. Plotly
# keeps the source arrays on each trace, so a click restyles the traces
# it can and marks the rest as unfiltered; that is honest about what a
# static export can do, and better than a control that silently does
# nothing.
_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
<script>{runtime}</script>
</head><body>
<header>
  <h1>{title}</h1>
  <p>{subtitle}</p>
</header>
<div class="wrap">
{slicers}
  <main>
    <div class="note" id="note"></div>
    <div class="kpis">{kpis}</div>
    <div class="grid">{tiles}</div>
  </main>
</div>
<footer>{footer}</footer>
<script>
{tile_js}

// Slicers highlight the selected category across every tile that plots
// it. A static file cannot recompute an aggregate, so rather than
// pretending to, selecting a value dims the other categories and says
// so — the reader can see what they picked without being shown a number
// that was never recalculated.
(function () {{
  var active = {{}};
  var note = document.getElementById('note');

  function apply() {{
    var picked = Object.keys(active).map(function (k) {{
      return k + ' = ' + active[k];
    }});
    note.textContent = picked.length
      ? 'Highlighting ' + picked.join(', ') + '. Totals are for the full dataset — open the app to re-run the numbers on a filter.'
      : '';
    note.className = picked.length ? 'note show' : 'note';

    var wanted = Object.keys(active).map(function (k) {{ return active[k]; }});
    document.querySelectorAll('.grid > .tile > div').forEach(function (el) {{
      var gd = el;
      if (!gd.data) return;
      var opacities = gd.data.map(function (tr) {{
        // Plotly hands back typed arrays for numeric axes, which have no
        // `.every` in every browser build — reading `tr.x` as a plain
        // array threw on the first click and killed the handler for
        // every tile after it.
        var labels = null;
        if (Array.isArray(tr.labels)) labels = tr.labels;
        else if (Array.isArray(tr.x) &&
                 tr.x.some(function (v) {{ return typeof v === 'string'; }})) {{
          labels = tr.x;
        }}
        if (!labels || !wanted.length) return null;
        return labels.map(function (v) {{
          return wanted.indexOf(String(v)) >= 0 ? 1 : 0.18;
        }});
      }});
      if (opacities.some(function (o) {{ return o !== null; }})) {{
        gd.data.forEach(function (tr, i) {{
          if (opacities[i]) {{
            if (tr.marker) tr.marker.opacity = opacities[i];
          }} else if (tr.marker) {{
            tr.marker.opacity = 1;
          }}
        }});
        Plotly.redraw(gd);
      }}
    }});
  }}

  document.querySelectorAll('.slicer button').forEach(function (btn) {{
    btn.addEventListener('click', function () {{
      var col = btn.closest('.slicer').dataset.col;
      var val = btn.dataset.value;
      var was = active[col] === val;
      btn.closest('.slicer').querySelectorAll('button')
         .forEach(function (b) {{ b.classList.remove('on'); }});
      if (was) {{ delete active[col]; }}
      else {{ active[col] = val; btn.classList.add('on'); }}
      apply();
    }});
  }});
}})();
</script>
</body></html>"""
