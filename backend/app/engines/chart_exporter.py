"""
engines/chart_exporter.py — compatibility shim.

Chart rendering lives in app/engines/charts/ now, split by the three
jobs this module used to do in 1,217 lines: style.py decides how a chart
looks and which aggregate a metric deserves, plots.py draws one,
selection.py decides which charts are worth a page at all. Every change
to any of the three used to touch the same file.

This keeps `from app.engines.chart_exporter import ...` working for the
API, the deck builder and the test suite. What it forwards is what
something actually imports — checked against the codebase, not guessed
— so it stays a compatibility layer rather than becoming a second public
API by accident. New code should import from app.engines.charts.
"""
from app.engines.charts.style import (                          # noqa: F401
    _agg_for_metric, _axis_label, _is_grouping_column, _pretty,
    _reference_line, _tick_budget,
)
from app.engines.charts.plots import (                          # noqa: F401
    fig_to_bytes, make_bar_chart, make_box_plot, make_bullet_chart,
    make_correlation_heatmap, make_driver_importance_chart, make_histogram,
    make_line_chart, make_pie_chart, make_ranked_bar_chart, make_risk_heatmap,
)
from app.engines.charts.selection import (                      # noqa: F401
    ChartSpec, _best_metric_by_category, _pick_best_metric, _rank_measures,
    generate_all_charts,
)

__all__ = ["generate_all_charts", "ChartSpec", "fig_to_bytes"]
