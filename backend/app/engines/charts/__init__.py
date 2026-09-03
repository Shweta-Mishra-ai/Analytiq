"""
engines/charts — static matplotlib chart rendering for reports and decks.

Three modules, three jobs: style.py decides how a chart looks and what
its axes mean, plots.py draws one, selection.py decides which are worth
drawing. app.engines.chart_exporter remains as the import path every
caller already uses.
"""
from app.engines.charts.plots import (                          # noqa: F401
    fig_to_bytes, make_bar_chart, make_box_plot, make_bullet_chart,
    make_correlation_heatmap, make_driver_importance_chart, make_histogram,
    make_line_chart, make_pie_chart, make_ranked_bar_chart, make_risk_heatmap,
)
from app.engines.charts.selection import (                      # noqa: F401
    ChartSpec, generate_all_charts,
)
