"""
core/engines/finance.py — Finance & Accounting domain engine.
Single responsibility: P&L, margin, budget variance, and cost driver insights.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from scipy import stats as scipy_stats

from app.engines.domains.base import build_insight
from app.services.stat_guards import MIN_N

from app.engines.column_roles import resolve

logger = logging.getLogger(__name__)



def _detect_finance_cols(df: pd.DataFrame) -> dict:
    """Detect finance columns by keyword matching. Returns dict of col roles."""
    def _find(*kws, excl=(), numeric_only=False):
        for c in df.columns:
            cl = c.lower()
            if numeric_only and not pd.api.types.is_numeric_dtype(df[c]):
                continue
            if any(k in cl for k in kws) and not any(e in cl for e in excl):
                return c
        return None

    # Rate/percentage suffixes that indicate a column is NOT an absolute
    # monetary value even if it contains "cost"/"revenue"/etc. — e.g.
    # 'cost_percentage', 'revenue_pct', 'margin_rate' must be excluded from
    # matchers looking for absolute money columns, or a value like 65 (%)
    # gets summed as if it were 65 (currency units), producing wildly wrong
    # margin/variance figures.
    RATE_EXCL = ("percentage", "percent", "_pct", "rate", "ratio", "ppm")

    _roles = resolve(df)

    return {
        "rev":    _find("revenue", "total_revenue", "income", "turnover", "net_sales",
                        excl=RATE_EXCL, numeric_only=True),
        "cost":   _find("cost", "cogs", "cost_of_goods", "expense", excl=RATE_EXCL,
                        numeric_only=True),
        "profit": _find("net_profit", "profit", "net_income", "ebitda", "operating_profit",
                        excl=RATE_EXCL, numeric_only=True),
        "budget": _find("budget", "plan", "target", "forecast", excl=RATE_EXCL,
                        numeric_only=True),
        "actual": _find("actual", "actuals", excl=RATE_EXCL, numeric_only=True),
        "opex":   _find("opex", "operating_expense", "operating_cost", excl=RATE_EXCL,
                        numeric_only=True),
        "period": _find("month", "quarter", "period", "date", "year"),
        # `_find` matches substrings with no qualifier check, so
        # `risk_category` and `forecast_category` were both eligible as
        # the reporting segment and the whole segment-profitability
        # section was then computed across a confidence band. The
        # resolver is asked first; the local match stays as the fallback
        # for the finance-specific names it knows about.
        "cat":    (_roles.product
                   or _find("department", "account", "cost_centre",
                            "cost centre", "business_unit")),
    }


def _finance_margin_insights(df, cols, stats, insights, findings, risks, opps) -> None:
    """Gross margin and profitability analysis."""
    rev_col, cost_col, profit_col = cols["rev"], cols["cost"], cols["profit"]

    if rev_col and cost_col:
        total_rev  = float(df[rev_col].sum())
        total_cost = float(df[cost_col].sum())
        if total_rev > 0:
            gross_margin = (total_rev - total_cost) / total_rev * 100
            finding_text = (
                f"Gross margin: {gross_margin:.1f}% "
                f"(Revenue: {total_rev:,.0f}, COGS: {total_cost:,.0f})"
            )
            findings.append(finding_text)

            # Cost increase (as % of current COGS) that would wipe out gross
            # profit: costs rise until they equal revenue. GM% = profit/rev, so
            # the buffer as a % of COGS is profit/cost = GM/(100-GM)*100.
            breakeven_cost_rise = (gross_margin / (100 - gross_margin) * 100
                                   if 0 < gross_margin < 100 else 0)
            if gross_margin < 10:
                risks.append(
                    f"Thin gross margin ({gross_margin:.1f}%) — leaves little buffer "
                    "for operating expenses and downturns (internal review threshold: 10%)."
                )
                insights.append(build_insight(
                    title=f"Gross Margin Low: {gross_margin:.1f}% (Below 10% Review Threshold)",
                    problem=f"Gross margin of {gross_margin:.1f}% leaves a narrow buffer.",
                    cause="Costs are consuming most of revenue — worth checking whether "
                          "COGS is rising faster than revenue in this data.",
                    evidence=f"Revenue={total_rev:,.0f} | Cost={total_cost:,.0f} | "
                             f"GM={gross_margin:.1f}%",
                    action="1. Audit COGS line items  2. Identify the top cost drivers  "
                           "3. Compare against your own historical periods  4. Review pricing",
                    impact=f"A COGS increase of about {breakeven_cost_rise:.0f}% would erase "
                           "gross profit entirely — that is the buffer at this margin.",
                    severity="critical", category="finance_margin"
                ))
            elif gross_margin < 25:
                risks.append(f"Gross margin {gross_margin:.1f}% is below the 25% internal "
                             "review threshold — worth a pricing/cost review.")
            else:
                opps.append(
                    f"Gross margin of {gross_margin:.1f}% is healthy. "
                    "Focus on volume growth and operating leverage."
                )
                insights.append(build_insight(
                    title=f"Gross Margin Healthy: {gross_margin:.1f}%",
                    problem=f"Gross margin is {gross_margin:.1f}%, above the 25% "
                            "internal review threshold.",
                    cause="Revenue is outpacing cost of goods sold at a healthy rate — "
                          "worth understanding what's driving it so it can be sustained.",
                    evidence=f"Revenue={total_rev:,.0f} | Cost={total_cost:,.0f} | "
                             f"GM={gross_margin:.1f}%",
                    action="1. Document what's driving this margin (pricing, mix, cost "
                           "control)  2. Set this as the internal benchmark for other "
                           "periods/segments  3. Watch for margin erosion as volume scales",
                    impact="A healthy margin gives room to invest in growth without "
                           "sacrificing profitability — worth protecting.",
                    severity="positive", category="finance_margin"
                ))

    if profit_col:
        try:
            total_profit = float(df[profit_col].sum())
            loss_rows    = int((df[profit_col] < 0).sum())
            loss_pct     = loss_rows / len(df) * 100
            findings.append(
                "Net profit is {:,.0f}, with {:,} of {:,} rows ({:.1f}%) "
                "recording a loss.".format(
                    total_profit, loss_rows, len(df), loss_pct)
            )
            if loss_rows > 0:
                risks.append(
                    "{:,} row{} ({:.1f}% of the file) record{} a loss — worth "
                    "checking whether the same periods or categories recur "
                    "rather than treating them as isolated.".format(
                        loss_rows, "" if loss_rows == 1 else "s", loss_pct,
                        "s" if loss_rows == 1 else "")
                )
        except Exception:
            logger.warning("Net profit analysis failed", exc_info=True)


def _finance_revenue_trend(df, cols, stats, insights, findings, risks, opps) -> None:
    """Period-over-period revenue trend analysis."""
    rev_col, period_col = cols["rev"], cols["period"]
    if not (rev_col and period_col):
        return
    try:
        period_rev = df.groupby(period_col)[rev_col].sum().sort_index()
        if len(period_rev) < 2:
            return
        first_half = float(period_rev.iloc[:len(period_rev)//2].mean())
        second_half = float(period_rev.iloc[len(period_rev)//2:].mean())
        if first_half > 0:
            trend_pct = (second_half - first_half) / first_half * 100
            # A 0.3% difference between two halves of a series is not
            # growth, and reporting it as "revenue growing 0.3%" invites
            # the reader to act on noise. Below 2% the honest description
            # is that it did not move.
            if abs(trend_pct) < 2.0:
                findings.append(
                    "Revenue held broadly flat across the {} periods covered "
                    "— the second half sits within {:.1f}% of the "
                    "first.".format(len(period_rev), abs(trend_pct))
                )
            else:
                findings.append(
                    "Revenue {} {:.1f}% between the first and second half of "
                    "the {} periods covered.".format(
                        "rose" if trend_pct > 0 else "fell",
                        abs(trend_pct), len(period_rev))
                )
            if trend_pct < -10:
                risks.append(
                    f"Revenue declining {abs(trend_pct):.1f}% period-over-period. "
                    "Investigate structural causes — market, pricing, or volume."
                )
                insights.append(build_insight(
                    title=f"Revenue Declining: {abs(trend_pct):.1f}% PoP",
                    problem=f"Revenue declined {abs(trend_pct):.1f}% comparing first vs second half.",
                    cause="Could be seasonal, structural, or customer churn-driven.",
                    evidence=f"H1 avg: {first_half:,.0f} | H2 avg: {second_half:,.0f}",
                    action="1. Decompose by category/product  2. Check customer retention  "
                           "3. Review pricing changes  4. Compare against your own prior periods",
                    impact=f"If the same rate of decline held, revenue would fall roughly "
                           f"{abs(trend_pct):.0f}% further next period — a straight-line "
                           f"projection, not a forecast; intervention changes it.",
                    severity="critical" if trend_pct < -20 else "warning",
                    category="finance_trend"
                ))
            elif trend_pct > 15:
                opps.append(
                    f"Revenue growing strongly ({trend_pct:.1f}% PoP). "
                    "Invest in capacity and margin protection before the next growth phase."
                )
    except Exception:
        logger.warning("Revenue trend analysis failed", exc_info=True)


def _finance_budget_variance(df, cols, stats, insights, findings, risks, opps) -> None:
    """Budget vs actual variance analysis."""
    budget_col, actual_col = cols["budget"], cols["actual"]
    # Almost no ledger export has a column called "actual" — the actual is
    # the revenue or cost line, and the budget sits beside it. Requiring
    # the literal name meant budget variance, a headline metric of the
    # finance report, was silently skipped on files that plainly carried a
    # budget column.
    basis = "the '{}' column".format(actual_col) if actual_col else ""
    if not actual_col:
        actual_col = cols["rev"] or cols["profit"] or cols["cost"]
        basis = ("'{}', taken as the actual because no column is named as "
                 "such".format(actual_col) if actual_col else "")
    if not (budget_col and actual_col and budget_col != actual_col):
        return
    try:
        total_budget = float(df[budget_col].sum())
        total_actual = float(df[actual_col].sum())
        if total_budget <= 0:
            return
        variance_pct = (total_actual - total_budget) / total_budget * 100
        over  = int((df[actual_col] > df[budget_col]).sum())
        under = int((df[actual_col] < df[budget_col]).sum())
        findings.append(
            "Actuals came in {:+.1f}% against '{}' ({:,} periods over, {:,} "
            "under), measured on {}.".format(
                variance_pct, budget_col, over, under, basis)
        )
        sev = "critical" if abs(variance_pct) > 20 else "warning" if abs(variance_pct) > 10 else "info"
        if abs(variance_pct) > 10:
            risks.append(
                f"Budget variance of {variance_pct:+.1f}% exceeds ±10% review threshold. "
                "Requires explanation and forecast update."
            )
            insights.append(build_insight(
                title=f"Budget Variance: {variance_pct:+.1f}%",
                problem=f"Actual spend/revenue is {abs(variance_pct):.1f}% {'above' if variance_pct > 0 else 'below'} budget.",
                cause="Forecast assumptions may not reflect current trading conditions.",
                evidence="Budget ('{}'): {:,.0f}. Actual ({}): {:,.0f}. "
                         "Variance: {:+,.0f} across {:,} rows.".format(
                             budget_col, total_budget, basis, total_actual,
                             total_actual - total_budget, len(df)),
                action="1. Identify top 3 variance drivers  2. Reforecast for remaining period  "
                       "3. Update planning assumptions  4. Communicate to stakeholders",
                impact="Persistent variance >10% undermines budgeting credibility and cash planning.",
                severity=sev, category="finance_budget"
            ))
    except Exception:
        logger.warning("Budget variance analysis failed", exc_info=True)


def _finance_cost_concentration(df, cols, stats, insights, findings, risks, opps) -> None:
    """Cost category concentration and operating expense ratio."""
    cost_col, cat_col, rev_col = cols["cost"], cols["cat"], cols["rev"]
    opex_col = cols["opex"]

    # Cost by category
    if cost_col and cat_col:
        try:
            cat_cost = df.groupby(cat_col)[cost_col].sum().sort_values(ascending=False)
            if len(cat_cost) > 1:
                top_cat     = str(cat_cost.index[0])
                top_cat_pct = float(cat_cost.iloc[0] / cat_cost.sum() * 100)
                findings.append(
                    f"Top cost category: '{top_cat}' = {top_cat_pct:.1f}% of total cost"
                )
                if top_cat_pct > 60:
                    risks.append(
                        f"'{top_cat}' represents {top_cat_pct:.1f}% of total costs — "
                        "high concentration creates supplier/vendor dependency risk."
                    )
                opps.append(
                    f"Pareto opportunity: targeting '{cat_cost.index[0]}' and "
                    f"'{cat_cost.index[1] if len(cat_cost)>1 else 'next'}' could address "
                    f"{float(cat_cost.iloc[:2].sum()/cat_cost.sum()*100):.0f}% of total cost."
                )
        except Exception:
            logger.warning("Cost concentration analysis failed", exc_info=True)

    # Operating expense ratio
    if opex_col and rev_col:
        try:
            total_opex = float(df[opex_col].sum())
            total_rev  = float(df[rev_col].sum())
            if total_rev > 0:
                oer = total_opex / total_rev * 100
                findings.append(f"Operating expense ratio (OER): {oer:.1f}%")
                if oer > 80:
                    risks.append(
                        f"OER of {oer:.1f}% is very high — only {100-oer:.1f}% of revenue "
                        "remains after operating expenses."
                    )
        except Exception:
            logger.warning("OER analysis failed", exc_info=True)


def _finance_revenue_concentration(df, cols, stats, insights, findings, risks, opps) -> None:
    """Revenue concentration by category and period-over-period cost growth."""
    rev_col, cat_col, cost_col, period_col = (
        cols["rev"], cols["cat"], cols["cost"], cols["period"])

    # Revenue concentration
    if rev_col and cat_col:
        try:
            cat_rev = df.groupby(cat_col)[rev_col].sum().sort_values(ascending=False)
            if len(cat_rev) > 1:
                top_pct = float(cat_rev.iloc[0] / cat_rev.sum() * 100)
                if top_pct > 50:
                    risks.append(
                        f"Revenue concentration: '{cat_rev.index[0]}' = {top_pct:.1f}% of total. "
                        f"Losing this segment would remove {top_pct:.1f}% of revenue — "
                        "diversification reduces the exposure."
                    )
                    insights.append(build_insight(
                        title=f"Revenue Concentration Risk: {top_pct:.1f}% in One Category",
                        problem=f"'{cat_rev.index[0]}' accounts for {top_pct:.1f}% of total revenue.",
                        cause="Over-reliance on one product/segment/client group.",
                        evidence=f"Top category: {float(cat_rev.iloc[0]):,.0f} of "
                                 f"total {float(cat_rev.sum()):,.0f}",
                        action="1. Map revenue by sub-segment  2. Build pipeline in adjacent segments  "
                               "3. Set concentration limit policy (<40% in one segment)",
                        impact=f"A 20% decline in this segment = "
                               f"{float(cat_rev.iloc[0])*0.2/cat_rev.sum()*100:.1f}% revenue loss.",
                        severity="warning", category="finance_concentration"
                    ))
        except Exception:
            logger.warning("Revenue concentration analysis failed", exc_info=True)

    # Cost growth PoP
    if cost_col and period_col:
        try:
            period_cost = df.groupby(period_col)[cost_col].sum().sort_index()
            if len(period_cost) >= 2:
                cost_growth = float((period_cost.iloc[-1] - period_cost.iloc[0]) /
                                     period_cost.iloc[0] * 100)
                if abs(cost_growth) > 15:
                    findings.append(
                        f"Cost growth first→last period: {cost_growth:+.1f}% "
                        f"({float(period_cost.iloc[0]):,.0f} → {float(period_cost.iloc[-1]):,.0f})"
                    )
                    if cost_growth > 15:
                        risks.append(
                            f"Costs grew {cost_growth:.1f}% over the data period. "
                            "Verify if matched by equivalent revenue growth."
                        )
        except Exception:
            logger.warning("Cost growth analysis failed", exc_info=True)


# ══════════════════════════════════════════════════════════
#  COST STRUCTURE AND BREAK-EVEN
# ══════════════════════════════════════════════════════════

def _finance_cost_structure(df, cols, stats, insights, findings, risks, opps) -> None:
    """Split cost into its fixed and variable parts, and derive break-even.

    Regressing total cost on revenue is the standard way to recover a cost
    structure from a P&L extract: the intercept estimates cost that does
    not move with trading, the slope estimates cost per unit of revenue.
    Break-even revenue follows as fixed / (1 - variable rate).

    The estimate is only reported when the line actually describes the
    data. A high-intercept fit through a scattered cloud produces a
    confident-looking break-even figure that is pure arithmetic on noise,
    which is exactly the sort of number that discredits a report when the
    client tests it against their own management accounts.
    """
    rev_col, cost_col = cols["rev"], cols["cost"]
    if not (rev_col and cost_col) or rev_col == cost_col:
        return
    try:
        pair = df[[rev_col, cost_col]].dropna()
        pair = pair[(pair[rev_col] > 0) & (pair[cost_col] >= 0)]
        n = len(pair)
        if n < MIN_N:
            return

        x = pair[rev_col].astype(float).to_numpy()
        y = pair[cost_col].astype(float).to_numpy()
        if x.std() == 0 or y.std() == 0:
            return

        fit = scipy_stats.linregress(x, y)
        r2 = float(fit.rvalue ** 2)
        slope = float(fit.slope)
        intercept = float(fit.intercept)

        # A cost structure needs a line that fits, a variable rate between
        # 0 and 1 (cost rising with revenue but not faster than it), and a
        # non-negative fixed component. Outside that, the model does not
        # describe a cost structure and no break-even is derivable.
        if r2 < 0.50 or fit.pvalue >= 0.05 or not (0 < slope < 1) or intercept <= 0:
            return

        contribution_margin = 1.0 - slope
        breakeven = intercept / contribution_margin
        # Per-period fixed cost, so the figure is comparable to the revenue
        # a period actually produces rather than to the whole extract.
        mean_rev = float(x.mean())
        headroom = (mean_rev - breakeven) / breakeven * 100 if breakeven > 0 else 0.0

        findings.append(
            f"Cost structure: {slope * 100:.0f}% of every unit of revenue is "
            f"consumed by variable cost, leaving a {contribution_margin * 100:.0f}% "
            f"contribution margin against {intercept:,.0f} of fixed cost per "
            f"record (R²={r2:.2f}, n={n})"
        )
        findings.append(
            f"Break-even revenue: {breakeven:,.0f} per record — current mean is "
            f"{mean_rev:,.0f} ({headroom:+.0f}% against break-even)"
        )

        if headroom < 15:
            sev = "critical" if headroom < 0 else "high"
            risks.append(
                f"Mean revenue per record sits {headroom:+.0f}% against an estimated "
                f"break-even of {breakeven:,.0f}. At a {contribution_margin * 100:.0f}% "
                f"contribution margin, a {abs(headroom) + 5:.0f}% revenue fall would "
                "take the average record below cost."
            )
            insights.append(build_insight(
                title=f"Break-even at {breakeven:,.0f} vs {mean_rev:,.0f} mean revenue",
                problem=f"Average revenue per record is {headroom:+.0f}% against the "
                        f"break-even implied by the fitted cost structure.",
                cause=f"{slope * 100:.0f}% of revenue is absorbed by variable cost and "
                      f"{intercept:,.0f} of cost per record does not move with trading. "
                      "Which specific costs are genuinely fixed is not identifiable from "
                      "this data — the split is inferred from how cost moves with revenue, "
                      "not from a cost classification.",
                evidence=f"OLS of {cost_col} on {rev_col}: slope={slope:.3f}, "
                         f"intercept={intercept:,.0f}, R²={r2:.2f}, p={fit.pvalue:.2g}, n={n}",
                action=f"1. Reconcile the {intercept:,.0f} fixed estimate against the actual "
                       f"fixed-cost line items  2. Test the {contribution_margin * 100:.0f}% "
                       "contribution margin on a known period  3. Set a revenue floor alert "
                       f"at {breakeven * 1.15:,.0f}",
                impact=f"Every unit of revenue above break-even contributes "
                       f"{contribution_margin * 100:.0f}% to profit; every unit below it "
                       "costs the same rate.",
                severity=sev, category="finance_structure",
            ))
        else:
            opps.append(
                f"Operating leverage: with a {contribution_margin * 100:.0f}% contribution "
                f"margin and break-even at {breakeven:,.0f}, each additional 10% of revenue "
                f"above break-even adds roughly {mean_rev * 0.10 * contribution_margin:,.0f} "
                "per record to profit — provided fixed cost holds."
            )
    except Exception:
        logger.warning("Cost structure analysis failed", exc_info=True)


# ══════════════════════════════════════════════════════════
#  MARGIN STABILITY
# ══════════════════════════════════════════════════════════

def _period_label(value) -> str:
    """A period as a reader would write it.

    `str()` on a Timestamp gives "2023-07-01 00:00:00", and a midnight
    time nobody supplied reads as machine output in a client report.
    """
    if isinstance(value, pd.Timestamp):
        if value.hour or value.minute or value.second:
            return value.strftime("%d %b %Y %H:%M")
        return value.strftime("%d %b %Y")
    return str(value)


def _finance_margin_volatility(df, cols, stats, insights, findings, risks, opps) -> None:
    """How much the margin moves between periods.

    A 22% average margin built from periods of 40% and 4% is a different
    business from a steady 22%, and only the second can be planned
    against. The average alone hides that completely.
    """
    rev_col, cost_col, period_col = cols["rev"], cols["cost"], cols["period"]
    if not (rev_col and cost_col and period_col):
        return
    try:
        grouped = df.groupby(period_col)[[rev_col, cost_col]].sum()
        grouped = grouped[grouped[rev_col] > 0]
        if len(grouped) < 6:      # fewer periods than this cannot show a pattern
            return

        margin = ((grouped[rev_col] - grouped[cost_col]) / grouped[rev_col] * 100)
        margin = margin.replace([np.inf, -np.inf], np.nan).dropna()
        if len(margin) < 6:
            return

        mean_m = float(margin.mean())
        sd_m = float(margin.std(ddof=1))
        if abs(mean_m) < 1e-9:
            return
        cv = abs(sd_m / mean_m)
        lo, hi = float(margin.min()), float(margin.max())
        worst_period = _period_label(margin.idxmin())

        findings.append(
            f"Margin averages {mean_m:.1f}% across {len(margin)} periods but ranges "
            f"{lo:.1f}% to {hi:.1f}% (SD {sd_m:.1f}pp, CV {cv:.2f})"
        )

        # A coefficient of variation above 0.30 means the period-to-period
        # swing is comparable to the margin itself.
        if cv > 0.30:
            risks.append(
                f"Margin varies by {sd_m:.1f} percentage points between periods "
                f"(CV {cv:.2f}); the {mean_m:.1f}% average is not a number to plan "
                f"against. Weakest period: {worst_period} at {lo:.1f}%."
            )
            insights.append(build_insight(
                title=f"Margin swings {lo:.1f}%–{hi:.1f}% across {len(margin)} periods",
                problem=f"The {mean_m:.1f}% average margin is drawn from periods ranging "
                        f"{lo:.1f}% to {hi:.1f}%.",
                cause="Period-to-period variation of this size usually traces to mix, "
                      "one-off costs or seasonality. Which of these applies is not "
                      "established by this data — it needs the period-level detail behind "
                      f"{worst_period}.",
                evidence=f"n={len(margin)} periods, mean {mean_m:.1f}%, SD {sd_m:.1f}pp, "
                         f"CV {cv:.2f}, range {lo:.1f}%–{hi:.1f}%",
                action=f"1. Open {worst_period} line by line against the strongest period  "
                       "2. Separate one-off from recurring items  3. Plan on the lower "
                       f"quartile ({float(margin.quantile(0.25)):.1f}%), not the mean",
                impact="Planning on an average margin this unstable systematically "
                       "overstates cash in the weak periods.",
                severity="high" if cv > 0.50 else "warning",
                category="finance_margin_stability",
            ))
        else:
            opps.append(
                f"Margin is stable at {mean_m:.1f}% (SD {sd_m:.1f}pp across "
                f"{len(margin)} periods), so it can be used as a planning assumption."
            )
    except Exception:
        logger.warning("Margin volatility analysis failed", exc_info=True)


# ══════════════════════════════════════════════════════════
#  LOSS-MAKING SEGMENTS
# ══════════════════════════════════════════════════════════

def _finance_loss_making_segments(df, cols, stats, insights, findings, risks, opps) -> None:
    """Segments that lose money, sized against the total.

    A profitable total routinely hides segments trading below cost. What
    matters is not that a segment is negative but how much of the group's
    profit it consumes, so the figure reported is the drag, not the rank.
    """
    rev_col, cost_col, profit_col, cat_col = (
        cols["rev"], cols["cost"], cols["profit"], cols["cat"])
    if not cat_col:
        return
    try:
        if profit_col:
            seg = df.groupby(cat_col)[profit_col].sum()
        elif rev_col and cost_col:
            g = df.groupby(cat_col)[[rev_col, cost_col]].sum()
            seg = g[rev_col] - g[cost_col]
        else:
            return

        seg = seg.dropna()
        if len(seg) < 2:
            return

        losers = seg[seg < 0].sort_values()
        if losers.empty:
            return

        total_positive = float(seg[seg > 0].sum())
        drag = float(-losers.sum())
        if total_positive <= 0:
            return
        drag_pct = drag / total_positive * 100
        # Below this the segment is a rounding error on the group result.
        if drag_pct < 1.0:
            return

        worst = str(losers.index[0])
        worst_val = float(losers.iloc[0])
        n_rows = {str(k): int(v) for k, v in df[cat_col].value_counts().items()}

        findings.append(
            f"{len(losers)} of {len(seg)} segments trade below cost, together "
            f"consuming {drag:,.0f} — {drag_pct:.0f}% of the profit the other "
            f"segments generate. Largest: '{worst}' at {worst_val:,.0f}."
        )
        risks.append(
            f"'{worst}' loses {abs(worst_val):,.0f} across "
            f"{n_rows.get(worst, 0):,} records. Group profit is "
            f"{drag_pct:.0f}% lower than the profitable segments alone would give."
        )
        insights.append(build_insight(
            title=f"{len(losers)} segments below cost, costing {drag:,.0f}",
            problem=f"'{worst}' and {len(losers) - 1} other segment(s) trade at a loss, "
                    f"offsetting {drag_pct:.0f}% of the profit earned elsewhere."
            if len(losers) > 1 else
            f"'{worst}' trades at a loss of {abs(worst_val):,.0f}, offsetting "
            f"{drag_pct:.0f}% of the profit earned elsewhere.",
            cause="Whether these segments are structurally unprofitable or carrying "
                  "allocated cost that belongs elsewhere is not answerable from this "
                  "extract; it needs the allocation basis.",
            evidence="; ".join(
                f"'{k}': {float(v):,.0f} over {n_rows.get(str(k), 0):,} records"
                for k, v in losers.head(4).items()),
            action=f"1. Confirm how shared cost is allocated to '{worst}'  "
                   "2. Separate genuinely loss-making volume from allocation effects  "
                   "3. Decide price, cost or exit per segment — in that order",
            impact=f"Eliminating the drag entirely would raise group profit by "
                   f"{drag_pct:.0f}%, before any revenue lost with the segment.",
            severity="critical" if drag_pct > 25 else "high" if drag_pct > 10 else "warning",
            category="finance_segment_loss",
        ))
    except Exception:
        logger.warning("Loss-making segment analysis failed", exc_info=True)


# ══════════════════════════════════════════════════════════
#  BENFORD FIRST-DIGIT TEST
# ══════════════════════════════════════════════════════════

# Expected share of each leading digit under Benford's law.
_BENFORD_P = np.log10(1 + 1 / np.arange(1, 10))


def _finance_benford(df, cols, stats, insights, findings, risks, opps) -> None:
    """First-digit conformity on a monetary column.

    Naturally occurring transaction amounts follow Benford's law closely.
    A departure is not evidence of anything on its own — rounded pricing,
    a threshold, or a narrow value range all produce one — so this is
    reported as a place to look, never as a finding of manipulation.

    Nigrini's mean absolute deviation is used rather than the chi-square
    p-value, because chi-square rejects conformity on almost any large
    accounting population and would flag every sizeable dataset.
    """
    col = cols["rev"] or cols["cost"] or cols["actual"]
    if not col:
        return
    try:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        s = s[s > 0]
        if len(s) < 300:      # below this the digit shares are too noisy
            return
        # Benford applies to values spanning orders of magnitude. A column
        # of prices between 90 and 110 will fail the test while being
        # entirely legitimate.
        span = np.log10(float(s.max()) / float(s.min())) if s.min() > 0 else 0
        if span < 2:
            return

        lead = s.astype(str).str.replace(r"[^1-9]", "", regex=True).str[:1]
        lead = lead[lead != ""].astype(int)
        if len(lead) < 300:
            return

        observed = np.array([(lead == d).sum() for d in range(1, 10)], dtype=float)
        n = observed.sum()
        expected_share = _BENFORD_P
        observed_share = observed / n
        mad = float(np.abs(observed_share - expected_share).mean())

        # Nigrini's published thresholds for first-digit MAD.
        if mad < 0.006:
            conformity = "close conformity"
        elif mad < 0.012:
            conformity = "acceptable conformity"
        elif mad < 0.015:
            conformity = "marginal conformity"
        else:
            conformity = "non-conformity"

        worst_i = int(np.argmax(np.abs(observed_share - expected_share)))
        worst_digit = worst_i + 1
        obs_pct = observed_share[worst_i] * 100
        exp_pct = expected_share[worst_i] * 100

        findings.append(
            f"Leading-digit distribution of {col} shows {conformity} with Benford's law "
            f"(MAD {mad:.4f}, n={int(n):,})"
        )

        if mad >= 0.015:
            insights.append(build_insight(
                title=f"{col} leading digits deviate from Benford (MAD {mad:.3f})",
                problem=f"Digit {worst_digit} leads {obs_pct:.1f}% of values against "
                        f"{exp_pct:.1f}% expected — the largest departure across the nine "
                        "digits.",
                cause="Rounded pricing, an approval threshold, a fixed fee schedule and a "
                      "narrow value band all produce this pattern, as does data entered "
                      "rather than transacted. This test cannot distinguish between them "
                      "and is not, on its own, an indication of anything improper.",
                evidence="; ".join(
                    f"{d}: {observed_share[d - 1] * 100:.1f}% vs {expected_share[d - 1] * 100:.1f}%"
                    for d in range(1, 10)) + f" (n={int(n):,}, MAD {mad:.4f})",
                action=f"1. Check whether values starting {worst_digit} cluster at a "
                       "threshold or a standard price  2. Confirm the population is "
                       "transactional rather than budgeted or rounded  3. Only if neither "
                       "explains it, sample those records for review",
                impact="Where an operational explanation exists, none. Where it does not, "
                       "the population warrants a sample check before the figures are "
                       "relied on.",
                severity="warning", category="finance_integrity",
            ))
    except Exception:
        logger.warning("Benford analysis failed", exc_info=True)


def _finance_margin_trend(df, cols, stats, insights, findings, risks, opps) -> None:
    """Whether the margin is holding, not just what it is.

    The engine reported the margin level and its volatility, and revenue
    over time — but never margin *over time*. That is the one a finance
    director asks second: a business can grow the top line while the
    margin erodes underneath it, and every figure in the report stays
    true while the trend that matters goes unmentioned.
    """
    rev_col, cost_col, period_col = cols["rev"], cols["cost"], cols["period"]
    if not (rev_col and cost_col and period_col):
        return
    try:
        work = df[[period_col, rev_col, cost_col]].dropna()
        by_period = work.groupby(period_col)[[rev_col, cost_col]].sum()
        by_period = by_period[by_period[rev_col] > 0]
        if len(by_period) < 6:
            return
        margin = ((by_period[rev_col] - by_period[cost_col])
                  / by_period[rev_col] * 100).sort_index()

        window = max(len(margin) // 3, 1)
        first = float(margin.iloc[:window].mean())
        last = float(margin.iloc[-window:].mean())
        change = last - first          # percentage points, not percent

        # A margin quoted in points that moved by less than half a point
        # is noise; calling that erosion sends someone looking for a
        # cause that is not there.
        if abs(change) < 0.5:
            findings.append(
                "Gross margin held at {:.1f}% across the {} periods "
                "covered, moving less than half a point between the first "
                "and last third.".format(float(margin.mean()), len(margin)))
            return

        direction = "eroded" if change < 0 else "improved"
        findings.append(
            "Gross margin {} {:.1f} points across the period, from {:.1f}% "
            "in the first third to {:.1f}% in the last.".format(
                direction, abs(change), first, last))

        # Erosion while revenue grows is the specific case worth naming:
        # both halves look healthy on their own.
        rev_first = float(by_period[rev_col].iloc[:window].mean())
        rev_last = float(by_period[rev_col].iloc[-window:].mean())
        growing = rev_first > 0 and (rev_last - rev_first) / rev_first > 0.05
        squeeze = (" Revenue grew {:.0f}% over the same window, so the "
                   "business is trading more at a thinner margin — the top "
                   "line alone would not show this.".format(
                       (rev_last - rev_first) / rev_first * 100)
                   if growing and change < 0 else "")

        if change < 0:
            # "Gross Margin Healthy: 33.7%" and "Gross Margin Down 8
            # Points" side by side are both true and read as a
            # contradiction. The level is the weaker claim once the
            # direction is known, so it steps down to a finding and says
            # what it is measured against.
            for existing in list(insights):
                if existing.title.startswith("Gross Margin Healthy"):
                    insights.remove(existing)
                    findings.append(
                        "The margin level is comfortable at {:.1f}% overall, "
                        "but that is an average across a period in which it "
                        "was falling — the trend below is the more useful "
                        "figure.".format(float(margin.mean())))
            opps[:] = [o for o in opps
                       if "margin of" not in o or "healthy" not in o.lower()]
            severity = "critical" if abs(change) >= 3 else "warning"
            insights.append(build_insight(
                title="Gross Margin Down {:.1f} Points Across the Period"
                      .format(abs(change)),
                problem="Margin fell from {:.1f}% to {:.1f}% between the "
                        "first and last third of the {} periods "
                        "covered.".format(first, last, len(margin)),
                cause="Margin moves on price, on input cost, or on mix. "
                      "This data shows the result of the three together "
                      "and cannot separate them.",
                evidence="Measured per '{}' on '{}' against '{}'. Margin by "
                         "period ranges {:.1f}% to {:.1f}%.{}".format(
                             period_col, rev_col, cost_col,
                             float(margin.min()), float(margin.max()), squeeze),
                action="1. Split the change into price, cost and mix before "
                       "acting — the remedy differs for each  "
                       "2. Check whether it is broad or concentrated in one "
                       "segment  3. Compare against the same periods last "
                       "year, not against a sector range",
                impact="Holding the opening margin across the closing "
                       "period's revenue would be worth about {:,.0f} — the "
                       "arithmetic of the gap, not a forecast.".format(
                           abs(change) / 100 * float(
                               by_period[rev_col].iloc[-window:].sum())),
                severity=severity, category="finance_margin",
            ))
            risks.append(
                "Gross margin is {:.1f} points below where it started; at "
                "the current revenue that is about {:,.0f} a period."
                .format(abs(change),
                        abs(change) / 100 * float(
                            by_period[rev_col].iloc[-window:].mean())))
        else:
            opps.append(
                "Gross margin improved {:.1f} points across the period — "
                "worth establishing what changed before it reverses."
                .format(change))
    except Exception:
        logger.warning("margin trend analysis failed", exc_info=True)


def _insights_finance(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    """
    Finance domain orchestrator.
    Delegates to 5 focused sub-functions — each independently testable.
    """
    insights, findings, risks, opps, actions = [], [], [], [], []

    cols = _detect_finance_cols(df)

    if not any(cols.values()):
        findings.append(
            "No standard finance columns detected. "
            "Rename columns to include: revenue, cost, profit, budget, actual, period, category."
        )
        return {"findings": findings, "risks": risks, "opportunities": opps,
                "actions": actions, "insights": insights}

    _finance_margin_insights(df, cols, stats, insights, findings, risks, opps)
    _finance_margin_trend(df, cols, stats, insights, findings, risks, opps)
    _finance_revenue_trend(df, cols, stats, insights, findings, risks, opps)
    _finance_budget_variance(df, cols, stats, insights, findings, risks, opps)
    _finance_cost_concentration(df, cols, stats, insights, findings, risks, opps)
    _finance_revenue_concentration(df, cols, stats, insights, findings, risks, opps)
    _finance_cost_structure(df, cols, stats, insights, findings, risks, opps)
    _finance_margin_volatility(df, cols, stats, insights, findings, risks, opps)
    _finance_loss_making_segments(df, cols, stats, insights, findings, risks, opps)
    _finance_benford(df, cols, stats, insights, findings, risks, opps)

    actions.extend([
        "Investigate the top 3 cost drivers — understand fixed vs variable split",
        "Set KPI targets from internal data: gross margin, OER, budget variance tolerance",
        "Build rolling 12-month trend dashboard for P&L — one number per period",
        "Define concentration policy: no single category >40% of revenue or cost",
    ])

    return {"findings": findings[:8], "risks": risks[:5], "opportunities": opps[:4],
            "actions": actions, "insights": insights}
