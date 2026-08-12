"""
core/engines/finance.py — Finance & Accounting domain engine.
Single responsibility: P&L, margin, budget variance, and cost driver insights.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from app.engines.domains.base import Insight, build_insight, col_stats

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
        "cat":    _find("category", "department", "account", "cost_centre", "segment"),
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
                f"Net profit: {total_profit:,.0f} | "
                f"Loss periods/items: {loss_rows} ({loss_pct:.1f}%)"
            )
            if loss_rows > 0:
                risks.append(
                    f"{loss_rows} loss-making records ({loss_pct:.1f}% of total). "
                    "Investigate which periods/categories are systematically unprofitable."
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
            direction = "growing" if trend_pct > 0 else "declining"
            findings.append(
                f"Revenue {direction} {abs(trend_pct):.1f}% comparing first vs second half "
                f"of available periods (n={len(period_rev)} periods)"
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
            f"Budget vs Actual: {variance_pct:+.1f}% overall "
            f"({over} over-budget, {under} under-budget periods/items)"
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
                evidence=f"Budget: {total_budget:,.0f} | Actual: {total_actual:,.0f} | "
                         f"Variance: {total_actual - total_budget:+,.0f}",
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
    _finance_revenue_trend(df, cols, stats, insights, findings, risks, opps)
    _finance_budget_variance(df, cols, stats, insights, findings, risks, opps)
    _finance_cost_concentration(df, cols, stats, insights, findings, risks, opps)
    _finance_revenue_concentration(df, cols, stats, insights, findings, risks, opps)

    actions.extend([
        "Investigate the top 3 cost drivers — understand fixed vs variable split",
        "Set KPI targets from internal data: gross margin, OER, budget variance tolerance",
        "Build rolling 12-month trend dashboard for P&L — one number per period",
        "Define concentration policy: no single category >40% of revenue or cost",
    ])

    return {"findings": findings[:8], "risks": risks[:5], "opportunities": opps[:4],
            "actions": actions, "insights": insights}
