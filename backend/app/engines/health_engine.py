"""
engines/health_engine.py — data-health scoring and niche insight cards
behind the client-facing Health Report.

Ported from dataforge-ai, where this logic lived inside the Streamlit
page (pages/11_Health_Report.py) rather than in a reusable module. It is
pure pandas/numpy — no UI framework — so it moves across unchanged apart
from the data_profiler import path.

compute_health() deliberately takes its headline score from
data_profiler.profile_dataset() rather than computing a second, independent
formula. Two different quality scores for the same file is a directly
client-visible contradiction when both PDFs are delivered together.

build_insights() returns plain dicts shaped
{tag, title, body, action, severity, border, bg, tag_color} — the contract
health_pdf_builder renders.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_health(df: pd.DataFrame) -> dict:
    rows, cols   = len(df), len(df.columns)
    missing_pct  = (df.isna().sum().sum() / max(df.size, 1)) * 100
    dup_pct      = df.duplicated().sum() / max(rows, 1) * 100
    num_cols     = df.select_dtypes(include="number").columns.tolist()
    # Use 1.5x IQR — the SAME threshold the analysis report and data_profiler
    # use. The old 3x IQR found almost nothing, so every dataset scored a
    # non-credible 100/100 while the same report elsewhere listed outlier
    # warnings. A skewed column also dents column health.
    outlier_cols = 0
    skewed_cols  = 0
    for c in num_cols:
        s = df[c].dropna()
        if len(s) > 10:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0 and ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).mean() > 0.05:
                outlier_cols += 1
            try:
                if abs(float(s.skew())) > 1.5:
                    skewed_cols += 1
            except Exception:
                logger.warning("skew calc failed for column '%s'", c, exc_info=True)
    outlier_pct = outlier_cols / max(len(num_cols), 1) * 100

    # Headline score: SAME computation the Main Report uses
    # (core.data_profiler.profile_dataset), not a second independent
    # formula. Two different scores for the identical dataset (this used
    # to compute its own subtractive score here — 53/100 on one real
    # dataset — while the Main Report's completeness/dedup/column-health
    # formula gave 86/100 for the SAME file) is a direct, client-visible
    # contradiction between the two PDFs a client receives together.
    from app.engines.data_profiler import profile_dataset
    try:
        score = float(profile_dataset(df).overall_quality_score)
    except Exception:
        logger.warning("profile_dataset() failed in compute_health — falling back "
                       "to a completeness-only estimate", exc_info=True)
        score = max(0.0, 100.0 - missing_pct)
    score = max(int(round(score)), 0)

    # The report's status colours (report_theme), not a separate bright set:
    # a neon-teal grade was the one colour on the page from no palette.
    grade_map = [(90,"A+","Excellent", "#067647"),
                 (80,"A", "Very Good", "#067647"),
                 (70,"B+","Good",      "#475467"),
                 (60,"B", "Fair",      "#B54708"),
                 (50,"C", "Needs Work","#B54708"),
                 (0, "D", "Poor",      "#B42318")]
    grade, label, color = next(
        (g, ln, c) for thresh, g, ln, c in grade_map if score >= thresh)

    return {
        "score":       score,
        "grade":       grade,
        "label":       label,
        "color":       color,
        "missing_pct": round(missing_pct, 1),
        "dup_pct":     round(dup_pct, 1),
        "outlier_pct": round(outlier_pct, 1),
        "rows":        rows,
        "cols":        cols,
        "num_cols":    len(num_cols),
    }


def build_insights(df: pd.DataFrame, niche: str) -> list:
    insights = []
    cols_lower = {c.lower(): c for c in df.columns}

    def _find(keywords, numeric_only=False):
        for kw in keywords:
            for cl, c in cols_lower.items():
                if numeric_only and not pd.api.types.is_numeric_dtype(df[c]):
                    continue
                if kw in cl:
                    return c
        return None

    def _ins(tag, title, body, action, severity):
        border, bg = _CARD_COLORS.get(severity, _CARD_COLORS["info"])
        return {"tag": tag, "title": title, "body": body, "action": action,
                "severity": severity, "border": border, "bg": bg, "tag_color": border}

    # Every card here states a measured figure. A gap between groups is
    # reported only when a significance test says it is larger than chance,
    # and no card names a cause or cites a source this dataset cannot back.
    # The cards this replaced quoted "Mercer 2024" and "McKinsey 2024" figures
    # nobody could trace, called a 6-point department gap with p=0.78 "the
    # highest attrition", and prescribed mentoring programmes and VIP tiers.
    from app.engines.chart_exporter import _measures, fmt
    from app.engines.domains.hr import (_exit_sentence, _find_left_mask,
                                        _segment_exit_rates, describe_rates,
                                        group_exit_gap)

    # ── Universal: skew ────────────────────────────────────
    for c in _measures(df)[:3]:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        skew = float(s.skew())
        if abs(skew) > 2:
            insights.append(_ins(
                "DATA SKEW",
                "'{}' is heavily skewed: mean {}, median {}".format(c, fmt(s.mean()), fmt(s.median())),
                "A few {} values pull the average {} the typical value "
                "(skewness {:.1f}).".format("large" if skew > 0 else "small",
                                           "above" if skew > 0 else "below", skew),
                "Quote the median for '{}' when describing a typical record.".format(c),
                "info"
            ))
            insights[-1]["column"] = str(c)

    # ── HR-specific ────────────────────────────────────────
    if niche == "hr":
        left = _find_left_mask(df)
        sat_col = _find(["satisfaction", "engagement"], numeric_only=True)

        if left is not None and len(left):
            rate = float(left.mean() * 100)
            band = ("above" if rate > 15 else "within" if rate >= 10 else "below")
            insights.append(_ins(
                "ATTRITION",
                "Attrition is {:.1f}%, {} the 10–15% planning band".format(rate, band),
                "{:,} of {:,} employees left. Healthy rates vary by industry; the band "
                "is a planning reference, not a benchmark for this sector.".format(
                    int(left.sum()), len(left)),
                "Interview recent leavers to learn why they left." if rate > 15 else
                "Keep tracking the rate by department so a rise shows early.",
                "warning" if rate > 15 else "info" if rate >= 10 else "positive"
            ))

        if sat_col:
            from app.engines.domains.base import infer_scale_bounds
            s = df[sat_col].dropna()
            lo, hi = infer_scale_bounds(float(s.min()), float(s.max()))
            pct = (float(s.mean()) - lo) / (hi - lo) * 100 if hi > lo else 0.0
            insights.append(_ins(
                "EMPLOYEE SATISFACTION",
                "Satisfaction averages {:.0f}% of the scale".format(pct),
                "Mean {:.2f} on a {:.0f}–{:.0f} scale.".format(float(s.mean()), lo, hi),
                "Break the scores down by department to see where they are lowest.",
                "warning" if pct < 55 else "info"
            ))

        # Tenure cohorts: only when cohorts really leave at different rates.
        tenure_col = _find(["tenure", "years_at", "seniority", "time_spend"], numeric_only=True)
        if tenure_col and left is not None:
            cohorts = pd.cut(df[tenure_col], bins=[-0.01, 2, 5, float("inf")],
                             labels=["0–2 years", "3–5 years", "6+ years"])
            gap = group_exit_gap(cohorts, left)
            if gap and gap["p"] < 0.05:
                insights.append(_ins(
                    "TENURE",
                    "'{}' tenure leaves at {:.0f}%, against {:.0f}% for '{}'".format(
                        gap["worst"], gap["rates"].iloc[0], gap["rates"].iloc[-1], gap["best"]),
                    "Exit rate by tenure: {}. Chi-square p={:.3f}. The data shows when "
                    "people leave, not why.".format(describe_rates(gap), gap["p"]),
                    "Ask leavers from the '{}' group what made them go.".format(gap["worst"]),
                    "warning"
                ))

        # Promotion: measured exit rates, not a cited multiplier.
        promo_col = _find(["promotion", "promoted"])
        if promo_col and left is not None:
            p = df[promo_col]
            p = (pd.to_numeric(p, errors="coerce") if pd.api.types.is_numeric_dtype(p)
                 else p.astype(str).str.lower().isin(["yes", "true", "1"]).astype(float))
            exits = _segment_exit_rates(df, p == 0)
            if exits and exits[2] == exits[2] and exits[2] < 0.05 and exits[0] > exits[1]:
                insights.append(_ins(
                    "PROMOTION",
                    "Employees without a recent promotion leave at {:.0f}%, against "
                    "{:.0f}%".format(exits[0], exits[1]),
                    "{:.1f}% of employees were promoted. {}".format(
                        float((p == 1).mean() * 100),
                        _exit_sentence("Those not promoted", exits)),
                    "Check whether the gap holds within each department and pay band "
                    "before treating promotion as a retention lever.",
                    "warning"
                ))

        # Flight risk is a label (bottom quartile of satisfaction); it is a
        # finding only if that group is measured to leave more.
        if sat_col and left is not None:
            from app.engines.domains.hr import compute_flight_risk
            n_flight, risk_pct, _ = compute_flight_risk(df, left_mask=left)
            q25 = float(df.loc[~left, sat_col].quantile(0.25))
            exits = _segment_exit_rates(df, df[sat_col] < q25)
            if n_flight and exits and exits[2] == exits[2] and exits[2] < 0.05 \
                    and exits[0] > exits[1]:
                insights.append(_ins(
                    "FLIGHT RISK",
                    "{:,} current employees sit in the satisfaction group that leaves most".format(
                        n_flight),
                    "{} {:,} current employees ({:.0f}%) score in that bottom quarter.".format(
                        _exit_sentence("Employees in the bottom quarter of satisfaction", exits),
                        n_flight, risk_pct),
                    "Start retention conversations with this group.",
                    "warning"
                ))

    # ── SALES-specific ────────────────────────────────────
    elif niche == "sales":
        rev_col    = _find(["revenue","amount","deal_value","deal_size","value","gmv","arr"], numeric_only=True)
        status_col = _find(["status","stage","outcome","result"])

        if rev_col:
            s     = df[rev_col].dropna()
            total = float(s.sum())
            k     = max(1, int(len(s) * 0.1))
            conc  = float(s.nlargest(k).sum()) / total * 100 if total > 0 else 0
            insights.append(_ins(
                "CONCENTRATION",
                "The largest 10% of records bring in {:.0f}% of {}".format(conc, rev_col),
                "The largest {:,} of {:,} records total {} of {}.".format(
                    k, len(s), fmt(float(s.nlargest(k).sum())), fmt(total)),
                "Check whether these records come from a few customers; if they do, "
                "that dependence is the risk to manage.",
                "warning" if conc > 60 else "info"
            ))

        if status_col:
            s    = df[status_col].dropna().astype(str).str.lower()
            won  = int(s.isin(["won","win","closed won","success"]).sum())
            lost = int(s.isin(["lost","lose","closed lost","loss","failed"]).sum())
            total = won + lost
            if total >= 20:
                win_rate = won / total * 100
                insights.append(_ins(
                    "WIN RATE",
                    "{:.0f}% of closed deals were won".format(win_rate),
                    "{:,} won and {:,} lost. A 25–40% win rate is a common planning "
                    "band; it varies a great deal by market.".format(won, lost),
                    "Compare won and lost deals by region, product and rep to see "
                    "where the rate drops.",
                    "warning" if win_rate < 25 else "info"
                ))

    # ── E-COMMERCE-specific ──────────────────────────────
    elif niche == "ecommerce":
        # Not "price": in a product catalogue that is a list price, and the
        # card called it "average order".
        order_val_col = _find(["order_value","gmv","amount","order_total","revenue","aov"],
                              numeric_only=True)
        customer_col  = _find(["customer","user","buyer","client"])

        if order_val_col:
            s = df[order_val_col].dropna()
            insights.append(_ins(
                "ORDER VALUE",
                "Average order {}; median {}".format(fmt(s.mean()), fmt(s.median())),
                "The top 20% of orders average {}.".format(
                    fmt(s.nlargest(max(1, int(len(s) * 0.2))).mean())),
                "Use the median when quoting a typical order; the average is pulled "
                "up by the largest ones." if s.mean() > 1.2 * s.median() else
                "The average is a fair summary of a typical order here.",
                "info"
            ))

        if customer_col and order_val_col and customer_col != order_val_col:
            try:
                spend = df.groupby(customer_col)[order_val_col].sum()
                if len(spend) >= 20 and spend.sum() > 0:
                    k = max(1, int(len(spend) * 0.2))
                    top = float(spend.nlargest(k).sum() / spend.sum() * 100)
                    insights.append(_ins(
                        "CUSTOMER CONCENTRATION",
                        "The top 20% of customers bring in {:.0f}% of revenue".format(top),
                        "{:,} of {:,} customers account for {} of {}.".format(
                            k, len(spend), fmt(float(spend.nlargest(k).sum())),
                            fmt(float(spend.sum()))),
                        "Look at what these customers buy and how often they return; "
                        "losing a few of them would show directly in revenue.",
                        "warning" if top > 80 else "info"
                    ))
            except Exception:
                logger.warning("Health Report section failure", exc_info=True)

    # ── FINANCE-specific ─────────────────────────────────
    elif niche == "finance":
        rev_col  = _find(["revenue","income","sales","turnover","gross"], numeric_only=True)
        cost_col = _find(["cost","expense","cogs","expenditure","opex"], numeric_only=True)

        if rev_col and cost_col:
            rev    = float(df[rev_col].dropna().sum())
            cost   = float(df[cost_col].dropna().sum())
            if rev > 0:
                margin = (rev - cost) / rev * 100
                insights.append(_ins(
                    "MARGIN",
                    "'{}' less '{}' leaves {:.1f}% of revenue".format(rev_col, cost_col, margin),
                    "Total {} {}, total {} {}.".format(rev_col, fmt(rev), cost_col, fmt(cost)),
                    "Break '{}' down by category to see which lines drive it.".format(cost_col),
                    "critical" if margin < 0 else "warning" if margin < 15 else "info"
                ))

    # ── Universal: Missing data risk ──────────────────────
    miss = df.isna().sum()
    bad_cols = miss[miss / len(df) > 0.3].sort_values(ascending=False)
    if len(bad_cols) > 0:
        col_list = ", ".join(
            ["'{}' ({:.0f}%)".format(c, miss[c]/len(df)*100) for c in bad_cols.index[:3]])
        worst_pct = float(bad_cols.iloc[0] / len(df))
        sev = "critical" if worst_pct > 0.5 else "warning"
        insights.append(_ins(
            "DATA QUALITY RISK",
            "{} column(s) have >30% missing data".format(len(bad_cols)),
            "Columns with high missing rates: **{}**. Any analysis or model trained on these columns will be statistically unreliable.".format(col_list),
            "Either impute with domain-appropriate values, or exclude from client-facing insights.",
            sev
        ))

    if not insights:
        insights.append(_ins(
            "DATA HEALTH",
            "Dataset is clean and analysis-ready",
            "No column is more than 30% empty and no measure is heavily skewed.",
            "No data-quality action is needed before analysis.",
            "positive"
        ))

    return insights


# ══════════════════════════════════════════════════════════
#  FULL INSIGHT SET  (domain engines + data-quality cards)
# ══════════════════════════════════════════════════════════

# The domain engines emit a richer severity ladder than the report's card
# renderer understands. Anything unmapped would fall back to "info" and be
# rendered as a neutral blue card, quietly demoting a critical finding.
_SEVERITY_TO_CARD = {
    "critical": "critical",
    "high":     "critical",
    "warning":  "warning",
    "medium":   "warning",
    "info":     "info",
    "low":      "info",
    "positive": "positive",
}

_CARD_COLORS = {
    "critical": ("#B42318", "#FEF3F2"),
    "warning":  ("#B54708", "#FFFAEB"),
    "positive": ("#067647", "#ECFDF3"),
    "info":     ("#475467", "#F2F4F7"),
}


def _insight_to_card(ins) -> Dict:
    """Adapt a domains/*.Insight dataclass into the flat card dict the
    Health Report renders.

    The engines already write in Problem → Cause → Evidence → Action →
    Impact form, which maps cleanly onto the report's
    What → Why it matters → What to do layout.
    """
    severity = _SEVERITY_TO_CARD.get(str(ins.severity).lower(), "info")
    border, bg = _CARD_COLORS[severity]

    # Each part is its own sentence; joined bare, "…in the bottom band" ran
    # into "Not identifiable from…" with no full stop between them.
    def _sentence(p) -> str:
        p = str(p).strip()
        return p if p[-1:] in ".!?" else p + "."
    body = " ".join(_sentence(p) for p in (ins.problem, ins.cause, ins.evidence)
                    if p and str(p).strip())
    if ins.impact and str(ins.impact).strip():
        body = f"{body} {_sentence(ins.impact)}"

    tag = str(getattr(ins, "category", "") or severity).upper().replace("_", " ")
    return {"tag": tag, "topic": str(getattr(ins, "category", "") or ""),
            "title": str(ins.title), "body": body,
            "action": str(ins.action), "severity": severity,
            "border": border, "bg": bg, "tag_color": border,
            "confidence": str(getattr(ins, "confidence", "") or "")}


def build_report_payload(df: pd.DataFrame, niche: str,
                          max_cards: int = 12) -> Dict:
    """Everything the Health Report renders, in one call.

    The domain engines produce four distinct kinds of output — insight
    cards, key findings, risks and opportunities — and several analyses
    contribute only to the latter three. Finance, for example, emits 6
    findings, 2 risks and 2 opportunities alongside just 3 cards; the
    cost-concentration and budget-variance analyses produce no card at
    all. Rendering cards alone therefore threw most of the analysis away.

    Returning all four (plus the executive summary and recommended
    actions) is also what makes the output read like a consulting
    deliverable rather than a list of alerts.
    """
    payload: Dict = {
        "executive_summary": "",
        "insights": [],
        "key_findings": [],
        "risks": [],
        "opportunities": [],
        "actions": [],
    }

    story = None
    try:
        from app.engines.story_engine import generate_story
        story = generate_story(df)
        payload["executive_summary"] = story.executive_summary or ""
        payload["key_findings"] = list(story.key_findings or [])
        payload["risks"] = list(story.business_risks or [])
        payload["opportunities"] = list(story.opportunities or [])
        payload["actions"] = list(story.recommended_actions or [])
    except Exception:
        logger.warning("story engine failed while building the health report "
                       "payload — narrative sections will be empty",
                       exc_info=True)

    payload["insights"] = build_full_insights(df, niche, max_cards=max_cards, story=story)
    return payload


def build_full_insights(df: pd.DataFrame, niche: str, max_cards: int = 12,
                        story=None) -> List[Dict]:
    """Every insight the Health Report should carry, best-first.

    Sources, in priority order:
      1. the per-domain engines in app/engines/domains/ — the deep business
         analysis (regional gaps, loss-making segments, quota attainment,
         rep spread, cohort/retention effects, …)
      2. this module's own niche cards — data-quality angles the domain
         engines don't cover (skew warnings, "use median not mean", …)

    Previously the report used source 2 alone, which produced a single
    insight card for a sales/ecommerce/finance dataset even when the domain
    engine found several critical findings in the same file. A one-card
    report is not a deliverable a client will pay for.
    """
    cards: List[Dict] = []

    try:
        if story is None:       # the caller's story, so it is not built twice
            from app.engines.story_engine import generate_story
            story = generate_story(df)
        for ins in story.top_insights:
            cards.append(_insight_to_card(ins))
    except Exception:
        logger.warning("domain insight engines failed for the health report — "
                       "falling back to data-quality cards only", exc_info=True)

    # The niche cards restate the headline figures the domain engines
    # already cover — "Attrition at 19.6%" and "Attrition rate is 19.6%",
    # and satisfaction as 51% of the scale in one card and 61% in another,
    # measured two ways. Keep the domain engine's card for those topics.
    covered = {c.get("topic") for c in cards}
    topic_of = {"ATTRITION": "attrition", "EMPLOYEE SATISFACTION": "satisfaction",
                "CONCENTRATION": "revenue",
                "CUSTOMER CONCENTRATION": "ecommerce_concentration",
                "MARGIN": "finance_margin"}
    # A skew card for a measure a domain card already describes says the
    # same thing twice ("median 160, mean 183" on both).
    domain_text = " ".join("{} {}".format(c["title"], c["body"]) for c in cards)
    try:
        cards.extend(c for c in build_insights(df, niche)
                     if topic_of.get(c.get("tag")) not in covered
                     and not (c.get("tag") == "DATA SKEW"
                              and "'{}'".format(c.get("column")) in domain_text))
    except Exception:
        logger.warning("niche data-quality cards failed", exc_info=True)

    # De-duplicate on title; keep the first (domain engines rank higher).
    seen, unique = set(), []
    for c in cards:
        key = c["title"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    unique.sort(key=lambda c: order.get(c["severity"], 9))
    return unique[:max_cards]
