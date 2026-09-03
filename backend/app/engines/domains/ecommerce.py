"""
core/engines/ecommerce.py — E-Commerce domain engine.
Single responsibility: rating, pricing, discount, and category insights.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import (build_insight, infer_scale_bounds)
from app.engines.domains.customer_analytics import run_customer_analytics
from app.engines.industry_benchmarks import lookup_benchmark

logger = logging.getLogger(__name__)


def _repeat_purchase_insights(df: pd.DataFrame, insights: List, findings: List,
                              risks: List, opps: List) -> None:
    """Repeat Purchase Rate — share of customers with more than one order.
    Not a column that exists in most flat order-level datasets; derived
    from a customer identifier column by counting orders per customer.
    """
    cust_kw = ("customer_id", "customerid", "customer_key", "buyer_id", "user_id", "client_id")
    order_kw = ("order_id", "orderid", "order_number", "transaction_id", "invoice_id")

    cust_col = next((c for c in df.columns if any(k in c.lower().replace(" ", "_")
                    for k in cust_kw)), None)
    if not cust_col:
        return

    try:
        # Prefer counting distinct ORDERS per customer if an order-id column
        # exists (multiple line items can share one order); otherwise fall
        # back to counting ROWS per customer.
        order_col = next((c for c in df.columns if any(k in c.lower().replace(" ", "_")
                          for k in order_kw) and c != cust_col), None)
        if order_col:
            orders_per_cust = df.groupby(cust_col)[order_col].nunique()
        else:
            orders_per_cust = df.groupby(cust_col).size()

        n_customers = len(orders_per_cust)
        if n_customers < 20:
            return

        repeat_customers = int((orders_per_cust > 1).sum())
        repeat_rate = repeat_customers / n_customers * 100
        avg_orders  = float(orders_per_cust.mean())

        findings.append(
            "Repeat purchase rate: {:.1f}% of {:,} customers placed more than one "
            "order (avg {:.1f} orders/customer).".format(
                repeat_rate, n_customers, avg_orders))

        bm = lookup_benchmark("ecommerce", "repeat_purchase_rate")
        bm_text = ""
        if bm:
            if repeat_rate < bm.low:
                bm_text = (" This is below the {:.0f}-{:.0f}% general guidance range — "
                          "retention/re-engagement may be an opportunity area.").format(
                              bm.low, bm.high)
            elif repeat_rate > bm.high:
                bm_text = " This is above the {:.0f}-{:.0f}% general guidance range.".format(
                    bm.low, bm.high)
            else:
                bm_text = " This falls within the {:.0f}-{:.0f}% general guidance range.".format(
                    bm.low, bm.high)

        insights.append(build_insight(
            title="Repeat Purchase Rate: {:.1f}% of Customers".format(repeat_rate),
            problem="{:.1f}% of {:,} customers have placed more than one order.".format(
                repeat_rate, n_customers),
            cause="Repeat-purchase behavior reflects product satisfaction, price "
                  "positioning, and retention/re-engagement efforts — not measured "
                  "directly here.",
            evidence="{:,} customers analyzed, {:,} ({:.1f}%) repeat buyers, "
                     "{:.1f} orders/customer on average.{}".format(
                         n_customers, repeat_customers, repeat_rate, avg_orders, bm_text),
            action="1. Segment repeat vs one-time buyers — what differs in their first "
                   "order?  2. Test a post-purchase re-engagement campaign for one-time "
                   "buyers  3. Track this rate as a trend over time, not just a snapshot",
            impact="Repeat customers are typically the cheapest revenue to generate — "
                   "moving this rate up compounds over time.",
            severity="warning" if bm and repeat_rate < bm.low * 0.5 else "info",
            category="repeat_purchase"
        ))

        if bm and repeat_rate < bm.low * 0.5:
            risks.append(
                "Repeat purchase rate ({:.1f}%) is well below the {:.0f}-{:.0f}% general "
                "guidance range — most customers are one-time buyers.".format(
                    repeat_rate, bm.low, bm.high))
        elif bm and repeat_rate > bm.high:
            opps.append(
                "Repeat purchase rate ({:.1f}%) exceeds general guidance — a genuine "
                "retention strength worth understanding and protecting.".format(repeat_rate))
    except Exception:
        logger.warning("repeat purchase rate analysis failed", exc_info=True)


def _rating_revenue_evidence(df: pd.DataFrame, rating_col, rev_col,
                             low_thr: float, high_thr: float) -> str:
    """
    Measured revenue comparison of low-rated (<low_thr) vs well-rated
    (>=high_thr) products in THIS catalog. Thresholds are scale-aware (passed
    in), so the comparison is valid on 1-5, 1-10 or 0-100 rating scales.
    Replaces invented conversion claims with a checkable number, or says
    nothing when the dataset can't support the comparison.
    """
    if not rating_col or not rev_col or rev_col not in df.columns:
        return ""
    try:
        pair = df[[rating_col, rev_col]].apply(pd.to_numeric, errors="coerce").dropna()
        low  = pair[pair[rating_col] < low_thr][rev_col]
        high = pair[pair[rating_col] >= high_thr][rev_col]
        if len(low) < 10 or len(high) < 10:
            return ""
        med_low, med_high = float(low.median()), float(high.median())
        if med_low <= 0 or med_high <= 0:
            return ""
        return ("In this catalog, median revenue for top-rated products is "
                "{:.2g} vs {:.2g} for low-rated ones ({:.1f}× difference, "
                "n={:,}/{:,}) — an association in this data, not a proven "
                "cause.".format(med_high, med_low, med_high / med_low,
                                 len(high), len(low)))
    except Exception:
        logger.warning("rating-revenue evidence failed", exc_info=True)
        return ""


def _category_revenue_share(df: pd.DataFrame, cat_col, rev_col, category) -> str:
    """How much of total revenue the underperforming category represents."""
    if not cat_col or not rev_col or rev_col not in df.columns:
        return ("Revenue impact depends on category mix — include a revenue "
                "column to quantify it.")
    try:
        rev = pd.to_numeric(df[rev_col], errors="coerce")
        total = float(rev.sum())
        if total <= 0:
            return ""
        share = float(rev[df[cat_col] == category].sum()) / total * 100
        return ("'{}' represents {:.1f}% of catalog revenue in this dataset — "
                "that is the exposure at stake.".format(category, share))
    except Exception:
        logger.warning("category revenue share failed", exc_info=True)
        return ""


def _insights_ecommerce(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    rating_col = next((c for c in df.columns if "rating" in c.lower()
                       and "count" not in c.lower()), None)
    price_col  = next((c for c in df.columns
                       if any(k in c.lower() for k in ["discounted_price","selling_price","price"])
                       and c in stats), None)
    _actual_col = next((c for c in df.columns
                       if "actual_price" in c.lower() or "mrp" in c.lower()), None)
    # Discount RATE column (percentage) — must NOT match "discounted_price"
    # (a price column). Prefer explicit percentage/pct/rate naming; exclude
    # any column that also contains price/amount/mrp (those are money values,
    # not a discount rate, even if "discount" appears in the name).
    disc_col = next((c for c in df.columns
                     if "discount" in c.lower() and c in stats
                     and not any(x in c.lower() for x in
                                ["price", "amount", "mrp", "value", "cost"])),
                    None)
    cat_col    = next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if "category" in c.lower() and df[c].nunique()<=30), None)
    rev_col    = next((c for c in df.columns
                       if any(k in c.lower() for k in ["revenue","sales","amount"]) and c in stats), None)

    # ── Rating Analysis (scale-aware: works on 1-5, 1-10, 0-100) ──────
    if rating_col and rating_col in stats:
        st     = stats[rating_col]
        mean_r = st.get("mean", 0)
        out_ct = st.get("outliers", 0)
        q1     = st.get("q1", 0)
        # Infer the theoretical rating scale rather than assuming /5, so a 1-10
        # or 0-100 rating column isn't mis-judged against 3.0/4.0 cut-offs.
        lo, hi = infer_scale_bounds(st.get("min", 0), st.get("max", 5))
        rng    = hi - lo
        poor   = lo + 0.50 * rng   # ≈ 3.0 on 1-5 — mediocre / bottom-box
        weak   = lo + 0.625 * rng  # ≈ 3.5 on 1-5 — emergency mean threshold
        good   = lo + 0.75 * rng   # ≈ 4.0 on 1-5 — "strong" target
        low_n  = int((df[rating_col].dropna() < poor).sum()) if rating_col in df.columns else 0
        rev_ev = _rating_revenue_evidence(df, rating_col, rev_col, weak, good)
        scale  = "{:.2f}/{:.0f}".format(mean_r, hi)
        pt     = "{:.1f}".format(good)   # target, in scale units

        if mean_r < weak:
            insights.append(build_insight(
                title="Average rating {} — {:,} products below {:.1f}".format(
                    scale, low_n, poor),
                problem="Average {} with {:,} products rated below {:.1f} on this "
                        "{:.0f}-point scale.".format(scale, low_n, poor, hi),
                cause="Bottom-rated products are pulling the catalogue average down — "
                      "quality, description, or delivery mismatch worth investigating",
                evidence=("Mean={:.2f} on a {:.0f}-{:.0f} scale. Planning target: {}+. "
                          "Bottom 25% rated below {:.1f}. {:,} critically low-rated "
                          "products. ".format(mean_r, lo, hi, pt, q1, low_n) + rev_ev),
                action="1. Immediate audit of the lowest-rated products  "
                       "2. Customer feedback analysis for bottom-rated items  "
                       "3. Supplier quality review for failing products  "
                       "4. Remove or improve within 14 days",
                impact="Low ratings are associated with lower revenue in this catalog "
                       "(see the comparison above); lifting them is a retention lever "
                       "to test, not a guaranteed gain.",
                severity="critical", category="rating"
            ))
            risks.append("Rating {} — {:,} products below {:.1f}; low-rated items show "
                         "lower median revenue in this catalog".format(scale, low_n, poor))
        elif mean_r < good:
            insights.append(build_insight(
                title="Rating Below Target: {} (Target {}+)".format(scale, pt),
                problem="{} average. Bottom 25% rated below {:.1f}. {:,} products below "
                        "{:.1f}.".format(scale, q1, low_n, poor),
                cause="Bottom-quartile products dragging overall performance",
                evidence=("Mean={:.2f} on a {:.0f}-{:.0f} scale. 25th percentile={:.1f}. "
                          "Target: {}+. ".format(mean_r, lo, hi, q1, pt) + rev_ev),
                action="1. Fix or remove bottom-quartile products  "
                       "2. Improve product descriptions and images  "
                       "3. Category-level quality audit",
                impact="Lifting the bottom quartile toward the catalog median raises the "
                       "average rating shoppers see at first glance.",
                severity="warning", category="rating"
            ))
        else:
            insights.append(build_insight(
                title="Strong Ratings: {} — Competitive Advantage".format(scale),
                problem="N/A — ratings are strong",
                cause="Quality products meeting customer expectations",
                evidence="Mean={:.2f} on a {:.0f}-{:.0f} scale. Above the {}+ planning "
                         "target. Only {:,} products below {:.1f}.".format(
                             mean_r, lo, hi, pt, low_n, poor),
                action="Leverage high ratings in all marketing. "
                       "Use as social proof in product listings.",
                impact="Strong ratings support premium-pricing experiments — A/B test "
                       "small increases on top-rated items and measure conversion.",
                severity="positive", category="rating"
            ))
            opps.append("Rating {} is strong — test a 5-10% price increase on top-rated "
                        "items and measure conversion".format(scale))

        if out_ct > 0:
            pct = st.get("outlier_pct",0)
            findings.append(
                "{:,} products have outlier ratings ({:.1f}% of catalog) — "
                "investigate immediately for quality or fraud issues".format(out_ct, pct))

    # ── Category Performance (significance-tested, scale-aware) ──────
    if cat_col and rating_col and rating_col in df.columns and rating_col in stats:
        cat_perf = df.groupby(cat_col)[rating_col].agg(["mean", "count"]).sort_values("mean")
        cat_perf = cat_perf[cat_perf["count"] >= 5]
        if len(cat_perf) >= 2:
            # Gap threshold relative to the rating SCALE, not a fixed 0.3 (which
            # assumes /5). 0.075 of the range ≈ 0.3 points on a 1-5 scale.
            _lo, _hi = infer_scale_bounds(stats[rating_col].get("min", 0),
                                          stats[rating_col].get("max", 5))
            gap_min = 0.075 * (_hi - _lo)
            worst_c = cat_perf.index[0]
            best_c  = cat_perf.index[-1]
            gap     = cat_perf.loc[best_c, "mean"] - cat_perf.loc[worst_c, "mean"]

            # Is the category difference real or sampling noise? Kruskal-Wallis.
            sig_txt, significant = "not statistically tested", True
            try:
                from scipy import stats as _sc
                groups = [g[rating_col].dropna().values
                          for _, g in df.groupby(cat_col) if len(g) >= 5]
                if len(groups) >= 2:
                    h, p = _sc.kruskal(*groups)
                    significant = bool(p < 0.05)
                    sig_txt = ("Kruskal-Wallis H={:.1f}, p<0.001".format(h) if p < 0.001
                               else "Kruskal-Wallis H={:.1f}, p={:.3f}".format(h, p))
            except Exception:
                logger.warning("category significance test failed", exc_info=True)

            if gap > gap_min and significant:
                insights.append(build_insight(
                    title="Category Gap: '{}' ({:.2f}) vs '{}' ({:.2f})".format(
                        worst_c, cat_perf.loc[worst_c, "mean"],
                        best_c, cat_perf.loc[best_c, "mean"]),
                    problem="'{}' category rates {:.2f} points below '{}' — a "
                            "statistically significant gap.".format(worst_c, gap, best_c),
                    cause="A gap this size is unlikely to be chance ({}). Likely drivers "
                          "to confirm: supplier quality, product complexity, or expectation "
                          "mismatch by category — not yet proven.".format(sig_txt),
                    evidence="{:.2f}-point gap across {} categories. '{}' avg={:.2f}, "
                             "'{}' avg={:.2f}. {}.".format(
                                 gap, len(cat_perf), worst_c, cat_perf.loc[worst_c, "mean"],
                                 best_c, cat_perf.loc[best_c, "mean"], sig_txt),
                    action="1. Quality audit of '{}' category suppliers  "
                           "2. Customer-complaint analysis for '{}' products  "
                           "3. Profile what '{}' does differently and pilot it in '{}'".format(
                               worst_c, worst_c, best_c, worst_c),
                    impact=("Closing half the gap adds +{:.2f} rating points to '{}'. ".format(
                                gap * 0.5, worst_c)
                            + _category_revenue_share(df, cat_col, rev_col, worst_c)),
                    severity="warning" if gap < 0.16 * (_hi - _lo) else "critical",
                    category="category_performance"
                ))
                findings.append("Category rating range: {} ({:.2f}) to {} ({:.2f}), "
                                "significant ({}).".format(
                                    worst_c, cat_perf.loc[worst_c, "mean"],
                                    best_c, cat_perf.loc[best_c, "mean"], sig_txt))

    # ── Pricing Analysis ───────────────────────────────────
    if price_col and price_col in stats:
        st   = stats[price_col]
        skew = st.get("skew",0)
        if skew > 1.5:
            findings.append(
                "Price distribution right-skewed (skew={:.1f}) — median {:.0f} vs mean {:.0f}. "
                "Most products are budget-range with few premium items. "
                "Consider expanding mid-market range.".format(
                    skew, st["median"], st["mean"]))
            opps.append("Mid-market price gap detected — products between median ({:.0f}) "
                        "and 75th percentile ({:.0f}) are underrepresented".format(
                            st["median"], st["q3"]))

    # ── Discount Analysis ──────────────────────────────────
    if disc_col and disc_col in stats:
        st       = stats[disc_col]
        avg_disc = st.get("mean",0)
        max_disc = st.get("max",0)
        if avg_disc > 40:
            insights.append(build_insight(
                title="High Avg Discount {:.0f}% — Potential Margin Erosion".format(avg_disc),
                problem="Average discount {:.0f}% with some products at {:.0f}% — profitability at risk".format(
                    avg_disc, max_disc),
                cause="Possible drivers to check: competitive pressure, or broad "
                      "discounting applied without per-product margin analysis.",
                evidence="Mean discount={:.0f}%, max={:.0f}%. "
                         "Reference band: 15–25% (indicative planning range — validate "
                         "against your category economics).".format(avg_disc, max_disc),
                action="1. Margin analysis per category — identify below-cost discounts  "
                       "2. Reduce discounts on 4.0+ rated products (they sell without discounts)  "
                       "3. Strategic discounting only: new launch, clearance, seasonal",
                impact="Every 10% unnecessary discount = direct margin loss. "
                       "High discounts also train customers to wait for sales.",
                severity="warning" if avg_disc<55 else "critical",
                category="pricing"
            ))
            risks.append("Avg discount {:.0f}% may be eroding margins — review per-product profitability".format(avg_disc))

    # ── Price-Rating Correlation ───────────────────────────
    for corr in corrs:
        cols = [corr["col_a"], corr["col_b"]]
        has_rating = any("rating" in c.lower() for c in cols)
        has_price  = any("price" in c.lower() for c in cols)
        if has_rating and has_price and abs(corr["r"])>=0.3:
            if corr["r"] < 0:
                risks.append(
                    "Higher-priced products tend to have LOWER ratings (r={:.2f}, "
                    "association not causation) — worth checking whether premium pricing "
                    "matches perceived value.".format(corr["r"]))
            else:
                opps.append(
                    "Higher-priced products tend to have HIGHER ratings (r={:.2f}, "
                    "association not causation) — a premium-pricing test on top-rated "
                    "items is worth trying, measuring conversion.".format(corr["r"]))

    # FIX-011: Column-gated actions — only recommend for columns that exist in dataset
    # Never generate recommendations for columns that are not present
    _ec_cols = [c.lower() for c in df.columns]

    if rating_col:
        actions.append("Weekly rating monitoring — alert when any product's rating "
                       "falls into the bottom quartile of your catalogue")
        actions.append("Review products that are both low-rated and thinly reviewed "
                       "(<50 reviews) — small samples move averages and are worth "
                       "fixing or de-listing first")
        if price_col:
            actions.append("A/B test a small price increase on your highest-rated "
                           "products and measure the effect on conversion before rolling out")
    if rev_col or any(k in _ec_cols for k in ["amount","sales","revenue"]):
        actions.append("Customer feedback loop — auto-survey buyers 7 days post-delivery to track satisfaction vs revenue")
    if cat_col:
        actions.append("Category manager review — monthly revenue and rating performance vs category target")
    else:
        actions.append("Segment your data by product type or channel — subgroup performance often tells a different story than averages")

    # ── Repeat Purchase Rate (derived from customer identifier) ─
    _repeat_purchase_insights(df, insights, findings, risks, opps)

    # ── Customer-level analysis: cohorts, RFM, concentration ───
    run_customer_analytics(df, insights, findings, risks, opps)

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  SALES INSIGHTS
# ══════════════════════════════════════════════════════════

