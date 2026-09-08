"""
engines/domains/realestate.py — property, lettings and portfolio.

Property data has a shape no other domain shares: almost every measure
is per-square-foot or per-unit, and a raw total is close to meaningless.
A £2m building is not expensive; £2m for 400 sq ft is. So the engine
normalises before it compares, and says so.

What a portfolio manager or agent asks:

**What is a unit worth here, once size is taken out?** Price per square
foot across locations is the comparison; total price across locations is
a statement about which buildings happen to be bigger.

**How long is stock sitting?** Days on market is the liquidity measure,
and its spread matters more than its average — a portfolio averaging 40
days where a quarter sit past 90 has a specific problem in a specific
segment.

**Where is the yield, and where is the vacancy?** Rental yield and
occupancy are the two numbers that decide whether a holding is working.

**Is the pricing consistent?** Wide price-per-unit variation inside one
location is either genuine quality spread or inconsistent valuation, and
naming which is the first question at a portfolio review.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import Insight, build_insight
from app.engines.domains._common import (
    benchmark_note, binary_rate, find_col, find_measure, fmt,
    grouping_columns, segment_gap, variability,
)

logger = logging.getLogger(__name__)

# Stock sitting longer than this is a pricing question, not a market one.
SLOW_STOCK_DAYS = 90
# Occupancy below this is a portfolio problem worth the headline.
OCCUPANCY_FLOOR = 90.0
# Price-per-unit variation above this inside one location suggests
# inconsistent valuation rather than genuine quality spread.
VALUATION_SPREAD_CV = 45.0


def _location_col(df: pd.DataFrame):
    for kw in ("location", "city", "suburb", "neighbourhood", "neighborhood",
               "district", "area", "region", "postcode", "zipcode", "market",
               "propertytype", "type"):
        for c in grouping_columns(df):
            if kw in c.lower().replace("_", "").replace(" ", ""):
                return c
    groups = grouping_columns(df)
    return groups[0] if groups else None


def _insights_realestate(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights: List[Insight] = []

    price = find_measure(df, ["price", "saleprice", "value", "valuation",
                              "listprice", "soldprice", "amount"],
                         exclude=["pricepersqft", "pricepersqm"])
    area = find_measure(df, ["sqft", "squarefeet", "sqm", "squaremetres",
                             "area", "size", "floorarea"])
    rent = find_measure(df, ["rent", "monthlyrent", "rentalincome",
                             "rentprice"])
    yield_col = find_measure(df, ["yield", "rentalyield", "caprate",
                                  "capitalisationrate"])
    dom = find_measure(df, ["daysonmarket", "dayslisted", "timeonmarket",
                            "dom", "daystosell"])
    occupancy = find_measure(df, ["occupancy", "occupancyrate", "occupied"])
    location = _location_col(df)
    sold = find_col(df, ["sold", "let", "leased", "transacted"])

    # ── Price, normalised ─────────────────────────────────
    per_unit = None
    if price and area:
        try:
            valid = df[(df[area] > 0) & df[price].notna()]
            if len(valid) >= 20:
                per_unit = (valid[price] / valid[area])
                mean_ppu = float(per_unit.mean())
                findings.append(
                    "Average price per unit of {} is {} across {:,} "
                    "properties — the comparable measure, where average "
                    "price alone mostly reflects which buildings are "
                    "bigger.".format(area, fmt(mean_ppu), len(valid)))
        except Exception:
            logger.debug("realestate price-per-unit failed", exc_info=True)

    if location and price and area and per_unit is not None:
        try:
            work = df[(df[area] > 0) & df[price].notna()].copy()
            work["_ppu"] = work[price] / work[area]
            gap = segment_gap(work, location, "_ppu", agg="mean")
            if gap and gap["ratio"] and gap["ratio"] >= 1.3:
                findings.append(
                    "'{}' averages {} per unit of {} against '{}' at {} — "
                    "{:.1f}x across {} {}s.".format(
                        gap["best"], fmt(gap["best_val"]), area, gap["worst"],
                        fmt(gap["worst_val"]), gap["ratio"], gap["n_groups"],
                        location))
                opps.append(
                    "A {:.1f}x price-per-unit spread between '{}' and '{}' "
                    "is where acquisition or disposal decisions get made, "
                    "and it is invisible in the headline price.".format(
                        gap["ratio"], gap["worst"], gap["best"]))
                insights.append(build_insight(
                    title="Price per {}: '{}' at {:.1f}x '{}'".format(
                        area, gap["best"], gap["ratio"], gap["worst"]),
                    problem="Price per unit of {} runs {:.1f}x higher in "
                            "'{}' than in '{}'".format(
                                area, gap["ratio"], gap["best"], gap["worst"]),
                    cause="Location premium, stock quality or age — the "
                          "comparison holds because size has already been "
                          "divided out",
                    evidence="{:,} properties in '{}' against {:,} in "
                             "'{}'".format(gap["n_best"], gap["best"],
                                           gap["n_worst"], gap["worst"]),
                    action="1. Confirm the stock is comparable on age and "
                           "condition  2. Check the spread against recent "
                           "transactions, not listings  3. Decide buy or "
                           "sell on the gap  4. Re-run at the next quarter",
                    impact="{} per unit of {} across the {:,} properties in "
                           "'{}'.".format(
                               fmt(gap["best_val"] - gap["worst_val"]), area,
                               gap["n_worst"], gap["worst"]),
                    severity="high", category="realestate_pricing",
                ))
        except Exception:
            logger.debug("realestate location comparison failed", exc_info=True)

    # ── How long stock sits ───────────────────────────────
    if dom:
        try:
            mean_dom = float(df[dom].mean())
            slow = int((df[dom] > SLOW_STOCK_DAYS).sum())
            slow_pct = slow / len(df) * 100 if len(df) else 0
            findings.append(
                "Stock averages {} days on market; {:,} properties ({:.0f}%) "
                "sit past {} days.".format(
                    fmt(mean_dom), slow, slow_pct, SLOW_STOCK_DAYS))
            if slow_pct >= 20:
                risks.append(
                    "{:.0f}% of stock has been listed longer than {} days. "
                    "Stock that old stops attracting viewings regardless of "
                    "price, so the discount needed grows the longer it "
                    "waits.".format(slow_pct, SLOW_STOCK_DAYS))
                insights.append(build_insight(
                    title="{:.0f}% of Stock Sits Past {} Days".format(
                        slow_pct, SLOW_STOCK_DAYS),
                    problem="{:,} properties have been on market longer than "
                            "{} days against an average of {}".format(
                                slow, SLOW_STOCK_DAYS, fmt(mean_dom)),
                    cause="Asking price against the local comparable, or a "
                          "segment with genuinely thin demand — the location "
                          "breakdown separates them",
                    evidence="{:,} of {:,} listings past {} days".format(
                        slow, len(df), SLOW_STOCK_DAYS),
                    action="1. Split the slow stock by location and type  "
                           "2. Re-price the segment where comparables have "
                           "moved  3. Withdraw and re-launch the rest rather "
                           "than letting it age  4. Re-measure in 30 days",
                    impact="{:,} properties are carrying holding cost with "
                           "no transaction in sight.".format(slow),
                    severity="high" if slow_pct >= 30 else "warning",
                    category="realestate_liquidity",
                ))
        except Exception:
            logger.debug("realestate days-on-market failed", exc_info=True)

    # ── Yield and vacancy ─────────────────────────────────
    if occupancy is not None:
        try:
            mean_occ = float(df[occupancy].mean())
            # A rate stored as 0-1 rather than 0-100 is common enough that
            # reading it wrong would print "occupancy is 0.9%".
            if 0 <= mean_occ <= 1.5:
                mean_occ *= 100
            note = benchmark_note("realestate", occupancy, mean_occ)
            findings.append("Average occupancy is {:.1f}%.{}".format(
                mean_occ, (" " + note) if note else ""))
            if mean_occ < OCCUPANCY_FLOOR:
                risks.append(
                    "Occupancy of {:.1f}% means roughly {:.0f}% of lettable "
                    "space earns nothing while still carrying cost.".format(
                        mean_occ, 100 - mean_occ))
                insights.append(build_insight(
                    title="Occupancy at {:.1f}%".format(mean_occ),
                    problem="{:.0f}% of space is vacant across the "
                            "portfolio".format(100 - mean_occ),
                    cause="Letting cycle, asking rent, or condition in a "
                          "specific asset rather than across the book",
                    evidence="{:.1f}% average occupancy across {:,} "
                             "records{}".format(
                                 mean_occ, len(df),
                                 (" — " + note) if note else ""),
                    action="1. Rank assets by vacant space, not by vacancy "
                           "rate  2. Check asking rent against the local "
                           "comparable on the worst two  3. Decide let-now "
                           "against hold-for-rate deliberately  4. Review "
                           "monthly",
                    impact="Each point of occupancy is about {:.0f}% of "
                           "rental income across the book.".format(
                               100 / max(mean_occ, 1)),
                    severity="high", category="realestate_yield",
                ))
        except Exception:
            logger.debug("realestate occupancy failed", exc_info=True)

    if yield_col:
        try:
            mean_yield = float(df[yield_col].mean())
            findings.append("Average {} is {}.".format(
                yield_col, fmt(mean_yield)))
        except Exception:
            logger.debug("realestate yield failed", exc_info=True)
    elif rent and price:
        opps.append(
            "'{}' and '{}' are both present but no yield is recorded — "
            "annualised rent over price is the measure that makes holdings "
            "comparable, and it can be derived here.".format(rent, price))

    # ── Consistent valuation? ─────────────────────────────
    if per_unit is not None:
        cv = variability(per_unit)
        if cv is not None and cv >= VALUATION_SPREAD_CV:
            findings.append(
                "Price per unit of {} varies by {:.0f}% around its mean — "
                "wide enough that a single average is not a valuation "
                "basis.".format(area, cv))
            actions.append(
                "Segment before valuing: price per unit of {} varies {:.0f}% "
                "across this book, so one average covers materially "
                "different stock.".format(area, cv))

    if sold:
        rate = binary_rate(df[sold])
        if rate is not None:
            findings.append("{:.1f}% of listings transacted.".format(rate))

    if not findings and not insights:
        logger.info("realestate engine found no domain signal in %d columns",
                    len(df.columns))

    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}
