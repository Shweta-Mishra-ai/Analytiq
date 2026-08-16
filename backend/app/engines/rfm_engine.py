"""
engines/rfm_engine.py — RFM (Recency, Frequency, Monetary) customer
segmentation.

Standard retail/ecommerce/SaaS analysis: scores each customer 1-5 on how
recently they transacted, how often, and how much they spent, then assigns
them to a named segment (Champions, At Risk, Lost, etc.) using the
industry-standard RFM segment map.

All computation is done from the submitted dataset only — no external
benchmarks. Column detection is automatic but can be overridden.

Ported from dataforge-ai's core/rfm_engine.py (pure pandas/numpy, no
framework coupling).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class RFMColumns:
    customer_id: str
    date_col:    str
    monetary_col: Optional[str] = None   # None → frequency-only monetary (count)
    quantity_col: Optional[str] = None   # optional units/quantity column
    price_col:    Optional[str] = None   # per-unit price — combined with
                                          # quantity_col to derive monetary
                                          # value when no direct total/revenue
                                          # column exists


@dataclass
class RFMSegmentSummary:
    segment:       str
    n_customers:   int
    pct_customers: float
    total_monetary: float
    pct_revenue:   float
    avg_recency:   float
    avg_frequency: float
    avg_monetary:  float
    description:   str
    action:        str
    color:         str


@dataclass
class RFMReport:
    customer_table:    pd.DataFrame            # per-customer R/F/M scores + segment
    segment_summary:   List[RFMSegmentSummary]
    columns_used:      RFMColumns
    analysis_date:     pd.Timestamp
    n_customers:       int
    n_transactions:    int
    total_revenue:     float
    top_segment:       str
    top_segment_pct:   float
    at_risk_pct:       float
    at_risk_revenue:   float
    champions_count:   int
    warnings:          List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
#  SEGMENT MAP  (industry-standard RFM segment definitions)
# ══════════════════════════════════════════════════════════
# Score pattern keys use (R, F) or (R, F, M) tuples/ranges checked in order.
# R/F/M are each 1 (worst) to 5 (best).

_SEGMENT_DEFS = [
    # (name, min_r, min_f, description, action, color)
    ("Champions",        4, 4, "Bought recently, buy often, spend the most.",
     "Reward them. Early access to new products, loyalty perks, ask for referrals.", "#10b981"),
    ("Loyal Customers",  3, 4, "Buy regularly and are responsive to promotions.",
     "Upsell higher-value products, engage in loyalty programs.", "#22c55e"),
    ("Potential Loyalists", 3, 2, "Recent customers with average frequency.",
     "Offer membership/loyalty programs, recommend related products.", "#3b82f6"),
    ("New Customers",    4, 1, "Bought most recently, but not often.",
     "Onboarding support, build relationship early, no aggressive selling yet.", "#60a5fa"),
    ("Promising",        3, 1, "Recent shoppers with low frequency and spend.",
     "Create brand awareness, offer free trials.", "#818cf8"),
    ("Need Attention",   2, 2, "Above-average recency/frequency, may be slipping.",
     "Make limited-time offers, recommend based on past purchases.", "#f59e0b"),
    ("About to Sleep",   2, 1, "Below-average recency and frequency.",
     "Reactivation campaign, share valuable resources, personalized offers.", "#fb923c"),
    ("At Risk",          1, 3, "Spent big and bought often, but long ago.",
     "Win them back via renewals, personalized outreach, don't lose to competition.", "#ef4444"),
    ("Cannot Lose Them", 1, 4, "Made biggest purchases and most frequently, but haven't returned.",
     "Reach out personally, aggressive win-back campaign, understand what changed.", "#dc2626"),
    ("Hibernating",      1, 1, "Last purchase long ago, low spend and frequency.",
     "Offer relevant products, reconnect via different channels, may not be worth pursuing.", "#94a3b8"),
    ("Lost",             1, 1, "Lowest recency, frequency, and monetary scores.",
     "Revive interest with reach-out, or deprioritize — often not cost-effective to win back.", "#64748b"),
]


def _classify_segment(r: int, f: int, m: int) -> Tuple[str, str, str, str]:
    """
    Classify a customer into an RFM segment based on standard RFM rules.
    Returns (segment_name, description, action, color).
    """
    # Champions: high everything
    if r >= 4 and f >= 4:
        return ("Champions", *_SEGMENT_DEFS[0][3:])
    # Loyal Customers
    if r >= 3 and f >= 4:
        return ("Loyal Customers", *_SEGMENT_DEFS[1][3:])
    # Cannot Lose Them: high F/M, low R
    if r <= 2 and f >= 4:
        return ("Cannot Lose Them", *_SEGMENT_DEFS[8][3:])
    # At Risk: mid-high F/M, low R
    if r <= 2 and f >= 3:
        return ("At Risk", *_SEGMENT_DEFS[7][3:])
    # Potential Loyalists
    if r >= 3 and f >= 2:
        return ("Potential Loyalists", *_SEGMENT_DEFS[2][3:])
    # New Customers
    if r >= 4 and f == 1:
        return ("New Customers", *_SEGMENT_DEFS[3][3:])
    # Promising
    if r >= 3 and f == 1:
        return ("Promising", *_SEGMENT_DEFS[4][3:])
    # Need Attention
    if r == 2 and f == 2:
        return ("Need Attention", *_SEGMENT_DEFS[5][3:])
    # About to Sleep
    if r == 2 and f == 1:
        return ("About to Sleep", *_SEGMENT_DEFS[6][3:])
    # Hibernating
    if r == 1 and f <= 2:
        return ("Hibernating", *_SEGMENT_DEFS[9][3:])
    # Lost (fallback — lowest everything)
    return ("Lost", *_SEGMENT_DEFS[10][3:])


# ══════════════════════════════════════════════════════════
#  COLUMN DETECTION
# ══════════════════════════════════════════════════════════

def detect_rfm_columns(df: pd.DataFrame) -> Optional[RFMColumns]:
    """
    Auto-detect customer_id, date, and monetary columns.
    Returns None if a customer_id or date column cannot be found —
    RFM analysis is not possible without both.
    """
    def _find(keywords: List[str], must_be_dtype=None) -> Optional[str]:
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in keywords):
                if must_be_dtype == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[c]):
                    # Try to parse it — failure here is expected and common
                    # (column matched a date keyword but isn't actually a
                    # parseable date, e.g. a "date_created_by" text column)
                    try:
                        pd.to_datetime(df[c], errors="raise")
                    except Exception:
                        logger.debug("Column '%s' matched date keyword but failed to parse as datetime", c)
                        continue
                return c
        return None

    customer_id = _find(["customerid", "customer_id", "custid", "cust_id",
                         "clientid", "client_id", "userid", "user_id"])
    if customer_id is None:
        # Fall back to any low-cardinality ID-like categorical column
        for c in df.columns:
            if "id" in c.lower() and df[c].nunique() < len(df) * 0.9 and df[c].nunique() > 1:
                customer_id = c
                break
    if customer_id is None:
        return None

    date_col = _find(["orderdate", "order_date", "transactiondate", "transaction_date",
                      "purchasedate", "purchase_date", "date", "invoicedate", "invoice_date"],
                     must_be_dtype="datetime")
    if date_col is None:
        return None

    monetary_col = _find(["totalprice", "total_price", "amount", "revenue", "sales",
                          "totalamount", "total_amount", "value", "spend"])
    quantity_col = _find(["quantity", "qty", "units"])
    price_col = None
    if monetary_col is None:
        # 'Price' alone (per-unit) wasn't in the monetary keyword list above
        # on purpose — summing raw per-unit prices across multiple line
        # items isn't real spend if quantities vary. Only used as a
        # monetary source in combination with quantity_col (price*qty),
        # never on its own.
        price_col = _find(["unitprice", "unit_price", "price"])

    return RFMColumns(customer_id=customer_id, date_col=date_col,
                      monetary_col=monetary_col, quantity_col=quantity_col,
                      price_col=price_col)


# ══════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ══════════════════════════════════════════════════════════

def run_rfm(
    df: pd.DataFrame,
    columns: Optional[RFMColumns] = None,
    analysis_date: Optional[pd.Timestamp] = None,
) -> RFMReport:
    """
    Run full RFM analysis. If `columns` is None, auto-detects them.

    Raises:
        ValueError — if required columns (customer_id, date) cannot be
        found or auto-detected, or if the dataset has too few customers
        for a meaningful segmentation (< 10 unique customers).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"run_rfm expects pd.DataFrame, got {type(df)}")

    warnings_list: List[str] = []

    if columns is None:
        columns = detect_rfm_columns(df)
        if columns is None:
            raise ValueError(
                "Could not detect customer ID and date columns required for "
                "RFM analysis. Ensure your dataset has a customer/client ID "
                "column and a transaction date column, or specify them manually."
            )

    need_price_qty = (columns.monetary_col is None and columns.price_col
                      and columns.quantity_col)
    cols_to_load = [columns.customer_id, columns.date_col]
    if columns.monetary_col:
        cols_to_load.append(columns.monetary_col)
    elif need_price_qty:
        cols_to_load += [columns.price_col, columns.quantity_col]
    work = df[cols_to_load].copy()

    # Parse dates
    work[columns.date_col] = pd.to_datetime(work[columns.date_col], errors="coerce")
    n_bad_dates = int(work[columns.date_col].isna().sum())
    if n_bad_dates > 0:
        warnings_list.append(
            f"{n_bad_dates} rows had unparseable dates and were excluded from RFM analysis."
        )
    work = work.dropna(subset=[columns.date_col])

    if work.empty:
        raise ValueError("No valid transaction dates found — cannot compute RFM.")

    n_customers_raw = work[columns.customer_id].nunique()
    if n_customers_raw < 10:
        raise ValueError(
            f"Only {n_customers_raw} unique customers found — RFM segmentation "
            f"requires at least 10 customers for meaningful segments."
        )

    # Monetary column — coerce to numeric, handle missing gracefully.
    # If there's no direct revenue/total column but price + quantity both
    # exist, derive line_total = price * quantity instead of falling back
    # to the frequency-only approximation.
    effective_monetary_col = columns.monetary_col
    if effective_monetary_col:
        work[effective_monetary_col] = pd.to_numeric(
            work[effective_monetary_col], errors="coerce"
        ).fillna(0)
        has_monetary = True
    elif need_price_qty:
        price_num = pd.to_numeric(work[columns.price_col], errors="coerce")
        qty_num = pd.to_numeric(work[columns.quantity_col], errors="coerce")
        work["_derived_line_total"] = (price_num * qty_num).fillna(0)
        effective_monetary_col = "_derived_line_total"
        has_monetary = True
        warnings_list.append(
            f"Monetary value derived from {columns.price_col} × "
            f"{columns.quantity_col} (no direct revenue/total column found)."
        )
    else:
        has_monetary = False
        warnings_list.append(
            "No monetary/revenue column detected — Monetary score approximated "
            "from transaction Frequency only. For accurate revenue-based "
            "segmentation, ensure your data has an amount/revenue column."
        )

    ref_date = analysis_date or (work[columns.date_col].max() + pd.Timedelta(days=1))

    # ── Aggregate to customer level ─────────────────────────────────────────
    agg_dict = {columns.date_col: ["max", "count"]}
    if has_monetary:
        agg_dict[effective_monetary_col] = "sum"

    cust = work.groupby(columns.customer_id).agg(agg_dict)
    cust.columns = ["_last_date", "_frequency"] + (["_monetary"] if has_monetary else [])
    cust = cust.reset_index()

    if not has_monetary:
        cust["_monetary"] = cust["_frequency"]  # proxy

    cust["_recency"] = (ref_date - cust["_last_date"]).dt.days

    # ── Score 1-5 using quantile binning (5 = best) ─────────────────────────
    def _qscore(series: pd.Series, ascending_is_better: bool) -> pd.Series:
        """Quantile-based 1-5 score. Handles duplicate edges gracefully."""
        try:
            ranks = series.rank(method="first")
            bins = pd.qcut(ranks, 5, labels=False, duplicates="drop") + 1
            if bins.max() < 5:
                # Too few unique values for 5 clean bins — pad up
                bins = bins.astype(float)
                bins = ((bins - bins.min()) / max(bins.max() - bins.min(), 1) * 4 + 1).round().astype(int)
            if not ascending_is_better:
                bins = 6 - bins
            return bins.astype(int)
        except Exception:
            logger.warning("Quantile scoring failed — falling back to rank-based tertiles", exc_info=True)
            med = series.median()
            return pd.Series(np.where(series >= med, 3, 2), index=series.index)

    # Recency: LOWER days = BETTER (so ascending_is_better=False, since raw
    # low values should map to score 5)
    cust["R"] = _qscore(cust["_recency"], ascending_is_better=False)
    cust["F"] = _qscore(cust["_frequency"], ascending_is_better=True)
    cust["M"] = _qscore(cust["_monetary"], ascending_is_better=True)
    cust["RFM_Score"] = cust["R"].astype(str) + cust["F"].astype(str) + cust["M"].astype(str)

    # ── Segment classification ──────────────────────────────────────────────
    seg_data = cust.apply(lambda row: _classify_segment(row["R"], row["F"], row["M"]), axis=1)
    cust["Segment"]     = seg_data.apply(lambda x: x[0])
    cust["_seg_desc"]   = seg_data.apply(lambda x: x[1])
    cust["_seg_action"] = seg_data.apply(lambda x: x[2])
    cust["_seg_color"]  = seg_data.apply(lambda x: x[3])

    cust = cust.rename(columns={
        columns.customer_id: "CustomerID",
        "_recency": "Recency", "_frequency": "Frequency", "_monetary": "Monetary",
        "_last_date": "LastPurchase",
    })

    total_revenue = float(cust["Monetary"].sum())

    # ── Segment summary table ────────────────────────────────────────────────
    summaries: List[RFMSegmentSummary] = []
    for seg_name in cust["Segment"].unique():
        sub = cust[cust["Segment"] == seg_name]
        desc   = sub["_seg_desc"].iloc[0]
        action = sub["_seg_action"].iloc[0]
        color  = sub["_seg_color"].iloc[0]
        summaries.append(RFMSegmentSummary(
            segment=seg_name,
            n_customers=len(sub),
            pct_customers=round(len(sub) / len(cust) * 100, 1),
            total_monetary=round(float(sub["Monetary"].sum()), 2),
            pct_revenue=round(float(sub["Monetary"].sum()) / total_revenue * 100, 1) if total_revenue > 0 else 0.0,
            avg_recency=round(float(sub["Recency"].mean()), 1),
            avg_frequency=round(float(sub["Frequency"].mean()), 1),
            avg_monetary=round(float(sub["Monetary"].mean()), 2),
            description=desc, action=action, color=color,
        ))
    summaries.sort(key=lambda s: -s.total_monetary)

    top_seg = summaries[0] if summaries else None
    at_risk_segs = [s for s in summaries if s.segment in ("At Risk", "Cannot Lose Them", "Hibernating")]
    at_risk_pct = sum(s.pct_customers for s in at_risk_segs)
    at_risk_rev = sum(s.total_monetary for s in at_risk_segs)
    champions_n = next((s.n_customers for s in summaries if s.segment == "Champions"), 0)

    display_cols = ["CustomerID", "Recency", "Frequency", "Monetary",
                    "R", "F", "M", "RFM_Score", "Segment"]

    return RFMReport(
        customer_table=cust[display_cols],
        segment_summary=summaries,
        columns_used=columns,
        analysis_date=ref_date,
        n_customers=len(cust),
        n_transactions=len(work),
        total_revenue=total_revenue,
        top_segment=top_seg.segment if top_seg else "N/A",
        top_segment_pct=top_seg.pct_customers if top_seg else 0.0,
        at_risk_pct=round(at_risk_pct, 1),
        at_risk_revenue=round(at_risk_rev, 2),
        champions_count=champions_n,
        warnings=warnings_list,
    )
