"""
Order-line exports (one row per product on an invoice, a quantity and a
unit price, no revenue column, cancellations as 'C' invoices) — the shape
of most shop data, and of the UCI Online Retail II file every defect here
was found on.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from app.engines import transactions as tx


def _lines(n_customers=120, months=26, seed=3):
    """Order lines across `months` months, ending on the 9th of a month."""
    rng = np.random.default_rng(seed)
    rows, inv = [], 500000
    start = pd.Timestamp("2009-12-01")
    end = start + pd.DateOffset(months=months - 1) + pd.Timedelta(days=8)
    for c in range(n_customers):
        first = start + pd.Timedelta(days=int(rng.integers(0, 300)))
        for _ in range(int(rng.integers(1, 9))):
            day = first + pd.Timedelta(days=int(rng.integers(0, 700)))
            if day > end:
                continue
            inv += 1
            for _ in range(int(rng.integers(2, 12))):          # lines per order
                rows.append({"Invoice": str(inv), "StockCode": f"S{rng.integers(1, 80)}",
                             "Description": f"ITEM {rng.integers(1, 80)}",
                             "Quantity": int(rng.integers(1, 24)),
                             "InvoiceDate": day, "Price": float(rng.choice([0.85, 1.25, 2.1, 4.95])),
                             "Customer ID": 12000.0 + c,
                             "Country": rng.choice(["United Kingdom"] * 8 + ["France", "EIRE"])})
            if rng.random() < 0.15:                            # a cancellation
                rows.append({**rows[-1], "Invoice": f"C{inv}", "Quantity": -rows[-1]["Quantity"]})
    df = pd.DataFrame(rows)
    df.loc[df.sample(frac=0.2, random_state=1).index, "Customer ID"] = np.nan
    return df


def test_revenue_is_quantity_times_price_and_net_of_returns():
    df, note = tx.derive_line_revenue(_lines())
    assert note and tx.is_derivation_note(note)
    assert np.allclose(df["Revenue"], (df["Quantity"] * df["Price"]).round(2))
    assert (df.loc[df["Quantity"] < 0, "Revenue"] < 0).all()
    # A file with its own amount keeps it.
    own = _lines().assign(**{"Line Total": 1.0})
    assert tx.derive_line_revenue(own)[1] is None


def test_a_plain_invoice_column_is_the_order_and_C_invoices_are_cancellations():
    df = _lines()
    assert tx.order_column(df) == "Invoice"
    mask = tx.cancellation_mask(df)
    assert mask.sum() == df["Invoice"].str.startswith("C").sum() > 0
    # Orders are invoices, not lines.
    assert tx.count_orders(df) == df["Invoice"].nunique() < len(df)


def test_the_upload_adds_revenue_and_the_report_says_how(client):
    df = _lines()
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    r = client.post("/api/datasets/upload",
                    files={"file": ("retail.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    meta = r.json()["meta"]
    assert any(tx.is_derivation_note(w) for w in meta["warnings"])
    ds = meta["dataset_id"]
    cols = client.get(f"/api/datasets/{ds}/preview?rows=5").json()
    assert "Revenue" in (cols.get("columns") or list(cols["rows"][0]))
    pdf = client.post(f"/api/reports/{ds}/pdf", json={"title": "Retail"})
    assert pdf.status_code == 200
    import pypdf
    text = " ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(pdf.content)).pages)
    text = " ".join(text.split())
    assert "was added as 'Quantity' × 'Price'" in text
    # The trend is of revenue; it used to be "Monthly average price".
    assert "monthly total revenue" in text.lower()
    assert "monthly average price" not in text.lower()


def test_loader_keeps_invoice_numbers_as_text():
    from app.engines.data_loader import load_file

    class _F:
        name = "lines.csv"

        def __init__(self, data):
            self.data, self.size = data, len(data)

        def read(self, *a):
            return self.data

        def seek(self, *a):
            pass

    df = _lines()
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    out = load_file(_F(buf.getvalue())).df
    # "489434.0" beside "C489449" matched nothing; both stay text.
    assert out["Invoice"].map(type).eq(str).all()
    assert not out["Invoice"].str.endswith(".0").any()


def test_cleaning_neither_blanks_mixed_ids_nor_imputes_a_customer():
    from app.engines.data_cleaner import auto_clean
    df = _lines()
    # A mixed column, as a loader that converted the numeric part leaves it.
    df["Invoice"] = [float(v) if not v.startswith("C") else v for v in df["Invoice"]]
    df["Invoice"] = df["Invoice"].astype(object)
    out, report = auto_clean(df)
    assert "Invoice" in out.columns
    assert out["Invoice"].notna().all()
    assert out["Customer ID"].isna().sum() == df.drop_duplicates()["Customer ID"].isna().sum()
    assert not any("median" in a.action for a in report.actions if a.column == "Customer ID")


def test_frequency_counts_orders_not_lines():
    from app.engines.rfm_engine import detect_rfm_columns, run_rfm
    df, _ = tx.derive_line_revenue(_lines())
    cols = detect_rfm_columns(df)
    assert cols.order_col == "Invoice"
    rep = run_rfm(df, cols)
    placed = df[~tx.cancellation_mask(df)].dropna(subset=["Customer ID"])
    per_customer = placed.groupby("Customer ID")["Invoice"].nunique()
    assert max(s.avg_frequency for s in rep.segment_summary) <= per_customer.max()
    lines_per_customer = placed.groupby("Customer ID").size().mean()
    assert all(s.avg_frequency < lines_per_customer for s in rep.segment_summary)


def test_scorecard_orders_exclude_cancellations_and_returns_are_reported():
    from app.engines.scorecard import build_scorecard
    df, _ = tx.derive_line_revenue(_lines(n_customers=300))
    sc = build_scorecard(df)
    by = {m.label: m for m in sc.metrics}
    assert "Orders" in by and "Average order value" in by
    assert any("Returns and cancellations" in m.title for m in sc.messages)


def test_retention_leaves_out_the_first_month_and_unfinished_return_months():
    from app.engines.domains.customer_analytics import cohort_retention
    df, _ = tx.derive_line_revenue(_lines(n_customers=900, seed=5))
    findings, insights = [], []
    cohort_retention(df, insights, findings, [], [])
    text = " ".join(findings) + " ".join(str(i) for i in insights)
    first = df["InvoiceDate"].min().to_period("M")
    last = df["InvoiceDate"].max().to_period("M")          # ends on the 9th
    for month in (first, last, last - 1):
        assert str(month) + ":" not in text


def test_arithmetic_pairs_are_not_findings():
    from app.engines.story_engine import _is_tautological_pair
    assert _is_tautological_pair("Quantity", "Revenue")
    assert _is_tautological_pair("Price", "Quantity")
    assert not _is_tautological_pair("Tenure", "Revenue")


def test_postage_and_fees_are_not_ranked_as_products():
    from app.engines.chart_exporter import build_report_charts
    df, _ = tx.derive_line_revenue(_lines(n_customers=200))
    df = pd.concat([df, df.head(400).assign(Description="DOTCOM POSTAGE", Quantity=1,
                                            Price=500.0, Revenue=500.0)])
    top = next(c for c in build_report_charts(df, domain="ecommerce", max_charts=6)
               if c.kind == "top_items")
    assert "POSTAGE" not in top.caption.split("Postage")[0]
    assert "left out of the ranking" in top.caption


def test_root_cause_does_not_name_a_column_the_target_is_computed_from():
    from app.engines.bi_engine import analyze_root_cause
    df, _ = tx.derive_line_revenue(_lines(n_customers=200))
    res = analyze_root_cause(df, "Revenue")
    factors = {d["factor"] for d in res.drivers}
    assert not factors & {"Quantity", "Price"}
