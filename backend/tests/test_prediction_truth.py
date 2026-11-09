"""
Predictions checked against data whose answer is known: a date is a
feature, not leakage; models with a date are scored on later rows; amounts
that are never negative are never predicted negative; a target that is a
formula of other columns is named as one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from app.engines.ml_engine import predict_what_if, run_ml_pipeline


def _trend(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2022-01-01", periods=n, freq="D")
    promo = rng.integers(0, 2, n)
    return pd.DataFrame({"Date": d, "Promo": promo, "Store": rng.choice(["A", "B"], n),
                         "Sales": 100 + 0.2 * np.arange(n) + rng.normal(0, 10, n) + 15 * promo})


def test_a_trending_date_is_used_not_dropped_as_leakage():
    r = run_ml_pipeline(_trend(), "Sales")
    assert "Date" in r.feature_cols
    assert not r.no_signal and not r.warnings


def test_a_model_with_dates_is_scored_on_later_rows():
    df = _trend()
    r = run_ml_pipeline(df, "Sales")
    assert "latest" in r.split_note and "latest" in r.summary
    # The reported score matches what the model achieves on genuinely later
    # dates; a random split reported 0.98 for a model that managed 0.78.
    cut = int(len(df) * 0.8)
    early = run_ml_pipeline(df.iloc[:cut], "Sales")
    later = df.iloc[cut:]
    preds = [predict_what_if(early, {"Date": str(x.Date.date()), "Promo": int(x.Promo),
                                     "Store": x.Store})["prediction"] for x in later.itertuples()]
    honest = r2_score(later.Sales, preds)
    assert abs(r.best_model.test_score - honest) < 0.12


def _orders(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    qty, price = rng.integers(1, 50, n), rng.lognormal(3, 1, n)
    return pd.DataFrame({"Qty": qty, "Category": rng.choice(["x", "y", "z"], n),
                         "Discount": rng.uniform(0, .3, n), "UnitPrice": price.round(2),
                         "Revenue": (qty * price * rng.lognormal(0, 0.2, n)).round(2)})


def test_an_amount_is_never_predicted_negative():
    r = run_ml_pipeline(_orders(), "Revenue")
    assert (r.y_pred >= 0).all()
    small = predict_what_if(r, {"Qty": 1, "Category": "x", "Discount": 0.3, "UnitPrice": 3})
    # Truth is about 3. The range is measured on orders of a similar size,
    # so it is a few rupees wide, not the ±200 of the whole book (an 80%
    # range, so one true value may fall just outside it).
    assert 0 <= small["lower"] <= small["prediction"] <= small["upper"] < 50
    if r.band_kind == "add":
        lo, hi = r.residual_band
        assert small["upper"] - small["lower"] < (hi - lo) / 3


def test_the_log_scale_is_chosen_only_when_it_predicts_better():
    rng = np.random.default_rng(3)
    n = 6000
    q, p = rng.integers(1, 50, n), rng.lognormal(3, 1, n)
    df = pd.DataFrame({"Qty": q, "Cat": rng.choice(list("xyz"), n),
                       "UnitPrice": p.round(2), "Revenue": (q * p * rng.lognormal(0, .2, n)).round(2)})
    r = run_ml_pipeline(df, "Revenue")
    names = [m.name for m in r.models]
    assert any("(log scale)" in nm for nm in names)
    best_raw = max(m.cv_score for m in r.models if "(log" not in m.name and m.model is not None)
    assert r.best_model.cv_score >= best_raw - 1e-9


def test_a_target_that_is_a_formula_is_named():
    rng = np.random.default_rng(1)
    n = 800
    cost, profit = rng.gamma(2, 50, n), rng.normal(30, 20, n)
    df = pd.DataFrame({"Cost": cost.round(2), "Profit": profit.round(2),
                       "Revenue": (cost + profit).round(2)})
    w = " ".join(run_ml_pipeline(df, "Revenue").warnings)
    assert "'Cost' + 'Profit'" in w
    q, p = rng.integers(1, 50, n), rng.lognormal(3, 1, n).round(2)
    df = pd.DataFrame({"Qty": q, "Price": p, "Disc": rng.uniform(0, .2, n),
                       "Revenue": (q * p).round(2)})
    w = " ".join(run_ml_pipeline(df, "Revenue").warnings)
    assert "'Qty' × 'Price'" in w


def test_noise_is_still_called_noise():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=1200), "b": rng.choice(list("pq"), 1200),
                       "Spend": rng.lognormal(3, 1, 1200)})
    assert run_ml_pipeline(df, "Spend").no_signal
