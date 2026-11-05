"""Synthetic data for the month-anomaly and Simpson's-paradox checks."""
import numpy as np
import pandas as pd


def spike_df(seed=1, spike_month=9, factor=2.2):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2024-01-01", "2024-12-31")
    df = pd.DataFrame({"Order Date": np.repeat(d, 5),
                       "Revenue": rng.normal(100, 10, len(d) * 5).round(2),
                       "Region": rng.choice(["North", "South"], len(d) * 5)})
    if spike_month:
        df.loc[df["Order Date"].dt.month == spike_month, "Revenue"] *= factor
    return df


def simpson_df(seed=1, n=2000):
    """B beats A inside each plan, but A sits mostly on the high-scoring
    plan, so A leads overall."""
    rng = np.random.default_rng(seed)
    plan = rng.choice(["Basic", "Pro"], n)
    grp = np.where(plan == "Basic", rng.choice(["A", "B"], n, p=[.85, .15]),
                   rng.choice(["A", "B"], n, p=[.15, .85]))
    result = np.where(plan == "Basic", 80, 40) + np.where(grp == "B", 5, 0) + rng.normal(0, 5, n)
    return pd.DataFrame({"Group": grp, "Plan": plan, "Result": result})


def daily_sales(seed=0, per_day=10, season=0.0, spike=None, factor=2.2,
                start="2022-01-01", end="2024-12-31"):
    """Orders a day (Poisson) with optional ±season seasonality and a
    one-month spike at (year, month)."""
    rng = np.random.default_rng(seed)
    d = pd.date_range(start, end)
    lam = per_day * (1 + season * np.sin(2 * np.pi * (d.month - 1) / 12))
    n = rng.poisson(lam)
    df = pd.DataFrame({"Order Date": np.repeat(d, n),
                       "Revenue": rng.gamma(2, 50, n.sum()).round(2)})
    if spike:
        y, m = spike
        mask = (df["Order Date"].dt.year == y) & (df["Order Date"].dt.month == m)
        df.loc[mask, "Revenue"] *= factor
    return df
