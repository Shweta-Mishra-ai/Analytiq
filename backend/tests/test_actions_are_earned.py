"""
tests/test_actions_are_earned.py — a recommendation needs a reason in the
data, and urgency needs evidence.

A sales file of plain transactions (amount, region, channel) was told, as
"[CRITICAL]", to "identify top 3 deals at risk in pipeline" and to review
revenue "vs target — per rep": it had no pipeline, target or rep column.
The CRITICAL label came from the action's position in a fixed list. The
same report put the skew down to "a few large deals (Pareto effect)" when
the top fifth of transactions held well under half the revenue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.engines.story_engine import detect_domain, generate_story


def _transactions(heavy: bool, n: int = 1000, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = (rng.lognormal(3, 2.0, n) if heavy else rng.gamma(2.0, 90.0, n)).round(2)
    # The columns of an ordinary order export, as the app classifies it: sales.
    return pd.DataFrame({
        "order_id": [f"O{100000 + i}" for i in range(n)],
        "customer_id": rng.integers(1, 150, n),
        "order_date": pd.date_range("2025-01-01", periods=n, freq="h"),
        "region": rng.choice(["North", "South", "East", "West", "Central"], n),
        "channel": rng.choice(["web", "store"], n),
        "amount": amount,
        "quantity": rng.integers(1, 6, n),
        "returned": rng.choice(["Yes", "No"], n, p=[0.1, 0.9]),
    })


def _overview(story):
    return next(i for i in story.top_insights
                if getattr(i, "title", "").startswith("Revenue Overview"))


def test_the_fixture_is_read_as_sales_data():
    assert detect_domain(_transactions(heavy=False))[0] == "sales"
    assert detect_domain(_transactions(heavy=True))[0] == "sales"


def test_no_action_names_a_column_the_data_does_not_have():
    story = generate_story(_transactions(heavy=False))
    text = " ".join(story.recommended_actions).lower()
    for absent in ("pipeline", "target", "per rep", "margin"):
        assert absent not in text, (absent, story.recommended_actions)


def test_actions_carry_no_invented_urgency():
    story = generate_story(_transactions(heavy=True))
    assert story.recommended_actions
    assert not any(a.lstrip().startswith("[") for a in story.recommended_actions)


def test_skewed_but_spread_revenue_is_not_called_a_pareto_effect():
    ins = _overview(generate_story(_transactions(heavy=False)))
    assert "Pareto" not in ins.cause
    assert "not concentrated" in ins.problem
    assert "of revenue" in ins.cause


def test_concentrated_revenue_is_measured_and_acted_on():
    story = generate_story(_transactions(heavy=True))
    ins = _overview(story)
    assert "concentrated in the largest transactions" in ins.problem
    assert any("concentration audit" in a.lower() for a in story.recommended_actions)
