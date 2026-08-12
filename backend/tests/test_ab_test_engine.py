"""Unit tests for engines/ab_test_engine.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.ab_test_engine import (
    required_sample_size,
    run_continuous_test,
    run_conversion_test,
)


def test_conversion_test_detects_clear_lift():
    result = run_conversion_test(conversions_a=100, n_a=1000, conversions_b=180, n_b=1000)
    assert result.test_type == "conversion"
    assert bool(result.is_significant) is True
    assert result.p_value < 0.05
    assert result.rate_b > result.rate_a
    assert "Ship" in result.recommendation


def test_conversion_test_no_difference_is_not_significant():
    result = run_conversion_test(conversions_a=100, n_a=1000, conversions_b=102, n_b=1000)
    assert bool(result.is_significant) is False
    assert "Do not ship" in result.recommendation


def test_conversion_test_small_sample_uses_fishers_exact():
    result = run_conversion_test(conversions_a=1, n_a=20, conversions_b=6, n_b=20)
    assert result.test_used == "Fisher's Exact Test"
    assert any("Fisher" in w for w in result.warnings)


def test_conversion_test_zero_baseline_uses_percentage_points():
    result = run_conversion_test(conversions_a=0, n_a=500, conversions_b=10, n_b=500)
    assert "percentage points" in result.verdict or result.warnings


def test_conversion_test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        run_conversion_test(conversions_a=0, n_a=0, conversions_b=1, n_b=10)
    with pytest.raises(ValueError):
        run_conversion_test(conversions_a=11, n_a=10, conversions_b=1, n_b=10)


def test_continuous_test_detects_clear_difference():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(100, 10, 200))
    b = pd.Series(rng.normal(115, 10, 200))
    result = run_continuous_test(a, b, metric_name="Order Value")
    assert result.test_type == "continuous"
    assert bool(result.is_significant) is True
    assert result.mean_b > result.mean_a


def test_continuous_test_small_sample_uses_mann_whitney():
    a = pd.Series([1, 2, 3, 4, 5])
    b = pd.Series([6, 7, 8, 9, 10])
    result = run_continuous_test(a, b)
    assert "Mann-Whitney" in result.test_used


def test_continuous_test_requires_minimum_observations():
    with pytest.raises(ValueError):
        run_continuous_test(pd.Series([1]), pd.Series([1, 2, 3]))


def test_continuous_test_nonpositive_baseline_uses_absolute_framing():
    a = pd.Series([-5.0, -3.0, -1.0, 0.0, 2.0] * 5)
    b = pd.Series([10.0, 12.0, 9.0, 11.0, 13.0] * 5)
    result = run_continuous_test(a, b)
    assert "relative % isn't meaningful" in result.verdict or result.relative_uplift == 0.0


def test_required_sample_size_positive_and_monotonic_in_effect_size():
    n_small_effect = required_sample_size(baseline_rate=0.1, min_detectable_effect_pct=5)
    n_large_effect = required_sample_size(baseline_rate=0.1, min_detectable_effect_pct=50)
    assert n_small_effect > n_large_effect > 0


def test_required_sample_size_rejects_invalid_baseline():
    with pytest.raises(ValueError):
        required_sample_size(baseline_rate=0)
    with pytest.raises(ValueError):
        required_sample_size(baseline_rate=1)
