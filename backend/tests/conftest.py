"""
tests/conftest.py — shared pytest fixtures for the backend test suite.

Forces single-user open dev mode (no admin key / password) so these tests
don't depend on whichever machine or .env they happen to run on — see
multi_tenant_test.py for the auth-enforced path, which manages its own
env vars per-test.
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATA_DIR", "/tmp/analytiq-pytest-data")
os.environ["APP_ADMIN_KEY"] = ""
os.environ["APP_PASSWORD"] = ""


def _make_hr_df(seed: int = 42, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "employee_id": range(1, n + 1),
        "department": rng.choice(["Sales", "Engineering", "HR", "Marketing"], n),
        "salary": rng.normal(60000, 15000, n).round(0),
        "age": rng.integers(22, 60, n),
        "tenure_years": rng.integers(0, 20, n),
        "satisfaction": rng.uniform(1, 5, n).round(1),
        "attrition": rng.choice(["Yes", "No"], n, p=[0.2, 0.8]),
        "hire_date": pd.date_range("2015-01-01", periods=n, freq="3D"),
    })


@pytest.fixture()
def hr_df() -> pd.DataFrame:
    """Synthetic HR dataset: good for survival (tenure_years/attrition)
    and A/B testing (department/salary, department/attrition)."""
    return _make_hr_df()


@pytest.fixture()
def hr_csv_bytes() -> bytes:
    return _make_hr_df().to_csv(index=False).encode()


@pytest.fixture()
def transactions_df() -> pd.DataFrame:
    """Synthetic ecommerce transaction log: good for RFM."""
    rng = np.random.default_rng(7)
    n = 800
    n_customers = 120
    return pd.DataFrame({
        "customer_id": rng.integers(1, n_customers + 1, n),
        "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 365, n), unit="D"),
        "amount": rng.gamma(shape=2.0, scale=40.0, size=n).round(2),
    })


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture()
def uploaded_dataset_id(client, hr_csv_bytes):
    """Uploads the synthetic HR dataset and returns its dataset_id."""
    r = client.post(
        "/api/datasets/upload",
        files={"file": ("hr_test.csv", io.BytesIO(hr_csv_bytes), "text/csv")},
    )
    assert r.status_code == 200, r.text
    return r.json()["meta"]["dataset_id"]
