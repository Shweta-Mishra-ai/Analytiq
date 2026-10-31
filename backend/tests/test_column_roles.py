"""
tests/test_column_roles.py — the user says once what each column is, and
every analysis uses it.

Guessed from names, "amount_paid" was an identifier (it contains "id"), a
store number was a measure to be summed, and "wait_minutes" was judged as if
more were better. The guesses stay as the starting point; a confirmed role
replaces them everywhere, survives cleaning, undo and reset, and changes the
cached results.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.engines.benchmarking import metric_direction
from app.engines.domains.base import is_id_column
from app.services import column_roles
from app.services.column_roles import RoleError, apply


def _upload(client, df, name="d.csv"):
    r = client.post("/api/datasets/upload", files={
        "file": (name, io.BytesIO(df.to_csv(index=False).encode()), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["meta"]["dataset_id"]


def _stores(n=120, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "order_ref": [f"R{1000 + i}" for i in range(n)],
        "branch": rng.integers(1, 9, n),     # a label stored as a number
        "wait_minutes": rng.gamma(2, 5, n).round(1),
        "score": rng.normal(70, 10, n).round(1),
        "region": rng.choice(["N", "S", "E"], n),
        "order_date": pd.date_range("2025-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
    })


def _roles(client, ds):
    r = client.get(f"/api/datasets/{ds}/columns")
    assert r.status_code == 200, r.text
    return {c["name"]: c for c in r.json()["columns"]}


def _save(client, ds, roles):
    return client.post(f"/api/datasets/{ds}/columns", json={"columns": roles})


# ── detection ─────────────────────────────────────────────

def test_the_starting_point_is_the_detected_role_of_every_column(client):
    cols = _roles(client, _upload(client, _stores()))
    assert cols["order_ref"]["role"] == "identifier"
    assert cols["order_date"]["role"] == "date"
    assert cols["region"]["role"] == "category"
    assert cols["wait_minutes"]["role"] == "measure"
    assert not any(c["confirmed"] for c in cols.values())
    assert cols["region"]["examples"]


# ── confirmed roles reach the analyses ────────────────────

def test_a_number_used_as_a_label_stops_being_analysed_as_a_measure(client):
    ds = _upload(client, _stores())
    before = client.get(f"/api/analytics/{ds}/stats").json()
    assert "branch" in before["numeric_cols"]
    roles = {n: {"role": c["role"], "direction": c["direction"]}
             for n, c in _roles(client, ds).items()}
    roles["branch"] = {"role": "category", "direction": 0}
    assert _save(client, ds, roles).status_code == 200
    after = client.get(f"/api/analytics/{ds}/stats").json()
    assert "branch" not in after["numeric_cols"]
    assert "branch" in after["categorical_cols"]
    # Labels, not 7.0
    preview = client.get(f"/api/datasets/{ds}/preview?rows=5").json()
    assert all("." not in str(r["branch"]) for r in preview["records"])


def test_a_confirmed_identifier_is_excluded_and_a_wrong_guess_is_overruled(client):
    df = _stores().assign(amount_paid=lambda d: np.arange(len(d)) + 1)
    ds = _upload(client, df)
    # A 1, 2, 3… integer sequence is guessed to be an identifier.
    assert "amount_paid" in client.get(f"/api/analytics/{ds}/stats").json()["identifier_cols"]
    assert _save(client, ds, {"amount_paid": {"role": "measure", "direction": 1},
                              "score": {"role": "identifier"}}).status_code == 200
    stats = client.get(f"/api/analytics/{ds}/stats").json()
    assert "amount_paid" in stats["numeric_cols"]
    assert "score" in stats["identifier_cols"] and "score" not in stats["numeric_cols"]


def test_a_confirmed_direction_replaces_the_guess_from_the_name():
    s = pd.Series([1.0, 2.0], name="reading")
    assert metric_direction("reading", s) == 0          # the name says nothing
    s.attrs[column_roles.ATTR] = {"reading": {"role": "measure", "direction": -1}}
    assert metric_direction("reading", s) == -1
    # "score" reads as higher-is-better; a golf score is the other way round.
    df = pd.DataFrame({"score": [1.0, 2.0]})
    assert metric_direction("score", df["score"]) == 1
    df.attrs[column_roles.ATTR] = {"score": {"role": "measure", "direction": -1}}
    assert metric_direction("score", df["score"].dropna()) == -1   # through a selection
    assert is_id_column("score", df["score"]) is False


# ── conversions ───────────────────────────────────────────

def test_text_dates_become_dates_and_the_unreadable_few_are_reported():
    df = pd.DataFrame({"when": [f"2024-03-{d:02d}" for d in range(1, 21)] + ["unknown"]})
    out, notes = apply(df, {"when": {"role": "date"}})
    assert pd.api.types.is_datetime64_any_dtype(out["when"])
    assert out["when"].isna().sum() == 1
    assert any("1 value(s) in 'when'" in n for n in notes)


def test_a_role_the_values_cannot_support_is_refused_with_examples():
    df = pd.DataFrame({"code": ["A1", "B2", "C3", "12", "13"]})
    with pytest.raises(RoleError, match="'A1'"):
        apply(df, {"code": {"role": "measure", "direction": 1}})


def test_numbers_with_thousands_separators_become_a_measure():
    df = pd.DataFrame({"revenue": ["1,200", "3,400.50", "75"]})
    out, _ = apply(df, {"revenue": {"role": "measure", "direction": 1}})
    assert out["revenue"].tolist() == [1200.0, 3400.5, 75.0]


def test_the_api_refuses_unknown_columns_roles_and_directions(client):
    ds = _upload(client, _stores())
    for body, needle in [({"nope": {"role": "measure"}}, "No column named 'nope'"),
                         ({"region": {"role": "kpi"}}, "is not a role"),
                         ({"score": {"role": "measure", "direction": 5}}, "must be 1, -1 or 0"),
                         ({"region": {"role": "date"}}, "read as dates")]:
        r = _save(client, ds, body)
        assert r.status_code == 422, body
        assert needle in r.json()["detail"], r.json()
    assert not any(c["confirmed"] for c in _roles(client, ds).values())


# ── persistence and invalidation ──────────────────────────

def test_roles_survive_cleaning_reset_and_a_restart(client):
    from app.services.dataset_store import DatasetStore, store
    ds = _upload(client, _stores())
    assert _save(client, ds, {"branch": {"role": "category"},
                              "wait_minutes": {"role": "measure", "direction": -1}}
                 ).status_code == 200
    assert client.post(f"/api/datasets/{ds}/clean", json={}).status_code == 200
    assert client.post(f"/api/datasets/{ds}/reset").status_code == 200
    cols = _roles(client, ds)
    assert cols["branch"]["confirmed"] and cols["branch"]["role"] == "category"
    assert cols["wait_minutes"]["direction"] == -1
    # A fresh process reading the same files.
    fresh = DatasetStore(store.base_dir)
    df = fresh.get_df("local", ds)
    assert not pd.api.types.is_numeric_dtype(df["branch"])
    assert metric_direction("wait_minutes", df["wait_minutes"]) == -1


def test_changing_a_role_back_undoes_its_conversion(client):
    ds = _upload(client, _stores())
    assert _save(client, ds, {"branch": {"role": "category"}}).status_code == 200
    assert _roles(client, ds)["branch"]["dtype"] != "int64"
    assert _save(client, ds, {"branch": {"role": "measure", "direction": 0}}).status_code == 200
    assert "branch" in client.get(f"/api/analytics/{ds}/stats").json()["numeric_cols"]


def test_saved_data_files_carry_no_roles(client):
    import pickle
    from app.services.dataset_store import store
    ds = _upload(client, _stores())
    _save(client, ds, {"branch": {"role": "category"}})
    client.post(f"/api/datasets/{ds}/clean", json={})
    with open(store._path("local", ds, "active.pkl"), "rb") as f:
        assert column_roles.ATTR not in pickle.load(f).attrs
