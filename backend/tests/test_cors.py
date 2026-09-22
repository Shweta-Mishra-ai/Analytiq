"""
Who is allowed to call this API from a browser.

The default used to be "*", and the reasoning for it was that the
frontend needs to reach the backend. It does not: in development Vite
proxies /api, and in production FastAPI serves the built frontend
itself, so the browser is same-origin in both and same-origin needs no
CORS headers at all.

What the wildcard did buy was an exfiltration path. Unset APP_ADMIN_KEY
with no accounts yet is single-user open mode — no authentication on any
route — so any page the operator visited could read every dataset on
their machine and post it elsewhere.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _app_with(monkeypatch, cors_origins: str):
    """Rebuild the app, because middleware is attached at import."""
    from app.config import config
    monkeypatch.setattr(config, "cors_origins", cors_origins, raising=False)

    import app.main as main
    importlib.reload(main)
    return main.app


@pytest.fixture(autouse=True)
def _restore_app():
    """Leave app.main as the rest of the suite expects to find it."""
    yield
    import app.main as main
    importlib.reload(main)


def test_the_default_allows_no_cross_origin_reader(monkeypatch):
    """The header's absence is the whole protection: without it the
    browser refuses to hand the response to the calling page."""
    client = TestClient(_app_with(monkeypatch, ""))

    res = client.get("/api/health", headers={"Origin": "https://evil.example"})

    assert res.status_code == 200, "same-origin callers must still work"
    assert "access-control-allow-origin" not in res.headers


def test_a_named_origin_is_allowed(monkeypatch):
    client = TestClient(_app_with(monkeypatch, "https://app.example.com"))

    res = client.get("/api/health",
                     headers={"Origin": "https://app.example.com"})

    assert res.headers.get("access-control-allow-origin") == \
        "https://app.example.com"


def test_naming_one_origin_does_not_admit_another(monkeypatch):
    client = TestClient(_app_with(monkeypatch, "https://app.example.com"))

    res = client.get("/api/health", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in res.headers


def test_several_origins_can_be_named(monkeypatch):
    client = TestClient(_app_with(
        monkeypatch, "https://a.example.com, https://b.example.com"))

    for origin in ("https://a.example.com", "https://b.example.com"):
        res = client.get("/api/health", headers={"Origin": origin})
        assert res.headers.get("access-control-allow-origin") == origin


def test_a_wildcard_still_works_but_says_so(monkeypatch, caplog):
    """An operator who means it gets it. They also get told what they
    just opened, because the combination with open mode is the one that
    loses a client's data."""
    import logging
    caplog.set_level(logging.WARNING)

    client = TestClient(_app_with(monkeypatch, "*"))
    res = client.get("/api/health", headers={"Origin": "https://evil.example"})

    assert res.headers.get("access-control-allow-origin") == "*"
    assert any("CORS_ORIGINS is '*'" in r.message for r in caplog.records), \
        "opening the API to every origin must not be silent"


def test_credentials_are_never_allowed_cross_origin(monkeypatch):
    """Tokens travel in an Authorization header the page sets itself, so
    there is nothing for the browser to attach automatically — and
    allow_credentials would additionally make a wildcard illegal."""
    client = TestClient(_app_with(monkeypatch, "https://app.example.com"))

    res = client.get("/api/health",
                     headers={"Origin": "https://app.example.com"})

    assert "access-control-allow-credentials" not in res.headers
