"""
tests/test_legacy_suites.py — pytest entry points for the existing
script-style regression suites (smoke_test.py, multi_tenant_test.py,
data_integrity_test.py, video_frames_test.py).

Each suite manages its own env vars / TestClient lifecycle and prints a
human-readable pass/fail summary, so it's run as a subprocess rather than
imported directly — that keeps its internal `sys.exit()` calls from
tearing down the whole pytest run, and keeps each suite's env var setup
isolated from the others (this matters most for multi_tenant_test.py,
which deliberately sets APP_ADMIN_KEY/APP_PASSWORD to non-open-mode
values that must not leak into the other suites' test client requests).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_suite(module: str) -> None:
    # DATA_DIR is deliberately dropped here even though conftest.py sets
    # it (via os.environ.setdefault, for the in-process `client` fixture):
    # subprocess.run() inherits the parent's environment, and each legacy
    # suite also does `os.environ.setdefault("DATA_DIR", ...)` with its
    # own distinct path for isolation. If conftest.py's value is inherited,
    # every suite's setdefault becomes a no-op and they all end up sharing
    # one DATA_DIR — e.g. multi_tenant_test's persisted admin/client
    # accounts then leak into data_integrity_test's run, flipping it out
    # of open dev mode and turning its upload call into a 401.
    env = {k: v for k, v in os.environ.items() if k != "DATA_DIR"}
    result = subprocess.run(
        [sys.executable, "-m", f"tests.{module}"],
        cwd=BACKEND_DIR, capture_output=True, text=True, timeout=300, env=env,
    )
    assert result.returncode == 0, (
        f"tests.{module} failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_smoke_suite():
    _run_suite("smoke_test")


def test_multi_tenant_suite():
    _run_suite("multi_tenant_test")


def test_data_integrity_suite():
    _run_suite("data_integrity_test")


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not on PATH",
)
def test_video_frames_suite():
    _run_suite("video_frames_test")
