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


def _run_suite(module: str, tmp_path) -> None:
    # Each suite must get a DATA_DIR of its own, and a fresh one per run.
    #
    # Two ways to get this wrong, both of which have bitten:
    #   - inheriting conftest.py's DATA_DIR makes every suite's own
    #     os.environ.setdefault() a no-op, so they share one directory and
    #     multi_tenant_test's persisted admin/client accounts leak into
    #     data_integrity_test, flipping it out of open dev mode and turning
    #     its upload into a 401;
    #   - dropping DATA_DIR entirely lets each suite fall back to its own
    #     hard-coded /tmp path, which survives the run. The second
    #     invocation then finds tenant_a and tenant_b already registered
    #     and fails on "admin creates tenant_b" — green on a clean CI
    #     runner, red on any developer's second `pytest`.
    #
    # A per-suite tmp_path satisfies both: distinct between suites, and
    # discarded by pytest afterwards.
    env = {k: v for k, v in os.environ.items() if k != "DATA_DIR"}
    env["DATA_DIR"] = str(tmp_path / module)
    result = subprocess.run(
        [sys.executable, "-m", f"tests.{module}"],
        cwd=BACKEND_DIR, capture_output=True, text=True, timeout=300, env=env,
    )
    assert result.returncode == 0, (
        f"tests.{module} failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )


def test_smoke_suite(tmp_path):
    _run_suite("smoke_test", tmp_path)


def test_multi_tenant_suite(tmp_path):
    _run_suite("multi_tenant_test", tmp_path)


def test_data_integrity_suite(tmp_path):
    _run_suite("data_integrity_test", tmp_path)


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not on PATH",
)
def test_video_frames_suite(tmp_path):
    _run_suite("video_frames_test", tmp_path)
