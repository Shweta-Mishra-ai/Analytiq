#!/usr/bin/env python3
"""
scripts/check_api_keys.py — verify GROQ_API_KEY and GEMINI_API_KEY
actually work, with one real, minimal, cheap call to each provider.

Why this exists: an AI assistant helping you set this up cannot reach
api.groq.com or generativelanguage.googleapis.com from its own sandbox
(both are blocked by its network policy) — so it can't test a key on
your behalf, even if you paste one into chat. This script runs the
same check *you* can actually run, using *your* network, in a few
seconds, with your key never leaving your machine.

Usage (from backend/):
    python3 scripts/check_api_keys.py

Reads GROQ_API_KEY / GEMINI_API_KEY from the environment, or from a
.env file in the current or parent directory (same as the app does).
"""
from __future__ import annotations

import os

# Quiet gRPC's own connection-retry logging (some of the Gemini SDK's
# internals still touch it) — otherwise a blocked/failing network can
# flood stderr with low-level handshake noise that has nothing to do
# with the actual key. Harmless no-op if nothing uses gRPC.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import queue
import sys
import threading
import time

# Load .env the same way the app does, without requiring the app's
# full dependency stack to be importable.
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

HARD_TIMEOUT_SEC = 20  # wall-clock cap, independent of whatever the SDK does


def _with_hard_timeout(fn, *args) -> tuple[bool, str]:
    """Runs fn in a daemon thread and gives up after HARD_TIMEOUT_SEC no
    matter what the underlying SDK does. This matters in practice: the
    google-genai SDK has open, unresolved upstream issues where it does
    not reliably honor its own timeout on a stalled/blocked connection
    (see googleapis/python-genai#911, #1893) — so trusting the SDK's
    timeout= alone can hang this script indefinitely on a bad network."""
    q: queue.Queue = queue.Queue(maxsize=1)

    def _run():
        try:
            q.put(fn(*args))
        except Exception as e:
            q.put((False, f"{type(e).__name__}: {e}"))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    try:
        return q.get(timeout=HARD_TIMEOUT_SEC)
    except queue.Empty:
        return False, (f"timed out after {HARD_TIMEOUT_SEC}s (network "
                        f"unreachable, or blocked by a firewall/proxy)")


def check_groq(key: str) -> tuple[bool, str]:
    if not key:
        return False, "GROQ_API_KEY not set"
    try:
        from groq import Groq
    except ImportError:
        return False, "groq package not installed — pip install -r requirements.txt"
    try:
        t0 = time.time()
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=5,
            timeout=15,
        )
        text = (resp.choices[0].message.content or "").strip()
        ms = int((time.time() - t0) * 1000)
        return True, f"responded in {ms}ms (model said: {text!r})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_gemini(key: str) -> tuple[bool, str]:
    if not key:
        return False, "GEMINI_API_KEY not set"
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return False, "google-genai package not installed — pip install -r requirements.txt"
    try:
        t0 = time.time()
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Reply with exactly: OK",
            config=types.GenerateContentConfig(max_output_tokens=5),
        )
        text = (resp.text or "").strip()
        ms = int((time.time() - t0) * 1000)
        return True, f"responded in {ms}ms (model said: {text!r})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()

    print("Checking GROQ_API_KEY  (powers Chat copilot + chart narratives)...")
    groq_ok, groq_msg = _with_hard_timeout(check_groq, groq_key)
    print(f"  {'✅' if groq_ok else '❌'} {groq_msg}")

    print("\nChecking GEMINI_API_KEY (powers RAG, image/video-to-dataset)...")
    gemini_ok, gemini_msg = _with_hard_timeout(check_gemini, gemini_key)
    print(f"  {'✅' if gemini_ok else '❌'} {gemini_msg}")

    print()
    if groq_ok and gemini_ok:
        print("🎉 Both keys work. All AI features are live.")
        return 0
    if not groq_ok and not gemini_ok:
        print("Neither key is working yet — see the messages above.")
    else:
        print("One key works, one doesn't — the app will run with that "
              "feature set partially enabled until the other is fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
