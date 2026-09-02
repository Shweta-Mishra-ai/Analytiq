#!/usr/bin/env python3
"""
scripts/check_api_keys.py — does every configured model provider
actually answer?

Why this exists: a key can be present, well-formed, and still not work
— expired, from the wrong account, out of quota, or blocked by the
network the app runs on. None of that shows up in the configuration,
and all of it looks the same from outside: the reports quietly come
back in the analysis engines' own wording instead of a model's.

Only a real call finds out, and only the machine holding the key can
make it. This script is the terminal version of what the running app
exposes at POST /api/admin/llm-check and on its System page — same
registry, same checks, so the two cannot drift apart. Use this one
locally; use the endpoint on a deployment, where the secrets actually
live (a Render environment variable, or a GitHub Actions secret the
workflow passes through, is not readable from anywhere else).

Usage (from backend/):
    python3 scripts/check_api_keys.py
    python3 scripts/check_api_keys.py groq openrouter   # only these

Keys are read from the environment or a .env file, exactly as the app
reads them, and never leave this machine.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai import providers                                    # noqa: E402
from app.ai.llm_client import fallback_order, task_routing      # noqa: E402
from app.config import config                                   # noqa: E402

GREEN, RED, DIM, YELLOW, RESET = (
    "\033[32m", "\033[31m", "\033[2m", "\033[33m", "\033[0m")


def _plain() -> bool:
    """Colour codes in a redirected file are noise, and CI logs are the
    place this output is most often read."""
    return not sys.stdout.isatty() or os.environ.get("NO_COLOR")


def paint(text: str, colour: str) -> str:
    return text if _plain() else f"{colour}{text}{RESET}"


def main() -> int:
    only = [a.strip().lower() for a in sys.argv[1:]] or None

    print(f"\n{config.app_name} — model provider check")
    print("=" * 62)

    results = providers.check_all(only=only)
    if not results:
        print(f"No provider matched {only}. Known: "
              f"{', '.join(p.name for p in providers.all_providers())}")
        return 2

    working = []
    for chk in results:
        if not chk.configured:
            print(f"\n{paint('○', DIM)} {chk.label} — {paint('not configured', DIM)}")
            print(f"  {chk.error}")
            if chk.hint:
                print(f"  {paint(chk.hint, DIM)}")
            continue

        if chk.ok:
            working.append(chk)
            print(f"\n{paint('✔', GREEN)} {chk.label} — "
                  f"{paint('working', GREEN)} ({chk.latency_ms} ms)")
            print(f"  model: {chk.model}")
            print(f"  {paint('replied: ' + chk.reply, DIM)}")
        else:
            print(f"\n{paint('✘', RED)} {chk.label} — {paint('failed', RED)}")
            print(f"  model: {chk.model}")
            print(f"  {chk.error}")
            if chk.hint:
                print(f"  {paint('→ ' + chk.hint, YELLOW)}")

    print("\n" + "=" * 62)
    if working:
        print(paint(f"{len(working)} provider(s) working: "
                    f"{', '.join(c.label for c in working)}", GREEN))
        chain = [n for n in fallback_order()
                 if any(c.name == n and c.ok for c in results)]
        if chain:
            print(f"Narratives will use: {' → '.join(chain)}")
        print(f"{paint('Routing:', DIM)} " + ", ".join(
            f"{task}={prov}" for task, prov in task_routing().items()
            if task != "default"))
    else:
        configured = [c for c in results if c.configured]
        if configured:
            print(paint("No provider answered. The app still works — every "
                        "figure and finding is computed by the analysis "
                        "engines, and only the prose would have come from a "
                        "model.", YELLOW))
        else:
            print(paint("No provider is configured. The app still works — "
                        "reports are built entirely by the analysis engines. "
                        "Add any one key to have a model phrase the prose.",
                        YELLOW))
    print()
    return 0 if working else 1


if __name__ == "__main__":
    raise SystemExit(main())
