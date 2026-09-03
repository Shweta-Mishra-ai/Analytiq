"""
tests/test_codebase_health.py — structural regression sweeps that catch
whole classes of silent failure, rather than one bug at a time.

Ported from dataforge-ai's equivalent suite. Analytiq had 54 silent
`except: pass`/`continue` swallows and 33 modules with no logger at all
when this was first run — every one of them a place where a genuine
failure vanished with no trace, which is exactly what "a feature doesn't
work but shows no error" looks like from the outside.

These tests scan the shipped source tree dynamically rather than using a
maintained file list, so a newly added module can't slip through simply
because nobody remembered to register it.
"""
from __future__ import annotations

import ast
import os
import re

import pytest

APP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")

_SILENT_BODIES = {"pass", "continue", "...", "break"}
_EXCEPT_HEADS = {"except Exception:", "except Exception as e:", "except:"}


def _iter_app_py_files():
    """Every shipped .py file under app/ (tests and caches excluded)."""
    skip_dirs = {"__pycache__", ".git", "node_modules", "venv"}
    for root, dirs, files in os.walk(APP_DIR):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


def _rel(path: str) -> str:
    return os.path.relpath(path, os.path.dirname(APP_DIR))


def _has_definitions(src: str) -> bool:
    """Skip trivial files (e.g. empty __init__.py) that legitimately need
    no logger because they contain no code that can fail."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return True  # a file that won't parse is a problem the syntax test reports
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for n in ast.walk(tree))


def test_all_app_modules_parse():
    """Every shipped module must at least be syntactically valid — guards
    against a bad automated refactor landing a broken file that only
    surfaces when some rarely-hit import path runs it."""
    broken = []
    for path in _iter_app_py_files():
        src = open(path, encoding="utf-8").read()
        try:
            ast.parse(src)
        except SyntaxError as e:
            broken.append(f"{_rel(path)}:{e.lineno}: {e.msg}")
    assert not broken, "Modules with syntax errors:\n" + "\n".join(broken)


def test_no_silent_except_app_wide():
    """No `except Exception:` / `except:` immediately followed by
    pass/continue/break/... anywhere under app/.

    A caught exception must leave *some* trace. Swallowing it silently is
    how a broken chart, a failed stat test, or a dead API call ends up
    looking identical to "no result available" — the single most common
    cause of 'the feature runs but nothing happens'.
    """
    violations = []
    for path in _iter_app_py_files():
        lines = open(path, encoding="utf-8").read().splitlines()
        for i, line in enumerate(lines):
            if line.strip() not in _EXCEPT_HEADS:
                continue
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt in _SILENT_BODIES:
                violations.append(f"{_rel(path)}:{i + 2}: {line.strip()} -> {nxt}")
    assert not violations, (
        f"{len(violations)} silent exception swallow(s) found — each must log "
        "(e.g. logger.debug(..., exc_info=True)) before pass/continue:\n"
        + "\n".join(violations)
    )


def _is_only_declarations(src: str) -> bool:
    """True for a module that declares shapes and does nothing else.

    A file of dataclasses has no failure path, so requiring a logger in
    it would be satisfying the letter of the rule below with a variable
    that can never be used. The test is narrow on purpose: no module-level
    functions and no exception handling anywhere. The moment such a module
    grows either, it needs a logger like everything else.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    if any(isinstance(n, (ast.Try, ast.ExceptHandler)) for n in ast.walk(tree)):
        return False
    return not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for n in tree.body)


def test_all_app_modules_with_code_have_a_logger():
    """Any module containing functions/classes must set up a module
    logger, so the log calls guarding its failure paths have somewhere
    to go. A module of pure declarations has no such paths."""
    missing = []
    for path in _iter_app_py_files():
        src = open(path, encoding="utf-8").read()
        if not _has_definitions(src):
            continue
        if _is_only_declarations(src):
            continue
        if "logger = logging.getLogger" not in src:
            missing.append(_rel(path))
    assert not missing, (
        "Modules with code but no `logger = logging.getLogger(__name__)`:\n"
        + "\n".join(missing)
    )


def test_no_print_statements_in_shipped_code():
    """`print()` in server code is a silent-in-production log — it goes to
    stdout unstructured and is invisible to any log level or handler.
    Engine/API/service code should use the module logger instead."""
    offenders = []
    for path in _iter_app_py_files():
        src = open(path, encoding="utf-8").read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                offenders.append(f"{_rel(path)}:{node.lineno}")
    assert not offenders, (
        "print() found in shipped code — use the module logger:\n"
        + "\n".join(offenders)
    )


# ── Engine output-quality regressions ────────────────────────────────────
# These assert that the analysis engines actually produce content, rather
# than degrading to an empty-but-successful response — the other way a
# feature silently "works" while telling the user nothing.

def test_story_engine_produces_findings(hr_df):
    from app.engines.story_engine import generate_story
    story = generate_story(hr_df)
    assert story.key_findings, "key_findings must never be empty"
    for f in story.key_findings:
        assert isinstance(f, str) and len(f) > 5, f"weak finding: {f!r}"


def test_story_engine_exec_summary_is_specific(hr_df):
    from app.engines.story_engine import generate_story
    story = generate_story(hr_df)
    assert len(story.executive_summary) > 30, "exec summary too short to be useful"
    assert re.search(r"\d", story.executive_summary), \
        "exec summary should cite at least one concrete number"


def test_insight_engine_returns_insights(hr_df):
    from app.engines.insight_engine import generate_insights
    insights = generate_insights(hr_df)
    assert insights, "generate_insights returned nothing for a rich dataset"
    for ins in insights:
        assert ins.get("title"), f"insight missing title: {ins}"
        assert ins.get("body"), f"insight missing body: {ins}"


def test_stats_engine_analyzes_every_numeric_measure(hr_df):
    """A numeric column that quietly falls out of the stats report is a
    silent failure: the section renders, just missing that column.

    Identifiers are the one exception, and they are named rather than
    dropped in silence — employee_id had been getting a mean, a skewness
    and a normality verdict of its own.
    """
    from app.engines.stats_engine import analyze
    stats = analyze(hr_df)
    numeric_cols = {c for c in hr_df.columns
                    if str(hr_df[c].dtype).startswith(("int", "float"))}
    accounted = set(stats.column_stats) | set(stats.identifier_cols)
    missed = numeric_cols - accounted
    assert not missed, f"stats engine silently skipped numeric columns: {missed}"
    assert "employee_id" in stats.identifier_cols, stats.identifier_cols
    assert "employee_id" not in stats.column_stats


@pytest.mark.parametrize("engine_import,fn_name", [
    ("app.engines.bi_engine", "run_bi"),
    ("app.engines.eda_engine", "run_eda"),
])
def test_heavy_engines_return_populated_report(engine_import, fn_name, hr_df):
    """A run that raises internally and returns an empty shell is a silent
    failure; assert the report carries actual content."""
    import importlib
    mod = importlib.import_module(engine_import)
    report = getattr(mod, fn_name)(hr_df)
    populated = [k for k, v in vars(report).items()
                 if not k.startswith("_") and v not in (None, [], {}, "")]
    assert populated, f"{fn_name} returned an entirely empty report"


def test_the_declarations_exemption_is_narrow():
    """The exemption above must not become a way to skip the logger rule.
    A module gains a failure path the moment it grows a function or an
    except, and must need a logger again at that point."""
    assert _is_only_declarations(
        "from dataclasses import dataclass\n@dataclass\nclass A:\n    x: int\n")
    assert not _is_only_declarations("def f():\n    return 1\n")
    assert not _is_only_declarations(
        "class A:\n    def m(self):\n        try:\n            pass\n"
        "        except Exception:\n            pass\n")
