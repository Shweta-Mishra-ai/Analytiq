"""
The runtime image installs requirements.txt and nothing else, so what
that file says is what a deployment gets.

It had drifted both ways: pytest and an HTTP client the app never
imports shipped to production, and python-pptx — which the deck builder
imports at module scope — sat under a '── Testing ──' heading, one
tidy-up away from taking PowerPoint export out of the image.
"""
from __future__ import annotations

import ast
import os
import re
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_APP = os.path.join(_BACKEND, "app")


def _requirements(name: str) -> set:
    path = os.path.join(_BACKEND, name)
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line or line.startswith("-r"):
                continue
            name = re.split(r"[<>=\[]", line)[0].strip().lower()
            # PEP 503: "_" and "-" name the same distribution.
            out.add(name.replace("_", "-"))
    return out


# Import name → the distribution that provides it, for the ones where
# they differ. Only packages the app actually imports belong here.
_DISTRIBUTION = {
    "pptx": "python-pptx", "docx": "python-docx", "sklearn": "scikit-learn",
    "PIL": "Pillow", "yaml": "PyYAML", "dateutil": "python-dateutil",
    "google": "google-genai", "multipart": "python-multipart",
    "fitz": "PyMuPDF", "cv2": "opencv-python", "bs4": "beautifulsoup4",
    # FastAPI is built on Starlette and pins a compatible range of it.
    # Pinning it again here would let the two disagree.
    "starlette": "fastapi",
}

# Imported behind a try/except or inside a function, on purpose: the app
# reports a missing one as a message rather than failing to start.
_OPTIONAL = {
    "boto3", "torch", "sentence_transformers", "psycopg", "psycopg2",
    "pymysql", "pyodbc", "snowflake", "rank_bm25", "faiss", "transformers",
    "duckdb", "openpyxl", "httpx", "redis",
}


def _top_level_imports() -> set:
    """Every module imported at the top level of a file under app/."""
    found = set()
    for root, _dirs, files in os.walk(_APP):
        if "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                try:
                    tree = ast.parse(fh.read(), filename=path)
                except SyntaxError:            # pragma: no cover
                    continue
            for node in tree.body:             # top level only
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        found.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        found.add(node.module.split(".")[0])
    return found


def test_everything_the_app_imports_at_import_time_is_a_runtime_dependency():
    """A module-scope import that is missing from requirements.txt does
    not degrade — the process fails to start."""
    runtime = _requirements("requirements.txt")
    missing = []

    for mod in sorted(_top_level_imports()):
        if mod in ("app", "__future__") or mod in _OPTIONAL:
            continue
        if mod in sys.stdlib_module_names:
            continue
        dist = _DISTRIBUTION.get(mod, mod).lower().replace("_", "-")
        if dist not in runtime:
            missing.append("{} (provided by {})".format(mod, dist))

    assert not missing, (
        "imported by app/ at module scope but absent from "
        "requirements.txt: " + ", ".join(missing))


@pytest.mark.parametrize("package", ["pytest", "httpx", "openpyxl"])
def test_test_only_packages_do_not_ship_in_the_runtime_image(package):
    """The Dockerfile installs requirements.txt into the image. Anything
    here is shipped to every deployment."""
    assert package not in _requirements("requirements.txt"), (
        "{} is only used by the tests — it belongs in "
        "requirements-dev.txt".format(package))
    assert package in _requirements("requirements-dev.txt")


def test_the_dev_file_builds_on_the_runtime_one():
    """Otherwise the two drift and CI stops testing what ships."""
    with open(os.path.join(_BACKEND, "requirements-dev.txt"),
              encoding="utf-8") as fh:
        assert "-r requirements.txt" in fh.read()
