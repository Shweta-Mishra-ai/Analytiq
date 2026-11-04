"""
The Claude Code plugin: its manifests point at files that exist, and the
launcher installs the pinned requirements minus the optional XGBoost.
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _json(*parts):
    with open(os.path.join(REPO, *parts)) as f:
        return json.load(f)


def _launcher():
    spec = importlib.util.spec_from_file_location(
        "analytiq_launch", os.path.join(REPO, "plugin", "launch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve(path: str) -> str:
    return os.path.join(REPO, path.replace("${CLAUDE_PLUGIN_ROOT}/", "").lstrip("./"))


def test_the_marketplace_lists_the_plugin_by_its_manifest_name():
    market = _json(".claude-plugin", "marketplace.json")
    plugin = _json(".claude-plugin", "plugin.json")
    entry, = market["plugins"]
    assert entry["name"] == plugin["name"] == "analytiq"
    assert entry["source"] == "./" and market["owner"]["name"]


def test_every_path_the_manifest_names_exists():
    plugin = _json(".claude-plugin", "plugin.json")
    server = plugin["mcpServers"]["analytiq"]
    assert os.path.isfile(_resolve(server["args"][0]))
    assert os.path.isfile(_resolve(plugin["hooks"]))
    for d in plugin["skills"]:
        assert os.path.isfile(os.path.join(_resolve(d), "analyze-data", "SKILL.md"))
    hook = _json("plugin", "hooks.json")["hooks"]["SessionStart"][0]["hooks"][0]
    assert hook["args"][-1] == "--prepare" and os.path.isfile(_resolve(hook["args"][0]))


def test_the_plugin_installs_the_pinned_requirements_without_xgboost():
    launch = _launcher()
    text = launch.plugin_requirements()
    names = {line.split("==")[0].lower() for line in text.splitlines()
             if line and not line.startswith((" ", "#"))}
    assert "xgboost" not in names and "nvidia-nccl-cu12" not in names
    assert {"pandas", "scikit-learn", "reportlab", "mcp"} <= names
    # Every other pin is kept exactly.
    with open(launch.REQUIREMENTS) as f:
        full = {line.split("==")[0].lower() for line in f
                if line.strip() and not line.startswith((" ", "#"))}
    assert full - names == {"xgboost", "nvidia-nccl-cu12"}


def test_a_changed_requirements_file_triggers_a_reinstall(tmp_path, monkeypatch):
    launch = _launcher()
    monkeypatch.setattr(launch, "STAMP", str(tmp_path / "stamp"))
    monkeypatch.setattr(launch, "venv_python", lambda: str(tmp_path / "python"))
    (tmp_path / "python").write_text("")
    (tmp_path / "stamp").write_text(launch.wanted())
    assert launch.ready()
    (tmp_path / "stamp").write_text("an older hash")
    assert not launch.ready()


def test_the_launcher_refuses_an_old_python(monkeypatch):
    launch = _launcher()
    monkeypatch.setattr(launch.sys, "version_info", (3, 10, 0))
    monkeypatch.setattr(launch.sys, "argv", ["launch.py"])
    with pytest.raises(SystemExit):
        launch.main()
