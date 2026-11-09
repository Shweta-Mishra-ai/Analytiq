#!/usr/bin/env python3
"""
Starts Analytiq's MCP server for the Claude Code plugin.

The plugin ships the source; its Python dependencies (pandas, scikit-learn,
ReportLab, ...) are installed once into a private virtual environment in the
plugin's data directory, which survives plugin updates, and reinstalled only
when backend/requirements.txt changes (XGBoost is left out; see SKIP). Standard library only, so it runs on
any Python 3.11+ before anything is installed.

  launch.py            install if needed, then serve over stdio (MCP)
  launch.py --prepare  start the install in the background and return at once
                       (run by the plugin's SessionStart hook, so the first
                       tool call does not wait minutes for pip)
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time

ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("CLAUDE_PLUGIN_DATA") or os.path.join(
    os.path.expanduser("~"), ".analytiq", "plugin")
BACKEND = os.path.join(ROOT, "backend")
REQUIREMENTS = os.path.join(BACKEND, "requirements.txt")
VENV = os.path.join(DATA, "venv")
STAMP = os.path.join(DATA, "requirements.sha256")
LOCK = os.path.join(DATA, "install.lock")
LOG = os.path.join(DATA, "install.log")
WAIT_FOR_INSTALL = 25          # seconds a starting server waits for a running install
PLUGIN_REQUIREMENTS = os.path.join(DATA, "requirements.plugin.txt")
# Left out of the plugin's environment: XGBoost and the NVIDIA library it
# pulls in on Linux are about 700 MB of a 1.5 GB install. The model
# comparison treats XGBoost as optional and runs the other three models.
SKIP = {"xgboost", "nvidia-nccl-cu12"}


def log(msg: str) -> None:
    print(f"[analytiq] {msg}", file=sys.stderr, flush=True)


def venv_python() -> str:
    sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return os.path.join(VENV, *sub)


def plugin_requirements() -> str:
    """The pinned requirements without the packages in SKIP."""
    keep = []
    with open(REQUIREMENTS) as f:
        for line in f:
            name = line.split("==")[0].split("[")[0].strip().lower()
            if line.strip() and not line.startswith((" ", "#")) and name in SKIP:
                continue
            keep.append(line)
    return "".join(keep)


def wanted() -> str:
    return hashlib.sha256(plugin_requirements().encode()).hexdigest()


def ready() -> bool:
    try:
        with open(STAMP) as f:
            return f.read().strip() == wanted() and os.path.exists(venv_python())
    except OSError:
        return False


def installing() -> bool:
    """A live lock: another process is installing (stale after an hour)."""
    try:
        return time.time() - os.path.getmtime(LOCK) < 3600
    except OSError:
        return False


def install() -> None:
    os.makedirs(DATA, exist_ok=True)
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))
    try:
        log(f"installing dependencies into {VENV} (first run takes a few minutes)")
        with open(LOG, "a") as out:
            if not os.path.exists(venv_python()):
                subprocess.run([sys.executable, "-m", "venv", VENV],
                               check=True, stdout=out, stderr=out)
            with open(PLUGIN_REQUIREMENTS, "w") as f:
                f.write(plugin_requirements())
            subprocess.run([venv_python(), "-m", "pip", "install", "--disable-pip-version-check",
                            "-q", "-r", PLUGIN_REQUIREMENTS], check=True, stdout=out, stderr=out)
        with open(STAMP, "w") as f:
            f.write(wanted())
        log("dependencies installed")
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


def prepare() -> None:
    """Begin the install detached from the hook, so session start is not held up."""
    if ready() or installing():
        return
    os.makedirs(DATA, exist_ok=True)
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
              "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008      # DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "--install"], **kwargs)


def serve() -> None:
    if not ready():
        deadline = time.time() + WAIT_FOR_INSTALL
        while installing() and time.time() < deadline:
            time.sleep(1)
        if installing():
            log("dependencies are still installing in the background; "
                f"progress in {LOG}. Reconnect with /mcp in a minute.")
            sys.exit(1)
        if not ready():
            install()
    env = dict(os.environ)
    env["PYTHONPATH"] = BACKEND + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("DATA_DIR", os.path.join(os.path.expanduser("~"), ".analytiq"))
    py = venv_python()
    if os.name == "nt":                           # no exec on Windows; relay the child
        sys.exit(subprocess.call([py, "-m", "app.mcp_server"], env=env, cwd=BACKEND))
    os.chdir(BACKEND)
    os.execve(py, [py, "-m", "app.mcp_server"], env)


def main() -> None:
    if sys.version_info < (3, 11):
        log(f"Python 3.11 or newer is required (found {sys.version.split()[0]}).")
        sys.exit(1)
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--prepare":
        prepare()
    elif arg == "--install":
        if not ready():
            install()
    else:
        serve()


if __name__ == "__main__":
    main()
