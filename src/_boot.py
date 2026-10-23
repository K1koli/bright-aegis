"""Self-bootstrapping for scripts in this repo.

Adds the workspace's package directories to ``sys.path`` so scripts never depend
on the shell environment (PYTHONPATH) -- this workspace path contains Chinese
characters and the host's Windows TLS stack is broken, so keeping path handling
inside Python is the only reliable option.

Import and call ``setup()`` before importing anything else from the project.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Order matters: later inserts land earlier in sys.path.
_SEARCH = (
    ".deps",  # third-party wheels installed by src/tools/install_wheel.py
    "src/.pydeps",  # our own small deps (pyarrow, packaging)
    "src",  # first-party packages (bright.*)
    "third_party/HarnessX",  # the HarnessX checkout (harnessx.*, benchmarks.*, recipe.*)
)

ENV_FILE = ROOT / ".env.local"


def load_env(path: Path | None = None) -> dict[str, str]:
    """Load ``.env.local`` into ``os.environ`` (real environment wins).

    Credentials must survive across separate shell invocations -- each tool call
    here is a fresh process with no inherited environment -- so they live in a
    file rather than in the shell.
    """
    p = Path(path) if path else ENV_FILE
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def quiet_logging(level: str = "WARNING") -> None:
    """Silence HarnessX's per-step INFO logging.

    The run loop emits ~10 loguru lines per model/tool step, which buries the
    driver's own round-by-round output (a 10-query smoke run produced >26 KB of
    it). Failures still surface at WARNING and above.
    """
    try:
        from loguru import logger as _logger

        _logger.remove()
        _logger.add(sys.stderr, level=level)
    except Exception:  # noqa: BLE001
        pass


def _bash_usable(path: str, timeout: float = 20.0) -> bool:
    """True if this bash can actually start and run a trivial command here."""
    import subprocess

    try:
        p = subprocess.run(
            [path, "-c", "echo harnessx-bash-probe"],
            capture_output=True,
            timeout=timeout,
        )
        return p.returncode == 0 and b"harnessx-bash-probe" in p.stdout
    except Exception:  # noqa: BLE001
        return False


def setup(quiet: bool = True) -> Path:
    # Force UTF-8 on stdout/stderr. The workspace path contains Chinese characters
    # while Windows Python defaults to cp1252, so any print() that echoes a path
    # raises UnicodeEncodeError -- that killed the final report of a 25-minute run
    # after the experiment itself had completed successfully.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    # HarnessX resolves its home as $HARNESSX_HOME or ~/.harnessx
    # (harnessx/home.py). ~/.harnessx is outside the workspace and therefore
    # denied by the file sandbox, which makes every harness.run() fail with
    # PermissionError. Point it inside the workspace instead.
    os.environ.setdefault("HARNESSX_HOME", str(ROOT / ".harnessx"))
    # HarnessX's Bash tool defaults its work dir to the literal "/tmp/harnessx-work",
    # which on Windows is a drive-relative "\tmp\..." that needs admin rights to
    # create -- so EVERY meta-agent Bash call died with WinError 5 before running
    # anything. That silently disabled the tools/processor evolution buckets (the
    # Evolver could not run its mandatory build->verify loop for new code and fell
    # back to prompt-only candidates). Point it inside the workspace instead.
    os.environ.setdefault("HARNESSX_WORK_DIR", str(ROOT / ".tmp" / "hx_bash"))
    # subprocess would otherwise use cmd.exe, which cannot run the POSIX syntax the
    # AEGIS templates assume. Prefer a real bash -- but only one that actually
    # works here: MSYS2/Cygwin bash needs a signal pipe (CreateNamedPipe), which
    # some sandboxes forbid, and a bash that dies on startup is worse than none.
    # Fall back to the platform default when the probe fails.
    if not os.environ.get("HARNESSX_BASH"):
        for cand in (
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        ):
            if cand.exists() and _bash_usable(str(cand)):
                os.environ["HARNESSX_BASH"] = str(cand)
                break
    os.environ.setdefault("HARNESSX_SHELL_NOTE", os.environ.get("HARNESSX_BASH") or "platform-default (cmd.exe on Windows)")
    for rel in _SEARCH:
        p = ROOT / rel
        if p.exists():
            sp = str(p)
            if sp in sys.path:
                sys.path.remove(sp)
            sys.path.insert(0, sp)
    load_env()
    if quiet:
        quiet_logging()
    return ROOT
