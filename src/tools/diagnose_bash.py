"""Diagnose why the meta-agent's Bash tool failed inside the smoke run.

The Evolver reported that *every* Bash invocation failed with
``[WinError 5] 拒绝访问`` and concluded it could not run the mandatory
build->verify loop for code candidates, so it shipped prompt-only. This script
exercises the same code path the meta-agent uses (``bash_tool`` through the tool
registry) and reports the real error, so we can tell apart:

* the DSH file/process sandbox around this session,
* HarnessX's LocalSandbox wrapper,
* a plain "no ``bash`` on PATH" problem on Windows.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from harnessx.tools.builtin.bash import bash_tool  # noqa: E402
from harnessx.tools.inmemory import InMemoryToolRegistry  # noqa: E402

COMMANDS = [
    "echo hello",
    "python --version",
    "dir",
]


def show(label: str, ok: bool | None, detail: str = "") -> None:
    mark = "OK " if ok else ("FAIL" if ok is False else "?? ")
    print(f"  [{mark}] {label}{(' -- ' + detail) if detail else ''}")


async def main() -> int:
    print("=== 1. what's on PATH ===")
    for exe in ("bash", "sh", "cmd", "powershell", "pwsh"):
        show(exe, shutil.which(exe) is not None, shutil.which(exe) or "not found")
    print(f"  OS={os.name}  cwd={os.getcwd()}")
    print(f"  DSH sandbox env: {[k for k in os.environ if k.startswith('DSH_')]}")

    print("\n=== 2. plain subprocess (control) ===")
    for cmd in (["cmd", "/c", "echo hello"], [sys.executable, "--version"]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            show(" ".join(cmd), p.returncode == 0, f"rc={p.returncode} out={p.stdout.strip()[:60]!r} err={p.stderr.strip()[:60]!r}")
        except Exception as exc:  # noqa: BLE001
            show(" ".join(cmd), False, f"{type(exc).__name__}: {exc}")

    print("\n=== 3. HarnessX bash_tool via the registry (what the meta-agent calls) ===")
    reg = InMemoryToolRegistry()
    reg.register(bash_tool)
    for cmd in COMMANDS:
        try:
            res = await reg.execute("Bash", {"command": cmd})
            err = res.error
            out = (res.output or "").strip()
            show(repr(cmd), not err, f"error={err!r} output={out[:120]!r}")
        except Exception as exc:  # noqa: BLE001
            show(repr(cmd), False, f"{type(exc).__name__}: {exc}")

    print("\n=== 4. bash_tool source: which shell does it invoke? ===")
    import inspect

    src = inspect.getsource(bash_tool.fn)
    for line in src.splitlines():
        s = line.strip()
        if any(k in s for k in ("subprocess", "shell=", "executable", "cmd.exe", "powershell", "bash", "Popen", "create_subprocess")):
            print(f"    {s[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
