"""Verify the AEGIS prompt templates still parse and that the new ``bright``
branch is actually selected.

The Evolver template is Jinja2. Adding an ``{% elif benchmark_context == "bright" %}``
branch is only safe if every conditional still balances, and the failure mode is
nasty: a broken template raises only when the meta-agent's stage is invoked,
i.e. mid-round, after budget has been spent. This test catches it for free.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from jinja2 import Template, TemplateSyntaxError  # noqa: E402

TEMPLATES = ROOT / "third_party" / "HarnessX" / "harnessx" / "aegis" / "templates"
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    names = sorted(p.name for p in TEMPLATES.glob("*.md"))
    print(f"templates found: {names}")

    print("\n1. every template parses")
    parsed: dict[str, Template] = {}
    for name in names:
        src = (TEMPLATES / name).read_text(encoding="utf-8")
        try:
            parsed[name] = Template(src)
            check(f"parses: {name}", True, f"{len(src)} chars")
        except TemplateSyntaxError as exc:
            check(f"parses: {name}", False, f"line {exc.lineno}: {exc.message}")

    print("\n2. evolver.md selects the bright branch")
    ev = parsed.get("evolver.md")
    if ev is None:
        check("evolver.md available", False)
    else:
        ctx = {
            "round": 1,
            "round_minus_1": 0,
            "ask_more_mode": False,
            "ask_more_candidate_id": "",
            "candidates_dir": "cands",
            "round_dir": "R1",
            "run_dir": "run",
            "benchmark_context": "bright",
        }
        out_bright = ev.render(**ctx)
        check(
            "bright marker present",
            "BRIGHT benchmark constraints" in out_bright,
            f"{len(out_bright)} chars",
        )
        check(
            "bright branch mentions the active-index accessor",
            "bright_harness.tools import get_active_index" in out_bright,
        )
        check(
            "bright branch forbids reading evaluation data",
            "Never touch evaluation data" in out_bright,
        )
        check("tau2 constraint NOT leaked into bright", "tau2 benchmark constraint" not in out_bright)
        check(
            "generic builtin tool list NOT leaked into bright",
            "builtin: [WebSearch, WebFetch, Browser, Read, Bash]" not in out_bright,
        )

        for other in ("", "tau2", "someotherbench"):
            try:
                out = ev.render(**{**ctx, "benchmark_context": other})
                leaked = "BRIGHT benchmark constraints" in out
                check(f"context {other!r} renders without the bright branch", not leaked, f"{len(out)} chars")
            except TemplateSyntaxError as exc:
                check(f"context {other!r} renders", False, f"line {exc.lineno}: {exc.message}")

        ctx_tau2 = {**ctx, "benchmark_context": "tau2"}
        out_tau2 = ev.render(**ctx_tau2)
        check(
            "tau2 branch still intact",
            "tau2 benchmark constraint" in out_tau2 and "BRIGHT benchmark constraints" not in out_tau2,
        )

    print("\n--- summary ---")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all template checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
