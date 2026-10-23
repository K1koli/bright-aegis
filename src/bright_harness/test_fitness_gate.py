"""Tests for the Tier-2 graded fitness gate.

The gate exists because AEGIS's stock no-regression check (the counterfactual
gate) compares ``final_output`` / ``exit_reason`` by exact equality on
previously-passing tasks. On a graded metric a query can fall from 0.62 to 0.21
while still "passing" a threshold, and the aggregate can rise at the same time.
``test_aggregate_up_but_one_query_collapses`` is the case that motivates the
whole change; everything else guards the skip paths and fail-closed behaviour.

Runs with no model and no network.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from harnessx.aegis.gates.fitness import (  # noqa: E402
    check_fitness_noregress,
    get_fitness_context,
    set_fitness_context,
)
from harnessx.aegis.stages.commit import _run_all_gates  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


BASE = {"q1": 0.60, "q2": 0.40, "q3": 0.20, "q4": 0.10}
PROBES = list(BASE)


def ctx(scores, *, delta=0.05, margin=0.0, fn=None, **extra):
    payload = {
        "probe_qids": PROBES,
        "incumbent_scores": dict(BASE),
        "fitness_fn": fn if fn is not None else (lambda path, qids: dict(scores)),
        "delta": delta,
        "margin": margin,
    }
    payload.update(extra)
    return payload


async def main() -> int:
    print("1. skip paths (must not block a stock binary benchmark)")
    for label, c in [
        ("no context", None),
        ("no probe queries", {**ctx(BASE), "probe_qids": []}),
        ("no fitness_fn", {**ctx(BASE), "fitness_fn": None}),
        ("no incumbent", {**ctx(BASE), "incumbent_scores": {}}),
    ]:
        r = await check_fitness_noregress("x.yaml", context=c)
        check(f"skip: {label}", r.ok and r.reason.startswith("skipped"), r.reason)

    print("\n2. accept / reject")
    r = await check_fitness_noregress("x.yaml", context=ctx(BASE))
    check("flat candidate accepted", r.ok, r.reason)

    r = await check_fitness_noregress("x.yaml", context=ctx({**BASE, "q1": 0.90, "q3": 0.30}))
    check("strict improvement accepted", r.ok, r.reason)

    # The motivating case: mean goes UP (0.325 -> 0.345) while q1 collapses.
    up_but_collapse = {"q1": 0.05, "q2": 0.60, "q3": 0.34, "q4": 0.39}
    mean_inc = sum(BASE.values()) / 4
    mean_cand = sum(up_but_collapse.values()) / 4
    r = await check_fitness_noregress("x.yaml", context=ctx(up_but_collapse))
    check(
        "aggregate up but one query collapses -> REJECTED",
        (not r.ok) and mean_cand > mean_inc,
        f"mean {mean_inc:.3f} -> {mean_cand:.3f}; {r.reason}",
    )

    r = await check_fitness_noregress("x.yaml", context=ctx({**BASE, "q1": 0.10, "q2": 0.10}))
    check("aggregate regression rejected", not r.ok, r.reason)

    just_inside = {**BASE, "q1": BASE["q1"] - 0.05}
    r = await check_fitness_noregress("x.yaml", context=ctx(just_inside, margin=-0.02))
    check("drop exactly == delta is tolerated (boundary)", r.ok, r.reason)

    r = await check_fitness_noregress("x.yaml", context=ctx({**BASE, "q1": BASE["q1"] - 0.06}))
    check("drop just beyond delta rejected", not r.ok, r.reason)

    print("\n3. margin semantics")
    r = await check_fitness_noregress("x.yaml", context=ctx(BASE, margin=0.01))
    check("flat candidate rejected when margin>0", not r.ok, r.reason)
    r = await check_fitness_noregress("x.yaml", context=ctx({**BASE, "q1": 0.70}, margin=0.02))
    check("improvement above margin accepted", r.ok, r.reason)

    print("\n4. fail-closed and async")
    r = await check_fitness_noregress("x.yaml", context=ctx(BASE, fn=lambda p, q: 1 / 0))
    check("fitness_fn raising -> rejected (fail-closed)", not r.ok, r.reason[:80])

    async def _async_fn(path, qids):
        return {k: v + 0.1 for k, v in BASE.items()}

    r = await check_fitness_noregress("x.yaml", context=ctx(BASE, fn=_async_fn))
    check("async fitness_fn supported", r.ok, r.reason)

    r = await check_fitness_noregress("x.yaml", context=ctx({**BASE, "q3": 0.2}, fn=lambda p, q: {"q1": 0.9}))
    check(
        "missing probe scores -> rejected",
        (not r.ok) and ("no score for" in r.reason),
        r.reason[:90],
    )

    print("\n5. wiring into Stage 4")
    tmp = ROOT / ".tmp" / "gate_test"
    tmp.mkdir(parents=True, exist_ok=True)
    manifest = tmp / "C-R1-01.md"
    manifest.write_text(
        "---\n"
        "candidate_id: C-R1-01\n"
        "bucket: prompt\n"
        "file_changes: []\n"
        "predicted_impact: {}\n"
        "---\n\n## Failure Evidence\nnothing\n",
        encoding="utf-8",
    )
    set_fitness_context(ctx({**BASE, "q1": 0.99}))
    check("module-level context is readable", get_fitness_context() is not None)
    results = await _run_all_gates(
        "C-R1-01", manifest, tmp / "nonexistent_config.yaml", refuted_signatures=set()
    )
    check("gate results include 'fitness'", "fitness" in results, str(list(results)))
    check(
        "fitness skipped when a cheap gate failed",
        not results["fitness"].ok and results["fitness"].reason.startswith("skipped"),
        results["fitness"].reason,
    )
    set_fitness_context(None)

    print("\n--- summary ---")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all fitness-gate checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
