"""Isolation tests for the per-session retrieval policy.

The concurrency change is safe only if two rollouts running at the same instant
cannot see each other's policy or share one ``max_retrieve_calls`` budget. Before
this change both lived in module globals, so query #2 would be told its retrieval
budget was already spent.

These tests exercise the exact shape the driver uses (``asyncio.gather`` with
interleaved awaits). Costs nothing: no model, no corpus, no network.

    python src/bright_harness/test_concurrency.py
"""
from __future__ import annotations

import asyncio
import contextvars
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from bright_harness import policy  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


async def _worker(sid: str, k: int, calls: int) -> dict:
    """One simulated rollout: own policy, own retrieval budget."""
    policy.begin_session(sid, retrieve_k=k, max_retrieve_calls=1, snippet_chars=100 + k)
    seen: list[int] = []
    try:
        for _ in range(calls):
            # Yield so the gather actually interleaves the workers.
            await asyncio.sleep(0.001)
            seen.append(policy.note_retrieve_call())
        pol = policy.get_policy()
        return {"sid": sid, "k": pol["retrieve_k"], "snippet": pol["snippet_chars"], "calls": seen}
    finally:
        policy.end_session()


async def main() -> int:
    print("1. concurrent sessions keep separate policies and budgets")
    workers = [("s-A", 10), ("s-B", 50), ("s-C", 100), ("s-D", 7)]
    results = await asyncio.gather(*[_worker(sid, k, 3) for sid, k in workers])
    by_sid = {r["sid"]: r for r in results}
    for sid, k in workers:
        r = by_sid[sid]
        check(f"{sid}: retrieve_k == {k}", r["k"] == k, str(r["k"]))
        check(f"{sid}: snippet_chars == {100 + k}", r["snippet"] == 100 + k, str(r["snippet"]))
        # Each session's own counter must start at 1 regardless of the others.
        check(f"{sid}: own counter is 1,2,3", r["calls"] == [1, 2, 3], str(r["calls"]))

    print("\n2. sessions are released (no leak after gather)")
    check("live_session_count() == 0", policy.live_session_count() == 0,
          str(policy.live_session_count()))

    print("\n3. counter really is per-session, not per-process")
    policy.begin_session("x1", retrieve_k=10, max_retrieve_calls=1)
    first = policy.note_retrieve_call()
    policy.end_session()
    policy.begin_session("x2", retrieve_k=10, max_retrieve_calls=1)
    second = policy.note_retrieve_call()
    policy.end_session()
    check("second session's first call is 1 (not 2)", first == 1 and second == 1, f"{first},{second}")

    print("\n4. a broken ContextVar fails loudly instead of mixing policies")
    # Two sessions live, but we read from a context that never opened one -- which
    # is what a tool sees if propagation breaks. It must raise, not silently return
    # another query's policy.
    policy.begin_session("live-1", retrieve_k=10)
    policy.begin_session("live-2", retrieve_k=99)
    raised = False
    try:
        contextvars.Context().run(policy.get_policy)  # fresh context, both sessions live
    except RuntimeError:
        raised = True
    check("unbound read while 2 sessions live raises RuntimeError", raised)
    policy.end_session("live-1")
    policy.end_session("live-2")
    check("live_session_count() back to 0", policy.live_session_count() == 0,
          str(policy.live_session_count()))

    print("\n5. serial/legacy path still works (no session opened)")
    contextvars.Context().run(lambda: policy.set_policy(retrieve_k=42))
    got = contextvars.Context().run(policy.get_policy)
    check("default state accepts set_policy", got["retrieve_k"] == 42, str(got["retrieve_k"]))
    contextvars.Context().run(lambda: policy.set_policy(**policy.DEFAULTS))

    print("\n--- summary ---")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all concurrency-isolation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
