"""Retrieval policy — the harness's tunable knobs.

H_0 originally had almost no edit surface. The AEGIS Planner read the source and
called it "a zero-degrees-of-freedom wrapper around BM25", correctly noting that:

* ``k`` was an argument the *model* passed to the tool, so only the prompt could
  influence it (``H0_RETRIEVE_K`` existed but was dead code);
* ``max_steps`` was hard-coded in the round driver, not read from the config;
* the thinking switch lived in an environment variable, outside any candidate's
  scratch directory.

That left only the ``prompt`` bucket reachable, which is not enough to test a
method whose whole claim is that harness *composition* evolves.

These values are now instance kwargs on a real processor
(:class:`bright_harness.processors.RetrievalPolicyProcessor`). That is what makes
them ``config``-bucket editable: ``harnessx/aegis/compose.py`` applies a config
edit by diffing kwargs on ``_target_``-matched processor entries, so a candidate
can change ``retrieve_k`` from 10 to 50 in its applied YAML with no new code.

**State is per session, keyed by the ContextVar set in ``on_task_start``.** That
is what makes concurrent rollouts safe: an earlier version kept one process-global
policy plus one global retrieval counter, so N concurrent queries would overwrite
each other's policy and share a single ``max_retrieve_calls`` budget -- the second
query to run would be told its retrieval budget was already exhausted.

The corpus index still needs its module-global fallback (see ``tools.py``), but
the policy does not: it is bound by a processor that runs *inside* the harness
run loop, and HarnessX propagates ContextVars from there to tools (its own
``get_current_sandbox()`` relies on exactly that).
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

__all__ = [
    "DEFAULTS",
    "set_policy",
    "get_policy",
    "reset_episode",
    "note_retrieve_call",
    "document_key",
    "begin_session",
    "end_session",
    "current_session",
    "live_session_count",
]

#: Default policy. These mirror H_0's original hard-coded behaviour so the
#: baseline is unchanged by introducing the knobs.
DEFAULTS: dict[str, Any] = {
    "retrieve_k": 10,  # passages returned per Retrieve call (1..100)
    "snippet_chars": 600,  # how much of each passage the agent sees
    "max_retrieve_calls": 1,  # retrieval budget per query, enforced by the tool
    "max_steps": 4,  # harness steps per query (read by the driver for BaseTask)
    "dedupe_by_document": False,  # keep only the best chunk per source document
}

_SESSION_CTX: ContextVar[str | None] = ContextVar("bright_policy_session", default=None)

#: session_id -> {"policy": {...}, "calls": int}. Small and bounded by the
#: number of (round, query, rollout) triples actually executed.
_SESSIONS: dict[str, dict[str, Any]] = {}

#: session_id -> number of task_start hooks still open. Used only to *detect* a
#: broken ContextVar while concurrent runs are live, so a silent mixing of two
#: queries' policies becomes a loud error instead.
_LIVE: dict[str, int] = {}

#: Serial / legacy fallback (unit tests, anything that never opened a session).
_DEFAULT: dict[str, Any] = {"policy": dict(DEFAULTS), "calls": 0}
_WARNED_FALLBACK = {"done": False}


def _merge(**kwargs: Any) -> dict[str, Any]:
    merged = dict(DEFAULTS)
    for key, value in kwargs.items():
        if key in DEFAULTS and value is not None:
            merged[key] = value
    return merged


def begin_session(session_id: str | None, **kwargs: Any) -> dict[str, Any]:
    """Open a policy session bound to ``session_id`` in the current context."""
    sid = session_id or f"anonymous-{len(_SESSIONS)}"
    state = {"policy": _merge(**kwargs), "calls": 0}
    _SESSIONS[sid] = state
    _LIVE[sid] = _LIVE.get(sid, 0) + 1
    _SESSION_CTX.set(sid)
    return state


def end_session(session_id: str | None = None) -> None:
    """Close the current context's session (idempotent)."""
    sid = session_id or _SESSION_CTX.get()
    if sid:
        depth = _LIVE.get(sid, 0) - 1
        if depth > 0:
            _LIVE[sid] = depth
        else:
            _LIVE.pop(sid, None)
    _SESSION_CTX.set(None)


def current_session() -> str | None:
    return _SESSION_CTX.get()


def live_session_count() -> int:
    """Number of sessions whose task_start hook has not yet closed."""
    return len(_LIVE)


def _state() -> dict[str, Any]:
    sid = _SESSION_CTX.get()
    if sid is not None:
        state = _SESSIONS.get(sid)
        if state is not None:
            return state
    if len(_LIVE) > 1:
        # Concurrency is on and the ContextVar did not reach the tool. Guessing
        # here would silently give this query another query's policy and share
        # one retrieval budget, so fail loudly at the first offending call.
        raise RuntimeError(
            "bright policy: no session bound in this context while "
            f"{len(_LIVE)} sessions are live -- the policy ContextVar did not "
            "reach the tool. Refusing to guess: concurrent runs would silently "
            "share one policy and one max_retrieve_calls budget."
        )
    if not _WARNED_FALLBACK["done"]:
        _WARNED_FALLBACK["done"] = True
        print(
            "[bright policy] no session bound; using the default policy state "
            "(serial/legacy path)",
            flush=True,
        )
    return _DEFAULT


def set_policy(**kwargs: Any) -> dict[str, Any]:
    """Replace the active policy, filling unspecified keys from DEFAULTS."""
    state = _state()
    state["policy"] = _merge(**kwargs)
    return dict(state["policy"])


def get_policy() -> dict[str, Any]:
    return dict(_state()["policy"])


def reset_episode() -> None:
    """Zero the per-query retrieval counter for the active session."""
    _state()["calls"] = 0


def note_retrieve_call() -> int:
    """Count a Retrieve call and return the new count (1-based)."""
    state = _state()
    state["calls"] += 1
    return state["calls"]


def document_key(doc_id: str) -> str:
    """Source-document key for a chunk id.

    BRIGHT chunk ids look like ``<family>/<document>_<n>.txt``; collapsing the
    trailing chunk index groups chunks that came from the same document. Passages
    from different domains use the same shape, so this stays domain-agnostic.
    """
    head, sep, tail = doc_id.rpartition("_")
    if sep and tail.endswith(".txt") and head:
        return head
    return doc_id
