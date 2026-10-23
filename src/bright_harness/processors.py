"""Harness processors for BRIGHT.

``RetrievalPolicyProcessor`` exists to give AEGIS a real ``config`` bucket. It
carries the retrieval knobs as instance kwargs, so a candidate can retune the
harness by editing its applied YAML — no new code, no gateway through the prompt.

Before this existed, the AEGIS Planner measured H_0 and concluded it was "a
zero-degrees-of-freedom wrapper around BM25 ... the agent has ~zero freedom", and
the Evolver could only propose prompt edits.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from harnessx.core.events import TaskStartEvent
from harnessx.core.processor import MultiHookProcessor

from bright_harness import policy

__all__ = ["RetrievalPolicyProcessor"]


class RetrievalPolicyProcessor(MultiHookProcessor):
    """Install the retrieval policy for each task.

    Every kwarg is a legitimate ``config``-bucket edit:

    * ``retrieve_k``          — passages returned per Retrieve call (1..100)
    * ``snippet_chars``       — how much of each passage the agent sees
    * ``max_retrieve_calls``  — retrieval budget per query, enforced by the tool
    * ``max_steps``           — harness steps per query (the driver reads this)
    * ``dedupe_by_document``  — keep only the best chunk per source document
    """

    _singleton_group = "bright.retrieval_policy"
    _order = 1

    def __init__(
        self,
        retrieve_k: int = policy.DEFAULTS["retrieve_k"],
        snippet_chars: int = policy.DEFAULTS["snippet_chars"],
        max_retrieve_calls: int = policy.DEFAULTS["max_retrieve_calls"],
        max_steps: int = policy.DEFAULTS["max_steps"],
        dedupe_by_document: bool = policy.DEFAULTS["dedupe_by_document"],
    ) -> None:
        self.retrieve_k = int(retrieve_k)
        self.snippet_chars = int(snippet_chars)
        self.max_retrieve_calls = int(max_retrieve_calls)
        self.max_steps = int(max_steps)
        self.dedupe_by_document = bool(dedupe_by_document)

    def as_policy(self) -> dict[str, Any]:
        return {
            "retrieve_k": self.retrieve_k,
            "snippet_chars": self.snippet_chars,
            "max_retrieve_calls": self.max_retrieve_calls,
            "max_steps": self.max_steps,
            "dedupe_by_document": self.dedupe_by_document,
        }

    async def on_task_start(self, event: TaskStartEvent) -> AsyncIterator[TaskStartEvent]:
        # Bound to this task's session id, so concurrent rollouts cannot see each
        # other's policy or share one retrieval budget. Re-installed per task so a
        # candidate's policy can never leak into the next run.
        policy.begin_session(getattr(event, "session_id", "") or "", **self.as_policy())
        yield event

    async def on_task_end(self, event) -> AsyncIterator:
        # Release the session so the concurrency guard in policy._state() can tell
        # "one live session" (fine) from "several live sessions" (ContextVar broken).
        policy.end_session()
        yield event
