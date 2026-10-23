"""Retrieval tools exposed to the task agent.

The active BM25 index is carried in a :class:`~contextvars.ContextVar` rather
than baked into the tool, mirroring HarnessX's own ``get_current_sandbox()``
pattern. That keeps the tool a normal module-level ``Tool`` object -- so it can
be referenced from a ``HarnessConfig`` YAML the way ``build_gaia_tools()`` tools
are -- while still being bound to one domain's corpus per run.

Evolved candidates may add their own tool files (the ``tools`` bucket); those
live in the round's scratch dir and are referenced as
``file://<scratch>/<name>.py::<symbol>``.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from harnessx.tools.base import tool

__all__ = ["set_active_index", "get_active_index", "Retrieve", "active_index"]

# The active corpus is held in BOTH a ContextVar and a plain module global.
#
# ContextVar alone is not enough: HarnessX's run loop executes tools in a
# different asyncio task, and a ContextVar set by the caller is not visible
# there, so every Retrieve call failed with "no active BRIGHT index" and the
# task agent obediently answered `[]` (measured: 50/50 queries scored 0).
# The module global covers that boundary; the ContextVar keeps the
# context-manager API honest for same-task callers.
_ACTIVE_INDEX: ContextVar[Any] = ContextVar("bright_active_index", default=None)
_ACTIVE_INDEX_GLOBAL: dict[str, Any] = {"index": None}

# The agent sees a bounded prefix of each passage. This is part of H_0's design:
# a small context window is what makes reranking/decomposition worth evolving.
SNIPPET_CHARS = 600


def set_active_index(index: Any):
    _ACTIVE_INDEX_GLOBAL["index"] = index
    return _ACTIVE_INDEX.set(index)


def get_active_index() -> Any:
    index = _ACTIVE_INDEX.get()
    if index is None:
        index = _ACTIVE_INDEX_GLOBAL["index"]
    if index is None:
        raise RuntimeError("no active BRIGHT index; call set_active_index() before running the harness")
    return index


class active_index:
    """Context manager binding a BM25 index for the duration of a run."""

    def __init__(self, index: Any):
        self.index = index
        self._token = None

    def __enter__(self):
        self._token = _ACTIVE_INDEX.set(self.index)
        _ACTIVE_INDEX_GLOBAL["index"] = self.index
        return self.index

    def __exit__(self, *exc):
        _ACTIVE_INDEX_GLOBAL["index"] = None
        if self._token is not None:
            _ACTIVE_INDEX.reset(self._token)
        return False


_RETRIEVE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query. May be the user's request verbatim or a reformulation.",
        },
        "k": {
            "type": "integer",
            "description": "How many passages to return (default 10, max 100).",
        },
    },
    "required": ["query"],
}


@tool(
    name="Retrieve",
    description=(
        "Search the domain corpus and return the top-ranked passages. Returns a ranked "
        "list where each entry is the passage id followed by a truncated excerpt of its "
        "text. Passages are ranked by the corpus's relevance model; lower rank = better."
    ),
    input_schema=_RETRIEVE_SCHEMA,
)
async def Retrieve(query: str, k: int = 10) -> str:
    from bright_harness import policy

    index = get_active_index()
    pol = policy.get_policy()

    calls = policy.note_retrieve_call()
    budget = int(pol["max_retrieve_calls"])
    if calls > budget:
        return (
            f"Retrieval budget exhausted ({budget} of {budget} calls already used). "
            "Do not retrieve again: publish your ranking now, using only the passages "
            "you have already received."
        )

    # The harness policy, not the tool argument, decides how many passages come
    # back: `retrieve_k` is a config-bucket knob, and the model's requested `k` is
    # only a request capped by it. (Before this, the model chose k freely and the
    # harness had no say -- which the AEGIS Planner flagged as a dead knob.)
    want = max(1, min(int(k or 10), int(pol["retrieve_k"]), 100))
    ranked = index.search(query, k=want)
    if not ranked:
        return "No passages matched the query."

    if pol["dedupe_by_document"]:
        seen: set[str] = set()
        kept: list[tuple[str, float]] = []
        for doc_id, score in ranked:
            key = policy.document_key(doc_id)
            if key in seen:
                continue
            seen.add(key)
            kept.append((doc_id, score))
        ranked = kept

    snippet_len = int(pol["snippet_chars"])
    lines: list[str] = []
    for rank, (doc_id, _score) in enumerate(ranked, start=1):
        snippet = index.text_of(doc_id)[:snippet_len].replace("\n", " ").strip()
        lines.append(f"{rank}. id={doc_id}\n   {snippet}")
    remaining = max(0, budget - calls)
    lines.append(f"\n[retrieval calls used: {calls}/{budget}; remaining: {remaining}]")
    return "\n".join(lines)
