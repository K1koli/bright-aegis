"""The starting harness H_0 and the BRIGHT task definition.

H_0 is deliberately naive but is not a strawman: it is a plain single-round
retrieval harness whose retriever is the same BM25 baseline BRIGHT itself
reports, so any later gain is attributable to harness composition rather than to
swapping in a stronger retriever.

    H_0 = QueryUnderstand(passthrough) -> Retrieve(BM25) -> Return(ranked ids)

Every component here is meant to be an evolution target:

* ``prompts/h0_system.md``      -- the ``prompt`` bucket
* the tool registry below       -- the ``tools`` bucket (add Rewrite/Rerank/...)
* ``Retrieve``'s ``k`` and the task's ``max_steps``
                                -- the ``config`` bucket
* the processor stack           -- the ``processor`` bucket (currently empty)

``make_bright_harness`` returns a plain ``HarnessConfig`` so the same object the
task agent executes is the object a candidate edits and the deterministic gate
canonicalizes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from harnessx.core.builder import HarnessBuilder
from harnessx.core.harness import BaseTask, HarnessConfig
from harnessx.processors.context.strategies.system_prompt.plain_markdown import (
    PlainMarkdownSystemPromptBuilder,
)
from harnessx.processors.context.system_prompt import SystemPromptProcessor
from harnessx.processors.context.user_wrapper import UserWrapperProcessor
from harnessx.tools.inmemory import InMemoryToolRegistry
from harnessx.tracing.journal import HarnessJournal
from harnessx.workspace.workspace import Workspace

from bright_harness.processors import RetrievalPolicyProcessor
from bright_harness.tools import Retrieve

ROOT = Path(__file__).resolve().parents[2]
H0_PROMPT = Path(__file__).resolve().parent / "prompts" / "h0_system.md"

# H_0's knobs. These are only FALLBACKS: the live values live in the
# RetrievalPolicyProcessor entry of the applied config, so AEGIS can retune them
# in the `config` bucket (see policy.py). The driver reads max_steps from the
# applied config, which is what makes the step budget evolvable.
# max_steps=2 was too tight in practice: the task agent calls Retrieve at step 0
# and often again at step 1, leaving no step to answer, so the run ended
# budget_exceeded with an empty ranking. 4 leaves room for a second retrieval
# plus the answer.
H0_MAX_STEPS = 4
H0_RETRIEVE_K = 10


def make_h0_tool_registry() -> InMemoryToolRegistry:
    registry = InMemoryToolRegistry()
    registry.register(Retrieve)
    return registry


def make_bright_harness(
    *,
    prompt_path: str | Path | None = None,
    tracer_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
    tool_registry: Any | None = None,
) -> HarnessConfig:
    """Build H_0 (or a variant of it) as a ``HarnessConfig``."""
    prompt = Path(prompt_path) if prompt_path else H0_PROMPT
    if not prompt.exists():
        raise FileNotFoundError(f"system prompt not found: {prompt}")

    tracer = HarnessJournal(
        base_dir=str(tracer_dir or (ROOT / ".tmp" / "hx_runs")),
        export_jsonl=True,
        silent=True,  # per-step INFO lines drown the driver's own output
    )
    workspace = Workspace(
        agent_id="bright",
        root=Path(workspace_root or (ROOT / ".tmp" / "bright_ws")),
        mode=None,
    )

    cfg = (
        HarnessBuilder()
        .slot(
            tool_registry=tool_registry if tool_registry is not None else make_h0_tool_registry(),
            tracer=tracer,
            workspace=workspace,
        )
        .add(SystemPromptProcessor(PlainMarkdownSystemPromptBuilder(str(prompt))))
        .add(UserWrapperProcessor())
        # Carries the retrieval knobs (retrieve_k, snippet_chars,
        # max_retrieve_calls, max_steps, dedupe_by_document) as instance kwargs,
        # which is what makes them `config`-bucket editable. Defaults reproduce the
        # original hard-coded behaviour, so the baseline is unchanged.
        .add(RetrievalPolicyProcessor())
        .build()
    )
    # The retrieval harness uses no filesystem tools, so seeding a workspace (which
    # copies the whole skills tree, ~hundreds of files) is pure overhead and
    # pollutes every round directory. A candidate that adds a filesystem tool can
    # legitimately turn this back on -- it is a real config field.
    cfg.init_workspace = False
    return cfg


def make_bright_task(
    query_text: str,
    *,
    qid: str = "",
    domain: str = "",
    max_steps: int = H0_MAX_STEPS,
    token_budget: int | None = 32_000,
) -> BaseTask:
    """Wrap one BRIGHT query as a task.

    Only the query text crosses the boundary -- never the gold ids, and never the
    human-written ``reasoning`` field, which would leak the intended retrieval
    target to the harness.
    """
    return BaseTask(
        description=query_text,
        success_criteria="",
        max_steps=max_steps,
        token_budget=token_budget,
        metadata={"qid": qid, "domain": domain, "benchmark": "bright"},
    )
