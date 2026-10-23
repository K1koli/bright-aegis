"""Aggregate a round's per-query fitness into the signals AEGIS consumes.

This module is deliberately the *only* place where a graded nDCG becomes a
binary flag, so the binarisation is auditable in one screen.

Two signals go to two different consumers:

* ``pass_flags_by_task`` -- ``{qid: [bool]}``. The engine's own machinery
  (Digester templates, ``actionability``, ``regressions.md``, ``scoreboard``
  hit-rates, the counterfactual gate) speaks only this vocabulary. A query
  "passes" at ``NDCG_PASS_TAU``.
* ``scores_by_task`` -- ``{qid: float}``. The graded truth, used by the Tier-2
  fitness gate and reported in ``curves.json``/``task_history.jsonl``.

``NDCG_PASS_TAU``, ``FITNESS_DELTA`` and ``FITNESS_MARGIN`` are registered a
priori and must not be tuned per round: they define what "no regression" means
for the whole experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean

# ---- a-priori registered constants (do not tune per round) -----------------
# A query counts as "passed" at nDCG@10 >= 0.5, which on BRIGHT means the top-10
# contains a gold chunk in a strong position.
NDCG_PASS_TAU = 0.5
# Per-query regression tolerance for the Tier-2 gate.
FITNESS_DELTA = 0.05
# Aggregate improvement the gate demands (0 = no worse than the incumbent).
FITNESS_MARGIN = 0.0


@dataclass
class QueryOutcome:
    qid: str
    domain: str
    ndcg10: float
    ranked: list[str]
    parse_ok: bool
    exit_reason: str = ""
    steps: int = 0
    cost_usd: float = 0.0
    total_tokens: int = 0
    latency_s: float = 0.0
    tool_calls: dict[str, int] = field(default_factory=dict)
    error: str = ""
    final_output_len: int = 0

    @property
    def passed(self) -> bool:
        return self.ndcg10 >= NDCG_PASS_TAU

    @property
    def failed_hard(self) -> bool:
        """No usable ranking at all: the answer was unparseable or the run errored."""
        return (not self.parse_ok) or bool(self.error)


@dataclass
class RoundMetrics:
    n_queries: int
    mean_ndcg: float
    pass_rate: float
    parse_rate: float
    n_passed: int
    n_regressed_vs: dict[str, float] = field(default_factory=dict)
    worst_delta: float = 0.0


def pass_flags(outcomes: list[QueryOutcome]) -> dict[str, list[bool]]:
    return {o.qid: [o.passed] for o in outcomes}


def scores_by_task(outcomes: list[QueryOutcome]) -> dict[str, float]:
    return {o.qid: o.ndcg10 for o in outcomes}


def aggregate(outcomes: list[QueryOutcome], *, incumbent: dict[str, float] | None = None) -> RoundMetrics:
    ndcgs = [o.ndcg10 for o in outcomes]
    passed = [o for o in outcomes if o.passed]
    parsed = [o for o in outcomes if o.parse_ok and not o.error]
    regressed: dict[str, float] = {}
    worst = 0.0
    if incumbent:
        for o in outcomes:
            if o.qid in incumbent:
                d = o.ndcg10 - incumbent[o.qid]
                worst = min(worst, d)
                if d < -FITNESS_DELTA:
                    regressed[o.qid] = d
    return RoundMetrics(
        n_queries=len(outcomes),
        mean_ndcg=fmean(ndcgs) if ndcgs else 0.0,
        pass_rate=(len(passed) / len(outcomes)) if outcomes else 0.0,
        parse_rate=(len(parsed) / len(outcomes)) if outcomes else 0.0,
        n_passed=len(passed),
        n_regressed_vs=regressed,
        worst_delta=worst,
    )


def task_history_records(outcomes: list[QueryOutcome], *, round_n: int) -> list[dict]:
    """One record per (round, task) in the shape ``append_task_history`` accepts.

    Note the engine's writer copies a FIXED key set, so unknown keys are dropped;
    the graded ``ndcg`` therefore also goes into ``curves.json`` (recipe-owned,
    and catalogued in INDEX.md) rather than relying on task_history alone.
    """
    return [
        {
            "round": round_n,
            "task_id": o.qid,
            "level": o.domain,
            "passed": o.passed,
            "passed_flags": [o.passed],
            "k": 1,
            "exit": o.exit_reason or ("error" if o.error else "done"),
            "steps": o.steps,
            "cost_usd": o.cost_usd,
            "final_output_len": o.final_output_len,
            "tools_used": sorted(o.tool_calls),
        }
        for o in outcomes
    ]
