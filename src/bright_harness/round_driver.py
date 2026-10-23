"""BRIGHT round driver: run the current harness, emit every artifact AEGIS needs,
then hand the round to the engine.

One "round" is:

    run H_t over the eval batch  ->  R{n}/raw/<qid>_r<i>.jsonl   (AEGIS Stage P input)
                                 ->  R{n}/trajectories/<qid>.md  (attribution evidence)
                                 ->  R{n}/config.yaml            (the incumbent H_t)
                                 ->  data/task_history.jsonl     (engine ledger)
                                 ->  curves.json                 (graded fitness, recipe-owned)
    then AegisAgent.evolve(..., raw_sessions_dir=R{n}/raw, round_n=n+1)

Two artifact details are load-bearing and easy to get wrong:

* **task_id recovery** -- Stage P does ``stem.rsplit("_r", 1)[0]``, so raw files
  MUST be named ``<qid>_r<i>.jsonl``. BRIGHT qids are bare integers, which keeps
  the rsplit unambiguous.
* **``kind`` rows** -- the counterfactual gate (the paper's "seesaw") replays rows
  whose top-level key is ``kind`` in {after_model, after_tool, task_end}, but no
  production code writes them, so upstream it is dormant for every benchmark. We
  append them alongside the journal's ``type`` rows: Stage P passes unknown keys
  through and ``extract_trace_facts`` ignores unknown ``type`` values, so one file
  serves both readers and the seesaw actually fires.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bright import data as bdata
from bright.retrieval import BM25Index
from bright_harness.harness import (
    H0_MAX_STEPS,
    make_bright_harness,
    make_bright_task,
)
from bright_harness.round_metrics import (
    FITNESS_DELTA,
    FITNESS_MARGIN,
    NDCG_PASS_TAU,
    QueryOutcome,
    aggregate,
    pass_flags,
    scores_by_task,
    task_history_records,
)
from bright_harness.scoring import score_output
from bright_harness.tools import active_index, set_active_index

ROOT = Path(__file__).resolve().parents[2]

# Frozen probe set size for the Tier-2 fitness gate.
DEFAULT_PROBE_SIZE = 20


def policy_kwargs_from_config(cfg) -> dict:
    """Read RetrievalPolicyProcessor's kwargs out of a HarnessConfig.

    This is what makes the retrieval knobs actually evolvable: the driver takes
    ``max_steps`` (and the rest) from the *applied* config, so a candidate that
    edits them in the ``config`` bucket changes how the next round really runs.
    Before this, ``max_steps`` was hard-coded here and ``k`` was only reachable
    through the prompt -- the AEGIS Planner flagged both as dead knobs.
    """
    for entry in getattr(cfg, "processors", None) or []:
        target = str((entry or {}).get("_target_", ""))
        if "RetrievalPolicyProcessor" in target:
            return {k: v for k, v in entry.items() if not k.startswith("_")}
    return {}


def max_steps_from_config(cfg, default: int = H0_MAX_STEPS) -> int:
    try:
        return int(policy_kwargs_from_config(cfg).get("max_steps", default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# artifact writers
# --------------------------------------------------------------------------- #
def _session_dir(sessions_root: Path, session_id: str) -> Path | None:
    """Locate the journal directory for one run.

    session_id must not contain path separators (they would nest the directory);
    we search for a directory whose name ends with it so the run_id filename
    stays the journal's business.
    """
    if not sessions_root.exists():
        return None
    for p in sessions_root.rglob("*.jsonl"):
        if p.parent.name.endswith(session_id):
            return p.parent
    return None


def _force_tracer_dir(cfg, sessions_root: Path) -> str:
    """Point the loaded config's tracer at THIS round's sessions directory.

    Only round 0 builds its harness through ``make_bright_harness(tracer_dir=...)``.
    From round 1 on the harness comes from ``HarnessConfig.from_yaml_file``, whose
    ``tracer.base_dir`` still names whichever round produced that YAML (usually
    R0). Without this override every round writes its journal into R0/sessions,
    R{n}/sessions stays empty, ``_flatten_session`` finds nothing, and R{n}/raw
    ends up holding only the appended ``kind`` rows -- so every later digest is a
    capture artifact rather than evidence.

    Measured consequence when this was missing: R0/raw = 45,816 B (healthy) while
    R1/raw and R2/raw were 3,073 B of pure ``kind`` rows, and the AEGIS Planner
    reported "R2's evidence base is broken ... the digest cluster's headline
    'failure modes' are capture artifacts, NOT agent behaviour".
    """
    tracer = getattr(cfg, "tracer", None)
    if tracer is None:
        return "no-tracer"
    target = str(sessions_root)
    if isinstance(tracer, dict):
        tracer["base_dir"] = target
        return "dict"
    for attr in ("base_dir", "dir", "root"):
        if hasattr(tracer, attr):
            try:
                setattr(tracer, attr, target)
                return f"attr:{attr}"
            except Exception:  # noqa: BLE001
                continue
    return "unpatched"


def _flatten_session(sessions_root: Path, session_id: str, dst: Path, *, fallback_root: Path | None = None) -> list[dict]:
    """Copy the run's journal JSONL to ``dst`` and return its parsed rows.

    Falls back to searching the whole run when the round's own sessions dir has
    nothing -- and says so loudly, because a silent empty result is exactly how the
    tracer bug above corrupted two rounds of evidence without anyone noticing.
    """
    sdir = _session_dir(sessions_root, session_id)
    if sdir is None and fallback_root is not None:
        found = _session_dir(fallback_root, session_id)
        if found is not None:
            import logging

            logging.getLogger("bright.round_driver").warning(
                "session %s was NOT under %s but under %s -- tracer.base_dir is stale; "
                "flattening from there (evidence would otherwise be empty)",
                session_id,
                sessions_root,
                found,
            )
            sdir = found
    if sdir is None:
        import logging

        logging.getLogger("bright.round_driver").warning(
            "no journal found for session %s (searched %s) -- this round's raw trace will be EMPTY",
            session_id,
            sessions_root,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("", encoding="utf-8")
        return []
    files = [p for p in sorted(sdir.glob("*.jsonl")) if not p.name.endswith("_trace.jsonl")]
    if not files:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("", encoding="utf-8")
        return []
    shutil.copy2(files[0], dst)
    rows: list[dict] = []
    for line in dst.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _tool_call_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("type") != "raw_assistant":
            continue
        msg = r.get("message") or {}
        for tc in msg.get("tool_calls") or []:
            name = str((tc or {}).get("name") or "?")
            counts[name] = counts.get(name, 0) + 1
    return counts


def _append_kind_rows(dst: Path, rows: list[dict], *, final_output: str, exit_reason: str) -> None:
    """Append the ``kind``-schema rows the counterfactual gate replays.

    Written *after* Stage P's future read of this file is irrelevant -- Stage P
    copies this file into ``R{n+1}/trajectories/`` and preserves unknown keys, so
    the gate (which globs ``R{n+1}/trajectories/<tid>*.jsonl``) will find them.
    """
    extra: list[dict] = []
    for r in rows:
        t = r.get("type")
        if t == "raw_assistant":
            msg = r.get("message") or {}
            extra.append(
                {
                    "kind": "after_model",
                    "step_id": r.get("step", 0),
                    "content": msg.get("content") or "",
                    "tool_calls": msg.get("tool_calls") or [],
                }
            )
        elif t == "raw_tool":
            msg = r.get("message") or {}
            extra.append(
                {
                    "kind": "after_tool",
                    "step_id": r.get("step", 0),
                    "tool_name": msg.get("name") or "",
                    "content": msg.get("content") or "",
                    "tool_call_id": msg.get("tool_call_id") or "",
                }
            )
    extra.append({"kind": "task_end", "final_output": final_output, "exit_reason": exit_reason})
    with open(dst, "a", encoding="utf-8") as fh:
        for row in extra:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_trajectory_md(
    path: Path,
    *,
    outcome: QueryOutcome,
    query_text: str,
    final_output: str,
    round_n: int,
    rollout: int,
    tools_used: list[str],
) -> None:
    """Human/attribution-facing trajectory.

    ``tool_call_counts`` MUST be a single-line JSON dict: attribution.py parses it
    with ``^tool_call_counts:\\s*(\\{[^}]*\\})`` under ``re.M``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    counts_line = json.dumps({k: int(v) for k, v in outcome.tool_calls.items()})
    fm = [
        "---",
        f'task_id: "{outcome.qid}"',
        f"domain: {outcome.domain}",
        f"round: {round_n}",
        f"rollout: {rollout}",
        f"ndcg10: {outcome.ndcg10:.6f}",
        f"eval_passed: {str(outcome.passed).lower()}",
        f"eval_score: {outcome.ndcg10:.6f}",
        f"exit_reason: {outcome.exit_reason or ('error' if outcome.error else 'done')}",
        f"steps: {outcome.steps}",
        f"cost_usd: {outcome.cost_usd:.6f}",
        f"latency_s: {outcome.latency_s:.3f}",
        f"n_ranked: {len(outcome.ranked)}",
        f"tool_call_counts: {counts_line}",
        f"tools_used: {json.dumps(tools_used)}",
        "---",
        "",
    ]
    body = [
        "# Query",
        "",
        query_text,
        "",
        "## Task agent final output",
        "",
        "```",
        final_output[:4000] if final_output else "(empty)",
        "```",
        "",
        "## Parsed ranking",
        "",
        "```",
        "\n".join(outcome.ranked) if outcome.ranked else "(nothing parsed)",
        "```",
        "",
        "## Deterministic score",
        "",
        f"nDCG@10 = {outcome.ndcg10:.6f}  (pass threshold {NDCG_PASS_TAU})",
        "",
    ]
    if outcome.error:
        body += ["## Error", "", "```", outcome.error[:2000], "```", ""]
    path.write_text("\n".join(fm + body), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# harness execution
# --------------------------------------------------------------------------- #
@dataclass
class RoundResult:
    round_n: int
    outcomes: list[QueryOutcome] = field(default_factory=list)
    mean_ndcg: float = 0.0
    pass_rate: float = 0.0
    parse_rate: float = 0.0
    raw_dir: Path | None = None
    config_path: Path | None = None
    n_regressed: int = 0
    #: Honest incumbent fitness on the FULL probe set (see run_round's probe_queries).
    probe_scores: dict[str, float] = field(default_factory=dict)
    tracer_target: str = ""

    def summary(self) -> dict:
        return {
            "round": self.round_n,
            "n_queries": len(self.outcomes),
            "mean_ndcg": round(self.mean_ndcg, 6),
            "pass_rate": round(self.pass_rate, 6),
            "parse_rate": round(self.parse_rate, 6),
            "n_passed": sum(1 for o in self.outcomes if o.passed),
            "tau": NDCG_PASS_TAU,
            "mean_cost_usd": round(statistics.fmean([o.cost_usd for o in self.outcomes]), 6) if self.outcomes else 0.0,
            "mean_latency_s": round(statistics.fmean([o.latency_s for o in self.outcomes]), 3) if self.outcomes else 0.0,
            "mean_tokens": round(statistics.fmean([o.total_tokens for o in self.outcomes]), 1) if self.outcomes else 0.0,
        }


async def _run_query(
    harness, q: bdata.Query, *, domain: str, round_n: int, rollout: int, max_steps: int = H0_MAX_STEPS
) -> tuple[QueryOutcome, str, str]:
    task = make_bright_task(q.text, qid=q.qid, domain=domain, max_steps=max_steps)
    session_id = f"bright-R{round_n}-{q.qid}-r{rollout}"
    t0 = time.time()
    try:
        result = await harness.run(task, session_id=session_id)
        final = result.final_output or ""
        exit_reason = getattr(result, "exit_reason", "") or ""
        err = ""
    except Exception as exc:  # noqa: BLE001
        final, exit_reason, err = "", "error", f"{type(exc).__name__}: {exc}"
    return (
        QueryOutcome(
            qid=q.qid,
            domain=domain,
            ndcg10=0.0,
            ranked=[],
            parse_ok=False,
            exit_reason=exit_reason,
            latency_s=time.time() - t0,
            error=err,
        ),
        final,
        session_id,
    )


async def run_round(
    *,
    domain: str,
    queries: list[bdata.Query],
    config_path: Path | None,
    round_n: int,
    run_root: Path,
    model_config,
    k_rollouts: int = 1,
    index: BM25Index | None = None,
    probe_queries: list[bdata.Query] | None = None,
    write_kind_rows: bool = True,
    query_concurrency: int = 1,
) -> RoundResult:
    """Execute one round of the current harness and write all AEGIS artifacts.

    ``probe_queries`` is the frozen Tier-2 probe set. Queries in it that are NOT in
    ``queries`` are evaluated too (their artifacts land under ``R{n}/probe_*`` and
    they are kept out of ``task_history`` so the engine's ledger stays about the
    eval batch). Without this, a probe set larger than the eval batch would have no
    honest incumbent score, and the gate compared candidates against 0.0.

    ``write_kind_rows`` controls whether the counterfactual gate can fire. Upstream,
    no production code writes the ``kind``-schema rows that gate replays, so it is
    dormant in every shipped recipe even though the paper describes it as a core
    constraint. Set this False to reproduce that dormant behaviour exactly.
    """
    from harnessx.core.harness import HarnessConfig

    round_dir = run_root / f"R{round_n}"
    raw_dir = round_dir / "raw"
    sessions_root = round_dir / "sessions"
    traj_dir = round_dir / "trajectories"
    for d in (raw_dir, sessions_root, traj_dir):
        d.mkdir(parents=True, exist_ok=True)

    if index is None:
        doc_ids, contents = bdata.load_corpus(domain)
        index = BM25Index.cached(domain, doc_ids, contents)
    valid_ids = set(index.doc_ids)

    # Bind the corpus BEFORE any harness runs. Without this the Retrieve tool
    # raises "no active BRIGHT index", the task agent receives an error string
    # instead of passages, and obediently answers `[]` -- which scored 0.0000 on
    # 50/50 queries while looking like a model failure. Binding is done through
    # the module global (see tools.py) because HarnessX executes tools in a
    # different asyncio task from the caller.
    set_active_index(index)

    if config_path is None:
        cfg = make_bright_harness(
            tracer_dir=sessions_root, workspace_root=run_root / "ws"
        )
    else:
        cfg = HarnessConfig.from_yaml_file(str(config_path))
    # Without this every round >= 1 journals into R0/sessions and its raw trace comes
    # out empty, so the next evolve round reasons over capture artifacts.
    tracer_target = _force_tracer_dir(cfg, sessions_root)
    # Always record the harness this round actually ran as R{n}/config.yaml.
    # Rounds >= 1 load their config from the previous round's artifact, and the old
    # code never wrote a per-round copy -- so R1/config.yaml simply did not exist and
    # the next round died with FileNotFoundError, while the report's "incumbent
    # config" resolution had nothing to diff against.
    round_config_path = round_dir / "config.yaml"
    if config_path is None or Path(config_path).resolve() != round_config_path.resolve():
        cfg.to_yaml_file(round_config_path)
    config_path = round_config_path
    harness = model_config.agentic(cfg)
    round_max_steps = max_steps_from_config(cfg)

    # ---- run the rollouts ---------------------------------------------------
    #
    # Rollouts are independent: each has its own session id, its own tracer dir
    # under R{n}/sessions/, and its own policy session (see bright_harness.policy).
    # So they can go concurrently. ALL artifact writing happens after the gather,
    # in the original query order, so the on-disk layout and the returned outcome
    # order are identical to a serial run.
    conc = max(1, int(query_concurrency))
    sem = asyncio.Semaphore(conc)

    async def _guarded(q: bdata.Query, i: int):
        if conc <= 1:
            return await _run_query(
                harness, q, domain=domain, round_n=round_n, rollout=i, max_steps=round_max_steps
            )
        async with sem:
            return await _run_query(
                harness, q, domain=domain, round_n=round_n, rollout=i, max_steps=round_max_steps
            )

    work = [(q, i) for q in queries for i in range(k_rollouts)]
    raw_results = await asyncio.gather(*[_guarded(q, i) for q, i in work])

    best_by_qid: dict[str, QueryOutcome] = {}
    for (q, i), (outcome, final, session_id) in zip(work, raw_results):
        scored = score_output(final, q.gold(), valid_ids=valid_ids, k=10)
        outcome.ranked = scored.ranked
        outcome.ndcg10 = scored.ndcg10
        outcome.parse_ok = scored.parse_ok
        outcome.final_output_len = len(final)

        raw_path = raw_dir / f"{q.qid}_r{i}.jsonl"
        rows = _flatten_session(sessions_root, session_id, raw_path, fallback_root=run_root)
        outcome.tool_calls = _tool_call_counts(rows)
        outcome.steps = sum(1 for r in rows if r.get("type") == "raw_assistant")
        for r in rows:
            if r.get("type") == "episode_end":
                outcome.steps = int(r.get("total_steps") or outcome.steps)
                break
        if raw_path.exists() and write_kind_rows:
            _append_kind_rows(
                raw_path, rows, final_output=final, exit_reason=outcome.exit_reason or "done"
            )
        _render_trajectory_md(
            traj_dir / f"{q.qid}_r{i}.md",
            outcome=outcome,
            query_text=q.text,
            final_output=final,
            round_n=round_n,
            rollout=i,
            tools_used=sorted(outcome.tool_calls),
        )
        # k>1: keep the best rollout as the query's outcome (optimistic, matching
        # the engine's documented any-k-pass semantics for `passed`).
        prev = best_by_qid.get(q.qid)
        if prev is None or outcome.ndcg10 > prev.ndcg10:
            best_by_qid[q.qid] = outcome

    outcomes: list[QueryOutcome] = [best_by_qid[q.qid] for q in queries]

    # ---- probe set: honest incumbent fitness for the Tier-2 gate ----------
    #
    # Probe queries that are not in the eval batch have never been run, so their
    # incumbent score is unknown. Filling those with 0.0 (as an earlier version did)
    # silently disabled the gate: every candidate would look like a large
    # improvement on them. So evaluate them for real. Their artifacts go under
    # R{n}/probe/ -- NOT R{n}/raw/, because a raw file without a matching
    # pass_flags_by_task entry makes the Digester raise on an UNKNOWN pattern.
    probe_qids = [q.qid for q in (probe_queries or [])]
    probe_scores: dict[str, float] = {o.qid: o.ndcg10 for o in outcomes if o.qid in set(probe_qids)}
    if probe_queries:
        seen = {o.qid for o in outcomes}
        extras = [q for q in probe_queries if q.qid not in seen]
        probe_dir = round_dir / "probe"
        (probe_dir / "trajectories").mkdir(parents=True, exist_ok=True)
        for q in extras:
            outcome, final, session_id = await _run_query(
                harness, q, domain=domain, round_n=round_n, rollout=0, max_steps=round_max_steps
            )
            scored = score_output(final, q.gold(), valid_ids=valid_ids, k=10)
            outcome.ranked = scored.ranked
            outcome.ndcg10 = scored.ndcg10
            outcome.parse_ok = scored.parse_ok
            probe_scores[q.qid] = scored.ndcg10
            _flatten_session(
                sessions_root, session_id, probe_dir / f"{q.qid}_r0.jsonl", fallback_root=run_root
            )
            _render_trajectory_md(
                probe_dir / "trajectories" / f"{q.qid}.md",
                outcome=outcome,
                query_text=q.text,
                final_output=final,
                round_n=round_n,
                rollout=0,
                tools_used=sorted(outcome.tool_calls),
            )
        _write_json(round_dir / "probe_scores.json", probe_scores)
        print(
            f"    probe: {len(probe_qids)} queries ({len(extras)} newly run), "
            f"incumbent mean = "
            f"{sum(probe_scores.values()) / max(1, len(probe_scores)):.4f}"
        )

    # ---- ledger + curves -------------------------------------------------
    from harnessx.aegis.data import ledger as aegis_ledger

    aegis_ledger.append_task_history(run_root, task_history_records(outcomes, round_n=round_n))

    metrics = aggregate(outcomes)
    curves_path = run_root / "curves.json"
    curves = []
    if curves_path.exists():
        try:
            curves = json.loads(curves_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            curves = []
    entry = metrics_dict = {
        "round": round_n,
        "mean_ndcg": round(metrics.mean_ndcg, 6),
        "pass_rate": round(metrics.pass_rate, 6),
        "parse_rate": round(metrics.parse_rate, 6),
        "n_queries": metrics.n_queries,
        "n_passed": metrics.n_passed,
        "tau": NDCG_PASS_TAU,
    }
    curves = [c for c in curves if c.get("round") != round_n] + [entry]
    curves.sort(key=lambda c: c.get("round", 0))
    _write_json(curves_path, curves)

    return RoundResult(
        round_n=round_n,
        outcomes=outcomes,
        mean_ndcg=metrics.mean_ndcg,
        pass_rate=metrics.pass_rate,
        parse_rate=metrics.parse_rate,
        raw_dir=raw_dir,
        config_path=config_path,
        n_regressed=len(metrics.n_regressed_vs),
        probe_scores=probe_scores,
        tracer_target=tracer_target,
    )


# --------------------------------------------------------------------------- #
# fitness context for the Tier-2 gate
# --------------------------------------------------------------------------- #
def make_fitness_fn(*, domain: str, index: BM25Index, model_config, scratch_dir: Path):
    """Return ``fitness_fn(config_path, qids) -> {qid: ndcg}`` for the gate.

    The gate calls this with a *candidate* config; we rebuild a harness from it and
    evaluate exactly the probe queries. Runs go to a scratch tracer dir so a probe
    never pollutes the round's real artifacts.
    """
    from harnessx.core.harness import HarnessConfig

    by_qid = {q.qid: q for q in bdata.load_queries(domain)}
    valid_ids = set(index.doc_ids)

    async def fitness_fn(candidate_config_path: Path, qids: list[str]) -> dict[str, float]:
        set_active_index(index)
        cfg = HarnessConfig.from_yaml_file(str(candidate_config_path))
        harness = model_config.agentic(cfg)
        probe_max_steps = max_steps_from_config(cfg)
        scores: dict[str, float] = {}
        for qid in qids:
            q = by_qid.get(qid)
            if q is None:
                continue
            task = make_bright_task(q.text, qid=q.qid, domain=domain, max_steps=probe_max_steps)
            try:
                result = await harness.run(task, session_id=f"probe-R-{qid}")
                final = result.final_output or ""
            except Exception:  # noqa: BLE001
                final = ""
            scores[qid] = score_output(final, q.gold(), valid_ids=valid_ids, k=10).ndcg10
        return scores

    return fitness_fn


def build_fitness_context(
    *,
    probe_qids: list[str],
    incumbent_scores: dict[str, float],
    fitness_fn,
    delta: float = FITNESS_DELTA,
    margin: float = FITNESS_MARGIN,
) -> dict:
    return {
        "probe_qids": list(probe_qids),
        "incumbent_scores": dict(incumbent_scores),
        "fitness_fn": fitness_fn,
        "delta": delta,
        "margin": margin,
    }
