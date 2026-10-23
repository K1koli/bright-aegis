"""Offline validation of the round artifacts against the REAL AEGIS Stage P.

No model, no network. Synthesises raw session files the same way the live driver
does, then feeds them to the engine's own code paths and asserts the engine can
actually read them:

* ``extract_trace_facts`` pairs the ``Retrieve`` call to its result (proves
  ``tool_call_id`` wiring) and reads ``episode_end``;
* ``run_stage_p`` cleans the raw files into ``R{n+1}/trajectories/`` and produces
  digests + ``summary.md`` with a usable actionability score (a stub Digester
  harness supplies the digest text, since the real one is an LLM);
* ``append_task_history`` accepts the ledger records and drops unknown keys;
* the **counterfactual gate** -- dormant for every upstream benchmark because no
  production code writes ``kind`` rows -- actually finds our rows and runs.

That last one is the point of appending ``kind`` rows next to the journal's
``type`` rows: one file, two readers, and the paper's seesaw finally fires.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from bright import data as bdata  # noqa: E402
from bright.metrics import ndcg_at_k  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402
from bright_harness.round_driver import _append_kind_rows  # noqa: E402
from bright_harness.round_metrics import (  # noqa: E402
    NDCG_PASS_TAU,
    QueryOutcome,
    pass_flags,
    task_history_records,
)
from harnessx.aegis.data import ledger as aegis_ledger  # noqa: E402
from harnessx.aegis.gates.counterfactual import check_counterfactual_replay  # noqa: E402
from harnessx.aegis.stages.preprocess import classify_pattern, run_stage_p  # noqa: E402
from harnessx.aegis.stages.trace_facts import extract_trace_facts  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def write_raw(path: Path, *, qid: str, query_text: str, ranked: list[str], ndcg: float) -> list[dict]:
    rows = [
        {"type": "raw_user", "step": 0, "message": {"role": "user", "content": query_text}},
        {
            "type": "raw_assistant",
            "step": 1,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"{qid}-c1", "name": "Retrieve", "input": {"query": query_text, "k": 10}}],
            },
        },
        {
            "type": "raw_tool",
            "step": 2,
            "message": {
                "role": "tool",
                "tool_call_id": f"{qid}-c1",
                "name": "Retrieve",
                "content": "1. id=" + (ranked[0] if ranked else "none") + "\n   snippet text",
            },
        },
        {"type": "raw_assistant", "step": 3, "message": {"role": "assistant", "content": json.dumps(ranked)}},
        {
            "type": "episode_end",
            "step": 4,
            "exit_reason": "done",
            "total_steps": 2,
            "passed": ndcg >= NDCG_PASS_TAU,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    _append_kind_rows(path, rows, final_output=json.dumps(ranked), exit_reason="done")
    return rows


class _StubDigesterHarness:
    """Stands in for the LLM Digester: writes a digest with the required frontmatter."""

    def __init__(self, inputs):
        self.inputs = inputs

    async def run(self, task):  # noqa: ARG002
        p = self.inputs.digest_out_path
        p.parent.mkdir(parents=True, exist_ok=True)
        anchor = f"[trajectories/{self.inputs.task_id}_r0.jsonl#step_1]"
        p.write_text(
            f"pattern: {self.inputs.pattern}\n"
            "failure_mode: query_vocabulary_mismatch\n"
            "\n"
            "## Failure Evidence\n"
            f"{anchor} the Retrieve call returned passages that do not answer the request.\n"
            "\n"
            "## Root Cause\nThe query is phrased as a claim; the relevant passage uses different wording.\n",
            encoding="utf-8",
        )


async def main() -> int:
    run_root = ROOT / ".tmp" / "offline_artifacts"
    if run_root.exists():
        import shutil

        shutil.rmtree(run_root)
    raw_dir = run_root / "R0" / "raw"
    traj_dir = run_root / "R1" / "trajectories"
    digests_dir = run_root / "R1" / "digests"
    summary_path = run_root / "R1" / "summary.md"

    domain = "biology"
    doc_ids, contents = bdata.load_corpus(domain)
    index = BM25Index.cached(domain, doc_ids, contents)
    queries = bdata.make_split(domain, n_dev=50).dev[:6]

    print(f"1. synthesise {len(queries)} raw sessions (BM25, no model)")
    outcomes: list[QueryOutcome] = []
    for q in queries:
        ranked = [d for d, _ in index.search(q.text, k=10)]
        ndcg = ndcg_at_k(ranked, q.gold(), 10)
        write_raw(raw_dir / f"{q.qid}_r0.jsonl", qid=q.qid, query_text=q.text, ranked=ranked, ndcg=ndcg)
        outcomes.append(
            QueryOutcome(
                qid=q.qid,
                domain=domain,
                ndcg10=ndcg,
                ranked=ranked,
                parse_ok=True,
                exit_reason="done",
                steps=2,
                tool_calls={"Retrieve": 1},
                final_output_len=len(json.dumps(ranked)),
            )
        )
    flags = pass_flags(outcomes)
    check("raw files written", len(list(raw_dir.glob("*.jsonl"))) == len(queries), f"{len(queries)} files")
    check("every stem has a pass_flags entry", all(f"{q.qid}_r0" .rsplit("_r", 1)[0] in flags for q in queries))

    print("\n2. extract_trace_facts reads our schema")
    facts = extract_trace_facts(queries[0].qid, [raw_dir / f"{queries[0].qid}_r0.jsonl"])
    check("a tool call was extracted", len(facts.tool_calls) == 1, f"{len(facts.tool_calls)} call(s)")
    if facts.tool_calls:
        fc = facts.tool_calls[0]
        check("tool name is Retrieve", fc.tool == "Retrieve", fc.tool)
        check(
            "tool_call_id paired to its result (not 'missing')",
            fc.return_type != "missing",
            f"return_type={fc.return_type} len={fc.return_len}",
        )
    check("exit fact extracted", len(facts.exits) == 1, f"{len(facts.exits)}")
    if facts.exits:
        ex = facts.exits[0]
        check("exit_reason round-trips", ex.exit_reason == "done", ex.exit_reason)
        check("total_steps round-trips", ex.total_steps == 2, str(ex.total_steps))
        check("terminal snippet captured", bool(ex.terminal_snippet), ex.terminal_snippet[:40])

    print("\n3. append_task_history accepts the records")
    records = task_history_records(outcomes, round_n=0)
    probe = {**records[0], "unknown_key_should_be_dropped": 123}
    aegis_ledger.append_task_history(run_root, [probe] + records[1:])
    hist_path = run_root / "data" / "task_history.jsonl"
    check("task_history.jsonl written", hist_path.exists(), str(hist_path.relative_to(ROOT)))
    lines = [json.loads(x) for x in hist_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    check("one record per query", len(lines) == len(queries), f"{len(lines)}")
    check("unknown key dropped by the engine", "unknown_key_should_be_dropped" not in lines[0], str(sorted(lines[0])))
    check(
        "exactly the 10 documented keys survive",
        set(lines[0]) == {
            "round", "task_id", "level", "passed", "passed_flags", "k",
            "exit", "steps", "cost_usd", "final_output_len", "tools_used",
        },
        str(sorted(lines[0])),
    )

    print("\n4. run_stage_p (real engine code, stubbed Digester)")
    passed_ids = [o.qid for o in outcomes if o.passed]
    print(f"     queries passing at tau={NDCG_PASS_TAU}: {len(passed_ids)}/{len(outcomes)}")
    stage_p = await run_stage_p(
        raw_dir=raw_dir,
        trajectories_dir=traj_dir,
        digests_dir=digests_dir,
        summary_path=summary_path,
        pass_flags_by_task=flags,
        harness_factory=lambda inputs: _StubDigesterHarness(inputs),
        concurrency=4,
    )
    cleaned = sorted(traj_dir.glob("*.jsonl"))
    check("Stage P cleaned trajectories into R1", len(cleaned) == len(queries), f"{len(cleaned)}")
    check("summary.md written", summary_path.exists())
    digests = sorted(digests_dir.glob("*.md"))
    check("digests written", len(digests) == len(queries), f"{len(digests)}")
    check(
        "actionability is usable (>0, so the round will not early-exit)",
        float(stage_p.get("actionability", 0.0)) > 0.0,
        f"actionability={stage_p.get('actionability')} reason={stage_p.get('actionability_reason')}",
    )
    patterns = {classify_pattern(flags[q.qid]) for q in queries}
    print(f"     classify_pattern values: {sorted(patterns)}")

    print("\n5. counterfactual gate finds our kind-rows (upstream: dormant)")
    cleaned_rows = [json.loads(x) for x in cleaned[0].read_text(encoding="utf-8").splitlines() if x.strip()]
    kinds = sorted({r.get("kind") for r in cleaned_rows if r.get("kind")})
    check("kind rows survived Stage P cleaning", set(kinds) == {"after_model", "after_tool", "task_end"}, str(kinds))
    check(
        "a task_end row carries the compared fields",
        any(r.get("kind") == "task_end" and "final_output" in r and "exit_reason" in r for r in cleaned_rows),
    )

    from pathlib import Path as _P

    cfg_path = run_root / "R0" / "config.yaml"
    sys.path.insert(0, str(ROOT / "src"))
    from bright_harness.harness import make_bright_harness

    cfg = make_bright_harness(
        tracer_dir=run_root / "R0" / "sessions", workspace_root=run_root / "R0" / "ws"
    )
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml_file(cfg_path)

    target_qid = queries[0].qid
    cf = await check_counterfactual_replay(
        new_config_yaml_text=cfg_path.read_text(encoding="utf-8"),
        passing_task_ids=[target_qid],
        trajectories_dir=traj_dir,
        k_samples=3,
    )
    check(
        "gate RAN (not skipped for lack of rows)",
        "no previously-passing" not in cf.reason and "skipped" not in cf.reason,
        cf.reason,
    )
    print(f"     gate verdict: ok={cf.ok} reason={cf.reason}")

    print("\n--- summary ---")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all offline artifact checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
