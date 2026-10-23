"""AEGIS pilot driver for BRIGHT.

This is the "recipe": it owns the loop the engine deliberately does not.

    for t in 0..T:
        run H_t over the eval batch        -> R{t}/raw, R{t}/config.yaml, curves.json
        build pass_flags + graded fitness  -> engine vocabulary + Tier-2 context
        AegisAgent.evolve(round_n=t+1)     -> R{t+1}/... ships H_{t+1}

Two things this driver does that upstream recipes do not:

* **Live counterfactual gate.** It writes AEGIS's ``kind``-schema rows next to the
  journal's ``type`` rows (see round_driver), so the paper's no-regression check
  actually fires instead of silently self-skipping. Pass ``--dormant-seesaw`` to
  reproduce upstream's shipped (dormant) behaviour instead.
* **Optional graded fitness gate** (``--fitness-gate``, OFF by default). When enabled
  it passes ``fitness_context`` so Stage 4 additionally requires the candidate to
  hold up on a frozen probe set under nDCG@10, covering the prompt-bucket candidates
  the counterfactual gate cannot see. With it off, the gate set is exactly HarnessX's
  own five.

    python src/bright_harness/evolve.py --rounds 3 --k-evolvers 1 --subset 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from bright import data as bdata  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402
from bright_harness.provider import (  # noqa: E402
    make_proposer_model_config,
    make_task_model_config,
    task_provider_summary,
)
from bright_harness.round_driver import (  # noqa: E402
    DEFAULT_PROBE_SIZE,
    build_fitness_context,
    make_fitness_fn,
    run_round,
)
from bright_harness.report import render_round, render_run  # noqa: E402
from bright_harness.round_metrics import (  # noqa: E402
    FITNESS_DELTA,
    FITNESS_MARGIN,
    NDCG_PASS_TAU,
    pass_flags,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class _StageCostCapture(logging.Handler):
    """Append the engine's per-stage summary lines to a durable file.

    Those lines (``Planner R1: exit=done steps=25 cost=$4.576 ... candidates=2``)
    are Python-logging records, not loguru, and they do not survive a PowerShell
    ``2>&1 | Tee-Object`` pipeline intact -- so per-stage cost and candidate counts
    were only ever visible in the live console and were lost once it scrolled.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(level=logging.INFO)
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if "cost=$" not in msg and "candidates=" not in msg and "landscape_written" not in msg:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"logger": record.name, "message": msg}, ensure_ascii=False) + "\n")
        except OSError:
            pass


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="biology", choices=list(bdata.DOMAINS))
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--subset", type=int, default=None,
                    help="queries per round (eval batch). Defaults to the WHOLE dev split "
                         "(len of make_split(--n-dev)); pass a number explicitly for a cheap "
                         "pilot.")
    ap.add_argument("--probe-size", type=int, default=20,
                    help="frozen probe set for the Tier-2 gate; must be >= --subset so "
                         "it is a strict superset of the eval batch and the extra queries "
                         "are genuine held-out verification")
    ap.add_argument("--rounds", type=int, default=3, help="T: evolve rounds after round 0")
    ap.add_argument("--k-evolvers", type=int, default=1, help="candidates per round")
    ap.add_argument("--k-rollouts", type=int, default=1)
    ap.add_argument(
        "--query-concurrency",
        type=int,
        default=1,
        help="Task-agent rollouts to run at once. DEFAULT 1 BECAUSE IT MEASURABLY DOES "
             "NOT HELP: the local vLLM server is already saturated by a single request, "
             "so width 4 bought about 14 percent and width 8 was SLOWER than serial "
             "(16 warm queries: 35s serial vs 48s at width 8). Per-query scores are "
             "identical either way (verified), so raising this is safe against a faster "
             "or multi-replica endpoint -- it just buys nothing here.",
    )
    ap.add_argument(
        "--digester-concurrency",
        type=int,
        default=8,
        help="Digester invocations to run at once. This IS the wall-time lever: digesters "
             "are straight DeepSeek agent loops (not local GPU work), and their cost is "
             "the dominant per-round term on a large task set. Forwarded to "
             "AegisAgent.max_concurrency; the engine already gates them with an "
             "asyncio.Semaphore (preprocess.py:265). Engine default is 4.",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Paper's early-stop rule (HarnessX 6.1): halt after P consecutive rounds that "
             "ship nothing, alongside the T=15 ceiling. 0 disables early stop.",
    )
    ap.add_argument(
        "--all-queries",
        action="store_true",
        help="Evolve AND report on the domain's entire task set (empty held-out), matching "
             "HarnessX 6.1 ('the full task set is evaluated every round, no subsampling'). "
             "NOTE: this consumes the in-domain held-out split. Other domains are untouched, "
             "so the cross-domain generalization test is unaffected.",
    )
    ap.add_argument(
        "--max-stage-cost",
        type=float,
        default=100.0,
        help=(
            "USD ceiling per AEGIS meta stage (Digester/Planner/Evolver/Critic). "
            "DEFAULT IS THE ENGINE'S OWN DEFAULT (100), chosen deliberately: measured "
            "stage costs are Planner ~$4.6, Evolver ~$16.3, Critic ~$6.6, so the "
            "engine's ceiling never binds and the method runs as published. Lowering it "
            "would truncate the Evolver and turn this into a test of a modified method. "
            "Control spend with --rounds and --subset instead."
        ),
    )
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="run every round but never call the proposer (no API cost); "
                         "round >=1 reloads the previous round's config, which exercises "
                         "the loaded-config path")
    ap.add_argument("--no-report", action="store_true", help="suppress the per-round readable report")
    ap.add_argument(
        "--dormant-seesaw",
        action="store_true",
        help="Reproduce upstream's SHIPPED behaviour for the counterfactual gate ('seesaw'): "
             "do not write the kind-schema rows it replays, so the gate self-skips. Upstream "
             "no production code writes those rows, so the gate is dormant in every published "
             "recipe even though the paper presents it as a core no-regression constraint. "
             "Default is LIVE (rows written), i.e. the method as designed. Flip this on to "
             "check whether a result is being propped up by that gate.",
    )
    ap.add_argument(
        "--fitness-gate",
        action="store_true",
        help="OPT IN to the Tier-2 graded fitness gate (patch 0001 -- NOT part of HarnessX). "
             "OFF BY DEFAULT so a run exercises exactly the published method: the gate set "
             "is HarnessX's own five (structure/novelty/canonicalize/counterfactual/replay). "
             "Switching it on also enables the probe set, because the probe exists only to "
             "feed this gate; with it off no extra queries are run.",
    )
    ap.add_argument("--verbose", action="store_true",
                    help="show harnessx's per-step INFO logs (default: suppressed -- they can "
                         "exceed 100 KB per round and hang a PowerShell console mid-run)")
    args = ap.parse_args()

    # harnessx re-installs its own loguru handler at import time, undoing the one
    # _boot.setup() installed BEFORE the import. Left alone, one round dumps >100 KB
    # of per-step INFO to the console -- enough to freeze a PowerShell window, which
    # is how we lost a Critic stage (exit=interrupted) on the previous run.
    _boot.quiet_logging("INFO" if args.verbose else "WARNING")

    # Enforced by the engine's CostGuardProcessor (AegisAgent.budget_per_round_usd
    # is accepted but never read, so it enforces nothing).
    os.environ["AEGIS_MAX_STAGE_COST_USD"] = str(args.max_stage_cost)

    # ---- resolve the eval batch -------------------------------------------
    #
    # --all-queries means "the whole task set", which also means "no subsampling",
    # so it drives --subset unless the caller passed one explicitly (sentinel None).
    _all = bdata.load_queries(args.domain)
    if args.subset is None:
        # Default to the WHOLE dev split, matching eval_config.py. The old default of 20
        # was a pilot convenience that silently shrank a real run -- pairing it with
        # --n-dev 76 would have evolved on 20 of the 76 dev queries while looking like a
        # 76-query run.
        subset = len(_all) if args.all_queries else args.n_dev
    else:
        subset = args.subset

    # Only meaningful when the Tier-2 gate is wired: with the gate off no probe
    # queries are run at all, so the relation between the two sizes is moot.
    if args.fitness_gate and args.probe_size < subset:
        print(
            f"WARNING: --probe-size {args.probe_size} < eval batch {subset}. The probe "
            "set would be a SUBSET of the eval batch, so the Tier-2 gate would do no "
            "held-out verification at all -- it would only check that the candidate does "
            "not regress on the very queries it is being scored on."
        )

    run_root = Path(args.run_root) if args.run_root else (
        ROOT / "runs" / f"aegis-{args.domain}-T{args.rounds}-seed{args.seed}"
    )
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    # Durable per-stage cost/candidate record (see _StageCostCapture).
    _root_logger = logging.getLogger()
    _root_logger.addHandler(_StageCostCapture(run_root / "stage_costs.jsonl"))
    if _root_logger.level in (logging.NOTSET, logging.WARNING):
        _root_logger.setLevel(logging.INFO)

    split = bdata.make_split(
        args.domain, n_dev=len(_all) if args.all_queries else args.n_dev,
        seed=args.seed, allow_full=args.all_queries,
    )
    eval_batch = split.dev[:subset]
    probe = split.dev[: args.probe_size]
    probe_qids = [q.qid for q in probe]

    doc_ids, contents = bdata.load_corpus(args.domain)
    index = BM25Index.cached(args.domain, doc_ids, contents)

    task_model = make_task_model_config()

    print("=== AEGIS / BRIGHT pilot ===")
    print(f"domain        : {args.domain}")
    print(f"eval batch    : {len(eval_batch)} queries" + ("  (ALL -- no subsampling)" if args.all_queries else ""))
    print(f"held-out      : {len(split.heldout)} queries" + ("  (consumed by --all-queries; cross-domain test unaffected)" if not split.heldout else "  (reserved in-domain)"))
    print(f"probe set     : {len(probe_qids)} queries" + (" (Tier-2 gate ON)" if args.fitness_gate else " (unused: Tier-2 gate off)"))
    print(f"gates         : " + ("HarnessX's five + Tier-2 fitness gate (patch 0001)" if args.fitness_gate else "HarnessX's own five (as published)"))
    print(f"rounds (T)    : {args.rounds}    patience (P): {args.patience}    k_evolvers: {args.k_evolvers}")
    print(f"concurrency   : queries {args.query_concurrency}    digesters {args.digester_concurrency}")
    print(f"seesaw        : " + ("DORMANT (upstream shipped behaviour)" if args.dormant_seesaw else "LIVE (kind rows written; method as designed)"))
    if args.fitness_gate:
        print(f"tau / delta / margin : {NDCG_PASS_TAU} / {FITNESS_DELTA} / {FITNESS_MARGIN}")
    print(f"max cost per stage   : ${args.max_stage_cost:.2f} (engine default; only binds if lowered)")
    print(f"run_root      : {run_root.relative_to(ROOT)}")
    print(f"task agent    : {task_provider_summary()}")
    print()

    config_path: Path | None = None
    history: list[dict] = []
    idle = 0  # consecutive rounds that shipped nothing (paper's P)
    t_start = time.time()

    for t in range(0, args.rounds + 1):
        print(f"--- round {t} : running {'H_0' if config_path is None else config_path.name} ---")
        res = await run_round(
            domain=args.domain,
            queries=eval_batch,
            config_path=config_path,
            round_n=t,
            run_root=run_root,
            model_config=task_model,
            k_rollouts=args.k_rollouts,
            index=index,
            # The probe set exists only to give the Tier-2 gate honest incumbent scores.
            # With the gate off we run no extra queries at all, so the round is exactly
            # the published evaluation loop.
            probe_queries=probe if args.fitness_gate else None,
            write_kind_rows=not args.dormant_seesaw,
            query_concurrency=args.query_concurrency,
        )
        scores = {o.qid: o.ndcg10 for o in res.outcomes}
        _write_json(run_root / f"R{t}" / "scores.json", scores)

        # Task-agent health gate, checked EVERY round. A dead endpoint (SSH tunnel
        # dropped, vLLM gone, budget expired) makes every request fail, so every query
        # scores 0 with no parseable answer. The engine cannot tell that apart from "the
        # domain is hard", so it happily digests the connection errors and evolves against
        # a constant-zero fitness signal. An R0-only check is not enough: the server can
        # die at round 3, and the tunnel dying mid-run is exactly what happened on
        # 2026-10-18 (a live tunnel pointing at a node whose vLLM had stopped returns
        # ConnectionReset, not ConnectionRefused, so it looks like a network fault).
        # Healthy rounds parse ~0.97-1.00; anything under 0.5 is a dead endpoint.
        if res.parse_rate < 0.5:
            raise SystemExit(
                f"\n*** ABORT at round {t}: task agent is not answering.\n"
                f"    parse_rate = {res.parse_rate:.3f} over {len(res.outcomes)} queries "
                f"(mean nDCG {res.mean_ndcg:.4f}).\n"
                f"    A healthy round parses ~0.97-1.00. This almost always means the local\n"
                f"    vLLM / SSH tunnel is down -- check:\n"
                f"      python -c \"import json,urllib.request;print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=10)))\"\n"
                f"    and confirm the tunnel's node still matches 'qstat -n'. No further\n"
                f"    evolution round was started and no digester or meta-agent spend followed."
            )

        summary = res.summary()
        if args.fitness_gate:
            summary["probe_mean_ndcg"] = (
                sum(res.probe_scores.get(q, 0.0) for q in probe_qids) / max(1, len(probe_qids))
            )
        summary["tracer_target"] = res.tracer_target
        history.append(summary)
        # NOTE: curves.json is owned by round_driver (it appends one entry per round,
        # already incrementally) -- do not write it here, the driver expects a list.
        probe_txt = f"   probe={summary['probe_mean_ndcg']:.4f}" if args.fitness_gate else ""
        print(f"    mean nDCG@10 = {res.mean_ndcg:.4f}   pass={summary['n_passed']}/{summary['n_queries']}"
              f"{probe_txt}   [{time.time() - t_start:.0f}s elapsed]")

        if t >= args.rounds:
            break
        if args.dry_run:
            # Keep going, but never call the proposer -- so this costs nothing while
            # still exercising the pipeline. Round t+1 reloads THIS round's config,
            # which is the "loaded config" path where the stale tracer.base_dir bug
            # lived; a plain repeat of round 0 would not reproduce it.
            config_path = run_root / f"R{t}" / "config.yaml"
            print(f"    (dry-run: no evolve; next round reloads {config_path.name})")
            continue

        # ---- build the two signals the engine needs -------------------------
        flags = pass_flags(res.outcomes)
        fitness_ctx = None
        if args.fitness_gate:
            fitness_ctx = build_fitness_context(
                probe_qids=probe_qids,
                # Honest incumbent scores harvested by run_round over the FULL probe set.
                # Using scores.get(q, 0.0) here (as an earlier version did) filled every
                # probe query outside the eval batch with 0.0 and silently disabled the gate.
                incumbent_scores={q: res.probe_scores[q] for q in probe_qids if q in res.probe_scores},
                fitness_fn=make_fitness_fn(
                    domain=args.domain, index=index, model_config=task_model,
                    scratch_dir=run_root / "probe_scratch",
                ),
            )
        else:
            print("    (gates: HarnessX's own five; Tier-2 fitness gate not wired)")

        from harnessx.aegis import AegisAgent

        # The AEGIS engine re-installs its own loguru handler when it is imported,
        # undoing the suppression applied at startup -- which is why a real round
        # still floods the console (>100 KB) and can freeze a PowerShell window,
        # while a --dry-run (no engine import) stays clean. Re-apply every round.
        _boot.quiet_logging("INFO" if args.verbose else "WARNING")

        aegis = AegisAgent(
            num_evolvers=args.k_evolvers,
            model_config=make_proposer_model_config(),
            replay_model=task_model,
            benchmark_context="bright",
            fitness_context=fitness_ctx,
            # Digester fan-out. The engine already gates these with an
            # asyncio.Semaphore (preprocess.py:265), so this is just the width.
            max_concurrency=max(1, int(args.digester_concurrency)),
        )
        # run_round now always writes R{t}/config.yaml, so this is the harness that
        # actually ran this round -- no need to fall back to the loaded path.
        current_config = run_root / f"R{t}" / "config.yaml"
        print(f"    evolving: {current_config.relative_to(run_root)} -> R{t+1}/")
        try:
            shipped = await aegis.evolve(
                current_config=current_config,
                trajectories_dir=run_root / f"R{t}" / "raw",
                output_dir=run_root / f"R{t + 1}",
                pass_flags_by_task=flags,
                round_n=t + 1,
                raw_sessions_dir=run_root / f"R{t}" / "raw",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    EVOLVE FAILED: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            break

        shipped = Path(shipped)
        if shipped.exists() and shipped.resolve() != current_config.resolve():
            config_path = shipped
            idle = 0
            print(f"    shipped -> {shipped.relative_to(run_root)}")
        else:
            config_path = current_config
            idle += 1
            print(
                f"    no-op round (nothing shipped) -- keeping the current harness"
                + (f"   [patience {idle}/{args.patience}]" if args.patience else "")
            )

        # The round's own section is only complete now: its candidates were written
        # by this evolve call, and its score by the run_round above it.
        if not args.no_report:
            print()
            print(render_round(run_root, t, show_diff=False))

        # HarnessX 6.1: "at most T=15 rounds, stopping early after P=3 consecutive
        # rounds without a shipped edit."
        if t == 0 and args.rounds > 1:
            per_iter = time.time() - t_start
            print(
                f"    [eta] first iteration (scoring + one evolve) took {per_iter/60:.1f} min "
                f"-> {args.rounds} iterations ~= {per_iter * args.rounds / 60:.0f} min "
                f"if nothing ships and early stop never fires"
            )
        if args.patience and idle >= args.patience:
            print(
                f"\n=== early stop: {idle} consecutive rounds shipped nothing "
                f"(patience P={args.patience}); stopped at round {t} ==="
            )
            break

    elapsed = time.time() - t_start
    _write_json(run_root / "pilot_history.json", history)
    print(f"\n=== pilot done in {elapsed/60:.1f} min ===")
    for h in history:
        probe_txt = f" probe={h['probe_mean_ndcg']:.4f}" if "probe_mean_ndcg" in h else ""
        print(f"  R{h['round']}: mean={h['mean_ndcg']:.4f} "
              f"passed={h['n_passed']}/{h['n_queries']}{probe_txt}")
    if not args.no_report:
        print()
        print(render_run(run_root, show_diff=True))
    print(f"\nhistory -> {(run_root / 'pilot_history.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
