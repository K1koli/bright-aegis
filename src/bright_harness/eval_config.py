"""Evaluate H_0 against a candidate harness on the SAME queries. Costs $0.

Only the frozen task agent runs, and it runs on our own GPU; no proposer is called.
That makes this the cheapest way to check whether a shipped candidate's gain is real
before spending another ~$31 on an evolve round.

    python src/bright_harness/eval_config.py \
        --config runs/fix-T1/R1/applied/merged.yaml --domain biology --subset 50

H_0 is always evaluated first on the identical query set, so the delta is paired and
free of split differences.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from bright import data as bdata  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402
from bright_harness.provider import make_task_model_config, task_provider_summary  # noqa: E402
from bright_harness.round_driver import run_round  # noqa: E402
from bright_harness.round_metrics import NDCG_PASS_TAU  # noqa: E402


async def _eval(label, domain, queries, config_path, run_root, model_config, index, k, concurrency=1):
    res = await run_round(
        domain=domain,
        queries=queries,
        config_path=config_path,
        round_n=0,
        run_root=run_root,
        model_config=model_config,
        k_rollouts=k,
        index=index,
        query_concurrency=concurrency,
    )
    scores = {o.qid: o.ndcg10 for o in res.outcomes}
    passed = sum(1 for o in res.outcomes if o.ndcg10 >= NDCG_PASS_TAU)
    print(f"  {label:<30} mean nDCG@10 = {res.mean_ndcg:.4f}   pass={passed}/{len(res.outcomes)}")
    return scores, res.mean_ndcg


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="candidate config.yaml to evaluate")
    ap.add_argument("--domain", default="biology")
    ap.add_argument("--subset", type=int, default=None,
                    help="queries to evaluate. Default: the whole split (all queries with "
                         "--all-queries, otherwise the 50-query dev split). Passing --subset "
                         "explicitly overrides that.")
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--split", choices=("dev", "heldout", "all"), default="dev",
                    help="which part of the domain's split to evaluate. 'dev' = the queries "
                         "evolution was allowed to see (contaminated by construction), "
                         "'heldout' = the ones it never saw (the only clean in-domain number), "
                         "'all' = dev+heldout. Uses --n-dev to decide the dev size, so a 3:1 "
                         "split on robotics (101) is --n-dev 76 --split heldout.")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the H_0 run. eval_config otherwise runs BOTH H_0 and the "
                         "candidate, so every cross-evaluation cell costs twice what it "
                         "needs to; use this once H_0's per-domain score is known.")
    ap.add_argument("--all-queries", action="store_true",
                    help="evaluate the domain's whole task set (empty held-out), matching "
                         "the paper's full-task-set protocol. Without it --subset is capped "
                         "at --n-dev (default 50).")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--k-rollouts", type=int, default=1)
    ap.add_argument(
        "--query-concurrency",
        type=int,
        default=1,
        help="rollouts to run at once. Use 1 vs 4 on the same queries to prove the "
             "concurrent path is score-identical to the serial one.",
    )
    ap.add_argument("--run-root", default="runs/eval-config")
    args = ap.parse_args()

    run_root = ROOT / args.run_root
    cand = Path(args.config)
    if not cand.is_absolute():
        cand = ROOT / cand
    if not cand.exists():
        print(f"config not found: {cand}")
        return 2

    use_all = args.all_queries or args.split == "all"
    split = bdata.make_split(
        args.domain,
        n_dev=(10**9 if use_all else args.n_dev),
        seed=args.seed,
        allow_full=use_all,
    )
    pool = split.heldout if args.split == "heldout" else split.dev
    # --subset defaults to the WHOLE selected pool. A fixed default of 50 silently
    # truncated --all-queries runs to 50 queries (the economics cross-domain check ran
    # on 50 of 103 and reported "queries: 50" while looking like a full-set evaluation).
    subset = args.subset if args.subset is not None else len(pool)
    queries = pool[:subset]
    doc_ids, contents = bdata.load_corpus(args.domain)
    index = BM25Index.cached(args.domain, doc_ids, contents)
    model_config = make_task_model_config()

    print("=== config evaluation (task agent only; no proposer, $0) ===")
    print(f"domain      : {args.domain}")
    print(f"queries     : {len(queries)}")
    print(f"candidate   : {cand}")
    print(f"task agent  : {task_provider_summary()}")
    print()
    print("--- running ---")

    h0_scores: dict[str, float] = {}
    h0_mean: float | None = None
    if args.no_baseline:
        print("(--no-baseline: H_0 skipped; only the candidate is scored)")
    else:
        h0_scores, h0_mean = await _eval(
            "H_0 (baseline)", args.domain, queries, None, run_root / "H0", model_config, index,
            args.k_rollouts, args.query_concurrency,
        )
    c_scores, c_mean = await _eval(
        f"CAND {cand.stem}", args.domain, queries, cand, run_root / "CAND", model_config, index,
        args.k_rollouts, args.query_concurrency,
    )

    print()
    print("=== per-query ===")
    if h0_mean is None:
        print(f"  {'qid':<10} {'CAND':>8}")
        for q in queries:
            print(f"  {q.qid:<10} {c_scores.get(q.qid, 0.0):>8.4f}")
        print()
        print(f"  {'mean':<10} {c_mean:>8.4f}   ({len(queries)} queries, candidate only)")
        print()
        print(f"=== CANDIDATE ONLY -- no baseline in this run ===")
        print(f"(artifacts: {run_root.relative_to(ROOT)})")
        return 0

    print(f"  {'qid':<10} {'H_0':>8} {'CAND':>8} {'delta':>9}")
    wins = losses = ties = 0
    for q in queries:
        a = h0_scores.get(q.qid, 0.0)
        b = c_scores.get(q.qid, 0.0)
        d = b - a
        if d > 1e-9:
            wins += 1
        elif d < -1e-9:
            losses += 1
        else:
            ties += 1
        print(f"  {q.qid:<10} {a:>8.4f} {b:>8.4f} {d:>+9.4f}")
    print()
    print(f"  {'mean':<10} {h0_mean:>8.4f} {c_mean:>8.4f} {c_mean - h0_mean:>+9.4f}")
    print(f"  wins={wins}  losses={losses}  ties={ties}")
    print()
    print(f"=== {'GAIN HOLDS' if c_mean > h0_mean else 'NO GAIN / REGRESSION'} ===")
    if losses and c_mean > h0_mean:
        print(f"note: {losses} quer{'y' if losses == 1 else 'ies'} regressed even though the mean rose")
    print(f"(artifacts: {run_root.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
