"""Run round 0: execute H_0 over a BRIGHT split and emit every AEGIS artifact.

This is the real starting measurement: the mean nDCG@10 here is the number every
later round and every OOD domain is compared against, and the artifacts under
``<run_root>/R0/`` are exactly what AEGIS Stage P consumes on the first evolve.

    python src/bright_harness/run_h0.py --domain biology --split dev --subset 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from bright import data as bdata  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402
from bright_harness.provider import make_task_model_config, task_provider_summary  # noqa: E402
from bright_harness.round_driver import DEFAULT_PROBE_SIZE, run_round  # noqa: E402
from bright_harness.round_metrics import NDCG_PASS_TAU  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="biology", choices=list(bdata.DOMAINS))
    ap.add_argument("--split", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--k-rollouts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--run-root", default=None)
    args = ap.parse_args()

    split = bdata.make_split(args.domain, n_dev=args.n_dev, seed=args.seed)
    queries = split.dev if args.split == "dev" else split.heldout
    if args.subset:
        queries = queries[: args.subset]

    run_root = Path(args.run_root) if args.run_root else (
        ROOT / "runs" / f"aegis-{args.domain}-{args.split}"
    )
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"domain      : {args.domain} / {args.split}")
    print(f"queries     : {len(queries)}")
    print(f"k_rollouts  : {args.k_rollouts}")
    print(f"run_root    : {run_root.relative_to(ROOT)}")
    print(f"pass tau    : {NDCG_PASS_TAU}")
    print(f"task agent  : {task_provider_summary()}")
    print(f"probe size  : {DEFAULT_PROBE_SIZE} (Tier-2 gate, used from round 1)")
    print()

    doc_ids, contents = bdata.load_corpus(args.domain)
    index = BM25Index.cached(args.domain, doc_ids, contents)
    print(f"corpus      : {index.n_docs} passages, {len(index.terms)} terms")
    print()

    model_config = make_task_model_config()
    t0 = time.time()
    result = await run_round(
        domain=args.domain,
        queries=queries,
        config_path=None,
        round_n=0,
        run_root=run_root,
        model_config=model_config,
        k_rollouts=args.k_rollouts,
        index=index,
    )
    elapsed = time.time() - t0

    print("=== H_0 results ===")
    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))
    print(f"\nwall clock: {elapsed/60:.1f} min  ({elapsed/max(1,len(queries)):.1f} s/query)")

    # Baseline reference: what BM25 alone scores on this exact split.
    from bright.metrics import ndcg_at_k

    bm25 = [ndcg_at_k([d for d, _ in index.search(q.text, k=10)], q.gold(), 10) for q in queries]
    mean_bm25 = sum(bm25) / len(bm25) if bm25 else 0.0
    print(f"\nBM25 alone (same split) : {mean_bm25:.4f}")
    print(f"H_0 through the agent   : {result.mean_ndcg:.4f}")
    print(f"obedience cost          : {mean_bm25 - result.mean_ndcg:+.4f}")

    print("\n=== artifacts ===")
    for p in sorted((run_root / "R0").rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(run_root)}  ({p.stat().st_size} B)")
    for extra in ("curves.json", "data/task_history.jsonl"):
        p = run_root / extra
        if p.exists():
            print(f"  {extra}  ({p.stat().st_size} B)")

    print("\n=== worst 8 queries (what the Digester will focus on) ===")
    for o in sorted(result.outcomes, key=lambda x: x.ndcg10)[:8]:
        print(f"  q{o.qid:<5} nDCG={o.ndcg10:.4f} ranked={len(o.ranked):2d} exit={o.exit_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
