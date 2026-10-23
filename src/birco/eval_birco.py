"""Evaluate BRIGHT-evolved harnesses on BIRCO.

Every arm sees the SAME candidate pool for a query; the only thing that differs is how
the passages get ordered. That isolates selection behaviour, which is the half of the
BRIGHT gain that is not tied to the corpus's chunking.

    # cheapest sanity check: no model at all, just the three reference orderings
    python src/birco/eval_birco.py --task arguana --arm echo=echo --arm bm25=bm25 --arm rand=random

    # the real thing: H_0 and the two evolved harnesses, all shown the full pool
    python src/birco/eval_birco.py --task arguana \
        --arm h0=builtin \
        --arm bio=runs/bio-full-T15/R9/applied/merged.yaml \
        --arm rob=runs/rob-3to1/R7/config.yaml \
        --force-k 50
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

ROOT = _boot.setup(quiet=True)

from bright.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402
from birco import data as bdata  # noqa: E402
from birco.pool_index import PoolIndex  # noqa: E402
from bright_harness.harness import make_bright_harness, make_bright_task  # noqa: E402
from bright_harness.provider import make_task_model_config, task_provider_summary  # noqa: E402
from bright_harness.round_driver import max_steps_from_config, policy_kwargs_from_config  # noqa: E402
from bright_harness.scoring import parse_ranked_ids  # noqa: E402
from bright_harness.tools import set_active_index  # noqa: E402


def _build_config(spec: str):
    """``echo`` / ``bm25`` / ``random`` need no model; ``builtin`` is stock H_0."""
    if spec == "builtin":
        return make_bright_harness()
    from harnessx.core.harness import HarnessConfig

    p = Path(spec)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"arm config not found: {p}")
    return HarnessConfig.from_yaml_file(str(p))


def _force_retrieve_k(cfg, k: int) -> None:
    """Make every arm see the whole pool.

    Without this the arms are not comparable: the biology ship asks for retrieve_k=30
    and the robotics ship for 50, so on a 50-candidate pool one arm would be shown a
    strict prefix of the other's input. Overriding is an explicit deviation from the
    as-shipped config and is reported as such.
    """
    for entry in getattr(cfg, "processors", None) or []:
        if "RetrievalPolicyProcessor" in str((entry or {}).get("_target_", "")):
            entry["retrieve_k"] = int(k)


async def _run_arm(name, spec, cfg, task: bdata.BircoTask, qids, model_config, full_index, *, force_k):
    # The MODE comes from the spec ("echo"/"bm25"/"random"/config path), not from the
    # arm's display name. Using the name meant `--arm rand=random` silently fell through
    # to the file-order branch and reported exactly the echo numbers.
    mode = spec
    if cfg is not None and force_k:
        _force_retrieve_k(cfg, force_k)
    k_visible = int(policy_kwargs_from_config(cfg).get("retrieve_k", 10)) if cfg is not None else 50
    max_steps = max_steps_from_config(cfg, 4) if cfg is not None else 4

    harness = None
    if cfg is not None:
        harness = model_config.agentic(cfg)

    ndcgs, rec5, ranks, parse_ok, t0 = [], [], [], 0, time.time()
    for qid in qids:
        pool = task.pools[qid]
        texts = {c: task.corpus[c] for c in pool}
        query = task.queries[qid]

        if mode == "random":
            idx = PoolIndex.from_random(pool, texts, seed=abs(hash(qid)) % 10_000)
        elif mode == "bm25":
            idx = PoolIndex.from_bm25(pool, texts, full_index, query)
        else:
            idx = PoolIndex.from_file_order(pool, texts)

        if harness is None:
            ranked = idx.doc_ids[: k_visible if mode not in ("bm25", "random") else 10]
        else:
            set_active_index(idx)
            bt = make_bright_task(query, qid=qid, domain=f"birco-{task.name}", max_steps=max_steps)
            try:
                res = await harness.run(bt, session_id=f"birco-{task.name}-{mode}-{qid}")
                out = res.final_output or ""
            except Exception as exc:  # noqa: BLE001
                out = ""
                print(f"    [{mode}] {qid} ERROR {type(exc).__name__}: {exc}")
            ranked = parse_ranked_ids(out, valid_ids=set(pool), k=10)
            parse_ok += bool(ranked)

        gold = task.gold[qid]
        ndcgs.append(ndcg_at_k(ranked, gold, 10))
        rec5.append(recall_at_k(ranked, gold, 5))
        # Only the first ten positions are graded, so the hit metric must stop at 10 —
        # counting the whole 50 compressed the real top-10 rate from 87% to 99%.
        hit = [i for i, d in enumerate(ranked[:10], 1) if d in gold]
        ranks.append(hit[0] if hit else 0)

    n = len(ndcgs)
    return {
        "arm": name,
        "n": n,
        "ndcg10": sum(ndcgs) / n,
        "recall5": sum(rec5) / n,
        "hit_in_top10": sum(1 for r in ranks if r) / n,
        "mean_rank_of_gold": (sum(r for r in ranks if r) / max(1, sum(1 for r in ranks if r))),
        "parse_rate": parse_ok / n if harness is not None else 1.0,
        "k_visible": k_visible,
        "elapsed_s": round(time.time() - t0, 1),
        # Per-query scores so paired bootstrap CIs are possible later; aggregates alone
        # cannot say whether a 0.05 gap on n=100 is real.
        "per_query": {q: round(v, 6) for q, v in zip(qids, ndcgs)},
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="arguana", choices=sorted(bdata.HF_TASKS))
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=SPEC",
                    help="SPEC: 'echo' | 'bm25' | 'random' (no model), 'builtin' (stock H_0), "
                         "or a config.yaml path")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N queries")
    ap.add_argument("--force-k", type=int, default=None,
                    help="override RetrievalPolicyProcessor.retrieve_k so every arm sees the "
                         "full pool (needed for a fair ranking comparison)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = bdata.load_task(args.task)
    print(f"=== BIRCO/{task.name} ===")
    print(f"  {bdata.describe(task)}")

    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm must be NAME=SPEC, got {spec!r}")
        name, spec2 = spec.split("=", 1)
        arms.append((name.strip(), spec2.strip()))

    qids = sorted(task.queries)[: args.limit] if args.limit else sorted(task.queries)
    print(f"  evaluating {len(qids)} queries")
    print(f"  arms: {[n for n, _ in arms]}")
    print(f"  force-k: {args.force_k}")

    model_config = make_task_model_config()
    full_index = BM25Index.build(list(task.corpus), list(task.corpus.values()))
    if any(s not in ("echo", "bm25", "random") for _, s in arms):
        print(f"  task agent: {task_provider_summary()}")
    print()

    rows = []
    for name, spec in arms:
        is_model = spec not in ("echo", "bm25", "random")
        cfg = _build_config(spec) if is_model else None
        r = await _run_arm(name, spec, cfg, task, qids, model_config, full_index, force_k=args.force_k)
        mode_note = "" if is_model else "  (no model)"
        print(f"  {name:<10} nDCG@10={r['ndcg10']:.4f}  Recall@5={r['recall5']:.4f}  "
              f"gold-in-top10={r['hit_in_top10']:.1%}  mean_rank={r['mean_rank_of_gold']:.1f}  "
              f"parse={r['parse_rate']:.2f}  k={r['k_visible']}  {r['elapsed_s']}s{mode_note}")
        rows.append(r)

    print()
    print("=== summary ===")
    print(f"  {'arm':<10} {'nDCG@10':>9} {'Recall@5':>10} {'top10 hit':>10} {'parse':>7}")
    for r in rows:
        print(f"  {r['arm']:<10} {r['ndcg10']:>9.4f} {r['recall5']:>10.4f} "
              f"{r['hit_in_top10']:>10.1%} {r['parse_rate']:>7.2f}")

    out = Path(args.out) if args.out else (ROOT / "runs" / "birco" / task.name)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"task": task.name, "n_queries": len(qids),
                    "force_k": args.force_k, "arms": rows}, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"wrote {(out / 'results.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    _boot.quiet_logging()
    raise SystemExit(asyncio.run(main()))
