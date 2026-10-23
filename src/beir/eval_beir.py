"""Evaluate the BRIGHT-evolved harnesses on a BEIR dataset.

Scoring is the same binary nDCG@10 the BRIGHT runs use, and for binary qrels BEIR's
own formula reduces to exactly that (gain = 2^1-1 = 1), so the numbers are method-
comparable with the paper's leaderboard.

Unlike the BIRCO run, ``retrieve_k`` is NOT forced here: on a real corpus, showing
the model more retrieved passages genuinely changes which documents are reachable, so
that lever is part of what was evolved and must be allowed to act.

    python src/beir/eval_beir.py --dataset scifact --arm bm25=bm25 --arm h0=builtin \
        --arm bio=runs/bio-full-T15/R9/applied/merged.yaml \
        --arm rob=runs/rob-3to1/R7/config.yaml
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
from beir import data as bdata  # noqa: E402
from bright_harness.harness import make_bright_harness, make_bright_task  # noqa: E402
from bright_harness.provider import make_task_model_config, task_provider_summary  # noqa: E402
from bright_harness.round_driver import max_steps_from_config, policy_kwargs_from_config  # noqa: E402
from bright_harness.scoring import parse_ranked_ids  # noqa: E402
from bright_harness.tools import set_active_index  # noqa: E402


def _build_config(spec: str):
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
    for entry in getattr(cfg, "processors", None) or []:
        if "RetrievalPolicyProcessor" in str((entry or {}).get("_target_", "")):
            entry["retrieve_k"] = int(k)


async def _run_arm(name, spec, cfg, task, qids, model_config, index, *, force_k):
    if cfg is not None and force_k:
        _force_retrieve_k(cfg, force_k)
    is_model = spec not in ("bm25",)
    k_visible = int(policy_kwargs_from_config(cfg).get("retrieve_k", 10)) if cfg is not None else 10
    max_steps = max_steps_from_config(cfg, 4) if cfg is not None else 4
    harness = model_config.agentic(cfg) if cfg is not None else None

    ndcgs, rec5, ranks, parse_ok, t0 = [], [], [], 0, time.time()
    for qid in qids:
        query = task.queries[qid]
        gold = task.gold[qid]

        if harness is None:
            ranked = [d for d, _ in index.search(query, k=10)]
        else:
            set_active_index(index)
            bt = make_bright_task(query, qid=qid, domain=f"beir-{task.name}", max_steps=max_steps)
            try:
                res = await harness.run(bt, session_id=f"beir-{task.name}-{name}-{qid}")
                out = res.final_output or ""
            except Exception as exc:  # noqa: BLE001
                out = ""
                print(f"    [{name}] {qid} ERROR {type(exc).__name__}: {exc}")
            ranked = parse_ranked_ids(out, valid_ids=set(task.corpus), k=10)
            parse_ok += bool(ranked)

        ndcgs.append(ndcg_at_k(ranked, gold, 10))
        rec5.append(recall_at_k(ranked, gold, 5))
        hit = [i for i, d in enumerate(ranked[:10], 1) if d in gold]
        ranks.append(hit[0] if hit else 0)

    n = len(ndcgs)
    return {
        "arm": name,
        "n": n,
        "ndcg10": sum(ndcgs) / n,
        "recall5": sum(rec5) / n,
        "hit_in_top10": sum(1 for r in ranks if r) / n,
        "parse_rate": parse_ok / n if harness is not None else 1.0,
        "k_visible": k_visible,
        "elapsed_s": round(time.time() - t0, 1),
        "per_query": {q: round(v, 6) for q, v in zip(qids, ndcgs)},
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact", choices=sorted(bdata.KNOWN))
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=SPEC",
                    help="SPEC: 'bm25' (no model), 'builtin' (stock H0), or a config.yaml path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-k", type=int, default=None,
                    help="only for an apples-to-apples visibility control; leave unset so the "
                         "evolved retrieve_k can act")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = bdata.load_task(args.dataset)
    print(f"=== BEIR/{task.name} ===")
    print(f"  {bdata.describe(task)}")

    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm must be NAME=SPEC, got {spec!r}")
        n, s = spec.split("=", 1)
        arms.append((n.strip(), s.strip()))

    qids = sorted(task.queries)
    if args.limit:
        qids = qids[: args.limit]
    print(f"  evaluating {len(qids)} queries | arms {[n for n, _ in arms]} | force-k {args.force_k}")

    model_config = make_task_model_config()
    index = BM25Index.build(list(task.corpus), list(task.corpus.values()))
    if any(s != "bm25" for _, s in arms):
        print(f"  task agent: {task_provider_summary()}")
    print()

    rows = []
    for name, spec in arms:
        cfg = _build_config(spec) if spec != "bm25" else None
        r = await _run_arm(name, spec, cfg, task, qids, model_config, index, force_k=args.force_k)
        note = "  (no model)" if cfg is None else ""
        print(f"  {name:<8} nDCG@10={r['ndcg10']:.4f}  Recall@5={r['recall5']:.4f}  "
              f"hit@10={r['hit_in_top10']:.1%}  parse={r['parse_rate']:.2f}  "
              f"k={r['k_visible']}  {r['elapsed_s']}s{note}")
        rows.append(r)

    print()
    print("=== summary ===")
    print(f"  {'arm':<8} {'nDCG@10':>9} {'Recall@5':>10} {'hit@10':>8} {'parse':>7} {'k':>4}")
    for r in rows:
        print(f"  {r['arm']:<8} {r['ndcg10']:>9.4f} {r['recall5']:>10.4f} "
              f"{r['hit_in_top10']:>8.1%} {r['parse_rate']:>7.2f} {r['k_visible']:>4}")

    out = Path(args.out) if args.out else (ROOT / "runs" / "beir" / task.name)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"dataset": task.name, "n_queries": len(qids),
                    "force_k": args.force_k, "arms": rows}, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"wrote {(out / 'results.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    _boot.quiet_logging()
    raise SystemExit(asyncio.run(main()))
