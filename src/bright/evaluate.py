"""Evaluate a retrieval harness on a BRIGHT domain and report nDCG@10.

Deterministic, zero LLM-judge cost: the fitness function for harness evolution
is exactly this script's mean nDCG@10 on a frozen query split.

    python src/bright/evaluate.py --domain biology --split dev --harness bm25
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from bright import data as bdata  # noqa: E402
from bright.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402


@dataclass
class QueryResult:
    qid: str
    domain: str
    ndcg10: float
    recall10: float
    mrr10: float
    n_gold: int
    ranked: list[str]
    latency_s: float = 0.0


@dataclass
class Report:
    harness: str
    domain: str
    split: str
    n_queries: int
    ndcg10: float
    recall10: float
    mrr10: float
    ndcg10_stdev: float
    elapsed_s: float
    per_query: list[QueryResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def select_queries(domain: str, split: str, n_dev: int, subset: int | None, seed: int):
    if split == "all":
        qs = bdata.load_queries(domain)
    else:
        s = bdata.make_split(domain, n_dev=n_dev, seed=seed)
        qs = s.dev if split == "dev" else s.heldout
    if subset:
        qs = qs[:subset]
    return qs


def run_bm25(domain: str, queries, *, candidates: int, k: int) -> tuple[list[QueryResult], BM25Index]:
    doc_ids, contents = bdata.load_corpus(domain)
    t0 = time.time()
    index = BM25Index.cached(domain, doc_ids, contents)
    build_s = time.time() - t0
    results: list[QueryResult] = []
    for q in queries:
        t1 = time.time()
        ranked_pairs = index.search(q.text, k=candidates, exclude=set(q.excluded_ids))
        ranked = [d for d, _ in ranked_pairs[:k]]
        gold = q.gold()
        results.append(
            QueryResult(
                qid=q.qid,
                domain=domain,
                ndcg10=ndcg_at_k(ranked, gold, k),
                recall10=recall_at_k(ranked, gold, k),
                mrr10=mrr_at_k(ranked, gold, k),
                n_gold=len(gold),
                ranked=ranked,
                latency_s=time.time() - t1,
            )
        )
    index.build_s = build_s  # type: ignore[attr-defined]
    return results, index


def summarize(harness: str, domain: str, split: str, results: list[QueryResult], elapsed: float, meta: dict) -> Report:
    ndcgs = [r.ndcg10 for r in results]
    return Report(
        harness=harness,
        domain=domain,
        split=split,
        n_queries=len(results),
        ndcg10=statistics.fmean(ndcgs) if ndcgs else 0.0,
        recall10=statistics.fmean([r.recall10 for r in results]) if results else 0.0,
        mrr10=statistics.fmean([r.mrr10 for r in results]) if results else 0.0,
        ndcg10_stdev=statistics.pstdev(ndcgs) if len(ndcgs) > 1 else 0.0,
        elapsed_s=elapsed,
        per_query=results,
        meta=meta,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="biology", choices=list(bdata.DOMAINS))
    ap.add_argument("--split", default="all", choices=["dev", "heldout", "all"])
    ap.add_argument("--harness", default="bm25")
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--subset", type=int, default=None, help="evaluate only the first N queries of the split")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--candidates", type=int, default=1000, help="BM25 candidate pool before truncation to k")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    queries = select_queries(args.domain, args.split, args.n_dev, args.subset, args.seed)
    if not queries:
        print("no queries selected", file=sys.stderr)
        return 2

    if args.harness != "bm25":
        print(f"unknown harness {args.harness}", file=sys.stderr)
        return 2

    t0 = time.time()
    results, index = run_bm25(args.domain, queries, candidates=args.candidates, k=args.k)
    elapsed = time.time() - t0
    meta = {
        "candidates": args.candidates,
        "k": args.k,
        "n_docs": index.n_docs,
        "index_build_s": round(getattr(index, "build_s", 0.0), 2),
    }
    rep = summarize(args.harness, args.domain, args.split, results, elapsed, meta)

    out = Path(args.out) if args.out else (
        ROOT / "runs" / "baselines" / f"{args.harness}-{args.domain}-{args.split}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(rep), indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.quiet:
        print(f"harness={rep.harness} domain={rep.domain} split={rep.split} n={rep.n_queries}")
        print(f"  nDCG@10   = {rep.ndcg10:.4f}  (stdev {rep.ndcg10_stdev:.4f})")
        print(f"  Recall@10 = {rep.recall10:.4f}")
        print(f"  MRR@10    = {rep.mrr10:.4f}")
        print(f"  docs={meta['n_docs']} index_build={meta['index_build_s']}s elapsed={elapsed:.1f}s")
        print(f"  -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
