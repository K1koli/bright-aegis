"""Deterministic retrieval metrics for BRIGHT.

BRIGHT scores with nDCG@10 over binary relevance: a retrieved chunk counts as
relevant iff its id is in the query's ``gold_ids`` (see the dataset card, which
documents ``gold_ids`` as "the ground truth documents, corresponding to the
indices of the ``documents`` subset").

    DCG@k  = sum_{i=1..k} rel_i / log2(i + 1)
    IDCG@k = sum_{i=1..min(|gold|, k)} 1 / log2(i + 1)
    nDCG@k = DCG@k / IDCG@k        (0.0 when the query has no gold ids)

Pure stdlib on purpose: the fitness function must be cheap, deterministic, and
auditable, and it must never be influenced by anything an evolved harness can
reach.
"""
from __future__ import annotations

import math

__all__ = ["dcg_at_k", "idcg_at_k", "ndcg_at_k", "recall_at_k", "mrr_at_k"]


def _discount(i: int) -> float:
    """Discount for 1-based rank ``i``."""
    return 1.0 / math.log2(i + 1)


def dcg_at_k(ranked_ids: list[str], gold: set[str], k: int = 10) -> float:
    total = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gold:
            total += _discount(i)
    return total


def idcg_at_k(gold: set[str], k: int = 10) -> float:
    ideal = min(len(gold), k)
    return sum(_discount(i) for i in range(1, ideal + 1))


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int = 10) -> float:
    ideal = idcg_at_k(gold, k)
    if ideal <= 0.0:
        return 0.0
    return dcg_at_k(ranked_ids, gold, k) / ideal


def recall_at_k(ranked_ids: list[str], gold: set[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    return len(set(ranked_ids[:k]) & gold) / len(gold)


def mrr_at_k(ranked_ids: list[str], gold: set[str], k: int = 10) -> float:
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gold:
            return 1.0 / i
    return 0.0
