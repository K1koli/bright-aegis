"""A BM25Index-compatible object over a FIXED candidate pool.

BIRCO hands you the candidates; there is nothing to search. So this class answers the
same two questions ``bright_harness.tools.Retrieve`` asks — ``search(query, k)`` and
``text_of(doc_id)`` — using the pool order as the ranking, in one of three modes:

``file``   the order the qrels file lists them (= the first-stage order BIRCO shipped)
``bm25``   re-scored with the repo's own BM25 over the FULL task corpus, then filtered
           to the pool (a lexical baseline that has no LLM in it)
``random`` a fixed-seed shuffle (the floor)

Keeping the interface identical is what lets every evolved config run unchanged: no
prompt edit, no tool rewrite, no processor change.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class PoolIndex:
    """Mimics ``bright.retrieval.BM25Index`` for one query's candidate pool."""

    doc_ids: list[str]
    texts: dict[str, str]
    #: higher score = earlier in the ranking, so the tool's head is the pool head
    scores: dict[str, float] = field(default_factory=dict)
    n_docs: int = 0

    def __post_init__(self) -> None:
        self.n_docs = len(self.doc_ids)
        if not self.scores:
            self.scores = {d: float(len(self.doc_ids) - i) for i, d in enumerate(self.doc_ids)}

    def text_of(self, doc_id: str) -> str:
        return self.texts.get(doc_id, "")

    def search(self, query: str, k: int = 10, *, candidates: int | None = None):
        """Return ``[(doc_id, score)]`` best-first. ``query`` is ignored on purpose.

        Ignoring the query is the point: the pool is fixed, so a harness that rewrites
        the query cannot change WHICH passages exist — only how they are ordered.
        """
        want = candidates or k
        ordered = sorted(self.scores.items(), key=lambda kv: -kv[1])
        return ordered[:want]

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_file_order(cls, pool: list[str], texts: dict[str, str]) -> "PoolIndex":
        return cls(doc_ids=list(pool), texts=texts)

    @classmethod
    def from_bm25(cls, pool: list[str], texts: dict[str, str], full_index, query: str) -> "PoolIndex":
        """Rank the pool with the repo's BM25 computed over the whole task corpus."""
        pool_set = set(pool)
        ranked = full_index.search(query, k=len(pool), candidates=full_index.n_docs)
        kept = [d for d, _ in ranked if d in pool_set]
        seen = set(kept)
        kept += [d for d in pool if d not in seen]  # anything BM25 dropped keeps pool order
        return cls(doc_ids=kept, texts=texts)

    @classmethod
    def from_random(cls, pool: list[str], texts: dict[str, str], seed: int = 0) -> "PoolIndex":
        shuffled = list(pool)
        random.Random(seed).shuffle(shuffled)
        return cls(doc_ids=shuffled, texts=texts)
