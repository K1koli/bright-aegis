"""Retrieval backends for the BRIGHT harness.

The H_0 harness retrieves with BM25 -- the same sparse baseline BRIGHT itself
reports -- so the starting point is a real, non-strawman retriever rather than a
placeholder. Scores and rankings are deterministic and reproducible.

The index is cached under ``data/bright/index/`` as a compressed postings
layout (term -> doc/tf slices) so repeated evaluation does not re-tokenize the
corpus.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / "data" / "bright" / "index"

# Pyserini/Lucene BM25 defaults, which is what the BRIGHT paper's sparse
# baseline uses.
K1 = 0.9
B = 0.4

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    doc_ids: list[str]
    doc_len: np.ndarray
    terms: list[str]
    term_ptr: np.ndarray
    term_docs: np.ndarray
    term_tfs: np.ndarray
    avgdl: float
    n_docs: int
    texts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.texts:
            self.texts = [""] * len(self.doc_ids)
        self._id_to_pos = {d: i for i, d in enumerate(self.doc_ids)}

    def text_of(self, doc_id: str) -> str:
        i = self._id_to_pos.get(doc_id)
        return self.texts[i] if i is not None else ""

    # ---- construction -------------------------------------------------
    @classmethod
    def build(cls, doc_ids: list[str], contents: list[str], *, verbose: bool = False) -> BM25Index:
        if len(doc_ids) != len(contents):
            raise ValueError("doc_ids and contents length mismatch")
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_len = np.zeros(len(contents), dtype=np.float32)
        for i, text in enumerate(contents):
            toks = tokenize(text)
            doc_len[i] = len(toks)
            if not toks:
                continue
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                postings.setdefault(t, []).append((i, c))
            if verbose and i and i % 10000 == 0:
                print(f"  tokenized {i}/{len(contents)}", flush=True)
        terms = sorted(postings)
        term_ptr = np.zeros(len(terms) + 1, dtype=np.int64)
        docs_all: list[int] = []
        tfs_all: list[int] = []
        for ti, t in enumerate(terms):
            plist = postings[t]
            term_ptr[ti + 1] = term_ptr[ti] + len(plist)
            docs_all.extend(d for d, _ in plist)
            tfs_all.extend(c for _, c in plist)
        return cls(
            doc_ids=doc_ids,
            doc_len=doc_len,
            terms=terms,
            term_ptr=term_ptr,
            term_docs=np.asarray(docs_all, dtype=np.int32),
            term_tfs=np.asarray(tfs_all, dtype=np.float32),
            avgdl=float(doc_len.mean()) if len(doc_len) else 0.0,
            n_docs=len(contents),
            texts=list(contents),
        )

    # ---- persistence ---------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            doc_ids=np.asarray(self.doc_ids, dtype=object),
            doc_len=self.doc_len,
            terms=np.asarray(self.terms, dtype=object),
            term_ptr=self.term_ptr,
            term_docs=self.term_docs,
            term_tfs=self.term_tfs,
            avgdl=np.asarray([self.avgdl], dtype=np.float64),
            n_docs=np.asarray([self.n_docs], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        z = np.load(path, allow_pickle=True)
        return cls(
            doc_ids=[str(x) for x in z["doc_ids"].tolist()],
            doc_len=z["doc_len"],
            terms=[str(x) for x in z["terms"].tolist()],
            term_ptr=z["term_ptr"],
            term_docs=z["term_docs"],
            term_tfs=z["term_tfs"],
            avgdl=float(z["avgdl"][0]),
            n_docs=int(z["n_docs"][0]),
        )

    @classmethod
    def cached(cls, domain: str, doc_ids: list[str], contents: list[str]) -> BM25Index:
        """Load the cached postings if present, else build and cache them.

        Passage text is *not* cached (it dominates the file size); it is
        re-attached from the caller's ``contents`` on every load.
        """
        path = CACHE / f"bm25-{domain}.npz"
        if path.exists():
            idx = cls.load(path)
            if idx.n_docs == len(contents):
                idx.texts = list(contents)
                return idx
        idx = cls.build(doc_ids, contents, verbose=True)
        idx.save(path)
        return idx

    # ---- search --------------------------------------------------------
    def _term_index(self, term: str) -> int:
        lo, hi = 0, len(self.terms)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.terms[mid] < term:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < len(self.terms) and self.terms[lo] == term else -1

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        candidates: int | None = None,
        exclude: set[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``[(doc_id, score), ...]`` of length <= ``candidates``."""
        pool = candidates or k
        scores = np.zeros(self.n_docs, dtype=np.float32)
        touched = False
        for term in tokenize(query):
            ti = self._term_index(term)
            if ti < 0:
                continue
            s, e = int(self.term_ptr[ti]), int(self.term_ptr[ti + 1])
            df = e - s
            idf = float(np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)))
            docs = self.term_docs[s:e]
            tf = self.term_tfs[s:e]
            dl = self.doc_len[docs]
            denom = tf + K1 * (1.0 - B + B * dl / (self.avgdl or 1.0))
            contrib = (idf * (tf * (K1 + 1.0)) / denom) * (weights.get(term, 1.0) if weights else 1.0)
            np.add.at(scores, docs, contrib.astype(np.float32))
            touched = True
        if not touched:
            return []
        order = np.argpartition(-scores, min(pool, self.n_docs - 1))[:pool]
        order = order[np.argsort(-scores[order], kind="stable")]
        out: list[tuple[str, float]] = []
        for d in order:
            doc_id = self.doc_ids[int(d)]
            if scores[d] <= 0.0:
                continue
            if exclude and doc_id in exclude:
                continue
            out.append((doc_id, float(scores[d])))
            if len(out) >= pool:
                break
        return out

    def doc_ids_for(self, ranked: list[tuple[str, float]]) -> list[str]:
        return [d for d, _ in ranked]


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
