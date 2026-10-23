"""Load a BIRCO task from its parquet files into plain dicts.

BIRCO's qrels file lists **one row per candidate in the pool** (score 0.0 for the
non-relevant ones), so the pool itself is defined by the qrels: every ``corpus-id``
that appears for a query is in that query's candidate pool. Verified against
BIRCO/ArguAna: 100 queries, 50 candidates each, exactly 1 relevant, binary scores.

Files are downloaded once from the Hugging Face mirror (parquet, no ``datasets``
dependency needed) and cached under ``data/birco/<task>/``.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

PARQUET = {"corpus": "corpus.parquet", "queries": "queries.parquet", "qrels": "default.parquet"}

HF_TASKS = {
    "arguana": "mteb/BIRCO-Arguana-Test",
    "wtb": "mteb/BIRCO-WTB-Test",
}


@dataclass
class BircoTask:
    name: str
    #: corpus id -> passage text (only the passages that appear in some pool)
    corpus: dict[str, str]
    #: query id -> query text
    queries: dict[str, str]
    #: query id -> candidate ids, in the order the qrels file lists them
    pools: dict[str, list[str]]
    #: query id -> set of relevant ids
    gold: dict[str, set[str]]

    def __len__(self) -> int:
        return len(self.queries)


def _download(task: str, root: pathlib.Path) -> None:
    if task not in HF_TASKS:
        raise SystemExit(f"unknown BIRCO task {task!r}; known: {sorted(HF_TASKS)}")
    import httpx

    root.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/{HF_TASKS[task]}/resolve/main"
    for part, fname in PARQUET.items():
        dst = root / fname
        if dst.exists():
            continue
        url = f"{base}/data/{part if part != 'qrels' else 'default'}/test-00000.parquet"
        with httpx.stream("GET", url, follow_redirects=True, timeout=180) as r:
            r.raise_for_status()
            with dst.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)


def load_task(task: str = "arguana", root: pathlib.Path | None = None) -> BircoTask:
    import pyarrow.parquet as pq

    root = root or pathlib.Path("data/birco") / task
    _download(task, root)

    corpus_df = pq.read_table(root / PARQUET["corpus"]).to_pandas()
    queries_df = pq.read_table(root / PARQUET["queries"]).to_pandas()
    qrels_df = pq.read_table(root / PARQUET["qrels"]).to_pandas()

    corpus = {str(r["_id"]): str(r["text"]) for _, r in corpus_df.iterrows()}
    queries = {str(r["_id"]): str(r["text"]) for _, r in queries_df.iterrows()}

    pools: dict[str, list[str]] = {}
    gold: dict[str, set[str]] = {}
    for qid, grp in qrels_df.groupby("query-id", sort=False):
        qid = str(qid)
        pools[qid] = [str(c) for c in grp["corpus-id"]]
        gold[qid] = {str(c) for c, s in zip(grp["corpus-id"], grp["score"]) if float(s) > 0}

    return BircoTask(name=task, corpus=corpus, queries=queries, pools=pools, gold=gold)


def describe(t: BircoTask) -> str:
    pool_sizes = [len(v) for v in t.pools.values()]
    gold_counts = [len(v) for v in t.gold.values()]
    return (
        f"BIRCO/{t.name}: {len(t.queries)} queries | corpus {len(t.corpus)} passages | "
        f"pool {min(pool_sizes)}-{max(pool_sizes)} (mean {sum(pool_sizes)/len(pool_sizes):.1f}) | "
        f"gold/query {min(gold_counts)}-{max(gold_counts)}"
    )
