"""BEIR adapter — real retrieval, different benchmark.

Unlike BIRCO, BEIR keeps the retrieval step: you get a corpus and must search it.
That is what makes BEIR/SciFact the condition that separates the two variables the
BRIGHT runs confounded — here there IS a retriever to search, but there is NO
chunking (each SciFact document is a whole abstract), so the "prefer another chunk
of the answering document" lever has nothing to act on.

Every BEIR dataset ships the same three files (corpus.jsonl / queries.jsonl /
qrels/test.tsv), so one loader serves all of them.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
KNOWN = ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid")


@dataclass
class BeirTask:
    name: str
    #: doc id -> "title text" (BEIR's own concatenation order)
    corpus: dict[str, str]
    #: query id -> query text
    queries: dict[str, str]
    #: query id -> set of relevant ids (only the test split)
    gold: dict[str, set[str]]

    def __len__(self) -> int:
        return len(self.queries)


def _ensure(dataset: str, root: pathlib.Path) -> pathlib.Path:
    if (root / "corpus.jsonl").exists():
        return root
    import httpx, io, zipfile, shutil

    url = f"{BASE_URL}/{dataset}.zip"
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_bytes():
            buf.write(chunk)
    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = root.parent / f"_{dataset}_zip"
    zipfile.ZipFile(buf).extractall(tmp)
    src = tmp / dataset
    root.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, root / f.name)
        else:
            shutil.copytree(f, root / f.name, dirs_exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return root


def load_task(dataset: str = "scifact", root: pathlib.Path | None = None) -> BeirTask:
    root = _ensure(dataset, root or pathlib.Path("data/beir") / dataset)

    corpus: dict[str, str] = {}
    for line in (root / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        title = (d.get("title") or "").strip()
        text = (d.get("text") or "").strip()
        # BEIR's GenericDataLoader joins title and text with a single space; keeping the
        # same order keeps the BM25 baseline comparable to published BEIR numbers.
        corpus[str(d["_id"])] = f"{title} {text}".strip() if title else text

    all_queries: dict[str, str] = {}
    for line in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        all_queries[str(d["_id"])] = str(d.get("text") or "")

    gold: dict[str, set[str]] = {}
    qrels_path = root / "qrels" / "test.tsv"
    lines = qrels_path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        qid, cid, score = parts[0], parts[1], float(parts[2])
        if score <= 0:
            continue
        gold.setdefault(qid, set()).add(cid)

    queries = {q: all_queries[q] for q in gold if q in all_queries}
    return BeirTask(name=dataset, corpus=corpus, queries=queries, gold=gold)


def describe(t: BeirTask) -> str:
    counts = [len(v) for v in t.gold.values()]
    lens = sorted(len(v) for v in t.corpus.values())
    return (
        f"BEIR/{t.name}: {len(t.queries)} test queries | corpus {len(t.corpus)} docs | "
        f"doc chars median {lens[len(lens)//2]} | gold/query mean {sum(counts)/len(counts):.2f} "
        f"(max {max(counts)})"
    )
