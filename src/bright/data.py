"""BRIGHT data loading and deterministic splits.

Layout on disk (downloaded by ``src/tools/fetch_bright.py``)::

    data/bright/raw/documents/<domain>-00000-of-00001.parquet   # id, content
    data/bright/raw/examples/<domain>-00000-of-00001.parquet    # query, gold_ids, ...

Splits are deterministic functions of (domain, seed, sizes), so every run of
the experiment sees the identical Biology-dev / Biology-heldout partition.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "bright" / "raw"

# Fixed by the experiment protocol: evolve on Biology-dev, report on
# Biology-heldout / Economics / Robotics. Economics additionally hosts a
# reverse-transfer run (evolve on Economics-dev -> test on Biology-heldout).
DOMAINS = ("biology", "economics", "robotics")


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    gold_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...] = ()
    domain: str = ""
    reasoning: str = ""

    def gold(self) -> set[str]:
        return set(self.gold_ids)


@dataclass
class Split:
    domain: str
    dev: list[Query] = field(default_factory=list)
    heldout: list[Query] = field(default_factory=list)

    def all_queries(self) -> list[Query]:
        return self.dev + self.heldout


def _split_seed(domain: str, seed: int) -> int:
    """Stable per-domain seed so one domain's split never shifts another's."""
    h = hashlib.sha256(f"{domain}|{seed}".encode()).hexdigest()[:16]
    return int(h, 16)


def load_queries(domain: str) -> list[Query]:
    path = RAW / "examples" / f"{domain}-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run src/tools/fetch_bright.py {domain}")
    df = pd.read_parquet(path)
    out: list[Query] = []
    for row in df.itertuples(index=False):
        gold = [g for g in list(row.gold_ids) if isinstance(g, str)]
        excl = [e for e in list(row.excluded_ids) if isinstance(e, str) and e != "N/A"]
        out.append(
            Query(
                qid=str(row.id),
                text=str(row.query),
                gold_ids=tuple(gold),
                excluded_ids=tuple(excl),
                domain=domain,
                reasoning=str(row.reasoning or ""),
            )
        )
    return out


def load_corpus(domain: str) -> tuple[list[str], list[str]]:
    """Return ``(doc_ids, contents)`` in parquet row order."""
    path = RAW / "documents" / f"{domain}-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run src/tools/fetch_bright.py {domain}")
    df = pd.read_parquet(path)
    return [str(x) for x in df["id"]], [str(x) for x in df["content"]]


def make_split(domain: str, *, n_dev: int = 50, seed: int = 20260101, allow_full: bool = False) -> Split:
    """Deterministic shuffle-then-cut. ``n_dev`` dev queries, rest held out.

    ``allow_full`` lets ``n_dev`` cover every query (empty held-out). The default
    reserves at least one query so a held-out set always exists; the paper-faithful
    full-task-set run needs ``allow_full=True`` because it deliberately evolves and
    reports on the same task set (HarnessX §6.1: "no held-out generalization is
    evaluated").
    """
    queries = load_queries(domain)
    rng = random.Random(_split_seed(domain, seed))
    shuffled = list(queries)
    rng.shuffle(shuffled)
    cap = len(shuffled) if allow_full else max(0, len(shuffled) - 1)
    n_dev = min(n_dev, cap)
    return Split(domain=domain, dev=shuffled[:n_dev], heldout=shuffled[n_dev:])


def write_split(split: Split, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "domain": split.domain,
        "dev": [q.qid for q in split.dev],
        "heldout": [q.qid for q in split.heldout],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path: Path, domain: str) -> Split:
    """Reload an on-disk split (qids only) and re-attach query payloads."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_qid = {q.qid: q for q in load_queries(domain)}
    missing = [q for q in payload["dev"] + payload["heldout"] if q not in by_qid]
    if missing:
        raise KeyError(f"split references unknown qids: {missing[:5]}")
    return Split(
        domain=domain,
        dev=[by_qid[q] for q in payload["dev"]],
        heldout=[by_qid[q] for q in payload["heldout"]],
    )


def gold_doc_ids(domain: str) -> set[str]:
    """Union of all gold ids; used to assert the corpus actually contains them."""
    out: set[str] = set()
    for q in load_queries(domain):
        out.update(q.gold_ids)
    return out
