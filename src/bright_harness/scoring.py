"""Parse the task agent's ranking and score it with deterministic nDCG@10.

Scoring is deliberately outside the harness: the harness never learns whether it
was right, and never sees a gold id. The only thing that crosses the boundary is
the agent's final text.

Parsing is defensive because a small task agent will not always obey the output
contract. A malformed answer scores 0 for that query, which is a real failure
mode the ``prompt`` and ``processor`` evolution buckets can legitimately fix.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from bright.metrics import ndcg_at_k, recall_at_k

__all__ = ["parse_ranked_ids", "score_output", "ScoreResult"]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_QUOTED_RE = re.compile(r"[\"']([^\"'\n]{1,300})[\"']")


@dataclass
class ScoreResult:
    ranked: list[str]
    ndcg10: float
    recall10: float
    parse_ok: bool
    n_gold: int


def _clean_json_candidates(text: str):
    for m in _FENCE_RE.finditer(text):
        yield m.group(1).strip()
    stripped = text.strip()
    if stripped:
        yield stripped


def _try_json_array(text: str) -> list[str] | None:
    for chunk in _clean_json_candidates(text):
        first, last = chunk.find("["), chunk.rfind("]")
        if first == -1 or last == -1 or last < first:
            continue
        try:
            data = json.loads(chunk[first : last + 1])
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, list):
            out = [str(x).strip() for x in data if isinstance(x, (str, int, float))]
            if out:
                return out
    return None


def parse_ranked_ids(text: str, *, valid_ids: set[str] | None = None, k: int = 10) -> list[str]:
    """Extract a ranked id list from the agent's final message.

    Resolution order: a JSON array (possibly fenced) -> ids that appear in the
    text and are known corpus ids, in order of appearance -> bare quoted
    tokens. Duplicates are dropped, first occurrence wins.
    """
    if not text:
        return []
    raw = _try_json_array(text)
    if raw is None and valid_ids:
        # Order by position in the text so the agent's implied ranking survives.
        hits: list[tuple[int, str]] = []
        for doc_id in valid_ids:
            pos = text.find(doc_id)
            if pos != -1:
                hits.append((pos, doc_id))
        hits.sort()
        raw = [d for _, d in hits]
    if raw is None:
        raw = [m.group(1).strip() for m in _QUOTED_RE.finditer(text)]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        item = item.strip().strip(",")
        if not item or item in seen:
            continue
        if valid_ids is not None and item not in valid_ids:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= k:
            break
    return out


def score_output(
    text: str,
    gold: set[str],
    *,
    valid_ids: set[str] | None = None,
    k: int = 10,
) -> ScoreResult:
    ranked = parse_ranked_ids(text, valid_ids=valid_ids, k=k)
    return ScoreResult(
        ranked=ranked,
        ndcg10=ndcg_at_k(ranked, gold, k),
        recall10=recall_at_k(ranked, gold, k),
        parse_ok=bool(ranked),
        n_gold=len(gold),
    )
