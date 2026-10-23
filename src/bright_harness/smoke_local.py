"""Local smoke test: everything except the LLM.

Verifies, with no model and no network:

1. the cached BM25 index loads and the ``Retrieve`` tool returns the documented
   shape;
2. H_0's ``HarnessConfig`` round-trips through ``to_yaml_file`` ->
   ``from_yaml_file`` -> ``canonicalize()`` -- the exact path AEGIS's
   deterministic gate uses, so a config that fails here can never ship;
3. the scoring path is self-consistent: feeding the agent's exact expected output
   format reproduces the BM25 ranking's nDCG@10 bit-for-bit.

(3) also yields the *echo ceiling*: what H_0 scores if the task agent perfectly
copies the tool's ranking. The real H_0 will sit below it by exactly the
obedience/parse failures, which is a quantity worth separating out.
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from bright import data as bdata  # noqa: E402
from bright.metrics import ndcg_at_k  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402
from bright_harness.harness import (  # noqa: E402
    make_bright_harness,
    make_bright_task,
    make_h0_tool_registry,
)
from bright_harness.scoring import parse_ranked_ids, score_output  # noqa: E402
from bright_harness.tools import active_index  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


async def main() -> int:
    domain = "biology"
    doc_ids, contents = bdata.load_corpus(domain)
    index = BM25Index.cached(domain, doc_ids, contents)
    valid_ids = set(doc_ids)

    print(f"index: {index.n_docs} docs, {len(index.terms)} terms, avgdl={index.avgdl:.1f}")

    # ---- 1. Retrieve tool ------------------------------------------------
    print("\n1. Retrieve tool")
    registry = make_h0_tool_registry()
    query = "why are insects attracted to light"
    with active_index(index):
        res = await registry.execute("Retrieve", {"query": query, "k": 10})
    out = res.output or ""
    check("tool returns text", bool(out), f"{len(out)} chars")
    check("no tool error", not res.error, str(res.error))
    check("output has 'id=' entries", out.count("id=") >= 8, f"{out.count('id=')} entries")
    first_id = out.split("id=", 1)[1].split("\n", 1)[0].strip() if "id=" in out else ""
    check("first id is a real corpus id", first_id in valid_ids, first_id)
    check("tool text lookup works", bool(index.text_of(first_id)))

    # ---- 2. HarnessConfig round-trip (the gate's canonicalize path) ------
    print("\n2. HarnessConfig round-trip (deterministic-gate path)")
    from harnessx.core.harness import HarnessConfig

    out_dir = ROOT / ".tmp" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "h0_config.yaml"
    cfg = make_bright_harness(tracer_dir=out_dir / "hx_runs", workspace_root=out_dir / "ws")
    cfg.to_yaml_file(cfg_path)
    check("config written", cfg_path.exists(), str(cfg_path.relative_to(ROOT)))
    reloaded = HarnessConfig.from_yaml_file(cfg_path)
    check("config reloads", reloaded is not None)
    canon = reloaded.canonicalize()
    check("config canonicalizes", canon is not None)
    reloaded.to_yaml_file(out_dir / "h0_config_roundtrip.yaml")

    def _norm(p: Path) -> str:
        return "\n".join(line.rstrip() for line in p.read_text(encoding="utf-8").splitlines()).strip()

    check("round-trip is byte-stable", _norm(cfg_path) == _norm(out_dir / "h0_config_roundtrip.yaml"))

    # ---- 3. Scoring path + echo ceiling ---------------------------------
    print("\n3. Scoring path and echo ceiling")
    split = bdata.make_split(domain, n_dev=50)
    echo_scores: list[float] = []
    direct_scores: list[float] = []
    parse_failures = 0
    for q in split.dev:
        ranked = [d for d, _ in index.search(q.text, k=10)]
        direct_scores.append(ndcg_at_k(ranked, q.gold(), 10))
        # Exact shape the prompt asks for:
        answer = json.dumps(ranked)
        scored = score_output(answer, q.gold(), valid_ids=valid_ids, k=10)
        if not scored.parse_ok:
            parse_failures += 1
        echo_scores.append(scored.ndcg10)
        check_ranked = parse_ranked_ids(answer, valid_ids=valid_ids, k=10)
        if check_ranked != ranked:
            FAILURES.append(f"parse mismatch on q{q.qid}")

    check("no parse failures on well-formed input", parse_failures == 0, f"{parse_failures} failures")
    check(
        "echo scores == direct BM25 scores",
        all(abs(a - b) < 1e-12 for a, b in zip(echo_scores, direct_scores)),
    )

    # Robustness of the parser against messy-but-plausible agent output.
    q0 = split.dev[0]
    r0 = [d for d, _ in index.search(q0.text, k=10)]
    messy = [
        ("fenced json", "```json\n" + json.dumps(r0[:5]) + "\n```"),
        ("numbered list", "\n".join(f"{i}. id={d}" for i, d in enumerate(r0[:5], 1))),
        ("prose + ids", "The best passages are " + ", ".join(r0[:3]) + "."),
        ("empty", ""),
    ]
    for label, text in messy:
        ids = parse_ranked_ids(text, valid_ids=valid_ids, k=10)
        print(f"      parser[{label}] -> {len(ids)} ids, first={ids[0] if ids else None}")

    print("\n--- summary ---")
    print(f"  BM25 nDCG@10 (biology-dev, n={len(direct_scores)}) = {statistics.fmean(direct_scores):.4f}")
    print(f"  echo ceiling (identical)                          = {statistics.fmean(echo_scores):.4f}")
    print(f"  task object: {make_bright_task('x', qid='0', domain=domain).metadata}")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("\nall local smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
