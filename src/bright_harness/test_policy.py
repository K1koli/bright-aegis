"""Tests for the retrieval-policy layer — i.e. that the ``config`` bucket is real.

The AEGIS Planner read H_0's source and concluded it was "a zero-degrees-of-
freedom wrapper around BM25", because ``k`` was a tool argument the model chose,
``max_steps`` was hard-coded in the driver, and nothing the Evolver could edit in
an applied YAML changed retrieval behaviour. This checks each knob is now a live
config field that survives the deterministic gate's canonicalize path, and that
the Retrieve tool actually obeys it.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from bright import data as bdata  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402
from bright_harness import policy  # noqa: E402
from bright_harness.harness import make_bright_harness, make_h0_tool_registry  # noqa: E402
from bright_harness.processors import RetrievalPolicyProcessor  # noqa: E402
from bright_harness.round_driver import max_steps_from_config, policy_kwargs_from_config  # noqa: E402
from bright_harness.tools import active_index  # noqa: E402
from harnessx.core.harness import HarnessConfig  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


async def retrieve(registry, query: str, k: int = 10) -> str:
    res = await registry.execute("Retrieve", {"query": query, "k": k})
    return res.output or ""


async def main() -> int:
    out = ROOT / ".tmp" / "policy_test"
    out.mkdir(parents=True, exist_ok=True)

    print("1. H_0 config carries the knobs")
    cfg = make_bright_harness(tracer_dir=out / "hx", workspace_root=out / "ws")
    kw = policy_kwargs_from_config(cfg)
    check("policy processor present in H_0", bool(kw), json.dumps(kw))
    check(
        "all five knobs exposed",
        set(kw) == {"retrieve_k", "snippet_chars", "max_retrieve_calls", "max_steps", "dedupe_by_document"},
        str(sorted(kw)),
    )
    check("defaults reproduce the old behaviour", kw.get("retrieve_k") == 10 and kw.get("max_steps") == 4, str(kw))
    check("driver reads max_steps from the config", max_steps_from_config(cfg) == 4)

    print("\n2. a config edit survives the gate's canonicalize path")
    base = out / "h0.yaml"
    cfg.to_yaml_file(base)
    text = base.read_text(encoding="utf-8")
    check("yaml mentions the policy processor", "RetrievalPolicyProcessor" in text)
    edited_path = out / "h0_edited.yaml"
    edited = (
        text.replace("retrieve_k: 10", "retrieve_k: 40")
        .replace("max_steps: 4", "max_steps: 8")
        .replace("dedupe_by_document: false", "dedupe_by_document: true")
    )
    check("edit actually changed the text", edited != text)
    edited_path.write_text(edited, encoding="utf-8")
    reloaded = HarnessConfig.from_yaml_file(edited_path)
    reloaded.canonicalize()  # exactly what aegis/gates/canonicalize.py does
    kw2 = policy_kwargs_from_config(reloaded)
    check("edited values survive reload+canonicalize", kw2.get("retrieve_k") == 40 and kw2.get("dedupe_by_document") is True, json.dumps(kw2))
    check("driver picks up the edited step budget", max_steps_from_config(reloaded) == 8)

    print("\n3. the Retrieve tool obeys the policy")
    doc_ids, contents = bdata.load_corpus("biology")
    index = BM25Index.cached("biology", doc_ids, contents)
    registry = make_h0_tool_registry()
    query = "why are insects attracted to light"

    policy.set_policy(retrieve_k=10, snippet_chars=600, max_retrieve_calls=1, max_steps=4, dedupe_by_document=False)
    with active_index(index):
        base_out = await retrieve(registry, query, k=10)
        n_base = base_out.count("id=")
        check("default returns 10 passages", n_base == 10, f"{n_base}")
        check("call budget is reported", "remaining: 0" in base_out, "budget line present")

    # Second call in the same episode must be refused.
    with active_index(index):
        second = await retrieve(registry, query, k=10)
    check("second call refused by the budget", "budget exhausted" in second.lower(), second[:80])

    # retrieve_k is authoritative: the model asking for 10 gets at most 5.
    policy.reset_episode()
    policy.set_policy(retrieve_k=5)
    with active_index(index):
        capped = await retrieve(registry, query, k=10)
    n_capped = capped.count("id=")
    check("model's k is capped by the harness policy", n_capped == 5, f"{n_capped} passages")

    # snippet_chars is honoured.
    policy.reset_episode()
    policy.set_policy(snippet_chars=80)
    with active_index(index):
        short = await retrieve(registry, query, k=3)
    longest = max((len(line.strip()) for line in short.splitlines() if line.startswith("   ")), default=0)
    check("snippet_chars truncates passages", longest <= 80, f"longest line {longest}")

    # dedupe_by_document collapses chunk-level duplicates.
    policy.reset_episode()
    policy.set_policy(snippet_chars=600, retrieve_k=10, dedupe_by_document=True)
    with active_index(index):
        deduped = await retrieve(registry, query, k=10)
    ids = [ln.split("id=", 1)[1].strip() for ln in deduped.splitlines() if "id=" in ln]
    keys = [policy.document_key(i) for i in ids]
    check("dedupe_by_document removes duplicate sources", len(set(keys)) == len(keys), f"{len(ids)} ids, {len(set(keys))} sources")

    print("\n4. processor exposes its policy")
    proc = RetrievalPolicyProcessor(retrieve_k=7, max_retrieve_calls=3)
    p = proc.as_policy()
    check("processor.as_policy() reflects kwargs", p["retrieve_k"] == 7 and p["max_retrieve_calls"] == 3, json.dumps(p))

    policy.set_policy()  # restore defaults for later imports
    print("\n--- summary ---")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all policy checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
