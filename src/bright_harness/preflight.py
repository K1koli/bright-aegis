"""Preflight: validate both model roles before spending any evolution budget.

Run this after filling in ``.env.local``::

    python src/bright_harness/preflight.py

Checks, in the order they can fail:

1. credentials are present and the task-agent endpoint answers a plain chat call;
2. **the task-agent endpoint supports OpenAI-style tool calling** -- without it
   H_0 can never call ``Retrieve``, so every query would score 0 and the whole
   experiment would silently measure obedience instead of retrieval;
3. a real H_0 run through HarnessX on a few BRIGHT queries: tool actually
   invoked, output parseable, nDCG computed, latency and token use reported;
4. the proposer endpoint answers, since AEGIS drives all four meta stages with it.

Exit code is 0 only when the task agent passes (1)-(3); the proposer (4) is
reported but does not gate a task-agent-only rerun.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from bright import data as bdata  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402
from bright_harness.harness import make_bright_harness, make_bright_task  # noqa: E402
from bright_harness.scoring import score_output  # noqa: E402
from bright_harness.tools import active_index  # noqa: E402

OK = "\u2713"
BAD = "\u2717"
WARN = "!"

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "Retrieve",
            "description": "Search the corpus and return the top-ranked passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "k": {"type": "integer", "description": "How many passages to return."},
                },
                "required": ["query"],
            },
        },
    }
]


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def report(label: str, ok: bool | None, detail: str = "") -> None:
    mark = OK if ok else (BAD if ok is False else WARN)
    print(f"  [{mark}] {label}{(' -- ' + detail) if detail else ''}")


def probe_chat(base_url: str, model: str, api_key: str, *, tools: bool, timeout: float = 60.0):
    """Raw OpenAI-SDK probe, independent of HarnessX."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)
    messages = [
        {"role": "system", "content": "You are a retrieval agent. Use the Retrieve tool when asked to find passages."},
        {"role": "user", "content": "Find passages about why insects are attracted to light."},
    ]
    kwargs = {"model": model, "messages": messages, "temperature": 0}
    if tools:
        kwargs["tools"] = TOOL_SCHEMA
        kwargs["tool_choice"] = "auto"
    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.time() - t0
    choice = resp.choices[0]
    msg = choice.message
    calls = getattr(msg, "tool_calls", None) or []
    return {
        "elapsed_s": elapsed,
        "finish_reason": choice.finish_reason,
        "content": (msg.content or "")[:200],
        "n_tool_calls": len(calls),
        "tool_names": [c.function.name for c in calls],
        "tool_args": [c.function.arguments for c in calls][:1],
        "usage": getattr(resp, "usage", None),
    }


async def run_h0(n_queries: int, *, domain: str = "biology") -> dict:
    from bright_harness.provider import make_task_model_config, task_provider_summary

    model_config = make_task_model_config()
    print(f"        task provider: {task_provider_summary()}")

    doc_ids, contents = bdata.load_corpus(domain)
    index = BM25Index.cached(domain, doc_ids, contents)
    valid_ids = set(doc_ids)

    run_dir = ROOT / ".tmp" / "preflight"
    cfg = make_bright_harness(tracer_dir=run_dir / "hx_runs", workspace_root=run_dir / "ws")
    harness = model_config.agentic(cfg)

    queries = bdata.make_split(domain, n_dev=50).dev[:n_queries]
    rows: list[dict] = []
    for q in queries:
        t0 = time.time()
        try:
            with active_index(index):
                result = await harness.run(make_bright_task(q.text, qid=q.qid, domain=domain))
            final = result.final_output or ""
            exit_reason = getattr(result, "exit_reason", "?")
            scored = score_output(final, q.gold(), valid_ids=valid_ids, k=10)
            rows.append(
                {
                    "qid": q.qid,
                    "ok": True,
                    "exit_reason": exit_reason,
                    "ndcg10": scored.ndcg10,
                    "n_ranked": len(scored.ranked),
                    "final_head": final[:160].replace("\n", " "),
                    "elapsed_s": time.time() - t0,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"qid": q.qid, "ok": False, "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.time() - t0})

    tool_calls = 0
    for p in sorted((run_dir / "hx_runs").rglob("*.jsonl")):
        try:
            tool_calls += p.read_text(encoding="utf-8", errors="replace").count('"Retrieve"')
        except OSError:
            pass
    return {"rows": rows, "tool_call_mentions": tool_calls, "run_dir": run_dir}


def main() -> int:
    print("=== BRIGHT / AEGIS preflight ===\n")

    print("0. credentials")
    task_url, task_model = env("BRIGHT_TASK_BASE_URL"), env("BRIGHT_TASK_MODEL")
    prop_url, prop_model = env("BRIGHT_PROPOSER_BASE_URL"), env("BRIGHT_PROPOSER_MODEL")
    report("BRIGHT_TASK_BASE_URL", bool(task_url), task_url or "missing")
    report("BRIGHT_TASK_MODEL", bool(task_model), task_model or "missing")
    report("BRIGHT_PROPOSER_BASE_URL", bool(prop_url), prop_url or "missing")
    report("BRIGHT_PROPOSER_MODEL", bool(prop_model), prop_model or "missing")
    report("BRIGHT_PROPOSER_API_KEY set", bool(env("BRIGHT_PROPOSER_API_KEY")), "value not printed")
    if not task_url or not task_model:
        print("\nFill in .env.local first (see .env.example). Aborting.")
        return 2

    task_ok = True

    print("\n1. task-agent endpoint: plain chat")
    try:
        r = probe_chat(task_url, task_model, env("BRIGHT_TASK_API_KEY"), tools=False)
        report("reachable", True, f"{r['elapsed_s']:.1f}s finish={r['finish_reason']}")
        print(f"        content: {r['content'][:120]!r}")
        if r["usage"]:
            print(f"        usage: {r['usage']}")
    except Exception as exc:  # noqa: BLE001
        report("reachable", False, f"{type(exc).__name__}: {exc}")
        return 1

    print("\n2. task-agent endpoint: OpenAI-style tool calling (H_0 depends on this)")
    try:
        r = probe_chat(task_url, task_model, env("BRIGHT_TASK_API_KEY"), tools=True)
        got = r["n_tool_calls"] > 0 and "Retrieve" in r["tool_names"]
        report("emits a Retrieve tool_call", got, f"{r['n_tool_calls']} call(s) {r['tool_names']}")
        if got:
            print(f"        args: {r['tool_args'][0][:160] if r['tool_args'] else ''}")
        else:
            report("fallback path", None, "harness may still work if the model writes JSON ids directly, but expect low scores")
            task_ok = False
    except Exception as exc:  # noqa: BLE001
        report("tool calling", False, f"{type(exc).__name__}: {exc}")
        task_ok = False

    print("\n3. real H_0 run through HarnessX (3 biology-dev queries)")
    try:
        out = asyncio.run(run_h0(3))
        rows = out["rows"]
        good = [r for r in rows if r.get("ok")]
        report("all queries completed", len(good) == len(rows), f"{len(good)}/{len(rows)}")
        report(
            "tool was actually invoked",
            out["tool_call_mentions"] > 0,
            f"{out['tool_call_mentions']} mention(s) of Retrieve in traces",
        )
        parseable = [r for r in good if r.get("n_ranked", 0) > 0]
        report("outputs parsed to a ranked list", len(parseable) == len(good) and bool(good), f"{len(parseable)}/{len(good)}")
        for r in rows:
            if r.get("ok"):
                print(
                    f"        q{r['qid']}: nDCG@10={r['ndcg10']:.4f} ranked={r['n_ranked']} "
                    f"exit={r['exit_reason']} {r['elapsed_s']:.1f}s"
                )
            else:
                print(f"        q{r['qid']}: ERROR {r['error'][:140]}")
        if good:
            print(f"        mean nDCG@10 over {len(good)} queries = {statistics.fmean([r['ndcg10'] for r in good]):.4f}")
        if not parseable:
            task_ok = False
    except Exception as exc:  # noqa: BLE001
        report("H_0 run", False, f"{type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        task_ok = False

    print("\n4. proposer endpoint (drives Digester/Planner/Evolver/Critic)")
    if not prop_url or not prop_model:
        report("proposer configured", None, "skipped -- not configured")
    else:
        try:
            r = probe_chat(prop_url, prop_model, env("BRIGHT_PROPOSER_API_KEY"), tools=False)
            report("reachable", True, f"{r['elapsed_s']:.1f}s finish={r['finish_reason']}")
            print(f"        content: {r['content'][:120]!r}")
        except Exception as exc:  # noqa: BLE001
            report("reachable", False, f"{type(exc).__name__}: {exc}")

    print("\n=== verdict ===")
    if task_ok:
        print(" task agent READY -- H_0 can retrieve, parse, and score.")
        return 0
    print(" task agent NOT ready -- fix the items marked above before evolving.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
