"""Measure the thinking-mode switch against the live endpoint.

Qwen3.5 emits a long  thinking trace by default. We need to know (a) whether the
OpenAI SDK's ``extra_body.chat_template_kwargs`` actually turns it off against
this vLLM build, and (b) how much wall-clock and tokens it saves. Guessing here
would cost hours of experiment time later.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from openai import OpenAI  # noqa: E402

from bright_harness.provider import task_provider_summary  # noqa: E402

CFG = task_provider_summary()
client = OpenAI(
    base_url=CFG["base_url"],
    api_key="EMPTY",
    timeout=300,
)

CASES = [
    ("thinking ON  (default)", {}),
    ("thinking OFF (extra_body)", {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
]

print(f"endpoint: {CFG['base_url']}  model: {CFG['model']}\n")
for label, extra in CASES:
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=CFG["model"],
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=4096,
            temperature=0,
            **extra,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: FAILED {type(exc).__name__}: {str(exc)[:200]}\n")
        continue
    dt = time.time() - t0
    ch = r.choices[0]
    msg = ch.message
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
    print(f"{label}")
    print(f"   elapsed            = {dt:.1f}s")
    print(f"   finish_reason      = {ch.finish_reason}")
    print(f"   completion_tokens  = {r.usage.completion_tokens}")
    print(f"   content            = {(msg.content or '')[:160]!r}")
    print(f"   reasoning present  = {bool(reasoning)}  len={len(reasoning)}")
    print()

print("=== tool-calling under thinking OFF ===")
payload = {
    "model": CFG["model"],
    "messages": [
        {"role": "system", "content": "You are a retrieval agent. Use the Retrieve tool when asked to find passages."},
        {"role": "user", "content": "Find passages about why insects are attracted to light."},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "Retrieve",
                "description": "Search the domain corpus and return the top-ranked passages.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                    "required": ["query"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "max_tokens": 512,
    "temperature": 0,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}
t0 = time.time()
r = client.chat.completions.create(**payload)
dt = time.time() - t0
ch = r.choices[0]
calls = getattr(ch.message, "tool_calls", None) or []
print(f"   elapsed={dt:.1f}s finish={ch.finish_reason} tokens={r.usage.completion_tokens}")
print(f"   tool_calls={len(calls)} names={[c.function.name for c in calls]}")
if calls:
    print(f"   args={calls[0].function.arguments}")
