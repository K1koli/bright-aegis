"""Unit test for the externalised-tool-payload fix.

The journal stores a large tool result in ``tool_results/<id>_raw.txt`` and leaves
``message.content`` empty, with ``meta.content_size`` recording the real size.
Before the fix, ``extract_trace_facts`` classified this as ``empty``/``len 0``,
which made the Digester report a phantom "empty tool return" failure mode and sent
the Evolver off to re-derive evidence (200 steps / $41 / 0 candidates). After the
fix it must be ``externalized`` with the real length.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

from harnessx.aegis.stages.trace_facts import extract_trace_facts  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


def main() -> int:
    tmp = ROOT / ".tmp" / "tf_test"
    tmp.mkdir(parents=True, exist_ok=True)

    def base(meta: dict) -> list[dict]:
        return [
            {
                "type": "raw_assistant",
                "step": 1,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "Retrieve", "input": {"query": "x", "k": 10}}],
                },
            },
            {
                "type": "raw_tool",
                "step": 2,
                "message": {"role": "tool", "tool_call_id": "c1", "name": "Retrieve", "content": ""},
                "meta": meta,
            },
            {"type": "episode_end", "step": 3, "exit_reason": "done", "total_steps": 1, "passed": False},
        ]

    print("1. externalised (empty content + meta.content_size) -> externalized")
    facts = extract_trace_facts("q1", [_write(tmp / "ext.jsonl", base({"content_ref": "tool_results/x_raw.txt", "content_size": 6067}))])
    c = facts.tool_calls[0]
    check("return_type == externalized", c.return_type == "externalized", c.return_type)
    check("return_len preserved", c.return_len == 6067, str(c.return_len))

    print("\n2. genuinely empty (no meta) -> still empty")
    facts = extract_trace_facts("q2", [_write(tmp / "empty.jsonl", base({}))])
    c = facts.tool_calls[0]
    check("return_type == empty", c.return_type == "empty", c.return_type)
    check("return_len == 0", c.return_len == 0, str(c.return_len))

    print("\n3. in-band text (no externalisation) -> text, unaffected")
    rows = base({})
    rows[1]["message"]["content"] = "1. id=abc\n   passage text"
    facts = extract_trace_facts("q3", [_write(tmp / "text.jsonl", rows)])
    c = facts.tool_calls[0]
    check("return_type == text", c.return_type == "text", c.return_type)
    check("return_len == real length", c.return_len > 0, str(c.return_len))

    print("\n--- summary ---")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all trace_facts checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
