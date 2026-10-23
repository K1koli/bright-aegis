"""Report the REAL cost of a run, and the prompt-cache hit rate.

Why this exists: HarnessX prices usage with Claude Sonnet's table
(``harnessx/core/runloop.py::_estimate_cost`` = $3/M input, $15/M output) and reports
that as ``cost=$...`` on every stage line. Against a DeepSeek endpoint the real bill is
one to two orders of magnitude smaller, so the stage logs are badly misleading. This
script sums the per-call usage the journal already records and prices it with the
endpoint's actual rates.

    python src/bright_harness/cost_report.py --run-root runs/fix-T1

Runs recorded before the ``openai_provider`` cache patch carry no cache figures; then
the cache-hit line reports "not recorded" and the estimate uses the all-miss bound.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

# DeepSeek Flash list prices, USD per 1M tokens (api-docs.deepseek.com/quick_start/pricing).
# Off-peak is half of peak; peak is 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri.
DEEPSEEK_FLASH = {
    "in_hit_offpeak": 0.003,
    "in_hit_peak": 0.006,
    "in_miss_offpeak": 0.15,
    "in_miss_peak": 0.30,
    "out_offpeak": 0.60,
    "out_peak": 1.20,
}
SONNET_ESTIMATE = {"in": 3.0, "out": 15.0}

# Measured output share of total tokens on this workload (150,609 out of 9,760,342 on
# the first complete round). Used only for stages whose input/output split is not
# journaled, i.e. Stage P -- ``build_digester_harness`` wires no tracer, so a digester's
# per-call usage reaches no journal at all and only its total token count is recoverable
# from the ``tokens=`` field on its stage line.
OUTPUT_SHARE = 0.0155
TOKENS_RE = re.compile(r"tokens=(\d+)")


def _stage_tokens_from_costs(run_root: Path) -> dict[str, tuple[int, int]]:
    """(total_tokens, n_calls) per stage, recovered from stage_costs.jsonl."""
    path = run_root / "stage_costs.jsonl"
    out: dict[str, tuple[int, int]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line).get("message") or ""
        except json.JSONDecodeError:
            continue
        m = TOKENS_RE.search(msg)
        if not m:
            continue
        head = msg.split(":", 1)[0].strip()
        key = head.split()[0] if head else "?"
        tot, calls = out.get(key, (0, 0))
        out[key] = (tot + int(m.group(1)), calls + 1)
    return out


def _iter_usage_records(run_root: Path):
    """Yield every ``usage`` dict the journals recorded, once per model call."""
    seen: set[tuple] = set()
    for path in sorted(run_root.rglob("meta_sessions/**/*.jsonl")):
        if path.name.endswith("_trace.jsonl") or path.name.endswith("_state.json"):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or '"usage"' not in line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # The journal nests per-call usage under meta.usage (journal.py:877-886);
            # older/other writers may put it at the top level or on the message.
            usage = None
            meta = row.get("meta")
            if isinstance(meta, dict):
                usage = meta.get("usage")
            if not isinstance(usage, dict):
                usage = row.get("usage")
            if not isinstance(usage, dict):
                msg = row.get("message")
                usage = msg.get("usage") if isinstance(msg, dict) else None
            if not isinstance(usage, dict) or "input_tokens" not in usage:
                continue
            key = (
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                usage.get("cache_read_tokens"),
                row.get("step"),
                str(path),
            )
            if key in seen:
                continue
            seen.add(key)
            yield usage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--peak", action="store_true", help="price at peak instead of off-peak")
    ap.add_argument(
        "--cache-hit-rate",
        type=float,
        default=None,
        help="force a hypothetical cache-hit fraction in [0,1] instead of using recorded data",
    )
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    if not run_root.exists():
        print(f"run root not found: {run_root}")
        return 2

    tin = tout = thit = 0
    calls = 0
    for u in _iter_usage_records(run_root):
        tin += int(u.get("input_tokens") or 0)
        tout += int(u.get("output_tokens") or 0)
        thit += int(u.get("cache_read_tokens") or 0)
        calls += 1

    if calls == 0:
        print(f"no usage records found under {run_root}/**/meta_sessions/")
        print("(the engine writes them only for AEGIS meta stages -- task-agent runs are journaled separately)")
        return 1

    suffix = "peak" if args.peak else "offpeak"
    p_in_hit = DEEPSEEK_FLASH[f"in_hit_{suffix}"]
    p_in_miss = DEEPSEEK_FLASH[f"in_miss_{suffix}"]
    p_out = DEEPSEEK_FLASH[f"out_{suffix}"]

    print(f"=== token totals ({calls} model calls) ===")
    print(f"  input tokens       : {tin:,}")
    print(f"  output tokens      : {tout:,}")
    if thit:
        rate = thit / tin if tin else 0.0
        print(f"  cache-READ tokens  : {thit:,}   ({rate:.1%} of input)")
        print(f"  cache-MISS tokens  : {tin - thit:,}")
    else:
        print("  cache-READ tokens  : not recorded (run predates the provider cache patch)")

    eff_rate = args.cache_hit_rate
    if eff_rate is None:
        eff_rate = (thit / tin) if (thit and tin) else 0.0
    hit = int(tin * eff_rate)
    miss = tin - hit

    real = (hit * p_in_hit + miss * p_in_miss + tout * p_out) / 1_000_000
    all_miss = (tin * p_in_miss + tout * p_out) / 1_000_000
    all_hit = (tin * p_in_hit + tout * p_out) / 1_000_000
    sonnet = (tin * SONNET_ESTIMATE["in"] + tout * SONNET_ESTIMATE["out"]) / 1_000_000

    print()
    print(f"=== real DeepSeek Flash cost ({'peak' if args.peak else 'off-peak'}) ===")
    if args.cache_hit_rate is not None:
        print(f"  at assumed cache-hit rate {eff_rate:.1%}     : ${real:.4f}")
        print(f"  bounds                              : ${all_hit:.4f} .. ${all_miss:.4f}")
    elif thit:
        print(f"  measured                            : ${real:.4f}")
        print(f"  worst case (no cache at all)        : ${all_miss:.4f}")
    else:
        print(f"  worst case (ALL input a cache miss) : ${all_miss:.4f}")
        print(f"  best case  (ALL input a cache hit)  : ${all_hit:.4f}")
        print(f"  a 80% cache-hit rate would give     : ${(0.8 * tin * p_in_hit + 0.2 * tin * p_in_miss + tout * p_out) / 1e6:.4f}")

    print()
    print("=== what HarnessX reported (Claude Sonnet prices) ===")
    print(f"  stage logs would say                : ${sonnet:.4f}")
    ratio = (sonnet / all_miss) if all_miss else 0.0
    print(f"  overstatement vs worst-case real    : {ratio:.1f}x")

    sc = run_root / "stage_costs.jsonl"
    if sc.exists():
        # ---- Stage P is only recoverable from the logged token totals ----------
        # build_digester_harness wires no tracer, so a digester's per-call usage is
        # journaled nowhere; its aggregate token count survives only in the ``tokens=``
        # field my digester log line writes. Skipping it would silently drop 77% of a
        # full-biology round.
        by_stage = _stage_tokens_from_costs(run_root)
        sp_tokens, sp_calls = by_stage.get("Digester", (0, 0))
        if sp_tokens:
            sp_out = int(sp_tokens * OUTPUT_SHARE)
            sp_in = sp_tokens - sp_out
            sp_miss = (sp_in * p_in_miss + sp_out * p_out) / 1_000_000
            sp_80 = (0.8 * sp_in * p_in_hit + 0.2 * sp_in * p_in_miss + sp_out * p_out) / 1_000_000
            print()
            print(f"=== Stage P (digesters): {sp_calls} calls, {sp_tokens:,} tokens ===")
            print("  no tracer on the digester harness -> recovered from the tokens= field")
            print(f"  worst case (all cache miss)         : ${sp_miss:.4f}")
            print(f"  a 80% cache-hit rate would give     : ${sp_80:.4f}")
            print()
            print("=== where HarnessX's reported cost goes ===")
            print(f"  stages 1-3 (journaled)              : ${sonnet:.4f}")
            print(f"  stage P   (reconstructed)           : ${(sp_tokens * SONNET_ESTIMATE['in']) / 1_000_000:.4f}")
            print(f"  combined (Sonnet prices)            : ${sonnet + sp_tokens * SONNET_ESTIMATE['in'] / 1_000_000:.4f}")
            print(f"  stage P share                       : {sp_tokens * SONNET_ESTIMATE['in'] / (sonnet * 1e6 + sp_tokens * SONNET_ESTIMATE['in']) * 100:.0f}%")

        print()
        print("=== per-stage totals as logged (Sonnet prices) ===")
        totals: dict[str, float] = {}
        for line in sc.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line).get("message") or ""
            m = re.search(r"cost=\$([0-9.]+)", msg)
            if not m:
                continue
            head = msg.split(":", 1)[0].strip()
            key = head.split()[0] if head else "?"
            totals[key] = totals.get(key, 0.0) + float(m.group(1))
        for key, val in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {key:<12} ${val:>10.3f}   (worst case ~${val/20:.2f} real)")
        print(f"  {'TOTAL':<12} ${sum(totals.values()):>10.3f}   (worst case ~${sum(totals.values())/20:.2f} real)")
        print()
        print("  NOTE: the worst-case column assumes ZERO cache hits. The measured figures")
        print("  above supersede it -- with a 96% hit rate the real ratio is ~160x, not 20x.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
