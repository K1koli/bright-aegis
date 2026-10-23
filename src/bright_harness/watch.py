"""Watch a run in flight: score curve + per-stage spend + liveness.

For a T=15 run this is the thing you keep in a side window. Reads only files the
driver already writes, so it is safe to run at any time and costs nothing.

    python src/bright_harness/watch.py --run-root runs/bio-full-T15
    python src/bright_harness/watch.py --run-root runs/bio-full-T15 --follow
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

COST_RE = re.compile(r"cost=\$([0-9.]+)")


def _curve(run_root: Path) -> list[dict]:
    path = run_root / "curves.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _costs(run_root: Path) -> list[dict]:
    path = run_root / "stage_costs.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def render(run_root: Path) -> str:
    out: list[str] = []
    curve = _curve(run_root)
    costs = _costs(run_root)

    out.append(f"run: {run_root}")
    out.append("")
    if curve:
        base = curve[0]["mean_ndcg"]
        out.append(f"  {'R':>3}  {'mean nDCG':>10}  {'delta':>8}  {'pass':>9}")
        for r in curve:
            d = r["mean_ndcg"] - base
            out.append(
                f"  {r['round']:>3}  {r['mean_ndcg']:>10.4f}  {d:>+8.4f}  "
                f"{r['n_passed']:>4}/{r['n_queries']:<4}"
            )
        best = max(curve, key=lambda r: r["mean_ndcg"])
        out.append("")
        out.append(
            f"  best so far: R{best['round']} at {best['mean_ndcg']:.4f}   "
            f"rounds scored: {len(curve)}"
        )
    else:
        out.append("  (no curves.json yet -- waiting for the first round to finish)")

    if costs:
        total = sum(float(COST_RE.search(r.get("message", "") or "").group(1))
                    for r in costs if COST_RE.search(r.get("message", "") or ""))
        by_stage: dict[str, float] = {}
        for r in costs:
            m = COST_RE.search(r.get("message", "") or "")
            if not m:
                continue
            stage = (r.get("logger") or "?").split(".")[-1]
            by_stage[stage] = by_stage.get(stage, 0.0) + float(m.group(1))
        out.append("")
        out.append("  last 6 stages:")
        for r in costs[-6:]:
            out.append(f"    {r.get('message', '')[:110]}")
        out.append("")
        out.append("  Sonnet-priced estimate by stage (divide by ~20 for the real DeepSeek bill;")
        out.append("  run cost_report.py for the real number once a round has finished):")
        for stage, val in sorted(by_stage.items()):
            out.append(f"    {stage:<10} ${val:>8.3f}")
        out.append(f"    {'TOTAL':<10} ${total:>8.3f}")

    digits = sorted(p.name for p in run_root.glob("R*") if p.is_dir())
    out.append("")
    out.append(f"  round dirs present: {', '.join(digits) if digits else '(none)'}")

    # ---- in-round progress -------------------------------------------------
    # curves.json only ticks once per round, and a full-biology round is hours
    # long, so the per-round artifacts are the only live feedback.
    rd = sorted(
        (p for p in run_root.glob("R*") if p.is_dir()),
        key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1,
    )
    if rd:
        cur = rd[-1]
        digests = list((cur / "digests").glob("*.md")) if (cur / "digests").exists() else []
        marks = [n for n in ("summary.md", "landscape.md", "decision.md") if (cur / n).exists()]
        applied_dir = cur / "applied"
        applied = [p for p in applied_dir.glob("*") if p.is_dir()] if applied_dir.exists() else []
        cands_dir = cur / "candidates"
        cands = list(cands_dir.glob("*.md")) if cands_dir.exists() else []
        # Expected digesters == tasks that failed in the round just scored.
        expected = ""
        if curve:
            last = curve[-1]
            expected = f"   (expected ~{last['n_queries'] - last['n_passed']} from R{last['round']} failures)"
        out.append("")
        out.append(f"  current round: {cur.name}")
        out.append(f"    digests written : {len(digests)}{expected}")
        if marks:
            out.append(f"    stage markers   : {', '.join(marks)}")
        else:
            out.append("    stage markers   : (none yet -- still in Stage P, the digester fan-out)")
        out.append(f"    candidates      : {len(cands)} manifest(s), {len(applied)} applied dir(s)")
        out.append(
            "    stage order     : digests -> summary.md -> landscape.md -> candidates -> "
            "decision.md -> R{n+1}"
        )

    newest = 0.0
    for p in run_root.rglob("*"):
        try:
            if p.is_file():
                newest = max(newest, p.stat().st_mtime)
        except OSError:
            continue
    if newest:
        age = time.time() - newest
        state = "ACTIVE" if age < 900 else ("quiet" if age < 3600 else "STALLED?")
        extra = "" if age < 3600 else "  <- no file written for over an hour; check bio-full.err"
        out.append("")
        out.append(f"  last write: {age/60:.1f} min ago   [{state}]{extra}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--follow", action="store_true", help="refresh every 60s until stopped")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    if not run_root.exists():
        print(f"run root not found: {run_root}")
        return 2

    if not args.follow:
        print(render(run_root))
        return 0

    try:
        while True:
            print("\033[2J\033[H", end="")  # clear screen
            print(render(run_root))
            print(f"\n  (refreshing every 60s -- Ctrl+C to stop)  {time.strftime('%H:%M:%S')}")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
