"""Run a handful of BIRCO queries across arms and dump EVERYTHING for inspection.

The scored eval only kept numbers, so this re-runs a few queries with the model's raw
output and the full per-step session journal preserved. It auto-picks the queries where
the un-evolved H0 beats both evolved harnesses by the widest margin — those are the
cases that need explaining.

    python src/birco/probe_birco.py --task arguana --auto 3
    python src/birco/probe_birco.py --task arguana --qid test-law-cppshbcjsfm-con04a
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup(quiet=True)

from bright.metrics import ndcg_at_k  # noqa: E402
from birco import data as bdata  # noqa: E402
from birco.pool_index import PoolIndex  # noqa: E402
from bright_harness.harness import make_bright_harness, make_bright_task  # noqa: E402
from bright_harness.provider import make_task_model_config  # noqa: E402
from bright_harness.round_driver import (  # noqa: E402
    _force_tracer_dir,
    max_steps_from_config,
    policy_kwargs_from_config,
)
from bright_harness.scoring import parse_ranked_ids  # noqa: E402
from bright_harness.tools import set_active_index  # noqa: E402

ARMS = {
    "h0": "builtin",
    "bio": "runs/bio-full-T15/R9/applied/merged.yaml",
    "rob": "runs/rob-3to1/R7/config.yaml",
}


def _build(spec: str):
    if spec == "builtin":
        return make_bright_harness()
    from harnessx.core.harness import HarnessConfig

    p = Path(spec)
    if not p.is_absolute():
        p = ROOT / p
    return HarnessConfig.from_yaml_file(str(p))


def _auto_qids(n: int) -> list[str]:
    """Queries where h0 beats MAX(bio, rob) most — the cases to explain."""
    src = ROOT / "runs/birco/arguana-k10/results.json"
    if not src.exists():
        raise SystemExit(f"need {src} first (run the scored k10 eval)")
    d = json.loads(src.read_text(encoding="utf-8"))
    per = {a["arm"]: a["per_query"] for a in d["arms"]}
    qs = sorted(per["h0"])
    gap = [(per["h0"][q] - max(per["bio"][q], per["rob"][q]), q) for q in qs]
    gap.sort(reverse=True)
    return [q for g, q in gap[:n] if g > 0]


def _session_records(sessions_root: Path) -> list[dict]:
    out = []
    for f in sorted(sessions_root.rglob("*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _text_of(rec: dict) -> str:
    m = rec.get("message")
    if isinstance(m, str):
        return m
    if isinstance(m, dict):
        for k in ("content", "text", "arguments"):
            v = m.get(k)
            if isinstance(v, str) and v.strip():
                return v
        try:
            return json.dumps(m, ensure_ascii=False)
        except Exception:
            return str(m)
    return ""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="arguana")
    ap.add_argument("--qid", action="append", default=None)
    ap.add_argument("--auto", type=int, default=3, help="auto-pick N queries h0 wins by most")
    ap.add_argument("--arms", default="h0,bio,rob")
    ap.add_argument("--out", default="runs/birco/probe")
    args = ap.parse_args()

    task = bdata.load_task(args.task)
    qids = args.qid or _auto_qids(args.auto)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== BIRCO/{task.name} probe: {len(qids)} queries x {len(arm_names)} arms ===")
    for q in qids:
        pool = task.pools[q]
        gold = task.gold[q]
        gpos = [i for i, c in enumerate(pool, 1) if c in gold]
        print(f"\n  qid={q}")
        print(f"    池子 {len(pool)} 条,gold 在池内第 {gpos} 位")
        print(f"    查询: {task.queries[q][:200]}")

    model_config = make_task_model_config()
    summary = {}

    for arm in arm_names:
        spec = ARMS.get(arm, arm)
        cfg = _build(spec)
        k = int(policy_kwargs_from_config(cfg).get("retrieve_k", 10))
        steps = max_steps_from_config(cfg, 4)
        harness = model_config.agentic(cfg)
        print(f"\n{'='*70}\n=== arm {arm}  (retrieve_k={k}, max_steps={steps}) ===\n{'='*70}")

        for q in qids:
            pool, gold = task.pools[q], task.gold[q]
            idx = PoolIndex.from_file_order(pool, {c: task.corpus[c] for c in pool})
            if k:
                # mirror the scored run: cap visibility at 10 so all arms saw the same
                for entry in getattr(cfg, "processors", None) or []:
                    if "RetrievalPolicyProcessor" in str((entry or {}).get("_target_", "")):
                        entry["retrieve_k"] = 10
                harness = model_config.agentic(cfg)

            sess = out_root / arm / q
            sess.mkdir(parents=True, exist_ok=True)
            _force_tracer_dir(cfg, sess)
            harness = model_config.agentic(cfg)
            set_active_index(idx)

            bt = make_bright_task(task.queries[q], qid=q, domain=f"birco-{task.name}",
                                 max_steps=min(steps, 4))
            try:
                res = await harness.run(bt, session_id=f"probe-{arm}-{q}")
                final = res.final_output or ""
                exitr = getattr(res, "exit_reason", "") or ""
            except Exception as exc:  # noqa: BLE001
                final, exitr = "", f"ERROR {type(exc).__name__}: {exc}"

            ranked = parse_ranked_ids(final, valid_ids=set(pool), k=10)
            score = ndcg_at_k(ranked, gold, 10)
            gpos = [i for i, c in enumerate(pool, 1) if c in gold][0]
            newpos = next((i for i, c in enumerate(ranked, 1) if c in gold), None)
            summary[(arm, q)] = {"score": score, "pool_rank": gpos, "out_rank": newpos}

            print(f"\n--- {q}  (gold 池内第 {gpos} 位 → 输出第 {newpos} 位 | nDCG={score:.3f} | exit={exitr})")
            print("  模型原始输出:")
            for line in (final or "(空)").splitlines():
                print(f"    | {line}")

            recs = _session_records(sess)
            if recs:
                print(f"  --- 会话轨迹({len(recs)} 条记录)---")
                for r in recs:
                    t = r.get("type")
                    if t in ("session_start", "system"):
                        continue
                    txt = _text_of(r)[:400].replace("\n", " ⏎ ")
                    if txt.strip():
                        print(f"    [{t} step={r.get('step')}] {txt}")

    print(f"\n{'='*70}\n=== 汇总 ===\n{'='*70}")
    print(f"  {'qid':<40} {'arm':<5} {'池内位次':>8} {'输出位次':>8} {'nDCG':>6}")
    for (arm, q), v in summary.items():
        print(f"  {q:<40} {arm:<5} {v['pool_rank']:>8} {str(v['out_rank']):>8} {v['score']:>6.3f}")
    print(f"\n  轨迹与输出已存到 {out_root.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    _boot.quiet_logging()
    raise SystemExit(asyncio.run(main()))
