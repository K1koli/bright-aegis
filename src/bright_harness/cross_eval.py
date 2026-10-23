"""Evaluate N evolved harnesses across a grid of (domain, split) cells.

Why this exists: evaluating a transfer matrix cell by cell with ``eval_config.py``
re-runs H_0 every single time. A 3-arm x 4-cell grid then costs 24 task-agent runs
where 16 would do (4 baselines + 12 candidates), and the baselines are the same
numbers every time. This driver runs each cell's H_0 ONCE, caches its per-query
scores, then scores every arm against that cache.

It also writes one run dir per cell, so cells stop overwriting each other in
``runs/eval-config`` (that is how the 50-query economics run was lost).

    # what will run, without running anything
    python src/bright_harness/cross_eval.py --list \
        --arm bio=runs/bio-full-T15/R9/applied/merged.yaml \
        --cell biology:all --cell economics:all

    # the real matrix
    python src/bright_harness/cross_eval.py \
        --arm bio=runs/bio-full-T15/R9/applied/merged.yaml \
        --arm econ=runs/econ-full/R9/applied/merged.yaml \
        --arm rob=runs/rob-3to1/R9/applied/merged.yaml \
        --cell biology:all \
        --cell economics:all \
        --cell robotics:dev:76 \
        --cell robotics:heldout:76 \
        --out runs/cross-eval
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

from bright import data as bdata  # noqa: E402
from bright.retrieval import BM25Index  # noqa: E402
from bright_harness.provider import make_task_model_config, task_provider_summary  # noqa: E402
from bright_harness.round_driver import run_round  # noqa: E402
from bright_harness.round_metrics import NDCG_PASS_TAU  # noqa: E402


def _parse_arm(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--arm must be NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"arm {name!r}: config not found: {p}")
    return name.strip(), p


def _parse_cell(spec: str) -> tuple[str, str, int | None]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"--cell must be DOMAIN:SPLIT[:N_DEV], got {spec!r}")
    domain, split = parts[0], parts[1]
    if split not in ("dev", "heldout", "all"):
        raise SystemExit(f"cell split must be dev|heldout|all, got {split!r}")
    n_dev = int(parts[2]) if len(parts) > 2 else None
    if domain not in bdata.DOMAINS:
        raise SystemExit(f"unknown domain {domain!r}; known: {list(bdata.DOMAINS)}")
    return domain, split, n_dev


def _resolve_queries(domain: str, split: str, n_dev: int | None, seed: int):
    """The query list for one cell, plus a human label."""
    if split == "all":
        s = bdata.make_split(domain, n_dev=10**9, seed=seed, allow_full=True)
        return list(s.dev) + list(s.heldout), f"{domain}/all"
    s = bdata.make_split(domain, n_dev=n_dev if n_dev is not None else 50, seed=seed)
    if split == "heldout":
        return list(s.heldout), f"{domain}/heldout"
    return list(s.dev), f"{domain}/dev"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH",
                    help="a config to evaluate; repeat for each arm of the matrix")
    ap.add_argument("--cell", action="append", required=True, metavar="DOMAIN:SPLIT[:N_DEV]",
                    help="an evaluation cell; split is dev|heldout|all. N_DEV defaults to 50 "
                         "and is ignored for 'all'.")
    ap.add_argument("--out", default="runs/cross-eval", help="root for per-cell artifacts")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--k-rollouts", type=int, default=1)
    ap.add_argument("--reuse-baseline", action="store_true",
                    help="reuse a cached H_0 score file for a cell that was already run")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    arms = [_parse_arm(a) for a in args.arm]
    cells = [_parse_cell(c) for c in args.cell]
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    plan = []
    for domain, split, n_dev in cells:
        queries, label = _resolve_queries(domain, split, n_dev, args.seed)
        plan.append((label, domain, split, queries))

    if args.list:
        print("=== plan (no inference) ===")
        print(f"task agent : {task_provider_summary()}")
        print(f"out        : {out_root.relative_to(ROOT) if out_root.is_relative_to(ROOT) else out_root}")
        print()
        total_runs = 0
        for label, domain, split, queries in plan:
            base = out_root / label.replace("/", "-")
            cached = (base / "_H0" / "scores.json").exists()
            print(f"  {label:<22} {len(queries):>3} queries")
            print(f"    H_0  -> {base.name}/_H0{'   (cached, will reuse with --reuse-baseline)' if cached else ''}")
            for name, _ in arms:
                print(f"    {name:<5}-> {base.name}/{name}")
            total_runs += len(arms) + (0 if cached and args.reuse_baseline else 1)
        print()
        print(f"  arms={len(arms)}  cells={len(plan)}  task-agent runs={total_runs}")
        naive = len(arms) * len(plan) * 2
        print(f"  (a naive per-cell eval_config.py loop would cost {naive} runs)")
        return 0

    model_config = make_task_model_config()
    matrix: dict[str, dict[str, float]] = {}
    matrix_pass: dict[str, dict[str, int]] = {}
    matrix_n: dict[str, int] = {}

    print("=== cross evaluation (task agent only; no proposer, $0) ===")
    print(f"task agent : {task_provider_summary()}")
    print(f"arms       : {[n for n, _ in arms]}")
    print()

    for label, domain, split, queries in plan:
        base = out_root / label.replace("/", "-")
        base.mkdir(parents=True, exist_ok=True)
        print(f"--- {label}: {len(queries)} queries ---")
        matrix_n[label] = len(queries)

        # corpus + index once per cell (domains differ, so this cannot be hoisted)
        doc_ids, contents = bdata.load_corpus(domain)
        index = BM25Index.cached(domain, doc_ids, contents)

        h0_path = base / "_H0" / "scores.json"
        if args.reuse_baseline and h0_path.exists():
            h0 = json.loads(h0_path.read_text(encoding="utf-8"))
            h0_mean = sum(h0.values()) / len(h0) if h0 else 0.0
            print(f"  {'H_0 (baseline)':<26} mean nDCG@10 = {h0_mean:.4f}   [cached]")
        else:
            res = await run_round(
                domain=domain, queries=queries, config_path=None, round_n=0,
                run_root=base / "_H0", model_config=model_config,
                k_rollouts=args.k_rollouts, index=index,
            )
            h0 = {o.qid: o.ndcg10 for o in res.outcomes}
            h0_mean = res.mean_ndcg
            h0_path.parent.mkdir(parents=True, exist_ok=True)
            h0_path.write_text(json.dumps(h0, indent=1), encoding="utf-8")
            print(f"  {'H_0 (baseline)':<26} mean nDCG@10 = {h0_mean:.4f}")
        matrix[label] = {"H_0": h0_mean}
        matrix_pass[label] = {"H_0": sum(1 for v in h0.values() if v >= NDCG_PASS_TAU)}

        for name, cfg in arms:
            res = await run_round(
                domain=domain, queries=queries, config_path=cfg, round_n=0,
                run_root=base / name, model_config=model_config,
                k_rollouts=args.k_rollouts, index=index,
            )
            scores = {o.qid: o.ndcg10 for o in res.outcomes}
            matrix[label][name] = res.mean_ndcg
            matrix_pass[label][name] = sum(1 for o in res.outcomes if o.ndcg10 >= NDCG_PASS_TAU)
            wins = losses = 0
            for q in queries:
                d = scores.get(q.qid, 0.0) - h0.get(q.qid, 0.0)
                if d > 1e-9:
                    wins += 1
                elif d < -1e-9:
                    losses += 1
            print(f"  {name:<26} mean nDCG@10 = {res.mean_ndcg:.4f}   "
                  f"delta={res.mean_ndcg - h0_mean:+.4f}  win={wins} loss={losses}")
        print()
        del index

    # ---- the matrix -------------------------------------------------------
    arm_names = [n for n, _ in arms]
    print("=== matrix: mean nDCG@10 (rows = eval cell, cols = arm) ===")
    w = max(12, *(len(x) for x in matrix) or 12)
    header = f"  {'cell':<{w}} {'H_0':>8}" + "".join(f"{n:>10}" for n in arm_names)
    print(header)
    for label in matrix:
        row = f"  {label:<{w}} {matrix[label]['H_0']:>8.4f}"
        for n in arm_names:
            row += f"{matrix[label][n]:>10.4f}"
        print(row)

    print()
    print("=== matrix: delta vs H_0 of the same cell ===")
    print(f"  {'cell':<{w}}" + "".join(f"{n:>10}" for n in arm_names))
    for label in matrix:
        row = f"  {label:<{w}}"
        for n in arm_names:
            row += f"{matrix[label][n] - matrix[label]['H_0']:>+10.4f}"
        print(row)

    print()
    print("=== matrix: passed / total (tau=%.1f) ===" % NDCG_PASS_TAU)
    print(f"  {'cell':<{w}} {'H_0':>8}" + "".join(f"{n:>10}" for n in arm_names))
    for label in matrix:
        n_q = matrix_n[label]
        row = f"  {label:<{w}} {matrix_pass[label]['H_0']:>8d}"
        for n in arm_names:
            row += f"{matrix_pass[label][n]:>10d}"
        print(row + f"   (of {n_q})")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "matrix.json").write_text(
        json.dumps({"mean": matrix, "passed": matrix_pass, "n_queries": matrix_n}, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"wrote {(out_root / 'matrix.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    _boot.quiet_logging()
    raise SystemExit(asyncio.run(main()))
