"""Readable per-round report for an AEGIS/Bright run.

While a run is going, the three questions you actually have are:

  1. what did this round score?
  2. what did the agents propose, and *which part* of the harness?
  3. what was finally applied — and if nothing was, which gate stopped it and why?

The answers are scattered across ``curves.json``, ``R{n}/scores.json``,
``R{n}/candidates/*.md``, ``R{n}/decision.md``, ``audit.jsonl`` and
``R{n}/applied/*/config.yaml``. This module joins them into one readable block,
including a unified diff of the incumbent config against each candidate's applied
config (which is the literal answer to "what did it change?").

    python src/bright_harness/report.py --run-root runs/smoke-T3b
    python src/bright_harness/report.py --run-root runs/smoke-T3b --round 1 --diff
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

__all__ = ["render_round", "render_run"]

_FM_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _frontmatter(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    m = _FM_RE.match(text)
    if not m:
        return {}
    try:
        import yaml

        return yaml.safe_load(m.group(1)) or {}
    except Exception:  # noqa: BLE001
        return {}


def _gate_results(run_root: Path) -> dict[int, dict[str, dict]]:
    """{round: {cid: {"results": {...}, "reasons": {...}}}} from audit.jsonl."""
    out: dict[int, dict[str, dict]] = {}
    path = run_root / "audit.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("stage")) != "4" or row.get("kind") != "gate":
            continue
        payload = row.get("payload") or {}
        cid = payload.get("cid")
        if not cid:
            continue
        out.setdefault(int(row.get("round", 0)), {})[cid] = payload
    return out


def _audit_events(run_root: Path) -> list[dict]:
    path = run_root / "audit.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _clean_for_diff(text: str) -> list[str]:
    """Drop serialization noise so a config diff shows real semantic changes.

    HarnessX rewrites ``_code_hash``, ``base_dir`` and ``_hook_`` on every save, so
    a prompt-only candidate otherwise looks like it rewrote the whole file.
    """
    noise = re.compile(r"^\s*(_code_hash:|base_dir:|session_id:|_hook_:)")
    return [ln for ln in text.splitlines() if ln.strip() and not noise.match(ln)]


def _diff(a: Path, b: Path, label: str, *, max_lines: int = 40) -> str:
    try:
        left = _clean_for_diff(a.read_text(encoding="utf-8", errors="replace"))
        right = _clean_for_diff(b.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ""
    if left == right:
        return f"    ({label}: identical apart from serialization noise)"
    body = list(difflib.unified_diff(left, right, fromfile=a.name, tofile=b.name, n=1, lineterm=""))
    if not body:
        return ""
    if len(body) > max_lines:
        body = body[:max_lines] + [f"... ({len(body) - max_lines} more diff lines)"]
    return f"    --- {label} ---\n" + "\n".join("      " + ln for ln in body if ln.strip())


def _incumbent_prompt(cfg_path: Path | None) -> Path | None:
    """The system-prompt template the incumbent config points at."""
    if cfg_path is None:
        return None
    try:
        text = cfg_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"template_path:\s*(.+)$", text, re.M)
    if not m:
        return None
    p = Path(m.group(1).strip().strip('"').strip("'"))
    return p if p.exists() else None


def _score_line(run_root: Path, round_n: int) -> str:
    curves = _read_json(run_root / "curves.json", []) or []
    entry = next((c for c in curves if c.get("round") == round_n), None)
    scores = _read_json(run_root / f"R{round_n}" / "scores.json", {}) or {}
    if entry is None and not scores:
        return "    (no score recorded for this round)"
    parts = []
    if entry:
        parts.append(
            f"mean nDCG@10 = {entry.get('mean_ndcg', 0):.4f}   "
            f"pass {entry.get('n_passed', 0)}/{entry.get('n_queries', 0)} at tau={entry.get('tau')}   "
            f"parsed {entry.get('parse_rate', 0) * 100:.0f}%"
        )
    if scores:
        worst = sorted(scores.items(), key=lambda kv: kv[1])[:4]
        best = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        parts.append(
            "    best: " + "  ".join(f"q{k}={v:.3f}" for k, v in best)
        )
        parts.append(
            "    worst: " + "  ".join(f"q{k}={v:.3f}" for k, v in worst)
        )
    return "\n".join(parts)


def _incumbent_config(run_root: Path, round_n: int) -> Path | None:
    """The config the round's candidates were proposed against.

    ``R{n}/config.yaml`` is written when H_n runs, so for the candidates stored in
    ``R{n}/`` the incumbent is ``R{n-1}``'s effective config — walking further back
    when a round shipped nothing (then H_n == H_{n-1}).
    """
    for back in (round_n - 1, round_n - 2, round_n):
        cfg = run_root / f"R{back}" / "config.yaml"
        if cfg.exists():
            return cfg
        merged = run_root / f"R{back}" / "applied" / "merged.yaml"
        if merged.exists():
            return merged
    return None


def render_round(run_root: Path, round_n: int, *, show_diff: bool = True) -> str:
    round_dir = run_root / f"R{round_n}"
    lines: list[str] = [f"=== ROUND {round_n} " + "=" * 46]

    lines.append("[1] score of the harness that ran this round")
    lines.append(_score_line(run_root, round_n))

    cand_dir = round_dir / "candidates"
    cands = sorted(cand_dir.glob("*.md")) if cand_dir.exists() else []
    decisions = round_dir / "decision.md"

    lines.append("")
    lines.append("[2] what was proposed")
    if not cands:
        lines.append("    (no candidates — the round ended before/at Stage 2)")
    for c in cands:
        fm = _frontmatter(c)
        bucket = fm.get("bucket") or "?"
        impact = fm.get("predicted_impact") or {}
        unlock = impact.get("tasks_will_unlock") or []
        risk = impact.get("tasks_at_risk") or []
        lines.append(f"    {fm.get('candidate_id', c.stem):<10} bucket={bucket}")
        if unlock or risk:
            lines.append(f"        predicts: unlock={unlock} at_risk={risk}")
        for fc in fm.get("file_changes") or []:
            action = fc.get("action", "?")
            path = str(fc.get("path", ""))
            short = path.replace(str(run_root) + "\\", "").replace(str(run_root) + "/", "")
            lines.append(f"        {action:<7} {short}")
            if fc.get("diff_summary"):
                lines.append(f"                {str(fc['diff_summary'])[:130]}")

    lines.append("")
    lines.append("[3] Critic decision")
    if decisions.exists():
        text = decisions.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"decision_type:\s*(\S+)", text)
        rank = re.findall(r"candidate_id:\s*(\S+)", text)
        lines.append(f"    decision_type = {m.group(1) if m else '?'}   ship_ranking = {rank}")
        sc = re.search(r"strategy_concern:\s*\|?\s*\n((?:\s{2,}.*\n)+)", text)
        if sc:
            concern = " ".join(x.strip() for x in sc.group(1).strip().splitlines())
            lines.append(f"    strategy_concern: {concern[:400]}")
    else:
        lines.append("    (no decision.md)")

    lines.append("")
    lines.append("[4] the deterministic gates (Tier-2 fitness gate appears only with --fitness-gate)")
    gates = _gate_results(run_root).get(round_n, {})
    if not gates:
        lines.append("    (no gate results recorded)")
    for cid, payload in gates.items():
        results = payload.get("results") or {}
        reasons = payload.get("reasons") or {}

        def _mark(name: str, ok: bool) -> str:
            # A gate that self-skipped returns ok=True, so "+fitness" was printed for a
            # gate that never ran (the Tier-2 gate with no fitness context wired). Mark
            # those "=" so a skipped gate cannot be misread as a pass.
            if ok and "skipped" in (reasons.get(name) or "").lower():
                return "=" + name
            return ("+" if ok else "-") + name

        marks = "  ".join(_mark(name, ok) for name, ok in results.items())
        lines.append(f"    {cid}: {marks}")
        for name, ok in results.items():
            reason = reasons.get(name) or ""
            if not ok and reason and "skipped" not in reason:
                lines.append(f"        {name}: {reason[:200]}")
            elif ok and "skipped" in reason.lower():
                lines.append(f"        {name}: {reason[:200]}")

    lines.append("")
    lines.append("[5] what was finally applied")
    applied = sorted((round_dir / "applied").glob("*/config.yaml")) if (round_dir / "applied").exists() else []
    merged = round_dir / "applied" / "merged.yaml"
    prev = _incumbent_config(run_root, round_n)
    ships = _audit_events(run_root)
    shipped = [
        e for e in ships
        if str(e.get("stage")) == "4" and e.get("kind") == "commit" and int(e.get("round", -1)) == round_n
    ]
    if merged.exists():
        lines.append(f"    SHIPPED -> {merged.relative_to(run_root)}")
        if show_diff and prev is not None:
            lines.append(_diff(prev, merged, "config diff (incumbent -> shipped)"))
            inc_prompt = _incumbent_prompt(prev)
            for extra in sorted((round_dir / "applied").glob("*/")):
                for f in sorted(extra.glob("*.md")):
                    if inc_prompt is not None:
                        lines.append(_diff(inc_prompt, f, f"prompt diff ({extra.name}/{f.name})"))
    elif applied:
        lines.append("    NOTHING SHIPPED — every candidate was rejected by a gate.")
        lines.append(f"    candidates applied-configs under R{round_n}/applied/: {len(applied)}")
        if show_diff:
            inc_prompt = _incumbent_prompt(prev)
            for cfg in applied:
                lines.append(_diff(prev, cfg, f"{cfg.parent.name}: config vs incumbent") if prev else "")
                for f in sorted(cfg.parent.glob("*.md")):
                    if inc_prompt is not None:
                        lines.append(_diff(inc_prompt, f, f"{cfg.parent.name}: prompt vs incumbent ({f.name})"))
    else:
        lines.append("    (nothing applied)")

    return "\n".join(lines)


def render_run(run_root: Path, *, rounds: list[int] | None = None, show_diff: bool = True) -> str:
    if rounds is None:
        found = sorted(
            int(p.name[1:]) for p in run_root.iterdir()
            if p.is_dir() and p.name.startswith("R") and p.name[1:].isdigit()
        )
        rounds = found
    blocks = [render_round(run_root, n, show_diff=show_diff) for n in rounds]
    header = [f"RUN {run_root}", f"rounds present: {rounds}"]
    history = _read_json(run_root / "pilot_history.json", None)
    if history:
        header.append("")
        header.append("score trajectory")
        for h in history:
            probe_txt = f" probe={h['probe_mean_ndcg']:.4f}" if "probe_mean_ndcg" in h else ""
            header.append(
                f"    R{h.get('round')}: mean={h.get('mean_ndcg', 0):.4f}"
                f"{probe_txt} passed={h.get('n_passed', 0)}/{h.get('n_queries', 0)}"
            )
    return "\n".join(header) + "\n\n" + "\n\n".join(blocks) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--no-diff", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit the score trajectory as JSON")
    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    if not run_root.exists():
        print(f"no such run: {run_root}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_read_json(run_root / "pilot_history.json", []), indent=2, ensure_ascii=False))
        return 0

    rounds = [args.round] if args.round is not None else None
    print(render_run(run_root, rounds=rounds, show_diff=not args.no_diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
