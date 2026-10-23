"""Draw the per-round score curve as a dependency-free SVG.

matplotlib is not installed in this runtime, and the round data is small enough that
hand-writing SVG is simpler and better: the output is vector, so it stays sharp at any
size in a paper, and there is nothing to install.

Markers distinguish rounds that actually shipped a new harness (filled) from rounds
that shipped nothing (hollow), so the figure shows *where the harness changed* rather
than only where the score moved.

    python src/bright_harness/plot_curve.py --run-root runs/bio-full-T15
    python src/bright_harness/plot_curve.py --run-root runs/bio-full-T15 --out fig4.svg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _boot  # noqa: E402

ROOT = _boot.setup()

W, H = 960, 540
ML, MR, MT, MB = 88, 150, 56, 78  # margins


def _curve(run_root: Path) -> list[dict]:
    path = run_root / "curves.json"
    if not path.exists():
        raise SystemExit(f"no curves.json under {run_root}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"curves.json is not a non-empty list: {path}")
    return sorted(data, key=lambda r: r["round"])


def _shipped_rounds(run_root: Path) -> set[int]:
    """Rounds whose evolve call produced a merged harness (R{n}/applied/merged.yaml)."""
    shipped = set()
    for d in run_root.glob("R*"):
        if not d.is_dir() or not d.name[1:].isdigit():
            continue
        if (d / "applied" / "merged.yaml").exists():
            shipped.add(int(d.name[1:]))
    return shipped


def build_svg(rows: list[dict], shipped: set[int], domain: str,
              subtitle: str | None = None, ymax_fixed: float | None = None) -> str:
    rounds = [r["round"] for r in rows]
    means = [r["mean_ndcg"] for r in rows]
    base = means[0]
    ymax = ymax_fixed if ymax_fixed is not None else max(0.10, max(means) * 1.18)
    ymin = 0.0

    pw = W - ML - MR  # plot width
    ph = H - MT - MB  # plot height

    def sx(i: int) -> float:
        return ML + (pw * i / max(1, len(rows) - 1))

    def sy(v: float) -> float:
        return MT + ph * (1 - (v - ymin) / (ymax - ymin))

    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">'
    )
    out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # title
    out.append(
        f'<text x="{ML}" y="30" font-size="19" font-weight="600" fill="#111">'
        f"AEGIS harness evolution on BRIGHT / {domain}</text>"
    )
    out.append(
        f'<text x="{ML}" y="48" font-size="12.5" fill="#555">'
        f"{subtitle or 'mean nDCG@10 per round (full task set, no subsampling)'}</text>"
    )

    # y gridlines + labels
    step = 0.05
    v = 0.0
    while v <= ymax + 1e-9:
        y = sy(v)
        out.append(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + pw}" y2="{y:.1f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{ML - 10}" y="{y + 4:.1f}" font-size="11.5" fill="#666" '
            f'text-anchor="end">{v:.2f}</text>'
        )
        v += step

    # baseline (H_0)
    yb = sy(base)
    out.append(
        f'<line x1="{ML}" y1="{yb:.1f}" x2="{ML + pw}" y2="{yb:.1f}" '
        f'stroke="#c0392b" stroke-width="1.4" stroke-dasharray="6 4" opacity="0.85"/>'
    )
    out.append(
        f'<text x="{ML + pw + 8}" y="{yb + 4:.1f}" font-size="11.5" fill="#c0392b">'
        f"H&#8320; = {base:.4f}</text>"
    )

    # x axis labels (thin out when many rounds)
    every = 1 if len(rows) <= 13 else 2
    for i, r in enumerate(rows):
        if r["round"] % every:
            continue
        out.append(
            f'<text x="{sx(i):.1f}" y="{MT + ph + 20}" font-size="11.5" fill="#555" '
            f'text-anchor="middle">{r["round"]}</text>'
        )
    out.append(
        f'<text x="{ML + pw / 2:.0f}" y="{H - 22}" font-size="12.5" fill="#333" '
        f'text-anchor="middle">evolution round</text>'
    )

    # axes
    out.append(
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + ph}" stroke="#999" stroke-width="1"/>'
    )
    out.append(
        f'<line x1="{ML}" y1="{MT + ph}" x2="{ML + pw}" y2="{MT + ph}" '
        f'stroke="#999" stroke-width="1"/>'
    )

    # the curve
    pts = " ".join(f"{sx(i):.1f},{sy(m):.1f}" for i, m in enumerate(means))
    out.append(
        f'<polyline points="{pts}" fill="none" stroke="#1f6feb" stroke-width="2.4" '
        f'stroke-linejoin="round"/>'
    )

    # markers: filled = this round's evolve shipped a new harness
    for i, r in enumerate(rows):
        cx, cy = sx(i), sy(r["mean_ndcg"])
        did = r["round"] in shipped
        fill = "#1f6feb" if did else "#ffffff"
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.6" fill="{fill}" '
            f'stroke="#1f6feb" stroke-width="1.8"><title>'
            f'R{r["round"]}: {r["mean_ndcg"]:.4f} ({r["n_passed"]}/{r["n_queries"]} pass)'
            f'{" -- shipped" if did else " -- no-op"}</title></circle>'
        )

    # annotate the peak
    best = max(rows, key=lambda r: r["mean_ndcg"])
    bi = rounds.index(best["round"])
    bx, by = sx(bi), sy(best["mean_ndcg"])
    out.append(
        f'<text x="{bx:.1f}" y="{by - 14:.1f}" font-size="12" font-weight="600" '
        f'fill="#1f6feb" text-anchor="middle">peak {best["mean_ndcg"]:.4f}</text>'
    )

    # legend
    lx = ML + pw + 10
    ly = MT + 12
    out.append(f'<circle cx="{lx + 6}" cy="{ly}" r="4.6" fill="#1f6feb" stroke="#1f6feb"/>')
    out.append(f'<text x="{lx + 18}" y="{ly + 4}" font-size="11.5" fill="#333">shipped</text>')
    ly += 20
    out.append(f'<circle cx="{lx + 6}" cy="{ly}" r="4.6" fill="#fff" stroke="#1f6feb" stroke-width="1.8"/>')
    out.append(f'<text x="{lx + 18}" y="{ly + 4}" font-size="11.5" fill="#333">no-op</text>')
    ly += 28
    out.append(
        f'<text x="{lx}" y="{ly}" font-size="11.5" fill="#666">'
        f'{len(shipped)} ships / {len(rows) - 1} evolves</text>'
    )

    out.append("</svg>")
    return "\n".join(out)


def write_png(rows: list[dict], shipped: set[int], domain: str, path: Path,
              subtitle: str | None = None, ymax_fixed: float | None = None,
              scale: int = 2) -> None:
    """Same figure as a raster PNG, for viewers that will not open SVG.

    Pillow only -- no matplotlib, nothing to install.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        f_title = ImageFont.load_default(size=19 * scale)
        f_sub = ImageFont.load_default(size=12 * scale)
        f_lab = ImageFont.load_default(size=11 * scale)
    except TypeError:  # Pillow < 10.1 has no size= on load_default
        f_title = f_sub = f_lab = ImageFont.load_default()

    w, h = W * scale, H * scale
    ml, mr, mt, mb = ML * scale, MR * scale, MT * scale, MB * scale
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    means = [r["mean_ndcg"] for r in rows]
    base = means[0]
    ymax = ymax_fixed if ymax_fixed is not None else max(0.10, max(means) * 1.18)
    pw, ph = w - ml - mr, h - mt - mb

    def sx(i: int) -> float:
        return ml + pw * i / max(1, len(rows) - 1)

    def sy(v: float) -> float:
        return mt + ph * (1 - v / ymax)

    d.text((ml, 12 * scale), f"AEGIS harness evolution on BRIGHT / {domain}", fill="#111", font=f_title)
    d.text((ml, 38 * scale), subtitle or "mean nDCG@10 per round (full task set, no subsampling)",
           fill="#555", font=f_sub)

    v = 0.0
    while v <= ymax + 1e-9:
        y = sy(v)
        d.line([(ml, y), (ml + pw, y)], fill="#e6e6e6", width=scale)
        d.text((ml - 8 * scale, y - 6 * scale), f"{v:.2f}", fill="#666", font=f_lab, anchor="ra")
        v += 0.05

    yb = sy(base)
    for x in range(int(ml), int(ml + pw), 10 * scale):
        d.line([(x, yb), (x + 6 * scale, yb)], fill="#c0392b", width=max(1, scale))
    d.text((ml + pw + 8 * scale, yb - 6 * scale), f"H0 = {base:.4f}", fill="#c0392b", font=f_lab)

    every = 1 if len(rows) <= 13 else 2
    for i, r in enumerate(rows):
        if r["round"] % every:
            continue
        d.text((sx(i), mt + ph + 8 * scale), str(r["round"]), fill="#555", font=f_lab, anchor="ma")
    d.text(((ml + ml + pw) / 2, h - 26 * scale), "evolution round", fill="#333",
           font=f_sub, anchor="ma")

    d.line([(ml, mt), (ml, mt + ph)], fill="#999", width=scale)
    d.line([(ml, mt + ph), (ml + pw, mt + ph)], fill="#999", width=scale)

    d.line([(sx(i), sy(m)) for i, m in enumerate(means)], fill="#1f6feb", width=2 * scale)

    r = 4 * scale
    for i, row in enumerate(rows):
        cx, cy = sx(i), sy(row["mean_ndcg"])
        did = row["round"] in shipped
        box = [cx - r, cy - r, cx + r, cy + r]
        if did:
            d.ellipse(box, fill="#1f6feb", outline="#1f6feb", width=2 * scale)
        else:
            d.ellipse(box, fill="white", outline="#1f6feb", width=2 * scale)

    best = max(rows, key=lambda x: x["mean_ndcg"])
    bi = [r["round"] for r in rows].index(best["round"])
    d.text((sx(bi), sy(best["mean_ndcg"]) - 22 * scale), f"peak {best['mean_ndcg']:.4f}",
           fill="#1f6feb", font=f_sub, anchor="ma")

    lx, ly = ml + pw + 10 * scale, mt + 8 * scale
    d.ellipse([lx, ly, lx + 2 * r, ly + 2 * r], fill="#1f6feb")
    d.text((lx + 3 * r, ly - 2 * scale), "shipped", fill="#333", font=f_lab)
    ly += 20 * scale
    d.ellipse([lx, ly, lx + 2 * r, ly + 2 * r], fill="white", outline="#1f6feb", width=2 * scale)
    d.text((lx + 3 * r, ly - 2 * scale), "no-op", fill="#333", font=f_lab)
    ly += 26 * scale
    d.text((lx, ly), f"{len(shipped)} ships / {len(rows) - 1} evolves", fill="#666", font=f_lab)

    img.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", default=None, help="output .svg (default: <run_root>/curve.svg)")
    ap.add_argument("--domain", default=None, help="label; defaults to the run dir name")
    ap.add_argument("--png", action="store_true", help="also write a raster PNG (Pillow)")
    ap.add_argument("--subtitle", default=None,
                    help="override the subtitle, e.g. when the run scored a dev split "
                         "rather than the full task set")
    ap.add_argument("--ymax", type=float, default=None,
                    help="fix the y-axis top so two runs can be compared visually")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    if not run_root.exists():
        print(f"run root not found: {run_root}")
        return 2

    rows = _curve(run_root)
    shipped = _shipped_rounds(run_root)
    domain = args.domain or "biology"
    svg = build_svg(rows, shipped, domain, subtitle=args.subtitle, ymax_fixed=args.ymax)

    out_path = Path(args.out) if args.out else (run_root / "curve.svg")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")

    print(f"wrote {out_path.relative_to(ROOT)}")
    print(f"  rounds  : {len(rows)}  (R{rows[0]['round']} .. R{rows[-1]['round']})")
    print(f"  baseline: {rows[0]['mean_ndcg']:.4f}")
    best = max(rows, key=lambda r: r["mean_ndcg"])
    print(f"  peak    : R{best['round']} at {best['mean_ndcg']:.4f}  "
          f"(+{best['mean_ndcg'] - rows[0]['mean_ndcg']:.4f})")
    print(f"  ships   : {sorted(shipped)}  ({len(shipped)} of {len(rows) - 1} evolves)")

    if args.png:
        png_path = out_path.with_suffix(".png")
        write_png(rows, shipped, domain, png_path, subtitle=args.subtitle,
                  ymax_fixed=args.ymax)
        print(f"wrote {png_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
