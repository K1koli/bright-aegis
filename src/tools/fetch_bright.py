"""Download selected BRIGHT domains (documents + examples) from HuggingFace.

Uses urllib (OpenSSL backend) because the host's Windows schannel stack is broken.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

REPO = "xlangai/BRIGHT"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
ROOT = Path(__file__).resolve().parents[2] / "data" / "bright" / "raw"

DOMAINS = ["biology", "economics", "robotics", "psychology", "stackoverflow"]
SPLITS = ["documents", "examples"]


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.name} {dest.stat().st_size/1e6:.1f} MB")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "bright-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
    tmp.replace(dest)
    print(f"  ok {dest.name} {got/1e6:.1f} MB (content-length {total/1e6:.1f} MB)")


def main() -> int:
    domains = sys.argv[1:] or DOMAINS
    for split in SPLITS:
        for dom in domains:
            name = f"{dom}-00000-of-00001.parquet"
            dest = ROOT / split / name
            print(f"{split}/{name}")
            try:
                fetch(f"{BASE}/{split}/{name}", dest)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {type(e).__name__}: {e}")
    print("\n--- local tree ---")
    for p in sorted(ROOT.rglob("*.parquet")):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
