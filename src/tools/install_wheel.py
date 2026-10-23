"""Dependency installer that bypasses pip.

The host's pip cannot write its own temp files (OSError EACCES on
``*.whl.metadata``), while direct file operations in this workspace work fine.
So we resolve wheels from PyPI, download them with urllib (OpenSSL), and
extract them straight into a target directory.

Usage:
    python install_wheel.py <spec|url> [<spec|url> ...] --target DIR [--with-deps]
    spec = "pyarrow" | "pyarrow==26.0.0" | "pyarrow>=15"

Notes:
    * Wheel selection prefers cp3XX-win_amd64, then py3-none-any/any.
    * ``--with-deps`` recurses over ``Requires-Dist`` using ``packaging`` for
      marker/specifier evaluation when it is importable.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

PYPI = "https://pypi.org/pypi/{name}/json"
UA = {"User-Agent": "wheel-fetch/1.0"}


def _tag_rank(filename: str, py_tag: str) -> int:
    """Lower is better. Handles abi3 wheels (e.g. ``cp39-abi3-win_amd64``)."""
    if f"{py_tag}-{py_tag}-win_amd64" in filename:
        return 0
    if "abi3-win_amd64" in filename:
        return 1
    if f"{py_tag}-" in filename and "win_amd64" in filename:
        return 2
    if "-none-win_amd64" in filename:
        return 3
    if "py3-none-any" in filename:
        return 4
    if "-none-any" in filename:
        return 5
    return 99


def _resolve(name: str, spec: str | None) -> tuple[str, str, list[str]]:
    """Return (filename, url, requires_dist) for the best matching wheel."""
    data = json.load(urllib.request.urlopen(urllib.request.Request(PYPI.format(name=name), headers=UA), timeout=60))
    version = data["info"]["version"]
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    candidates: list[tuple[int, str, str]] = []
    for f in data["urls"]:
        fn = f["filename"]
        if not fn.endswith(".whl"):
            continue
        if f.get("yanked"):
            continue
        rank = _tag_rank(fn, py_tag)
        if rank >= 99:
            continue
        candidates.append((rank, fn, f["url"]))
    if not candidates and data.get("releases"):
        for rel_ver, files in data["releases"].items():
            for f in files:
                fn = f["filename"]
                if not fn.endswith(".whl") or f.get("yanked"):
                    continue
                rank = _tag_rank(fn, py_tag)
                if rank < 99:
                    candidates.append((rank, fn, f["url"]))
                    version = rel_ver
    if not candidates:
        raise RuntimeError(f"no compatible wheel for {name} {version}")
    candidates.sort(key=lambda c: c[0])
    _, fn, url = candidates[0]
    reqs = [r for r in (data["info"].get("requires_dist") or [])]
    return fn, url, reqs


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def _metadata_requires(wheel: Path) -> tuple[list[str], str, str]:
    """Read (Requires-Dist, Name, Version) from a wheel's METADATA."""
    with zipfile.ZipFile(wheel) as z:
        name = next((n for n in z.namelist() if n.endswith(".dist-info/METADATA")), None)
        if not name:
            return [], "", ""
        raw = z.read(name).decode("utf-8", "replace")
    reqs = re.findall(r"^Requires-Dist:\s*(.+)$", raw, re.M)
    nm = re.search(r"^Name:\s*(.+)$", raw, re.M)
    ver = re.search(r"^Version:\s*(.+)$", raw, re.M)
    return reqs, (nm.group(1).strip() if nm else ""), (ver.group(1).strip() if ver else "")


def _extract(wheel: Path, target: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        z.extractall(target)
    return names


def _applies(req: str) -> bool:
    """Evaluate a Requires-Dist environment marker; drop extras."""
    if ";" not in req:
        return True
    _, marker = req.split(";", 1)
    marker = marker.strip()
    if "extra ==" in marker or 'extra=="' in marker:
        return False
    try:
        from packaging.markers import Marker  # type: ignore

        return Marker(marker).evaluate()
    except Exception:  # noqa: BLE001
        low = marker.lower()
        if "sys_platform" in low and "win32" not in low:
            return False
        if "platform_system" in low and "windows" not in low:
            return False
        return "python_version" not in low


def _pkg_name(req: str) -> tuple[str, str | None]:
    req = req.split(";", 1)[0].strip()
    m = re.match(r"^([A-Za-z0-9._-]+)\s*(\[[^\]]*\])?\s*(\(|==|>=|<=|~=|!=|>|<)?\s*([^,;]*)?", req)
    if not m:
        return req, None
    name = m.group(1)
    op, ver = m.group(3), (m.group(4) or "").strip().rstrip(")")
    spec = f"{op}{ver}" if op and ver else None
    return name, spec


def main(argv: list[str]) -> int:
    target = Path(".")
    with_deps = "--with-deps" in argv
    argv = [a for a in argv if a != "--with-deps"]
    if "--target" in argv:
        i = argv.index("--target")
        target = Path(argv[i + 1])
        del argv[i : i + 2]
    target.mkdir(parents=True, exist_ok=True)

    import importlib.util

    done: set[str] = set()
    queue: list[tuple[str, str | None]] = []
    for a in argv:
        if a.startswith(("http://", "https://")):
            fn = a.rsplit("/", 1)[-1]
            dest = target.parent / ".wheels" / fn
            _download(a, dest)
            _extract(dest, target)
            print(f"[url] {fn}")
        else:
            queue.append(_pkg_name(a))

    failed: list[str] = []
    while queue:
        name, spec = queue.pop(0)
        key = name.lower().replace("_", "-")
        if key in done:
            continue
        if importlib.util.find_spec(name.replace("-", "_")) and not spec:
            done.add(key)
            continue
        try:
            fn, url, reqs = _resolve(name, spec)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{name}: {e}")
            print(f"  FAIL {name}: {e}")
            continue
        dest = target.parent / ".wheels" / fn
        if not dest.exists():
            _download(url, dest)
        _extract(dest, target)
        _, meta_name, meta_ver = _metadata_requires(dest)
        done.add(key)
        print(f"  ok {meta_name or name} {meta_ver}  ({fn})")
        if with_deps:
            for r in reqs:
                if _applies(r):
                    queue.append(_pkg_name(r))
    if failed:
        print("\n--- failures ---")
        for f in failed:
            print(" ", f)
    print(f"\ntarget: {target}  packages installed: {len(done)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
