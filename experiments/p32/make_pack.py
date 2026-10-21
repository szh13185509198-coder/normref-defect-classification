#!/usr/bin/env python3
"""P32: build the small add-on pack for the device (P32 code + the input images and metadata), to be unpacked into the
P31 bundle root (``repo/`` and ``MVTEC/``) without overwriting anything. Model weights and the rest of the code come
from the P31 bundle already on the device. Every file is listed with its SHA256 in ``P32_MANIFEST.json``.

Usage:
    python -m experiments.p32.make_pack --inputs <list from device_extract --list-inputs> --mvtec-root <root> --out <tar>
    python -m experiments.p32.make_pack --verify <bundle root on the device>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE = ("experiments/p32/__init__.py", "experiments/p32/device_extract.py", "experiments/p32/make_pack.py",
        "experiments/p33/__init__.py", "experiments/p33/power_profile.py",
        "experiments/p26/cost_profile.py")   # replaces the P31 copy (FP16 blocks added; see the dispatch)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(inputs: Path, mvtec_root: Path, out: Path) -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    files = {}
    for line in inputs.read_text(encoding="utf-8").splitlines():
        p = Path(line.strip()).resolve()
        for base, prefix in ((mvtec_root.resolve(), "MVTEC"), (PROJECT_ROOT, "repo")):
            try:
                files[f"{prefix}/{p.relative_to(base).as_posix()}"] = p
                break
            except ValueError:
                continue
        else:
            raise SystemExit(f"outside the repo and the MVTec root: {p}")
    manifest = {"schema_version": "p32_pack_v1", "git_head": head, "files": {}}
    with tarfile.open(out, "w") as tar:
        for rel in CODE:   # committed bytes, as in the P31 bundle
            data = subprocess.run(["git", "-c", "core.autocrlf=false", "show", f"HEAD:{rel}"], cwd=PROJECT_ROOT,
                                  capture_output=True, check=True).stdout
            info = tarfile.TarInfo(f"repo/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            manifest["files"][f"repo/{rel}"] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, path in sorted(files.items()):
            manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            tar.add(str(path), arcname=name)
        blob = json.dumps(manifest, indent=1).encode("utf-8")
        info = tarfile.TarInfo("P32_MANIFEST.json")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    print(f"pack {out} ({out.stat().st_size / 2 ** 20:.0f} MiB): git {head[:8]}, {len(manifest['files'])} files; "
          f"sha256 {sha256(out)}")
    return 0


def verify(root: Path) -> int:
    manifest = json.loads((root / "P32_MANIFEST.json").read_text(encoding="utf-8"))
    bad = [n for n, v in manifest["files"].items() if not (root / n).is_file() or sha256(root / n) != v["sha256"]]
    print(f"{len(manifest['files']) - len(bad)}/{len(manifest['files'])} files OK" + (f"; BAD: {bad[:10]}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--mvtec-root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", type=Path)
    a = ap.parse_args(argv)
    if a.verify:
        return verify(a.verify)
    return build(a.inputs, a.mvtec_root, a.out)


if __name__ == "__main__":
    sys.exit(main())
