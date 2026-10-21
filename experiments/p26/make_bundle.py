#!/usr/bin/env python3
"""P26: build the offline bundle for timing on another machine (e.g. a rented RTX 4090 or a Jetson).

No GitHub is involved: the code comes from ``git archive`` of the local HEAD (the committed tree only),
and the data, weights and ignored third-party code are exactly the files that a traced local P26 run
opened (``P26_TRACE_OPENS``), plus the two third-party trees imported as code. Every file is listed
with its SHA256 in ``BUNDLE_MANIFEST.json`` inside the tarball; ``--verify`` re-checks an extracted
bundle against it.

Usage:
    python -m experiments.p26.make_bundle --trace <opens.txt> --mvtec-root D:\\intergration\\MVTEC --out <bundle.tar>
    python -m experiments.p26.make_bundle --verify <extracted dir>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP_CACHE = Path(os.path.expanduser("~")) / ".cache" / "clip"
TORCH_HUB_CACHE = Path(os.path.expanduser("~")) / ".cache" / "torch" / "hub"
THIRD_PARTY = ("third_party/dinov2", "third_party/CLIP-LoRA")   # ignored by git, imported as code
SKIP_PARTS = ("__pycache__",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def classify(path: Path, mvtec_root: Path, tracked: set):
    """-> (bundle-relative name) for files we ship, or None (git-tracked files come from git archive)."""
    p = path.resolve()
    for base, prefix in ((PROJECT_ROOT, "repo"), (mvtec_root.resolve(), "MVTEC"), (CLIP_CACHE.resolve(), "clip_cache"),
                         (TORCH_HUB_CACHE.resolve(), "torch_hub_cache")):
        try:
            rel = p.relative_to(base)
        except ValueError:
            continue
        if any(part in SKIP_PARTS for part in rel.parts) or p.suffix in (".pyc",):
            return None
        if prefix == "repo" and (rel.parts[:2] == ("ckpt", "p26") or rel.as_posix() in tracked):
            return None   # outputs of the traced run itself, or already in the git archive
        return f"{prefix}/{rel.as_posix()}"
    return None


def build(trace: Path, mvtec_root: Path, out: Path) -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments", "tests"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"commit first; uncommitted changes:\n{dirty}")
    tracked = set(subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                 check=True).stdout.splitlines())
    files = {}
    for line in trace.read_text(encoding="utf-8").splitlines():
        path = Path(line.strip())
        if path.is_file():
            name = classify(path, mvtec_root, tracked)
            if name:
                files[name] = path
    for tp in THIRD_PARTY:
        for path in (PROJECT_ROOT / tp).rglob("*"):
            if path.is_file() and not any(part in SKIP_PARTS for part in path.parts):
                files[f"repo/{path.relative_to(PROJECT_ROOT).as_posix()}"] = path
    # core.autocrlf=false: ship the committed bytes (plus explicit .gitattributes eol rules), i.e. exactly what the
    # traced laptop run read. With autocrlf=true, git archive rewrote 496 text files to CRLF in bundle v3 and every
    # SHA-checked input (episode file, data manifests) was refused on the RTX 4090.
    code = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=tar", "--prefix=repo/", "HEAD"],
                          cwd=PROJECT_ROOT, capture_output=True, check=True).stdout
    modified = {line[3:].strip().strip('"') for line in subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, capture_output=True, text=True,
        check=True).stdout.splitlines()}
    archived, mismatched = {}, []
    with tarfile.open(fileobj=io.BytesIO(code)) as src:
        for member in src.getmembers():
            if member.isfile():
                data = src.extractfile(member).read()
                rel = member.name[len("repo/"):]
                archived[member.name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                local = PROJECT_ROOT / rel
                # a CRLF working copy (core.autocrlf=true) of an LF commit is the same file; we ship the committed bytes
                if rel not in modified and local.is_file() and \
                        local.read_bytes().replace(b"\r\n", b"\n") != data.replace(b"\r\n", b"\n"):
                    mismatched.append(rel)
    if mismatched:
        raise SystemExit(f"{len(mismatched)} archived files differ from the working copy, e.g. {mismatched[:5]}")
    manifest = {"schema_version": "p26_bundle_v2", "built": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": head,
                "code": "git -c core.autocrlf=false archive HEAD (tracked files; byte-identical to the working copy)",
                "layout": {"repo/": "repository root (code + traced data/ckpt/weights + third_party)",
                           "MVTEC/": "MVTec AD root (only traced files)",
                           "clip_cache/": "copy into ~/.cache/clip on the target machine",
                           "torch_hub_cache/": "copy into ~/.cache/torch/hub on the target machine"},
                "files": {}}
    with tarfile.open(out, "w") as tar:
        with tarfile.open(fileobj=io.BytesIO(code)) as src:
            for member in src.getmembers():
                tar.addfile(member, src.extractfile(member) if member.isfile() else None)
        manifest["files"].update(archived)
        for name, path in sorted(files.items()):
            manifest["files"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            tar.add(str(path), arcname=name)
        blob = json.dumps(manifest, indent=1).encode("utf-8")
        info = tarfile.TarInfo("BUNDLE_MANIFEST.json")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    total = sum(manifest["files"][n]["bytes"] for n in files)
    print(f"bundle {out} ({out.stat().st_size / 2 ** 30:.2f} GiB): git {head[:8]}, {len(files)} data/weight/third-party "
          f"files ({total / 2 ** 30:.2f} GiB); sha256 {sha256(out)}")
    return 0


def verify(root: Path) -> int:
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    bad = [n for n, v in manifest["files"].items() if not (root / n).is_file() or sha256(root / n) != v["sha256"]]
    print(f"{len(manifest['files']) - len(bad)}/{len(manifest['files'])} files OK" + (f"; BAD: {bad[:10]}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--mvtec-root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--verify", type=Path)
    a = ap.parse_args(argv)
    if a.verify:
        return verify(a.verify)
    if not (a.trace and a.mvtec_root and a.out):
        raise SystemExit("--trace, --mvtec-root and --out are required")
    return build(a.trace, a.mvtec_root, a.out)


if __name__ == "__main__":
    sys.exit(main())
