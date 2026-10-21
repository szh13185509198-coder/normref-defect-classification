#!/usr/bin/env python3
"""Fetch Real-IAD (512-px version + annotation jsons) for the P19 blind test. Never decodes an image.

Downloads realiad_jsons.zip, LICENSE/README and realiad_512/*.zip from the gated Hugging Face
dataset Real-IAD/Real-IAD (the user's cached login token is used), records the SHA256 of every
archive, extracts them under data/Real-IAD/, and writes an inventory: per object and per path
component, the number of files and the SHA256 of every extracted file. Image contents are only
hashed, never decoded or displayed.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe experiments/p19/fetch_realiad.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "Real-IAD"
ARCHIVES = ROOT / "hf"
PATTERNS = ["realiad_jsons.zip", "LICENSE.txt", "README.md", "ReadMe.txt", "realiad_512/*.zip"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    from huggingface_hub import snapshot_download

    if (ROOT / "realiad_inventory.json").exists():
        raise SystemExit(f"{ROOT / 'realiad_inventory.json'} exists - already fetched; STOP and report")
    snapshot_download(repo_id="Real-IAD/Real-IAD", repo_type="dataset", allow_patterns=PATTERNS, local_dir=str(ARCHIVES))
    archives = sorted(ARCHIVES.rglob("*.zip"))
    if len(archives) != 31:
        raise SystemExit(f"expected 31 archives (jsons + 30 objects), found {len(archives)}")
    archive_sha = {str(a.relative_to(ARCHIVES)).replace("\\", "/"): sha256(a) for a in archives}
    for a in archives:
        target = ROOT / ("realiad_jsons" if a.name == "realiad_jsons.zip" else "realiad_512")
        with zipfile.ZipFile(a) as z:
            z.extractall(target)
        print(f"extracted {a.name}", flush=True)
    files = {}
    for part in ("realiad_jsons", "realiad_512"):
        for f in sorted((ROOT / part).rglob("*")):
            if f.is_file():
                files[str(f.relative_to(ROOT)).replace("\\", "/")] = sha256(f)
    tree = Counter("/".join(rel.split("/")[:4]) for rel in files if rel.startswith("realiad_512/"))
    inventory = {"source": "huggingface.co/datasets/Real-IAD/Real-IAD", "license": "CC BY-NC-SA 4.0 (research only)",
                 "archives_sha256": archive_sha, "file_count": len(files),
                 "file_count_by_first_four_path_components": dict(sorted(tree.items())), "files_sha256": files}
    (ROOT / "realiad_inventory.json").write_text(json.dumps(inventory, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"archives {len(archives)}, extracted files {len(files)}")
    for k, v in sorted(tree.items())[:60]:
        print(f"{v:7d}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
