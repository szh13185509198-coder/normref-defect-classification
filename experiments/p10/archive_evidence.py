#!/usr/bin/env python3
"""Preserve Revision-C / P10 evidence: SHA256 manifest (tracked) + tar.gz of result files.

Large feature caches are hashed and listed but not copied into the archive.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.archive_evidence --archive-dir D:\\intergration\\evidence_archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.revision import revision_common as rc  # noqa: E402

EVIDENCE_DIRS = (
    "ckpt/revision_c", "ckpt/revision_c/c3_resource",
    "ckpt/p10_pilot", "ckpt/p10_pilot/dev_features", "ckpt/p10_pilot/normal_features",
    "ckpt/p10_formal/results", "ckpt/p10_formal/features", "ckpt/p10_formal/supplement_s",
    "ckpt/p10_external/results", "ckpt/p10_external/normal_aware", "ckpt/p10_cable/results",
)
FEATURE_SUFFIXES = (".pt",)  # hashed, not archived
MANIFEST = PROJECT_ROOT / "docs" / "paper" / "P10_EVIDENCE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive-dir", required=True)
    args = parser.parse_args(argv)
    entries, missing = [], []
    for rel in EVIDENCE_DIRS:
        folder = PROJECT_ROOT / rel
        if not folder.is_dir():
            missing.append(rel)
            continue
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            entries.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "bytes": path.stat().st_size,
                            "sha256": sha256(path), "archived": path.suffix not in FEATURE_SUFFIXES})
    provenance = rc.git_provenance()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"p10_evidence_{(provenance['head'] or 'nohead')[:7]}_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for e in entries:
            if e["archived"]:
                tar.add(PROJECT_ROOT / e["path"], arcname=e["path"])
        for doc in ("docs/agent/RESEARCH_LEDGER.md", "docs/P6B_REVISION_ADDENDUM_C.md",
                    "docs/P10_ANNOTATION_FREE_LOCALIZATION_PILOT_PROTOCOL.md",
                    "docs/P10_FORMAL_EVALUATION_PROTOCOL.md", "docs/P10_EXTERNAL_VISA_MTD_PROTOCOL.md",
                    "docs/P10_SUPPLEMENT_S_PROTOCOL.md", "docs/P10_INDUSTRIAL_RELIABILITY_PROTOCOL.md"):
            tar.add(PROJECT_ROOT / doc, arcname=doc)
    manifest = {"schema_version": "p10_evidence_manifest_v1", "created_utc": stamp, "provenance": provenance,
                "archive_file": str(archive), "archive_sha256": sha256(archive),
                "missing_dirs": missing, "files": entries}
    rc.write_json(MANIFEST, manifest)
    print(f"{len(entries)} files hashed ({sum(e['archived'] for e in entries)} archived); missing dirs: {missing}")
    print(f"archive: {archive}  sha256: {manifest['archive_sha256']}")
    print(f"manifest: {MANIFEST.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
