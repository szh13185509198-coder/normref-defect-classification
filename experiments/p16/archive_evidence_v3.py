#!/usr/bin/env python3
"""Evidence manifest + archive covering Revision-C and P10–P15 plus figures v2–v4.

Usage:
    D:\Anaconda\envs\cole\python.exe -m experiments.p16.archive_evidence_v3 --archive-dir D:\intergration\evidence_archive
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p10 import archive_evidence as v1  # noqa: E402
from experiments.p12 import archive_evidence_v2 as v2  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

EVIDENCE_DIRS = v2.EVIDENCE_DIRS + (
    "ckpt/p13_pilot/dev", "ckpt/p13_pilot/cable_explore",
    "ckpt/p14_formal/mvtec", "ckpt/p14_formal/external", "ckpt/p14_formal/cable", "ckpt/p14_resource",
    "ckpt/p15/dev", "ckpt/p15/external", "ckpt/p15/cable_test", "docs/paper/figures_v4",
)
DOCS = v2.DOCS + ("docs/P13_RESIDUAL_CONSENSUS_MULTIREGION_PILOT_PROTOCOL.md", "docs/P14_FINAL_FORMAL_ADDENDUM.md",
                  "docs/P15_TWO_STAGE_SCREENING_PROTOCOL.md", "docs/P16_FIGURES_V4_PROTOCOL.md")
MANIFEST = PROJECT_ROOT / "docs" / "paper" / "P16_EVIDENCE_MANIFEST.json"


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
                            "sha256": v1.sha256(path), "archived": path.suffix not in v1.FEATURE_SUFFIXES})
    provenance = rc.git_provenance()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = Path(args.archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"p16_evidence_{(provenance['head'] or 'nohead')[:7]}_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for e in entries:
            if e["archived"]:
                tar.add(PROJECT_ROOT / e["path"], arcname=e["path"])
        for doc in DOCS:
            tar.add(PROJECT_ROOT / doc, arcname=doc)
    manifest = {"schema_version": "p16_evidence_manifest_v1", "created_utc": stamp, "provenance": provenance,
                "archive_file": str(archive), "archive_sha256": v1.sha256(archive),
                "missing_dirs": missing, "files": entries}
    rc.write_json(MANIFEST, manifest)
    print(f"{len(entries)} files hashed ({sum(e['archived'] for e in entries)} archived); missing dirs: {missing}")
    print(f"archive: {archive}  sha256: {manifest['archive_sha256']}")
    print(f"manifest: {MANIFEST.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
