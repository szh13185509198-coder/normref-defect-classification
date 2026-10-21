#!/usr/bin/env python3
"""Evidence manifest + archive covering Revision-C, P10, P11 and P12 (P12 protocol, section E).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p12.archive_evidence_v2 --archive-dir D:\\intergration\\evidence_archive
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
from experiments.revision import revision_common as rc  # noqa: E402

EVIDENCE_DIRS = v1.EVIDENCE_DIRS + (
    "ckpt/p10_pilot/sensitivity", "ckpt/p11_pilot", "ckpt/p11_formal/mvtec", "ckpt/p11_formal/external",
    "ckpt/p12/threshold_dev", "ckpt/p12/threshold_mvtec", "ckpt/p12/threshold_external",
    "ckpt/p12/cable", "ckpt/p12/normal_aware", "docs/paper/figures_v2", "docs/paper/figures_v3",
)
DOCS = ("docs/agent/RESEARCH_LEDGER.md", "docs/P6B_REVISION_ADDENDUM_C.md",
        "docs/P10_ANNOTATION_FREE_LOCALIZATION_PILOT_PROTOCOL.md", "docs/P10_FORMAL_EVALUATION_PROTOCOL.md",
        "docs/P10_EXTERNAL_VISA_MTD_PROTOCOL.md", "docs/P10_SUPPLEMENT_S_PROTOCOL.md",
        "docs/P10_INDUSTRIAL_RELIABILITY_PROTOCOL.md", "docs/P10_FIGURES_AND_SENSITIVITY_PROTOCOL.md",
        "docs/P11_ADAPTIVE_REGION_AND_GUIDED_MATCHING_PILOT_PROTOCOL.md", "docs/P11_FORMAL_ADDENDUM.md",
        "docs/P12_THRESHOLD_CONTROLS_RELIABILITY_FIGURES_PROTOCOL.md")
MANIFEST = PROJECT_ROOT / "docs" / "paper" / "P12_EVIDENCE_MANIFEST.json"


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
    archive = archive_dir / f"p12_evidence_{(provenance['head'] or 'nohead')[:7]}_{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for e in entries:
            if e["archived"]:
                tar.add(PROJECT_ROOT / e["path"], arcname=e["path"])
        for doc in DOCS:
            tar.add(PROJECT_ROOT / doc, arcname=doc)
    manifest = {"schema_version": "p12_evidence_manifest_v1", "created_utc": stamp, "provenance": provenance,
                "archive_file": str(archive), "archive_sha256": v1.sha256(archive),
                "missing_dirs": missing, "files": entries}
    rc.write_json(MANIFEST, manifest)
    print(f"{len(entries)} files hashed ({sum(e['archived'] for e in entries)} archived); missing dirs: {missing}")
    print(f"archive: {archive}  sha256: {manifest['archive_sha256']}")
    print(f"manifest: {MANIFEST.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
