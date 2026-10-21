#!/usr/bin/env python3
"""Build the full P6.B Stage 1 support-manifest bundle (195 manifests).

Reads ONLY the frozen feasibility outputs (``p6b_mvtec_fs_rgb_feasibility.json``
and ``p6b_mvtec_fs_rgb_image_records.jsonl``). Query-side handling is
firewall-minimized: JSONL lines are parsed, but query label/path fields are
not accessed, retained, or used for any manifest/class-universe/training
decision; the only query-side information consumed is
product/source_pools/rgb_sha256 (overlap audit). No image pixels, no model
code (stdlib only).

Hardening contract (locked):

- Frozen-source gate: the JSONL SHA256 must equal ``feasibility.output_hashes``
  and both feasible-product lists must equal the locked 13 products exactly.
- Bundle-level transactional write: an existing target directory is rejected;
  everything is completed in a sibling staging directory and published with a
  single ``os.replace``; any mid-build failure leaves nothing behind.
- Git preflight: the formal CLI requires a resolvable HEAD and a clean tree;
  ``--allow-dirty`` is an explicit diagnostic switch whose completion record
  is marked ``status=diagnostic`` / ``formal_eligible=false``.

Locked contract: ``docs/P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md`` §2/§5/§7/§8.
Any integrity violation is fail-closed (non-zero exit, no partial trust).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import (  # noqa: E402
    EXPECTED_MANIFEST_COUNT,
    LOCKED_K,
    LOCKED_PRODUCTS,
    LOCKED_SEEDS,
    P6BFailClosedError,
    atomic_write_json,
    build_support_manifest,
    finalize_manifest,
    load_frozen_records,
    manifest_self_sha256,
    query_sha_set_by_product,
    sha256_file,
    support_candidates_by_class,
)

FEASIBILITY_SCHEMA = "p6b_mvtec_fs_rgb_feasibility_v2"
COMPLETION_SCHEMA = "p6b_support_manifest_build_v1"
STAGING_SUFFIX = ".p6b_staging"


def git_commit_and_dirty(project_root: Path) -> tuple:
    """Resolve HEAD and dirty state; fail-closed if git is unusable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty_text = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError, FileNotFoundError) as exc:
        raise P6BFailClosedError(f"git preflight failed: {exc}") from exc
    if not commit:
        raise P6BFailClosedError("git preflight failed: empty commit hash")
    return commit, bool(dirty_text)


def verify_frozen_source(feasibility: dict, records_path: Path) -> None:
    """Locked frozen-source gate: hash pin + exact feasible-product sets."""
    expected_sha = (feasibility.get("output_hashes") or {}).get(records_path.name)
    if not expected_sha:
        raise P6BFailClosedError(
            "feasibility JSON lacks output_hashes for the records JSONL")
    actual_sha = sha256_file(records_path)
    if actual_sha != expected_sha:
        raise P6BFailClosedError(
            f"records JSONL sha256 {actual_sha} != frozen output_hashes "
            f"{expected_sha}")
    locked = sorted(LOCKED_PRODUCTS)
    common_products = feasibility.get("common_products_feasible_for_all_shots")
    if sorted(common_products or []) != locked:
        raise P6BFailClosedError(
            "common_products_feasible_for_all_shots != locked 13 products")
    by_shot = feasibility.get("feasible_products_by_shot") or {}
    for shot in ("1", "3", "5"):
        if sorted(by_shot.get(shot) or []) != locked:
            raise P6BFailClosedError(
                f"feasible_products_by_shot[{shot!r}] != locked 13 products")


def manifest_relative_path(product: str, k_shot: int, seed: int) -> Path:
    return Path("p6b_support_manifests") / product / f"k{k_shot}_s{seed}.json"


def build_all(feasibility_dir: Path, output_dir: Path, provenance: dict,
              diagnostic: bool) -> dict:
    """Build the full bundle into ``output_dir`` (caller-owned transaction)."""
    feasibility_path = feasibility_dir / "p6b_mvtec_fs_rgb_feasibility.json"
    records_path = feasibility_dir / "p6b_mvtec_fs_rgb_image_records.jsonl"
    if not feasibility_path.is_file():
        raise P6BFailClosedError(f"missing feasibility JSON: {feasibility_path}")
    if not records_path.is_file():
        raise P6BFailClosedError(f"missing records JSONL: {records_path}")

    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    if feasibility.get("schema_version") != FEASIBILITY_SCHEMA:
        raise P6BFailClosedError(
            f"unexpected feasibility schema_version: "
            f"{feasibility.get('schema_version')!r}"
        )
    if feasibility.get("formal_test_used") is not False:
        raise P6BFailClosedError("feasibility JSON must record formal_test_used=false")
    verify_frozen_source(feasibility, records_path)

    records = load_frozen_records(records_path)
    query_shas = query_sha_set_by_product(records)

    manifest_count = 0
    per_product_audit = {}
    for product in LOCKED_PRODUCTS:
        candidates = support_candidates_by_class(records, product)
        if not candidates:
            raise P6BFailClosedError(f"product {product!r}: no support candidates")

        support_shas = {
            record["rgb_sha256"]
            for class_candidates in candidates.values()
            for record in class_candidates
        }
        overlap = sorted(support_shas & query_shas.get(product, set()))
        if overlap:
            raise P6BFailClosedError(
                f"product {product!r}: support/query SHA overlap != 0 "
                f"({len(overlap)} records)"
            )

        product_manifests = {}
        for k_shot in LOCKED_K:
            for seed in LOCKED_SEEDS:
                manifest = finalize_manifest(
                    build_support_manifest(records, product, k_shot, seed)
                )
                if manifest["manifest_sha256"] != manifest_self_sha256(manifest):
                    raise P6BFailClosedError(
                        f"product {product!r} K={k_shot} seed={seed}: "
                        "self-hash verification failed"
                    )
                relative = manifest_relative_path(product, k_shot, seed)
                atomic_write_json(output_dir / relative, manifest)
                product_manifests[f"k{k_shot}_s{seed}"] = {
                    "relative_path": str(relative).replace("\\", "/"),
                    "manifest_sha256": manifest["manifest_sha256"],
                }
                manifest_count += 1

        per_product_audit[product] = {
            "class_universe": sorted(candidates.keys()),
            "support_candidate_count": {
                class_name: len(class_candidates)
                for class_name, class_candidates in sorted(candidates.items())
            },
            "support_query_sha_overlap": 0,
            "query_sha_metadata_count": len(query_shas.get(product, set())),
            "manifests": product_manifests,
        }

    if manifest_count != EXPECTED_MANIFEST_COUNT:
        raise P6BFailClosedError(
            f"manifest count {manifest_count} != expected {EXPECTED_MANIFEST_COUNT}"
        )

    # Firewall semantics of the completion flags:
    # - query_pixels_read=False: no image bytes are ever opened by Stage 1.
    # - query_labels_read=False means "not semantically consumed": JSONL lines
    #   are parsed, but query label/path fields are not accessed, retained, or
    #   used for any manifest/class-universe/training decision; it does NOT
    #   claim the raw JSON lines were never parsed.
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "diagnostic" if diagnostic else "completed",
        "formal_eligible": not diagnostic,
        "allow_dirty": bool(diagnostic),
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "products": list(LOCKED_PRODUCTS),
        "K_values": list(LOCKED_K),
        "seeds": list(LOCKED_SEEDS),
        "manifest_count": manifest_count,
        "feasibility_dir": str(feasibility_dir).replace("\\", "/"),
        "feasibility_json_sha256": sha256_file(feasibility_path),
        "records_jsonl_sha256": sha256_file(records_path),
        "project_provenance": provenance,
        "per_product": per_product_audit,
    }
    atomic_write_json(output_dir / "p6b_manifest_build_complete.json", completion)
    return completion


def build_bundle(feasibility_dir: Path, target_dir: Path, project_root: Path,
                 diagnostic: bool) -> dict:
    """Bundle-level transaction: preflight, stage everything, publish atomically.

    On any failure the official target directory never exists and the staging
    directory is removed; no partial or stale artifacts may remain.
    """
    commit, dirty = git_commit_and_dirty(project_root)
    if dirty and not diagnostic:
        raise P6BFailClosedError(
            "formal build requires a clean git tree; use --allow-dirty only "
            "for explicit diagnostic runs (formal_eligible=false)")
    if target_dir.exists():
        raise P6BFailClosedError(f"target directory already exists: {target_dir}")
    staging = target_dir.parent / (target_dir.name + STAGING_SUFFIX)
    if staging.exists():
        raise P6BFailClosedError(f"staging directory already exists: {staging}")
    try:
        completion = build_all(
            feasibility_dir, staging,
            provenance={"commit": commit, "dirty": dirty},
            diagnostic=diagnostic,
        )
        os.replace(staging, target_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feasibility-dir",
        default=str(PROJECT_ROOT / "data" / "p6b_probe_output_v2"),
        help="Frozen P6.B feasibility output directory (read-only input).",
    )
    parser.add_argument("--output-dir", required=True,
                        help="Official bundle directory (must NOT exist yet).")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Explicit diagnostic switch: tolerate a dirty git tree; the "
             "completion record is marked status=diagnostic/formal_eligible=false.",
    )
    args = parser.parse_args()

    feasibility_dir = Path(args.feasibility_dir).resolve()
    target_dir = Path(args.output_dir).resolve()
    if target_dir == feasibility_dir or feasibility_dir in target_dir.parents:
        raise P6BFailClosedError("output dir must not escape into the feasibility dir")

    completion = build_bundle(feasibility_dir, target_dir, PROJECT_ROOT,
                              diagnostic=args.allow_dirty)
    print(
        f"[P6.B Stage 1] wrote {completion['manifest_count']} manifests "
        f"(13 products x 3 K x 5 seeds) to {target_dir} "
        f"(status={completion['status']}, formal_eligible={completion['formal_eligible']})"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except P6BFailClosedError as exc:
        print(f"[P6.B Stage 1] FAIL-CLOSED: {exc}", file=sys.stderr)
        sys.exit(2)
