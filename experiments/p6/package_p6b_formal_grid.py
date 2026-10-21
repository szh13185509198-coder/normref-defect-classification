#!/usr/bin/env python3
"""Validate and package a P6.B formal support grid (or an explicit shard).

The archive contains only validated completion files and their registered
artifacts.  It never opens query records, query pixels, or query labels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import P6BFailClosedError
from experiments.p6 import build_p6b_support_manifests as builder
from experiments.p6.p6b_stage3_common import (
    GRID_COMPLETION,
    SUPPORT_COMPLETION,
    file_sha256,
    expected_units,
    load_json,
    validate_formal_grid,
    validate_support_unit,
)
from experiments.p6.run_p6b_formal_support_grid import selected_units


PACKAGE_SCHEMA = "p6b_formal_support_shard_package_v1"
PACKAGE_MANIFEST = "p6b_shard_package_manifest.json"


def _safe_relative_file(unit_dir: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise P6BFailClosedError(f"invalid registered artifact path: {relative!r}")
    root = unit_dir.resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise P6BFailClosedError(f"registered artifact escapes unit or is missing: {relative}")
    return candidate


def build_package_plan(grid_root: Path, shard_count=None, shard_index=None) -> dict:
    """Deep-validate a full grid or explicit shard and return its manifest."""
    grid_root = Path(grid_root).resolve()
    if not grid_root.is_dir():
        raise P6BFailClosedError(f"grid root is not a directory: {grid_root}")
    packaging_commit, dirty = builder.git_commit_and_dirty(PROJECT_ROOT)
    if dirty:
        raise P6BFailClosedError("shard packaging requires a clean git tree")

    full_grid = shard_count is None and shard_index is None
    if (shard_count is None) != (shard_index is None):
        raise P6BFailClosedError(
            "shard-count and shard-index must be supplied together, or both omitted")
    if full_grid:
        validated = validate_formal_grid(grid_root, require_global_completion=True)
        execution_commit = validated["execution_commit"]
        units = list(expected_units())
    else:
        if (grid_root / GRID_COMPLETION).exists():
            raise P6BFailClosedError(
                "global grid completion exists; package the full grid without shard arguments")
        units = selected_units(shard_count, shard_index)
        execution_commit = None
    unit_records = []
    file_records = []
    archive_paths = set()
    for unit in units:
        unit_dir = unit.directory(grid_root)
        # Full-grid validation above has already performed the deep artifact
        # hash pass. Avoid hashing large model artifacts a second time merely
        # to construct the package manifest.
        if full_grid:
            completion = load_json(unit_dir / SUPPORT_COMPLETION)
        else:
            raw_completion = load_json(unit_dir / SUPPORT_COMPLETION)
            candidate_commit = (raw_completion.get("execution") or {}).get("commit")
            if execution_commit is None:
                execution_commit = candidate_commit
            completion = validate_support_unit(unit, unit_dir, execution_commit)
        completion_path = unit_dir / SUPPORT_COMPLETION
        unit_files = []
        completion_relative = completion_path.relative_to(grid_root).as_posix()
        completion_record = {
            "relative_path": completion_relative,
            "size_bytes": completion_path.stat().st_size,
            "sha256": file_sha256(completion_path),
        }
        source_records = [completion_record]
        for artifact in completion["artifacts"]:
            path = _safe_relative_file(unit_dir, artifact["relative_path"])
            source_records.append({
                "relative_path": path.relative_to(grid_root).as_posix(),
                "size_bytes": artifact["size_bytes"],
                "sha256": artifact["sha256"],
            })

        for record in source_records:
            relative = record["relative_path"]
            if relative in archive_paths:
                raise P6BFailClosedError(f"duplicate archive path: {relative}")
            archive_paths.add(relative)
            file_records.append(record)
            unit_files.append(record)
        unit_records.append({
            "key": unit.key,
            "completion_sha256": file_sha256(completion_path),
            "files": unit_files,
        })

    if full_grid:
        global_path = grid_root / GRID_COMPLETION
        file_records.append({
            "relative_path": GRID_COMPLETION,
            "size_bytes": global_path.stat().st_size,
            "sha256": file_sha256(global_path),
        })

    return {
        "schema_version": PACKAGE_SCHEMA,
        "status": "completed",
        "formal_eligible": True,
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "execution_commit": execution_commit,
        "packaging_commit": packaging_commit,
        "package_scope": "full_grid" if full_grid else "shard",
        "shard_count": shard_count,
        "shard_index": shard_index,
        "unit_count": len(units),
        "file_count": len(file_records),
        "grid_directory_name": grid_root.name,
        "units": unit_records,
        "files": file_records,
    }


def _canonical_json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _write_sha256_sidecar(archive: Path) -> Path:
    sidecar = Path(str(archive) + ".sha256")
    line = f"{file_sha256(archive)}  {archive.name}\n"
    temporary = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    temporary.write_text(line, encoding="ascii", newline="\n")
    os.replace(temporary, sidecar)
    return sidecar


def create_archive(grid_root: Path, output: Path, plan: dict, compression_level: int) -> dict:
    """Create a zstd-compressed GNU-tar archive from an already validated plan."""
    grid_root = Path(grid_root).resolve()
    output = Path(output).resolve()
    if output.suffixes[-2:] != [".tar", ".zst"]:
        raise P6BFailClosedError("output must end in .tar.zst")
    if output == grid_root or grid_root in output.parents:
        raise P6BFailClosedError("archive output must be outside the grid root")
    if output.exists() or Path(str(output) + ".sha256").exists():
        raise P6BFailClosedError("archive or SHA256 sidecar already exists")
    if not 1 <= compression_level <= 19:
        raise P6BFailClosedError("zstd compression level must be in [1, 19]")
    if shutil.which("tar") is None or shutil.which("zstd") is None:
        raise P6BFailClosedError("GNU tar and zstd must both be installed")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p6b-package-") as temp_name:
        temp = Path(temp_name)
        manifest_path = temp / PACKAGE_MANIFEST
        manifest_path.write_bytes(_canonical_json_bytes(plan))
        file_list = temp / "files.txt"
        # Paths are newline-safe because production artifact names are locked;
        # reject any drift instead of relying on ambiguous tar list parsing.
        members = [f"{grid_root.name}/{record['relative_path']}" for record in plan["files"]]
        package_member = f"{grid_root.name}/{PACKAGE_MANIFEST}"
        if any("\n" in value or "\r" in value for value in members):
            raise P6BFailClosedError("archive member contains a newline")
        file_list.write_text("\n".join(members) + "\n", encoding="utf-8", newline="\n")

        # Stage only the tiny package manifest beside the source tree. All
        # research artifacts remain in place and are read directly by tar.
        staged_manifest = grid_root / PACKAGE_MANIFEST
        if staged_manifest.exists():
            raise P6BFailClosedError(f"reserved package manifest already exists: {staged_manifest}")
        try:
            staged_manifest.write_bytes(manifest_path.read_bytes())
            with file_list.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(package_member + "\n")
            command = [
                "tar", "-I", f"zstd -{compression_level} -T0", "-cf", str(output),
                "-C", str(grid_root.parent), "--verbatim-files-from",
                "--files-from", str(file_list),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0 or not output.is_file():
                if output.exists():
                    output.unlink()
                raise P6BFailClosedError(
                    f"tar/zstd packaging failed with exit code {completed.returncode}")
        finally:
            if staged_manifest.exists():
                staged_manifest.unlink()

    sidecar = _write_sha256_sidecar(output)
    return {
        "archive": str(output),
        "archive_size_bytes": output.stat().st_size,
        "archive_sha256": file_sha256(output),
        "sha256_sidecar": str(sidecar),
        "unit_count": plan["unit_count"],
        "file_count": plan["file_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", required=True)
    parser.add_argument("--shard-count", type=int, default=None,
                        help="only for a shard stored on a separate machine")
    parser.add_argument("--shard-index", type=int, default=None,
                        help="only for a shard stored on a separate machine")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compression-level", type=int, default=3)
    return parser


def main(argv=None) -> int:
    try:
        args = build_parser().parse_args(argv)
        plan = build_package_plan(args.grid_root, args.shard_count, args.shard_index)
        result = create_archive(args.grid_root, args.output, plan, args.compression_level)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    except (P6BFailClosedError, OSError) as exc:
        print(f"[P6.B package] FAIL-CLOSED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
