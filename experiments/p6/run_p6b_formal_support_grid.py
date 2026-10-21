#!/usr/bin/env python3
"""P6.B formal support-grid orchestrator (1,755 isolated support-only runs)."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

# Direct-file launch (VS Code/debugpy/cloud shell) starts with
# ``experiments/p6`` on sys.path rather than the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import P6BFailClosedError, atomic_write_json
from experiments.p6 import build_p6b_support_manifests as builder
from experiments.p6.p6b_stage3_common import (
    GRID_COMPLETION, GRID_SCHEMA, METHODS, PRODUCTS, UNIT_COUNT,
    expected_units, file_sha256, validate_formal_grid, validate_support_unit,
)

def selected_units(shard_count: int, shard_index: int):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise P6BFailClosedError("invalid shard-count/shard-index")
    return [unit for index, unit in enumerate(expected_units())
            if index % shard_count == shard_index]


def schedule_units(units):
    """Interleave methods without changing shard membership or grid identity.

    The canonical grid index remains method-major. This order is operational:
    it prevents a cloud from spending hours on tiny prototype-only units while
    its GPU is idle, by mixing trainable and frozen methods for each config.
    """
    method_rank = {name: index for index, name in enumerate(METHODS)}
    product_rank = {name: index for index, name in enumerate(PRODUCTS)}
    return sorted(units, key=lambda unit: (
        product_rank[unit.product], unit.k_shot, unit.seed,
        method_rank[unit.method]))


def child_environment(cpu_threads_per_worker: int):
    if cpu_threads_per_worker < 1:
        raise P6BFailClosedError("cpu-threads-per-worker must be positive")
    env = os.environ.copy()
    threads = str(cpu_threads_per_worker)
    # Never inherit an empty/non-integer cloud setting: libgomp treats it as
    # invalid and every worker emits a warning. Bound CPU libraries explicitly
    # so GPU workers do not oversubscribe the host.
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads
    return env


def build_command(args, unit, output_dir):
    command = [
        args.python, "-m", "experiments.p6.run_p6b_stage2_support",
        "--bundle-dir", str(Path(args.bundle_dir).resolve()),
        "--image-root", str(Path(args.image_root).resolve()),
        "--product", unit.product, "--k", str(unit.k_shot),
        "--seed", str(unit.seed), "--method", unit.method,
        "--output-dir", str(output_dir.resolve()),
    ]
    if args.device:
        command += ["--device", args.device]
    return command


def run_one_unit(args, unit, output_dir, commit, ordinal, total):
    """Run and validate one isolated unit; safe for bounded thread dispatch."""
    command = build_command(args, unit, output_dir)
    print(f"[P6.B grid] run {ordinal}/{total} {unit.key}", flush=True)
    completed = subprocess.run(
        command, cwd=str(PROJECT_ROOT), check=False,
        env=child_environment(args.cpu_threads_per_worker))
    if completed.returncode != 0:
        raise P6BFailClosedError(
            f"support subprocess failed ({completed.returncode}): {unit.key}")
    validate_support_unit(unit, output_dir, commit)
    print(f"[P6.B grid] completed {ordinal}/{total} {unit.key}", flush=True)
    return unit.key


def run_grid(args) -> dict:
    root = Path(args.output_root).resolve()
    commit, dirty = builder.git_commit_and_dirty(PROJECT_ROOT)
    if dirty:
        raise P6BFailClosedError("formal grid orchestration requires a clean git tree")
    if root.exists() and not root.is_dir():
        raise P6BFailClosedError("output root exists and is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    if (root / GRID_COMPLETION).exists():
        validated = validate_formal_grid(root, require_global_completion=True)
        if validated["execution_commit"] != commit:
            raise P6BFailClosedError(
                "completed grid belongs to a different execution commit")
        return {"status": "already_completed", **validated}

    if args.max_parallel < 1:
        raise P6BFailClosedError("max-parallel must be positive")
    units = [] if args.finalize_only else schedule_units(
        selected_units(args.shard_count, args.shard_index))
    pending = []
    for ordinal, unit in enumerate(units, 1):
        output_dir = unit.directory(root)
        if output_dir.exists():
            validate_support_unit(unit, output_dir, commit)
            print(f"[P6.B grid] resume-valid {ordinal}/{len(units)} {unit.key}", flush=True)
            continue
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        pending.append((ordinal, unit, output_dir))

    # Threads only supervise independent child processes. Each unit retains
    # its own Python process, RNG stream, model instance, staging directory and
    # artifact validation. Concurrency is operational scheduling, not a change
    # to any scientific definition.
    failures = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_parallel, thread_name_prefix="p6b-unit") as pool:
        futures = {
            pool.submit(run_one_unit, args, unit, output_dir, commit,
                        ordinal, len(units)): unit
            for ordinal, unit, output_dir in pending
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except BaseException as exc:
                failures.append((futures[future].key, exc))
    if failures:
        detail = "; ".join(f"{key}: {exc}" for key, exc in failures[:5])
        raise P6BFailClosedError(
            f"{len(failures)} concurrent support unit(s) failed; {detail}")

    # A shard never publishes a scientific completion. The last shard (or an
    # explicit finalize-only invocation) may publish only after all 1,755 pass.
    try:
        validated = validate_formal_grid(root, require_global_completion=False)
    except P6BFailClosedError:
        if args.finalize_only:
            raise
        return {"status": "shard_completed", "shard_count": args.shard_count,
                "shard_index": args.shard_index, "unit_count": len(units)}
    # Cross-shard provenance gate: every pre-existing unit must come from this
    # exact clean execution commit. Do not publish first and discover drift.
    for unit in expected_units():
        validate_support_unit(unit, unit.directory(root), commit)
    records = validated["units"]
    global_record = {
        "schema_version": GRID_SCHEMA,
        "status": "completed",
        "formal_eligible": True,
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "execution_commit": commit,
        "unit_count": UNIT_COUNT,
        "backbone_identities": validated["backbone_identities"],
        "units": records,
    }
    atomic_write_json(root / GRID_COMPLETION, global_record)
    # Re-read and validate exact bytes/index after publication.
    validate_formal_grid(root, require_global_completion=True)
    return global_record


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="bounded independent support subprocesses; default 1")
    parser.add_argument("--cpu-threads-per-worker", type=int, default=2,
                        help="explicit OMP/MKL/OpenBLAS threads per child; default 2")
    parser.add_argument("--finalize-only", action="store_true",
                        help="run no models; publish completion only if all 1,755 units validate")
    return parser


def main(argv=None):
    try:
        result = run_grid(build_parser().parse_args(argv))
        print(f"[P6.B grid] {result['status']}")
        return 0
    except (P6BFailClosedError, OSError) as exc:
        print(f"[P6.B grid] FAIL-CLOSED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
