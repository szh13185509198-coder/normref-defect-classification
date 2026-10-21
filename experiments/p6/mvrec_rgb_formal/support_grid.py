"""MVREC-RGB formal support-grid orchestrator (195 isolated support units).

Schedules the single-method MVREC-RGB support grid exactly like the P6.B
grid orchestrator: independent subprocess per unit, sharding, resume
validation, and a cross-shard provenance gate before the global completion
is published.  A shard never publishes a scientific completion; only a
full-grid validation (or an explicit finalize-only run) may publish.

Execution is authorized for the support-only smoke phase (explicit
``--unit-filter``); the full 195-unit grid remains gated on main-agent
authorization: without ``--allow-full-grid`` the orchestrator refuses to
schedule more units than the smoke filter.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import (  # noqa: E402
    P6BFailClosedError,
    atomic_write_json,
)
from experiments.p6.mvrec_rgb_formal.execution_manifest import (  # noqa: E402
    GRID_COMPLETION,
    GRID_SCHEMA,
    METHOD,
    UNIT_COUNT,
    expected_units,
)
from experiments.p6.mvrec_rgb_formal.support_runner import (  # noqa: E402
    git_commit_and_dirty,
)
from experiments.p6.mvrec_rgb_formal.grid_audit import (  # noqa: E402
    validate_formal_grid,
    validate_support_unit,
)


def selected_units(shard_count: int, shard_index: int, unit_filter=None):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise P6BFailClosedError("invalid shard-count/shard-index")
    units = [unit for index, unit in enumerate(expected_units())
             if index % shard_count == shard_index]
    if unit_filter:
        allowed = set(unit_filter)
        units = [u for u in units if u.key in allowed]
        missing = allowed - {u.key for u in units}
        if missing:
            raise P6BFailClosedError(f"unit filter keys not in this shard: {sorted(missing)}")
    return units


def child_environment(cpu_threads_per_worker: int):
    if cpu_threads_per_worker < 1:
        raise P6BFailClosedError("cpu-threads-per-worker must be positive")
    env = os.environ.copy()
    threads = str(cpu_threads_per_worker)
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads
    env["MVREC_ALPHA_L14_CKPT"] = env.get("MVREC_ALPHA_L14_CKPT", "")
    return env


def build_command(args, unit, output_dir):
    command = [
        args.python, "-m", "experiments.p6.mvrec_rgb_formal.support_runner",
        "--manifest-dir", str(Path(args.bundle_dir).resolve()),
        "--image-root", str(Path(args.image_root).resolve()),
        "--alpha-ckpt", str(Path(args.alpha_ckpt).resolve()),
        "--product", unit.product, "--k", str(unit.k_shot),
        "--seed", str(unit.seed),
        "--output-dir", str(output_dir.resolve()),
    ]
    if args.device:
        command += ["--device", args.device]
    return command


def run_one_unit(args, unit, output_dir, commit, ordinal, total):
    command = build_command(args, unit, output_dir)
    print(f"[mvrec-rgb grid] run {ordinal}/{total} {unit.key}", flush=True)
    completed = subprocess.run(
        command, cwd=str(PROJECT_ROOT), check=False,
        env=child_environment(args.cpu_threads_per_worker))
    if completed.returncode != 0:
        raise P6BFailClosedError(
            f"support subprocess failed ({completed.returncode}): {unit.key}")
    validate_support_unit(unit, output_dir, commit)
    print(f"[mvrec-rgb grid] completed {ordinal}/{total} {unit.key}", flush=True)
    return unit.key


def run_grid(args) -> dict:
    root = Path(args.output_root).resolve()
    commit, dirty = git_commit_and_dirty(PROJECT_ROOT)
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

    units = [] if args.finalize_only else selected_units(
        args.shard_count, args.shard_index, args.unit_filter)
    if not args.finalize_only and not args.allow_full_grid:
        if len(units) > 1:
            raise P6BFailClosedError(
                "full 195-unit grid requires explicit --allow-full-grid "
                "(main-agent authorization); only single-unit support-only "
                "smoke is permitted otherwise")
    pending = []
    for ordinal, unit in enumerate(units, 1):
        output_dir = unit.directory(root)
        if output_dir.exists():
            validate_support_unit(unit, output_dir, commit)
            print(f"[mvrec-rgb grid] resume-valid {ordinal}/{len(units)} {unit.key}", flush=True)
            continue
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        pending.append((ordinal, unit, output_dir))

    failures = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_parallel, thread_name_prefix="mvrec-rgb-unit") as pool:
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

    try:
        validated = validate_formal_grid(root, require_global_completion=False)
    except P6BFailClosedError:
        if args.finalize_only:
            raise
        return {"status": "shard_completed", "shard_count": args.shard_count,
                "shard_index": args.shard_index, "unit_count": len(units)}

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
        "method": METHOD,
        "backbone_identities": validated["backbone_identities"],
        "units": records,
    }
    atomic_write_json(root / GRID_COMPLETION, global_record)
    validate_formal_grid(root, require_global_completion=True)
    return global_record


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--alpha-ckpt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=2)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--unit-filter", action="append", default=None,
                        help="restrict to exact unit keys "
                             "(e.g. mvrec_rgb/bottle/k1_s111); repeatable")
    parser.add_argument("--allow-full-grid", action="store_true",
                        help="explicit main-agent authorization for the full "
                             "195-unit grid")
    return parser


def main(argv=None):
    try:
        result = run_grid(build_parser().parse_args(argv))
        print(f"[mvrec-rgb grid] {result['status']}")
        return 0
    except (P6BFailClosedError, OSError) as exc:
        print(f"[mvrec-rgb grid] FAIL-CLOSED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
