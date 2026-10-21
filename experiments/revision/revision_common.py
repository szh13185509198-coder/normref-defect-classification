"""Shared paths, grid constants and metric helpers for Addendum C analyses.

Torch-free on purpose: C1 (bootstrap) must run without loading any model.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
from pathlib import Path

from experiments.p6.p6b_execution_common import (
    LOCKED_K,
    LOCKED_PRODUCTS,
    LOCKED_SEEDS,
    P6BFailClosedError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "ckpt" / "revision_c"

FORMAL_QUERY_DIR = PROJECT_ROOT / "ckpt" / "p6b_formal_query_v1"
FORMAL_AUDIT = FORMAL_QUERY_DIR / "p6b_formal_query_audit.json"
FORMAL_PREDICTIONS = FORMAL_QUERY_DIR / "p6b_formal_query_predictions.jsonl"
SUPPORT_GRID = PROJECT_ROOT / "ckpt" / "p6b_formal_support_grid_v3"
RECORDS_PATH = PROJECT_ROOT / "data" / "p6b_probe_output_v2" / "p6b_mvtec_fs_rgb_image_records.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "mvtec-fs"
STAGE1_BUNDLE = PROJECT_ROOT / "data" / "p6b_stage1_manifest_bundle"

QUERY_FEATURES = OUT_ROOT / "c2_fusion_query_features.pt"
FIDELITY_REPORT = OUT_ROOT / "c2_fidelity_report.json"
C2_RESULTS = OUT_ROOT / "c2_frozen_feature_baselines.json"
C2_PREDICTIONS = OUT_ROOT / "c2_frozen_feature_predictions.jsonl"
C1_RESULTS = OUT_ROOT / "c1_paired_bootstrap.json"
C1_TABLE = OUT_ROOT / "c1_paired_bootstrap_table.md"
C3_DIR = OUT_ROOT / "c3_resource"

PRODUCTS = tuple(LOCKED_PRODUCTS)
K_VALUES = tuple(LOCKED_K)
SEEDS = tuple(LOCKED_SEEDS)
METRICS = ("balanced_accuracy", "macro_f1", "top1_accuracy")

__all__ = [
    "P6BFailClosedError", "PROJECT_ROOT", "OUT_ROOT",
    "PRODUCTS", "K_VALUES", "SEEDS", "METRICS",
]


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_provenance() -> dict:
    def _run(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = _run("status", "--porcelain", "--untracked-files=no")
    return {"head": _run("rev-parse", "HEAD"),
            "tracked_tree_dirty": bool(status) if status is not None else None}


def metric_grid_from_rows(rows) -> dict:
    """{method: {(product, K, seed): {metric: value}}} with duplicate/NaN checks."""
    grid = {}
    for row in rows:
        key = (row["product"], int(row["K"]), int(row["seed"]))
        method_grid = grid.setdefault(row["method"], {})
        if key in method_grid:
            raise P6BFailClosedError(f"duplicate metric row {row['method']}/{key}")
        values = {}
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise P6BFailClosedError(f"non-finite {metric} at {row['method']}/{key}")
            values[metric] = value
        method_grid[key] = values
    return grid


def require_complete(grid: dict, method: str) -> None:
    method_grid = grid.get(method)
    if not method_grid:
        raise P6BFailClosedError(f"missing metric grid for {method}")
    missing = [(p, k, s) for p in PRODUCTS for k in K_VALUES for s in SEEDS
               if (p, k, s) not in method_grid]
    if missing:
        raise P6BFailClosedError(f"{method}: {len(missing)} missing units, e.g. {missing[:3]}")


def load_formal_metric_grid() -> dict:
    audit = load_json(FORMAL_AUDIT)
    if audit.get("status") != "completed" or audit.get("formal_eligible") is not True:
        raise P6BFailClosedError("formal P6.B audit is not completed/eligible")
    return metric_grid_from_rows(audit["raw_metrics"])


def product_macro_summary(grid: dict, method: str) -> dict:
    """Locked aggregation: equal-product macro per (K, seed), then mean and ddof=1 SD."""
    require_complete(grid, method)
    out = {}
    for k_shot in K_VALUES:
        per_seed = {metric: [] for metric in METRICS}
        for seed in SEEDS:
            for metric in METRICS:
                per_seed[metric].append(statistics.fmean(
                    grid[method][(p, k_shot, seed)][metric] for p in PRODUCTS))
        out[str(k_shot)] = {
            metric: {"mean": statistics.fmean(vals), "std_ddof1": statistics.stdev(vals),
                     "per_seed": {str(s): v for s, v in zip(SEEDS, vals)}}
            for metric, vals in per_seed.items()
        }
    return out


def write_json(path: Path, obj) -> None:
    """Human-readable atomic JSON (string keys only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)
