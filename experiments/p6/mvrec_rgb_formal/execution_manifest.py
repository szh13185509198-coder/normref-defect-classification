"""MVREC-RGB formal support-grid execution manifest (195 units).

The MVREC-RGB formal comparison is the single-method adaptation grid:
13 locked P6.B products x K in {1,3,5} x seeds {111,222,333,444,555} =
195 support units, each reusing the frozen P6.B support manifest of the
pinned Stage 1 bundle (Protocol A section 8).

This module is pure infrastructure: it reads manifest metadata (class
universe, support identities) but never opens an image, never loads a model,
never reads query pixels/labels/metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.p6.p6b_execution_common import (
    EXPECTED_MANIFEST_COUNT,
    LOCKED_K,
    LOCKED_PRODUCTS,
    LOCKED_SEEDS,
    P6BFailClosedError,
    atomic_write_json,
    sha256_file,
)

METHOD = "mvrec_rgb"
UNIT_COUNT = len(LOCKED_PRODUCTS) * len(LOCKED_K) * len(LOCKED_SEEDS)  # 195
MANIFEST_SCHEMA = "mvrec_rgb_formal_execution_manifest_v1"
SUPPORT_COMPLETION = "mvrec_rgb_support_complete.json"
GRID_COMPLETION = "mvrec_rgb_formal_support_grid_complete.json"
GRID_SCHEMA = "mvrec_rgb_formal_support_grid_v1"


@dataclass(frozen=True, order=True)
class MvrecRgbUnit:
    product: str
    k_shot: int
    seed: int

    @property
    def method(self) -> str:
        return METHOD

    @property
    def key(self) -> str:
        return f"{self.method}/{self.product}/k{self.k_shot}_s{self.seed}"

    @property
    def manifest_relative_path(self) -> str:
        return f"p6b_support_manifests/{self.product}/k{self.k_shot}_s{self.seed}.json"

    def directory(self, root) -> Path:
        return Path(root) / self.method / self.product / f"k{self.k_shot}_s{self.seed}"


def expected_units():
    for product in LOCKED_PRODUCTS:
        for k_shot in LOCKED_K:
            for seed in LOCKED_SEEDS:
                yield MvrecRgbUnit(product, k_shot, seed)


def load_frozen_manifest(bundle_dir: Path, unit: MvrecRgbUnit) -> dict:
    """Load one frozen P6.B support manifest with self-SHA verification."""
    path = Path(bundle_dir) / unit.manifest_relative_path
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P6BFailClosedError(f"cannot read frozen manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != "p6b_support_manifest_v1":
        raise P6BFailClosedError(f"{unit.key}: manifest schema drift")
    if manifest.get("product") != unit.product or int(manifest.get("K")) != unit.k_shot \
            or int(manifest.get("seed")) != unit.seed:
        raise P6BFailClosedError(f"{unit.key}: manifest identity mismatch")
    declared = manifest.get("manifest_sha256")
    if not declared:
        raise P6BFailClosedError(f"{unit.key}: manifest carries no self SHA")
    from experiments.p6.p6b_execution_common import canonical_json_bytes, sha256_bytes

    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    actual = sha256_bytes(canonical_json_bytes(body))
    if actual != declared:
        raise P6BFailClosedError(f"{unit.key}: manifest self SHA mismatch")
    return manifest


def build_execution_manifest(bundle_dir, out_path=None) -> dict:
    """Build the 195-unit job list from the frozen Stage 1 bundle.

    Returns the execution manifest object; optionally atomically written to
    ``out_path``.  Fail-closed on any manifest read/SHA problem.
    """
    bundle_dir = Path(bundle_dir)
    units = []
    for unit in expected_units():
        manifest = load_frozen_manifest(bundle_dir, unit)
        class_order = manifest.get("class_order")
        if not isinstance(class_order, list) or not class_order:
            raise P6BFailClosedError(f"{unit.key}: empty class_order")
        units.append({
            "key": unit.key,
            "method": METHOD,
            "product": unit.product,
            "K": unit.k_shot,
            "seed": unit.seed,
            "manifest_relative_path": unit.manifest_relative_path,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": sha256_file(bundle_dir / unit.manifest_relative_path),
            "class_order": list(class_order),
            "support_identities": [
                {"class_name": cls, "rel_path": item["rel_path"],
                 "rgb_sha256": item["rgb_sha256"]}
                for cls in class_order
                for item in manifest["classes"][cls]["items"]
            ],
        })
    if len(units) != UNIT_COUNT:
        raise P6BFailClosedError(f"execution manifest has {len(units)} != {UNIT_COUNT} units")
    keys = [u["key"] for u in units]
    if len(set(keys)) != len(keys):
        raise P6BFailClosedError("duplicate unit keys in execution manifest")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "built",
        "formal_eligible": True,
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "performance_metrics_generated": False,
        "method": METHOD,
        "unit_count": UNIT_COUNT,
        "products": list(LOCKED_PRODUCTS),
        "K_values": list(LOCKED_K),
        "seeds": list(LOCKED_SEEDS),
        "units": units,
    }
    if out_path is not None:
        atomic_write_json(out_path, manifest)
    return manifest
