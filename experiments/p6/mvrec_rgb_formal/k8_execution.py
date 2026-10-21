"""MVREC-RGB / Protocol B K8 execution infrastructure (support-only).

The K8 nested shot-scaling extension adds K=8 support units for the nine
availability-qualified products:

- methods: Ours, MVREC-RGB adaptation, Frozen-Fusion;
- grid: 9 products x 5 seeds x 3 methods = 135 method-units;
- K1/K3/K5 are NOT re-run: their frozen formal results are re-aggregated
  on the same nine-product subset;
- each K8 support unit is trained support-only with the frozen method
  semantics and the final-checkpoint rule (Protocol B section 4);
- the query gate stays CLOSED until the support audit has no blocker.

This module provides the 135-unit execution manifest, the K8 manifest
loader (schema `p6b_k8_support_manifest_v1`, self-SHA verified), and the
per-unit subprocess command builders.  No query pixels/labels/metrics are
read anywhere here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.p6.p6b_execution_common import (
    P6BFailClosedError,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from experiments.p6.mvrec_rgb_formal.k8_builder import (
    K8,
    K8_PRODUCTS,
    K8_SCHEMA,
)
from experiments.p6.mvrec_rgb_formal.k8_scaling_evaluator import (
    METHODS,
    SEEDS,
)

UNIT_COUNT = len(METHODS) * len(K8_PRODUCTS) * len(SEEDS)  # 135
MANIFEST_SCHEMA = "mvrec_rgb_k8_execution_manifest_v1"
GRID_COMPLETION = "mvrec_rgb_k8_support_grid_complete.json"
GRID_SCHEMA = "mvrec_rgb_k8_support_grid_v1"


@dataclass(frozen=True, order=True)
class K8MethodUnit:
    method: str
    product: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.method}/{self.product}/k{K8}_s{self.seed}"

    @property
    def manifest_relative_path(self) -> str:
        return f"p6b_k8_support_manifests/{self.product}/k{K8}_s{self.seed}.json"

    def directory(self, root) -> Path:
        return Path(root) / self.method / self.product / f"k{K8}_s{self.seed}"


def expected_k8_units():
    for method in METHODS:
        for product in K8_PRODUCTS:
            for seed in SEEDS:
                yield K8MethodUnit(method, product, seed)


def load_k8_manifest(k8_bundle_dir: Path, unit: K8MethodUnit) -> dict:
    """Load one K8 manifest (schema p6b_k8_support_manifest_v1) with
    self-SHA verification and identity checks."""
    path = Path(k8_bundle_dir) / unit.manifest_relative_path
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P6BFailClosedError(f"cannot read K8 manifest {path}: {exc}") from exc
    if manifest.get("schema_version") != K8_SCHEMA:
        raise P6BFailClosedError(f"{unit.key}: K8 manifest schema drift")
    if manifest.get("product") != unit.product or int(manifest.get("K")) != K8 \
            or int(manifest.get("seed")) != unit.seed:
        raise P6BFailClosedError(f"{unit.key}: K8 manifest identity mismatch")
    declared = manifest.get("manifest_sha256")
    if not declared:
        raise P6BFailClosedError(f"{unit.key}: K8 manifest carries no self SHA")
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    actual = sha256_bytes(canonical_json_bytes(body))
    if actual != declared:
        raise P6BFailClosedError(f"{unit.key}: K8 manifest self SHA mismatch")
    return manifest


def build_k8_execution_manifest(k8_bundle_dir, out_path=None) -> dict:
    """Build the 135-unit K8 job list from the locked K8 manifest bundle."""
    k8_bundle_dir = Path(k8_bundle_dir)
    units = []
    for unit in expected_k8_units():
        manifest = load_k8_manifest(k8_bundle_dir, unit)
        units.append({
            "key": unit.key,
            "method": unit.method,
            "product": unit.product,
            "K": K8,
            "seed": unit.seed,
            "manifest_relative_path": unit.manifest_relative_path,
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": sha256_file(
                k8_bundle_dir / unit.manifest_relative_path
            ),
            "class_order": list(manifest["class_order"]),
            "support_identities": [
                {"class_name": cls, "rel_path": item["rel_path"],
                 "rgb_sha256": item["rgb_sha256"]}
                for cls in manifest["class_order"]
                for item in manifest["classes"][cls]["items"]
            ],
        })
    if len(units) != UNIT_COUNT:
        raise P6BFailClosedError(
            f"K8 execution manifest has {len(units)} != {UNIT_COUNT} units"
        )
    keys = [u["key"] for u in units]
    if len(set(keys)) != len(keys):
        raise P6BFailClosedError("duplicate K8 unit keys")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "built",
        "formal_eligible": True,
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "performance_metrics_generated": False,
        "methods": list(METHODS),
        "unit_count": UNIT_COUNT,
        "products": list(K8_PRODUCTS),
        "K": K8,
        "seeds": list(SEEDS),
        "K1_K3_K5_not_rerun": True,
        "units": units,
    }
    if out_path is not None:
        atomic_write_json(out_path, manifest)
    return manifest


def k8_mvrec_rgb_command(python, bundle_dir, image_root, alpha_ckpt, unit,
                         output_dir, device):
    """Subprocess command for one MVREC-RGB K8 support unit (official
    support pipeline, K recovered from the K8 manifest)."""
    return [
        python, "-m", "experiments.p6.mvrec_rgb_formal.support_runner",
        "--manifest-dir", str(Path(bundle_dir).resolve()),
        "--image-root", str(Path(image_root).resolve()),
        "--alpha-ckpt", str(Path(alpha_ckpt).resolve()),
        "--product", unit.product, "--k", str(K8),
        "--seed", str(unit.seed),
        "--k8",
        "--output-dir", str(Path(output_dir).resolve()),
    ] + (["--device", device] if device else [])


def k8_stage2_command(python, k8_bundle_dir, image_root, unit, output_dir,
                      device):
    """Subprocess command for one Ours/Frozen-Fusion K8 support unit via the
    P6.B Stage 2 runner (frozen method semantics, final-checkpoint rule)."""
    return [
        python, "-m", "experiments.p6.run_p6b_stage2_support",
        "--bundle-dir", str(Path(k8_bundle_dir).resolve()),
        "--image-root", str(Path(image_root).resolve()),
        "--product", unit.product, "--k", str(K8),
        "--seed", str(unit.seed), "--method", unit.method,
        "--output-dir", str(Path(output_dir).resolve()),
    ] + (["--device", device] if device else [])
