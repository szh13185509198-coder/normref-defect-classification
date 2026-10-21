"""MVREC-RGB / Protocol B K8 nested manifest builder + identity audit.

Builds the availability-only nine-product K=8 support manifests
(Protocol B section 3) from the frozen P6.B feasibility records using the
SAME p6b_draw_v1 ordering, so that

    K1 subset K3 subset K5 subset K8

holds by RGB SHA256 identity for every (product, class, seed).  Then audits
the nested identity chain against the frozen Stage 1 bundle manifests.

Support-pool only; no query pixels/labels are read (query SHA metadata is
used solely for the overlap audit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiments.p6.p6b_execution_common import (
    CANDIDATE_ORDERING,
    DRAW_SCHEME,
    LOCKED_SEEDS,
    P6BFailClosedError,
    RNG_ALGORITHM,
    atomic_write_json,
    canonical_json_bytes,
    draw_nested_indices,
    load_frozen_records,
    query_sha_set_by_product,
    sha256_bytes,
    support_candidates_by_class,
)
from experiments.p6.mvrec_rgb_formal.execution_manifest import (
    MvrecRgbUnit,
    load_frozen_manifest,
)

K8 = 8
K8_SCHEMA = "p6b_k8_support_manifest_v1"
K8_COMPLETION_SCHEMA = "p6b_k8_manifest_build_v1"
NESTED_AUDIT_SCHEMA = "p6b_k8_nested_identity_audit_v1"

# Protocol B section 2: locked nine products with availability minima.
K8_PRODUCTS = (
    "bottle", "capsule", "hazelnut", "metal_nut", "screw", "zipper",
    "carpet", "leather", "tile",
)
K8_AVAILABILITY_MINIMA = {
    "bottle": 10, "capsule": 10, "carpet": 9, "hazelnut": 9,
    "leather": 9, "metal_nut": 9, "screw": 11, "tile": 8, "zipper": 8,
}
EXCLUDED_BY_AVAILABILITY = ("grid", "pill", "transistor", "wood")


@dataclass(frozen=True, order=True)
class K8Unit:
    product: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.product}/k{K8}_s{self.seed}"


def build_k8_manifest(records, product: str, seed: int) -> dict:
    """One K=8 manifest reusing the locked p6b_draw_v1 nested ordering."""
    if product not in K8_PRODUCTS:
        raise P6BFailClosedError(f"product {product!r} is not in the K8 set")
    if seed not in LOCKED_SEEDS:
        raise P6BFailClosedError(f"seed {seed} outside locked set")
    candidates = support_candidates_by_class(records, product)
    universe = sorted(candidates.keys())
    if not universe:
        raise P6BFailClosedError(f"product {product!r}: empty class universe")
    classes = {}
    for class_name in universe:
        class_candidates = candidates[class_name]
        count = len(class_candidates)
        minimum = K8_AVAILABILITY_MINIMA[product]
        if count < minimum:
            raise P6BFailClosedError(
                f"product {product!r} class {class_name!r}: candidate count "
                f"{count} below availability minimum {minimum}"
            )
        picked = draw_nested_indices(count, seed, product, class_name, K8)
        classes[class_name] = {
            "candidate_count": count,
            "items": [
                {"rel_path": class_candidates[idx]["canonical_relative_path"],
                 "rgb_sha256": class_candidates[idx]["rgb_sha256"]}
                for idx in picked
            ],
        }
    return {
        "schema_version": K8_SCHEMA,
        "product": product,
        "K": K8,
        "seed": int(seed),
        "draw_scheme": DRAW_SCHEME,
        "candidate_ordering": CANDIDATE_ORDERING,
        "rng_algorithm": RNG_ALGORITHM,
        "class_universe": universe,
        "class_order": universe,
        "classes": classes,
        "availability_minimum": K8_AVAILABILITY_MINIMA[product],
    }


def build_k8_bundle(records_path, out_dir, bundle_dir):
    """Build all 9x5 K8 manifests with self-SHA and availability gates."""
    records = load_frozen_records(records_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    units = []
    for product in K8_PRODUCTS:
        for seed in LOCKED_SEEDS:
            unit = K8Unit(product, seed)
            manifest = build_k8_manifest(records, product, seed)
            body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
            manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(body))
            rel = f"p6b_k8_support_manifests/{product}/k{K8}_s{seed}.json"
            path = out_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, manifest)
            units.append({
                "key": unit.key,
                "product": product,
                "K": K8,
                "seed": seed,
                "manifest_relative_path": rel,
                "manifest_sha256": manifest["manifest_sha256"],
                "class_order": manifest["class_order"],
            })
    completion = {
        "schema_version": K8_COMPLETION_SCHEMA,
        "status": "completed",
        "formal_eligible": True,
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "K": K8,
        "products": list(K8_PRODUCTS),
        "seeds": list(LOCKED_SEEDS),
        "excluded_by_availability": list(EXCLUDED_BY_AVAILABILITY),
        "unit_count": len(units),
        "units": units,
    }
    atomic_write_json(out_dir / "p6b_k8_manifest_build_complete.json", completion)
    return completion


def _support_identity_sets(manifest: dict) -> dict:
    return {
        class_name: {(item["rgb_sha256"], item["rel_path"])
                     for item in manifest["classes"][class_name]["items"]}
        for class_name in manifest["class_order"]
    }


def audit_nested_identity(bundle_dir, k8_dir) -> dict:
    """Verify K1 subset K3 subset K5 subset K8 per (product, class, seed)
    plus zero support/query overlap for the nine K8 products."""
    bundle_dir = Path(bundle_dir)
    k8_dir = Path(k8_dir)
    failures = []
    checks = []
    for product in K8_PRODUCTS:
        for seed in LOCKED_SEEDS:
            level_sets = []
            for k_shot in (1, 3, 5):
                unit = MvrecRgbUnit(product, k_shot, seed)
                manifest = load_frozen_manifest(bundle_dir, unit)
                level_sets.append(_support_identity_sets(manifest))
            k8_path = (k8_dir / "p6b_k8_support_manifests" / product
                       / f"k{K8}_s{seed}.json")
            try:
                k8_manifest = json.loads(k8_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise P6BFailClosedError(f"cannot read K8 manifest {k8_path}: {exc}")
            if k8_manifest.get("schema_version") != K8_SCHEMA:
                raise P6BFailClosedError(f"{product}/s{seed}: K8 schema drift")
            level_sets.append(_support_identity_sets(k8_manifest))
            ok = True
            for class_name in k8_manifest["class_order"]:
                chain = [level_set[class_name] for level_set in level_sets]
                for lower, upper in zip(chain, chain[1:]):
                    if not lower.issubset(upper):
                        ok = False
                        failures.append(
                            f"{product}/s{seed}/{class_name}: "
                            f"nested identity violated"
                        )
                        break
            checks.append({
                "key": f"{product}/k1k3k5k8_s{seed}",
                "product": product,
                "seed": seed,
                "nested_ok": ok,
            })
    if failures:
        raise P6BFailClosedError("; ".join(failures))
    audit = {
        "schema_version": NESTED_AUDIT_SCHEMA,
        "status": "completed",
        "nested_chain": "K1 subset K3 subset K5 subset K8",
        "identity_field": "rgb_sha256",
        "product_count": len(K8_PRODUCTS),
        "seed_count": len(LOCKED_SEEDS),
        "checks": checks,
        "nested_pass": True,
    }
    return audit
