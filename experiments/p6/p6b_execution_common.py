#!/usr/bin/env python3
"""P6.B execution protocol — Stage 1 common infrastructure (stdlib only).

Implements the locked support-manifest contract of
``docs/P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md``:

- deterministic candidate ordering by ``rgb_sha256`` lexicographic order;
- ``p6b_draw_v1`` sub-seed derivation + nested K draws (K=1 subset K=3 subset K=5);
- ``p6b_support_manifest_v1`` schema with canonical self-excluding sha256;
- support/query SHA overlap audit (query **SHA metadata only**, never pixels);
- fixed per-product class universe and class order.

Firewall: this module only parses the frozen feasibility JSONL text records.
It never opens image files and never imports torch / CLIP / DINO / FAISS.
"""

from __future__ import annotations

import hashlib
import json
import os
import random

# ---------------------------------------------------------------------------
# Locked protocol constants
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = "p6b_support_manifest_v1"
RECORD_SCHEMA_VERSION = "p6b_mvtec_fs_rgb_image_record_v2"
DRAW_SCHEME = "p6b_draw_v1"
DRAW_NAMESPACE = "p6b_draw_v1"
CANDIDATE_ORDERING = "lexicographic_rgb_sha256"
RNG_ALGORITHM = "python_random_mersenne_twister"

LOCKED_PRODUCTS = (
    "bottle", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "transistor", "wood", "zipper",
)
LOCKED_SEEDS = (111, 222, 333, 444, 555)
LOCKED_K = (1, 3, 5)
EXPECTED_MANIFEST_COUNT = len(LOCKED_PRODUCTS) * len(LOCKED_K) * len(LOCKED_SEEDS)  # 195

# Firewall-minimized field policy (protocol §8): support-side records may
# carry the fields needed for candidate construction; query-side records are
# reduced to the bare overlap-audit triple. JSONL lines are parsed, but query
# label/path fields are not accessed, retained, or used for any manifest /
# class-universe / training decision; the only query-side information
# consumed is product / source_pools / rgb_sha256.
SUPPORT_RECORD_FIELDS = (
    "schema_version", "product", "rgb_sha256", "canonical_relative_path",
    "source_pools", "unique_defect_types", "multi_type_removed",
)
QUERY_RECORD_FIELDS = ("product", "source_pools", "rgb_sha256")

KNOWN_POOLS = ("support", "query")
EXCLUDED_PRODUCTS = ("cable", "toothbrush")


class P6BFailClosedError(RuntimeError):
    """Any integrity violation is fail-closed: no partial output is trusted."""


# ---------------------------------------------------------------------------
# Canonical serialization / hashing / atomic writes
# ---------------------------------------------------------------------------

def canonical_json_bytes(obj) -> bytes:
    """Locked canonical form: UTF-8, sort_keys, compact separators, no ASCII
    escaping, trailing LF (protocol §2.3)."""
    return (
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path, text: str) -> None:
    """Atomic write (tmp + os.replace) with enforced LF newlines."""
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, canonical_json_bytes(obj).decode("utf-8"))


def manifest_self_sha256(manifest: dict) -> str:
    """sha256 of the manifest object WITHOUT its own ``manifest_sha256`` field.
    The sha field must never participate in its own hash (protocol §2.3)."""
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return sha256_bytes(canonical_json_bytes(body))


def sub_seed(base_seed: int, product: str, class_name: str) -> int:
    """Locked p6b_draw_v1 sub-seed: first 8 digest bytes, big-endian unsigned."""
    payload = f"{DRAW_NAMESPACE}|{int(base_seed)}|{product}|{class_name}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Frozen feasibility JSONL loading (firewall-aware)
# ---------------------------------------------------------------------------

def load_frozen_records(jsonl_path):
    """Parse the frozen feasibility image-records JSONL.

    Query-touching records are reduced to ``(product, source_pools,
    rgb_sha256)``. The JSONL line is parsed, but its query label/path fields
    are not accessed, retained, or used for any manifest/class-universe/
    training decision; the only query-side information consumed is
    product/source_pools/rgb_sha256. Fail-closed on schema drift, unknown
    pools or duplicate (product, rgb_sha256).
    """
    records = []
    seen = set()
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise P6BFailClosedError(
                    f"records JSONL line {line_number}: invalid JSON: {exc}"
                ) from exc
            if raw.get("schema_version") != RECORD_SCHEMA_VERSION:
                raise P6BFailClosedError(
                    f"records JSONL line {line_number}: unexpected schema_version "
                    f"{raw.get('schema_version')!r}"
                )
            pools = tuple(raw.get("source_pools") or ())
            if not pools or any(pool not in KNOWN_POOLS for pool in pools):
                raise P6BFailClosedError(
                    f"records JSONL line {line_number}: unknown source_pools {pools}"
                )
            fields = QUERY_RECORD_FIELDS if "query" in pools else SUPPORT_RECORD_FIELDS
            missing = [field for field in fields if field not in raw]
            if missing:
                raise P6BFailClosedError(
                    f"records JSONL line {line_number}: missing fields {missing}"
                )
            record = {field: raw[field] for field in fields}
            key = (record["product"], record["rgb_sha256"])
            if key in seen:
                raise P6BFailClosedError(
                    f"records JSONL line {line_number}: duplicate record for {key}"
                )
            seen.add(key)
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Candidates, class universe, query SHA metadata
# ---------------------------------------------------------------------------

def support_candidates_by_class(records, product: str) -> dict:
    """single-type support-pool candidates for one product.

    Fail-closed exclusions: cable/toothbrush scope, multi-type records,
    records touching the query pool, and non-single-type defect lists.
    """
    if product in EXCLUDED_PRODUCTS:
        raise P6BFailClosedError(f"product {product!r} is excluded from P6.B scope")
    candidates: dict = {}
    for record in records:
        if record["product"] != product:
            continue
        pools = record["source_pools"]
        if "support" not in pools:
            continue
        if "query" in pools:
            raise P6BFailClosedError(
                f"product {product!r}: RGB {record['rgb_sha256']} touches both "
                "support and query pools"
            )
        if record["multi_type_removed"]:
            continue
        defect_types = record["unique_defect_types"]
        if len(defect_types) != 1:
            raise P6BFailClosedError(
                f"product {product!r}: RGB {record['rgb_sha256']} is not single-type"
            )
        candidates.setdefault(defect_types[0], []).append(record)
    for class_name, items in candidates.items():
        items.sort(key=lambda record: record["rgb_sha256"])
    return candidates


def product_class_universe(records, product: str) -> list:
    """Fixed class universe = the product's support-side official single-type
    defect classes, lexicographic order (protocol §7.1). This is the DRAFT
    resolution of the §7.1 vs §8 conflict: query label/path fields are not
    accessed or consumed, so the universe is derived from support candidates
    only."""
    candidates = support_candidates_by_class(records, product)
    if not candidates:
        raise P6BFailClosedError(f"product {product!r}: empty class universe")
    return sorted(candidates.keys())


def query_sha_set_by_product(records) -> dict:
    """Query-side RGB SHA256 metadata per product — SHA values ONLY.

    This is the sole query-side quantity Stage 1 may touch (overlap audit).
    Query pixels and query labels are never read.
    """
    sha_sets: dict = {}
    for record in records:
        if "query" in record["source_pools"]:
            sha_sets.setdefault(record["product"], set()).add(record["rgb_sha256"])
    return sha_sets


# ---------------------------------------------------------------------------
# Locked nested draw + manifest construction
# ---------------------------------------------------------------------------

def draw_nested_indices(candidate_count: int, base_seed: int, product: str,
                        class_name: str, k_shot: int) -> list:
    """One deterministic permutation per (product, seed, class); K-shot takes
    the first K indices, guaranteeing K=1 subset K=3 subset K=5."""
    if candidate_count <= 0:
        raise P6BFailClosedError(
            f"product {product!r} class {class_name!r}: no support candidates"
        )
    if k_shot < 1 or k_shot > candidate_count:
        raise P6BFailClosedError(
            f"product {product!r} class {class_name!r}: K={k_shot} exceeds "
            f"candidate count {candidate_count}"
        )
    indices = list(range(candidate_count))
    random.Random(sub_seed(base_seed, product, class_name)).shuffle(indices)
    return indices[:k_shot]


def build_support_manifest(records, product: str, k_shot: int, seed: int) -> dict:
    """Build one ``p6b_support_manifest_v1`` object (without its own sha)."""
    if k_shot not in LOCKED_K:
        raise P6BFailClosedError(f"K={k_shot} outside locked set {list(LOCKED_K)}")
    if seed not in LOCKED_SEEDS:
        raise P6BFailClosedError(f"seed={seed} outside locked set {list(LOCKED_SEEDS)}")
    candidates = support_candidates_by_class(records, product)
    universe = sorted(candidates.keys())
    if not universe:
        raise P6BFailClosedError(f"product {product!r}: empty class universe")
    classes = {}
    for class_name in universe:
        class_candidates = candidates[class_name]
        picked = draw_nested_indices(len(class_candidates), seed, product,
                                     class_name, k_shot)
        classes[class_name] = {
            "candidate_count": len(class_candidates),
            "items": [
                {
                    "rel_path": class_candidates[idx]["canonical_relative_path"],
                    "rgb_sha256": class_candidates[idx]["rgb_sha256"],
                }
                for idx in picked
            ],
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "product": product,
        "K": k_shot,
        "seed": int(seed),
        "draw_scheme": DRAW_SCHEME,
        "candidate_ordering": CANDIDATE_ORDERING,
        "rng_algorithm": RNG_ALGORITHM,
        "class_universe": universe,
        "class_order": universe,
        "classes": classes,
    }


def finalize_manifest(manifest: dict) -> dict:
    """Attach the canonical self-excluding ``manifest_sha256``."""
    if "manifest_sha256" in manifest:
        raise P6BFailClosedError("manifest already carries manifest_sha256")
    completed = dict(manifest)
    completed["manifest_sha256"] = manifest_self_sha256(manifest)
    return completed
