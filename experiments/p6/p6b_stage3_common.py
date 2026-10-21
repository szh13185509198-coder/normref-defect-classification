#!/usr/bin/env python3
"""Fail-closed contracts shared by the P6.B Stage 3 grid and evaluator."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from experiments.p6.p6b_execution_common import (
    P6BFailClosedError,
    atomic_write_json,
)
from experiments.p6 import p6b_stage2_execution as stage2
from experiments.p6 import p6b_stage2_models as models
from experiments.p6 import run_p6b_stage2_support as support


METHODS = tuple(support.METHODS)
PRODUCTS = tuple(support.LOCKED_PRODUCTS)
K_VALUES = tuple(support.LOCKED_K)
SEEDS = tuple(support.LOCKED_SEEDS)
UNIT_COUNT = len(METHODS) * len(PRODUCTS) * len(K_VALUES) * len(SEEDS)
SUPPORT_COMPLETION = "p6b_stage2_support_complete.json"
GRID_COMPLETION = "p6b_formal_support_grid_complete.json"
GRID_SCHEMA = "p6b_formal_support_grid_v1"
PREDICTION_SCHEMA = "p6b_formal_query_prediction_v1"
PREDICTION_MANIFEST_SCHEMA = "p6b_formal_query_predictions_manifest_v1"
UNIT_PREDICTION_MANIFEST_SCHEMA = "p6b_formal_query_unit_prediction_manifest_v1"
AUDIT_SCHEMA = "p6b_formal_query_audit_v1"
EXPECTED_QUERY_RECORDS_SHA256 = (
    "a3deaeaae58e54a2a2ed75211690d26db7136a25d314bc343c3a51619bf3c2aa"
)
EXPECTED_RAW_QUERY_COUNT = 549
EXPECTED_EXCLUDED_MULTI_TYPE = 19
EXPECTED_FORMAL_QUERY_COUNT = 530


@dataclass(frozen=True, order=True)
class GridUnit:
    method: str
    product: str
    k_shot: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.method}/{self.product}/k{self.k_shot}_s{self.seed}"

    def directory(self, root: Path) -> Path:
        return Path(root) / self.method / self.product / f"k{self.k_shot}_s{self.seed}"


def expected_units():
    for method in METHODS:
        for product in PRODUCTS:
            for k_shot in K_VALUES:
                for seed in SEEDS:
                    yield GridUnit(method, product, k_shot, seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P6BFailClosedError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise P6BFailClosedError(f"JSON root must be an object: {path}")
    return value


def _contained_file(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative:
        raise P6BFailClosedError(f"invalid relative path: {relative!r}")
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise P6BFailClosedError(f"path escapes root: {relative}")
    if not candidate.is_file():
        raise P6BFailClosedError(f"file missing: {candidate}")
    return candidate


def extract_backbone_identities(completion: dict, unit_key: str) -> dict:
    """Normalize path-independent CLIP/DINO identities across both schemas."""
    method = completion.get("method")
    provenance = completion.get("model_provenance") or {}
    result = {}
    if method in {"ours", "p30_frozen_proto"}:
        if provenance.get("mode") != "real-local-weights":
            raise P6BFailClosedError(f"{unit_key}: injected/non-real model provenance")
        clip_node = provenance.get("clip_backbone") or {}
        dino_node = provenance.get("dino_checkpoint") or {}
        fingerprints = provenance.get("loaded_state_fingerprints") or {}
        sources = (("clip", clip_node, fingerprints.get("clip_backbone")),
                   ("dino", dino_node, fingerprints.get("dino_backbone")))
    else:
        sources = []
        for name in ("clip", "dino"):
            node = provenance.get(name) or {}
            if node.get("not_loaded") is True:
                continue
            sources.append((name, node, node.get("loaded_state")))
    for name, node, fingerprint in sources:
        weight = node.get("weight_file") or {}
        if (node.get("network_download") is not False or
                weight.get("exists") is not True or
                not isinstance(weight.get("sha256"), str) or
                not isinstance(weight.get("size_bytes"), int) or
                not isinstance(fingerprint, dict) or
                not isinstance(fingerprint.get("sha256"), str)):
            raise P6BFailClosedError(f"{unit_key}: incomplete {name} provenance")
        identity = {
            "weight_filename": weight.get("filename"),
            "weight_sha256": weight["sha256"],
            "weight_size_bytes": weight["size_bytes"],
            "loaded_state_sha256": fingerprint["sha256"],
            "parameter_names_sha256": fingerprint.get("parameter_names_sha256"),
            "parameter_count": fingerprint.get("parameter_count"),
            "tensor_count": fingerprint.get("tensor_count"),
        }
        if any(value is None for value in identity.values()):
            raise P6BFailClosedError(f"{unit_key}: incomplete {name} identity")
        if name == "dino":
            expected = node.get("expected_checkpoint_sha256", node.get("expected_sha256"))
            actual = node.get("actual_sha256", weight["sha256"])
            if expected != models.EXPECTED_DINO_SHA256 or actual != expected:
                raise P6BFailClosedError(f"{unit_key}: DINO pin mismatch")
        result[name] = identity
    required = {
        "clip_proto": {"clip"}, "dino_proto": {"dino"},
        "fusion_proto": {"clip", "dino"}, "linear_probe": {"clip", "dino"},
        "ours": {"clip", "dino"}, "zero_shot_clip": {"clip"},
        "tip_adapter": {"clip"}, "p30_frozen_proto": {"clip", "dino"},
        "tip_adapter_f": {"clip"},
    }[method]
    if set(result) != required:
        raise P6BFailClosedError(f"{unit_key}: backbone set {set(result)} != {required}")
    return result


def validate_support_unit(unit: GridUnit, unit_dir: Path, expected_commit=None) -> dict:
    """Validate a formal support unit deeply enough for safe resume/query gating."""
    unit_dir = Path(unit_dir)
    completion_path = unit_dir / SUPPORT_COMPLETION
    completion = load_json(completion_path)
    exact = {
        "schema_version": support.COMPLETION_SCHEMA,
        "status": "completed",
        "formal_eligible": True,
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "method": unit.method,
        "product": unit.product,
        "K": unit.k_shot,
        "seed": unit.seed,
    }
    for key, wanted in exact.items():
        if completion.get(key) != wanted:
            raise P6BFailClosedError(
                f"{unit.key}: completion {key}={completion.get(key)!r}, expected {wanted!r}"
            )
    execution = completion.get("execution") or {}
    commit = execution.get("commit")
    if execution.get("dirty") is not False or execution.get("allow_dirty_diagnostic") is not False:
        raise P6BFailClosedError(f"{unit.key}: execution is not clean formal mode")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise P6BFailClosedError(f"{unit.key}: invalid execution commit")
    if expected_commit is not None and commit != expected_commit:
        raise P6BFailClosedError(f"{unit.key}: commit {commit} != grid commit {expected_commit}")

    bundle = completion.get("protocol_bundle") or {}
    expected_bundle = {
        "version": support.PROTOCOL_BUNDLE_VERSION,
        "main_protocol_sha256": support.EXPECTED_DOCUMENT_SHAS_CANONICAL_LF["protocol"],
        "addendum_a_sha256": support.EXPECTED_DOCUMENT_SHAS_CANONICAL_LF["addendum_a"],
        "addendum_b_sha256": support.EXPECTED_DOCUMENT_SHAS_CANONICAL_LF["addendum_b"],
        "stage1_completion_sha256": stage2.STAGE1_COMPLETION_SHA256,
        "hash_mode": "canonical-lf",
    }
    for key, wanted in expected_bundle.items():
        if bundle.get(key) != wanted:
            raise P6BFailClosedError(f"{unit.key}: protocol bundle mismatch at {key}")

    class_order = completion.get("class_order")
    if not isinstance(class_order, list) or not class_order or len(set(class_order)) != len(class_order):
        raise P6BFailClosedError(f"{unit.key}: invalid class_order")
    if class_order != sorted(class_order):
        raise P6BFailClosedError(f"{unit.key}: class_order is not locked lexicographic order")

    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise P6BFailClosedError(f"{unit.key}: empty artifact registry")
    seen = set()
    for record in artifacts:
        relative = record.get("relative_path") if isinstance(record, dict) else None
        if not isinstance(relative, str) or relative in seen:
            raise P6BFailClosedError(f"{unit.key}: duplicate/invalid artifact record")
        seen.add(relative)
        artifact = _contained_file(unit_dir, relative)
        if artifact.stat().st_size != record.get("size_bytes"):
            raise P6BFailClosedError(f"{unit.key}: artifact size mismatch: {relative}")
        if file_sha256(artifact) != record.get("sha256"):
            raise P6BFailClosedError(f"{unit.key}: artifact SHA mismatch: {relative}")
    required = {"p6b_support_meta.jsonl", "support_run_config.json", "support_artifact_manifest.json"}
    method_required = {
        "clip_proto": {"clip_proto_prototypes.pt"},
        "dino_proto": {"dino_proto_prototypes.pt"},
        "fusion_proto": {"fusion_proto_prototypes.pt"},
        "linear_probe": {"linear_probe_head.pt", "linear_probe_step_audit.json"},
        "ours": {"multiclass_epoch_19.pth", "ours_step_audit.json"},
        "zero_shot_clip": {"zero_shot_clip_text_weights.pt"},
        "tip_adapter": {"tip_adapter_cache.pt"},
        "p30_frozen_proto": {"p30_frozen_proto_prototypes.pt"},
        "tip_adapter_f": {"tip_adapter_f_cache.pt", "tip_adapter_f_step_audit.json"},
    }
    required.update(method_required[unit.method])
    if not required.issubset(seen):
        raise P6BFailClosedError(f"{unit.key}: required support artifacts missing")

    run_config = load_json(unit_dir / "support_run_config.json")
    config_checks = {
        "schema_version": support.RUN_CONFIG_SCHEMA, "method": unit.method,
        "product": unit.product, "K": unit.k_shot, "seed": unit.seed,
        "class_order": class_order,
    }
    for key, wanted in config_checks.items():
        if run_config.get(key) != wanted:
            raise P6BFailClosedError(f"{unit.key}: support_run_config {key} mismatch")
    metadata = run_config.get("support_metadata") or {}
    meta_path = unit_dir / "p6b_support_meta.jsonl"
    if metadata.get("sha256") != file_sha256(meta_path):
        raise P6BFailClosedError(f"{unit.key}: support metadata SHA mismatch")
    counts = {name: 0 for name in class_order}
    meta_rows = []
    with meta_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise P6BFailClosedError(f"{unit.key}: invalid support meta line {line_no}") from exc
            name, label = row.get("class_name"), row.get("label")
            if name not in counts or label != class_order.index(name):
                raise P6BFailClosedError(f"{unit.key}: support meta class mapping drift")
            counts[name] += 1
            meta_rows.append(row)
    if len(meta_rows) != metadata.get("records") or any(count != unit.k_shot for count in counts.values()):
        raise P6BFailClosedError(f"{unit.key}: support meta does not contain exact K per class")

    artifact_manifest = load_json(unit_dir / "support_artifact_manifest.json")
    manifest_checks = {
        "schema_version": support.ARTIFACT_MANIFEST_SCHEMA,
        "method": unit.method, "product": unit.product,
        "K": unit.k_shot, "seed": unit.seed,
    }
    for key, wanted in manifest_checks.items():
        if artifact_manifest.get(key) != wanted:
            raise P6BFailClosedError(f"{unit.key}: artifact manifest {key} mismatch")
    completion_without_self = [r for r in artifacts
                               if r["relative_path"] != "support_artifact_manifest.json"]
    if artifact_manifest.get("artifacts") != completion_without_self:
        raise P6BFailClosedError(f"{unit.key}: artifact manifest registry mismatch")
    extract_backbone_identities(completion, unit.key)
    return completion


def validate_formal_grid(grid_root: Path, require_global_completion=True) -> dict:
    """Validate all 1,755 support units and the immutable global index."""
    grid_root = Path(grid_root).resolve()
    global_record = load_json(grid_root / GRID_COMPLETION) if require_global_completion else None
    expected_commit = None if global_record is None else global_record.get("execution_commit")
    records = []
    identity_sets = {"clip": set(), "dino": set()}
    identity_values = {"clip": None, "dino": None}
    for unit in expected_units():
        completion = validate_support_unit(unit, unit.directory(grid_root), expected_commit)
        identities = extract_backbone_identities(completion, unit.key)
        for name, identity in identities.items():
            encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            identity_sets[name].add(encoded)
            identity_values[name] = identity
        records.append({
            "key": unit.key,
            "completion_relative_path": str(
                (unit.directory(Path(".")) / SUPPORT_COMPLETION).as_posix()
            ),
            "completion_sha256": file_sha256(unit.directory(grid_root) / SUPPORT_COMPLETION),
            "class_order": completion["class_order"],
        })
    if len(records) != UNIT_COUNT:
        raise P6BFailClosedError(f"formal grid count {len(records)} != {UNIT_COUNT}")
    for name in ("clip", "dino"):
        if len(identity_sets[name]) != 1:
            raise P6BFailClosedError(
                f"formal grid has {len(identity_sets[name])} distinct {name} identities")
    if global_record is not None:
        checks = {
            "schema_version": GRID_SCHEMA,
            "status": "completed",
            "formal_eligible": True,
            "formal_test_used": False,
            "query_pixels_read": False,
            "query_labels_read": False,
            "unit_count": UNIT_COUNT,
        }
        for key, wanted in checks.items():
            if global_record.get(key) != wanted:
                raise P6BFailClosedError(f"global grid {key} mismatch")
        if global_record.get("units") != records:
            raise P6BFailClosedError("global grid unit index is incomplete or stale")
        if global_record.get("backbone_identities") != identity_values:
            raise P6BFailClosedError("global grid backbone identity index is stale")
    return {"execution_commit": expected_commit, "units": records,
            "backbone_identities": identity_values}


def load_query_contract(records_path: Path, dataset_root: Path, stage1_bundle: Path,
                        *, verify_rgb=True) -> tuple[list[dict], dict]:
    """Open the frozen query contract only after the caller has gated the grid."""
    records_path = Path(records_path)
    if file_sha256(records_path) != EXPECTED_QUERY_RECORDS_SHA256:
        raise P6BFailClosedError("frozen image-records JSONL SHA mismatch")
    stage1 = load_json(Path(stage1_bundle) / "p6b_manifest_build_complete.json")
    if file_sha256(Path(stage1_bundle) / "p6b_manifest_build_complete.json") != stage2.STAGE1_COMPLETION_SHA256:
        raise P6BFailClosedError("Stage 1 completion SHA mismatch")
    stage1_checks = {
        "schema_version": "p6b_support_manifest_build_v1", "status": "completed",
        "formal_eligible": True, "formal_test_used": False,
        "query_pixels_read": False, "query_labels_read": False,
        "records_jsonl_sha256": EXPECTED_QUERY_RECORDS_SHA256,
    }
    for key, wanted in stage1_checks.items():
        if stage1.get(key) != wanted:
            raise P6BFailClosedError(f"Stage 1 completion {key} mismatch")
    raw_query = []
    selected = []
    excluded = []
    seen = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise P6BFailClosedError(f"query JSONL line {line_no}: {exc}") from exc
            if record.get("schema_version") != "p6b_mvtec_fs_rgb_image_record_v2":
                raise P6BFailClosedError(f"query JSONL line {line_no}: schema drift")
            if "query" not in (record.get("source_pools") or []):
                continue
            if record.get("product") not in PRODUCTS:
                continue
            required = ("rgb_sha256", "canonical_relative_path",
                        "unique_defect_types", "multi_type_removed")
            if any(key not in record for key in required):
                raise P6BFailClosedError(f"query JSONL line {line_no}: missing contract field")
            if record.get("source_pools") != ["query"]:
                raise P6BFailClosedError(f"query JSONL line {line_no}: query record touches another pool")
            if not isinstance(record.get("multi_type_removed"), bool):
                raise P6BFailClosedError(f"query JSONL line {line_no}: invalid multi_type_removed")
            raw_query.append(record)
            sha = record.get("rgb_sha256")
            if (not isinstance(sha, str) or len(sha) != 64 or
                    any(char not in "0123456789abcdef" for char in sha)):
                raise P6BFailClosedError(f"query JSONL line {line_no}: invalid RGB SHA")
            key = (record.get("product"), sha)
            if key in seen:
                raise P6BFailClosedError(f"duplicate query identity: {key}")
            seen.add(key)
            if record.get("multi_type_removed") is True:
                excluded.append({"product": key[0], "rgb_sha256": sha, "reason": "multi_type_removed"})
                continue
            labels = record.get("unique_defect_types")
            if not isinstance(labels, list) or len(labels) != 1:
                raise P6BFailClosedError(f"single-type query has invalid labels: {key}")
            class_order = stage1["per_product"][key[0]]["class_universe"]
            if labels[0] not in class_order:
                raise P6BFailClosedError(f"query label outside frozen class universe: {key}")
            relative = record.get("canonical_relative_path")
            image_path = _contained_file(dataset_root, relative)
            if verify_rgb and file_sha256(image_path) != sha:
                raise P6BFailClosedError(f"query RGB SHA mismatch: {relative}")
            selected.append({
                "product": key[0], "rgb_sha256": sha, "relative_path": relative,
                "image_path": str(image_path), "true_label": labels[0],
                "class_order": class_order,
            })
    if len(raw_query) != EXPECTED_RAW_QUERY_COUNT:
        raise P6BFailClosedError(f"raw query count {len(raw_query)} != {EXPECTED_RAW_QUERY_COUNT}")
    if len(excluded) != EXPECTED_EXCLUDED_MULTI_TYPE:
        raise P6BFailClosedError(f"multi-type exclusion count {len(excluded)} != {EXPECTED_EXCLUDED_MULTI_TYPE}")
    if len(selected) != EXPECTED_FORMAL_QUERY_COUNT:
        raise P6BFailClosedError(f"formal query count {len(selected)} != {EXPECTED_FORMAL_QUERY_COUNT}")
    raw_by_product = {product: 0 for product in PRODUCTS}
    for record in raw_query:
        raw_by_product[record["product"]] += 1
    for product in PRODUCTS:
        expected = stage1.get("per_product", {}).get(product, {}).get("query_sha_metadata_count")
        if raw_by_product[product] != expected:
            raise P6BFailClosedError(
                f"raw query count drift for {product}: {raw_by_product[product]} != {expected}")
    selected.sort(key=lambda r: (PRODUCTS.index(r["product"]), r["rgb_sha256"]))
    return selected, {
        "raw_query_count": len(raw_query),
        "multi_type_excluded_count": len(excluded),
        "formal_query_count": len(selected),
        "excluded": sorted(excluded, key=lambda r: (r["product"], r["rgb_sha256"])),
    }


def classification_metrics(true_labels, predicted_labels, class_order) -> dict:
    if len(true_labels) != len(predicted_labels) or not true_labels:
        raise P6BFailClosedError("metric vectors must be aligned and non-empty")
    class_order = list(class_order)
    if not class_order or len(set(class_order)) != len(class_order):
        raise P6BFailClosedError("invalid metric class_order")
    if any(label not in class_order for label in true_labels + predicted_labels):
        raise P6BFailClosedError("metric label outside class_order")
    recalls, f1s = [], []
    for label in class_order:
        tp = sum(t == label and p == label for t, p in zip(true_labels, predicted_labels))
        fn = sum(t == label and p != label for t, p in zip(true_labels, predicted_labels))
        fp = sum(t != label and p == label for t, p in zip(true_labels, predicted_labels))
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = recalls[-1]
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "top1_accuracy": sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels),
        "balanced_accuracy": statistics.fmean(recalls),
        "macro_f1": statistics.fmean(f1s),
    }


def aggregate_raw_metrics(raw_metrics: list[dict]) -> dict:
    """Locked order: equal-product macro per (method,K,seed), then ddof=1."""
    metric_names = ("top1_accuracy", "balanced_accuracy", "macro_f1")
    by_key = {}
    for row in raw_metrics:
        key = (row["method"], int(row["K"]), int(row["seed"]))
        if (row["method"] not in METHODS or row["product"] not in PRODUCTS or
                int(row["K"]) not in K_VALUES or int(row["seed"]) not in SEEDS):
            raise P6BFailClosedError("raw metric key outside locked grid")
        product_rows = by_key.setdefault(key, {})
        if row["product"] in product_rows:
            raise P6BFailClosedError(
                f"duplicate raw metric: {row['method']}/{row['product']}/K{row['K']}/s{row['seed']}")
        product_rows[row["product"]] = row
        if any(not math.isfinite(float(row[name])) for name in metric_names):
            raise P6BFailClosedError("non-finite raw metric")
    macro_rows = []
    for method in METHODS:
        for k_shot in K_VALUES:
            for seed in SEEDS:
                product_rows = by_key.get((method, k_shot, seed), {})
                if set(product_rows) != set(PRODUCTS):
                    raise P6BFailClosedError(f"incomplete product macro: {method}/K{k_shot}/s{seed}")
                macro_rows.append({
                    "method": method, "K": k_shot, "seed": seed,
                    **{name: statistics.fmean(product_rows[p][name] for p in PRODUCTS)
                       for name in metric_names},
                })
    summaries = []
    for method in METHODS:
        for k_shot in K_VALUES:
            rows = [r for r in macro_rows if r["method"] == method and r["K"] == k_shot]
            summaries.append({
                "method": method, "K": k_shot, "seed_count": len(rows),
                **{f"{name}_mean": statistics.fmean(r[name] for r in rows)
                   for name in metric_names},
                **{f"{name}_std_ddof1": statistics.stdev(r[name] for r in rows)
                   for name in metric_names},
            })
    per_product = []
    for method in METHODS:
        for product in PRODUCTS:
            for k_shot in K_VALUES:
                rows = [r for r in raw_metrics if r["method"] == method and r["product"] == product and int(r["K"]) == k_shot]
                if len(rows) != len(SEEDS):
                    raise P6BFailClosedError("incomplete per-product seed summary")
                per_product.append({
                    "method": method, "product": product, "K": k_shot,
                    **{f"{name}_mean": statistics.fmean(r[name] for r in rows)
                       for name in metric_names},
                    **{f"{name}_std_ddof1": statistics.stdev(r[name] for r in rows)
                       for name in metric_names},
                })
    return {"product_macro_per_seed": macro_rows, "method_k_summary": summaries,
            "per_product_five_seed": per_product}
