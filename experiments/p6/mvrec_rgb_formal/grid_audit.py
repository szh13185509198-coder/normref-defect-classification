"""Fail-closed audit contracts for the MVREC-RGB formal support grid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.p6.p6b_execution_common import (
    P6BFailClosedError,
)
from experiments.p6.mvrec_rgb.protocol_lock import (
    MVREC_UPSTREAM_COMMIT,
    compute_protocol_hashes,
)
from experiments.p6.mvrec_rgb.semantic_smoke import EXPECTED_TRAINABLE
from experiments.p6.mvrec_rgb_formal.execution_manifest import (
    GRID_COMPLETION,
    GRID_SCHEMA,
    UNIT_COUNT,
    expected_units,
)
from experiments.p6.mvrec_rgb_formal.support_runner import (
    ARTIFACT_MANIFEST,
    ARTIFACT_MANIFEST_SCHEMA,
    COMPLETION,
    COMPLETION_SCHEMA,
    HEAD_CHECKPOINT,
    META_JSONL,
    RUN_CONFIG,
    RUN_CONFIG_SCHEMA,
    STEP_AUDIT,
    STEP_AUDIT_SCHEMA,
    SUPPORT_PROTO,
)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path) -> dict:
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


def validate_support_unit(unit, unit_dir, expected_commit=None) -> dict:
    """Validate one formal MVREC-RGB support unit deeply enough for safe
    resume/query gating."""
    unit_dir = Path(unit_dir)
    completion = load_json(unit_dir / COMPLETION)
    exact = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "completed",
        "formal_eligible": True,
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "performance_metrics_generated": False,
        "method": unit.method,
        "product": unit.product,
        "K": unit.k_shot,
        "seed": unit.seed,
    }
    for key, wanted in exact.items():
        if completion.get(key) != wanted:
            raise P6BFailClosedError(
                f"{unit.key}: completion {key}={completion.get(key)!r}, "
                f"expected {wanted!r}"
            )
    execution = completion.get("execution") or {}
    commit = execution.get("commit")
    if execution.get("dirty") is not False or execution.get("allow_dirty_diagnostic") is not False:
        raise P6BFailClosedError(f"{unit.key}: execution is not clean formal mode")
    if not isinstance(commit, str) or len(commit) != 40 or \
            any(c not in "0123456789abcdef" for c in commit):
        raise P6BFailClosedError(f"{unit.key}: invalid execution commit")
    if expected_commit is not None and commit != expected_commit:
        raise P6BFailClosedError(
            f"{unit.key}: commit {commit} != grid commit {expected_commit}"
        )

    # Protocol bundle: the three locked protocol documents (canonical-LF).
    expected_hashes = compute_protocol_hashes()
    bundle = completion.get("protocol_bundle") or {}
    if bundle.get("hash_mode") != "canonical-lf":
        raise P6BFailClosedError(f"{unit.key}: protocol hash mode drift")
    for key, wanted in expected_hashes.items():
        if bundle.get(key) != wanted:
            raise P6BFailClosedError(f"{unit.key}: protocol bundle mismatch at {key}")

    # Model provenance: official classifier + locked trainable whitelist.
    provenance = completion.get("model_provenance") or {}
    if provenance.get("classifier") != "EchoClassfierF":
        raise P6BFailClosedError(f"{unit.key}: classifier drift")
    if provenance.get("backbone_family") != "AlphaCLIP ViT-L/14":
        raise P6BFailClosedError(f"{unit.key}: backbone drift")
    alpha_sha = provenance.get("alpha_ckpt_sha256")
    if not isinstance(alpha_sha, str) or len(alpha_sha) != 64:
        raise P6BFailClosedError(f"{unit.key}: missing alpha checkpoint SHA")
    trainable = provenance.get("trainable_parameter_names")
    if set(trainable or []) != EXPECTED_TRAINABLE:
        raise P6BFailClosedError(f"{unit.key}: trainable whitelist drift")

    # Locked resource measurements (decision rule section 5.1): every unit
    # must report the full resource block with finite scalar counts.
    resource = completion.get("resource_measurements") or {}
    for key in ("device", "torch_version", "precision",
                "trainable_parameter_tensor_count",
                "trainable_parameter_scalar_count",
                "total_parameter_scalar_count",
                "support_training_gpu_seconds",
                "peak_allocated_gpu_memory_bytes",
                "gpu_name", "cuda_version"):
        if key not in resource:
            raise P6BFailClosedError(f"{unit.key}: missing resource field {key}")
    if not isinstance(resource["trainable_parameter_tensor_count"], int) \
            or resource["trainable_parameter_tensor_count"] < 1:
        raise P6BFailClosedError(f"{unit.key}: invalid trainable tensor count")
    for key in ("trainable_parameter_scalar_count",
                "total_parameter_scalar_count"):
        if not isinstance(resource[key], int) or resource[key] < 1:
            raise P6BFailClosedError(f"{unit.key}: invalid scalar count {key}")
    if resource["total_parameter_scalar_count"] < \
            resource["trainable_parameter_scalar_count"]:
        raise P6BFailClosedError(f"{unit.key}: total < trainable scalar count")
    gpu_seconds = resource["support_training_gpu_seconds"]
    if not isinstance(gpu_seconds, (int, float)) or gpu_seconds < 0:
        raise P6BFailClosedError(f"{unit.key}: invalid support GPU seconds")
    peak = resource["peak_allocated_gpu_memory_bytes"]
    if peak is not None and (not isinstance(peak, int) or peak < 0):
        raise P6BFailClosedError(f"{unit.key}: invalid peak memory bytes")

    class_order = completion.get("class_order")
    if not isinstance(class_order, list) or not class_order or \
            len(set(class_order)) != len(class_order):
        raise P6BFailClosedError(f"{unit.key}: invalid class_order")

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
    required = {HEAD_CHECKPOINT, SUPPORT_PROTO, STEP_AUDIT, RUN_CONFIG,
                META_JSONL, ARTIFACT_MANIFEST}
    if not required.issubset(seen):
        raise P6BFailClosedError(f"{unit.key}: required support artifacts missing")

    step_audit = load_json(unit_dir / STEP_AUDIT)
    for key, wanted in {
        "schema_version": STEP_AUDIT_SCHEMA,
        "method": unit.method, "product": unit.product,
        "K": unit.k_shot, "seed": unit.seed,
        "expected_steps": 500, "actual_steps": 500,
        "optimizer_ownership_ok": True,
        "ce_loss_finite": True,
    }.items():
        if step_audit.get(key) != wanted:
            raise P6BFailClosedError(f"{unit.key}: step audit {key} mismatch")

    run_config = load_json(unit_dir / RUN_CONFIG)
    for key, wanted in {
        "schema_version": RUN_CONFIG_SCHEMA, "method": unit.method,
        "product": unit.product, "K": unit.k_shot, "seed": unit.seed,
        "class_order": class_order, "ft_epo": 500, "zip_config_index": 6,
    }.items():
        if run_config.get(key) != wanted:
            raise P6BFailClosedError(f"{unit.key}: run config {key} mismatch")
    metadata = run_config.get("support_metadata") or {}
    if metadata.get("sha256") != file_sha256(unit_dir / META_JSONL):
        raise P6BFailClosedError(f"{unit.key}: support metadata SHA mismatch")
    counts = {name: 0 for name in class_order}
    rows = 0
    with (unit_dir / META_JSONL).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise P6BFailClosedError(
                    f"{unit.key}: invalid support meta line {line_no}"
                ) from exc
            name, label = row.get("class_name"), row.get("label")
            if name not in counts or label != class_order.index(name):
                raise P6BFailClosedError(f"{unit.key}: support meta class mapping drift")
            counts[name] += 1
            rows += 1
    if rows != metadata.get("records") or any(c != unit.k_shot for c in counts.values()):
        raise P6BFailClosedError(
            f"{unit.key}: support meta does not contain exact K per class"
        )

    artifact_manifest = load_json(unit_dir / ARTIFACT_MANIFEST)
    for key, wanted in {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA, "method": unit.method,
        "product": unit.product, "K": unit.k_shot, "seed": unit.seed,
    }.items():
        if artifact_manifest.get(key) != wanted:
            raise P6BFailClosedError(f"{unit.key}: artifact manifest {key} mismatch")
    completion_without_self = [r for r in artifacts
                               if r["relative_path"] != ARTIFACT_MANIFEST]
    if artifact_manifest.get("artifacts") != completion_without_self:
        raise P6BFailClosedError(f"{unit.key}: artifact manifest registry mismatch")
    return completion


def validate_formal_grid(grid_root, require_global_completion=True) -> dict:
    """Validate all 195 MVREC-RGB support units and the immutable index."""
    grid_root = Path(grid_root).resolve()
    global_record = (load_json(grid_root / GRID_COMPLETION)
                     if require_global_completion else None)
    expected_commit = None if global_record is None else global_record.get("execution_commit")
    records = []
    alpha_sha_set = set()
    alpha_sha_value = None
    for unit in expected_units():
        completion = validate_support_unit(
            unit, unit.directory(grid_root), expected_commit
        )
        alpha_sha = completion["model_provenance"]["alpha_ckpt_sha256"]
        alpha_sha_set.add(alpha_sha)
        alpha_sha_value = alpha_sha
        records.append({
            "key": unit.key,
            "completion_relative_path": str(
                (unit.directory(Path(".")) / COMPLETION).as_posix()
            ),
            "completion_sha256": file_sha256(unit.directory(grid_root) / COMPLETION),
            "class_order": completion["class_order"],
        })
    if len(records) != UNIT_COUNT:
        raise P6BFailClosedError(f"formal grid count {len(records)} != {UNIT_COUNT}")
    if len(alpha_sha_set) != 1:
        raise P6BFailClosedError(
            f"formal grid has {len(alpha_sha_set)} distinct AlphaCLIP weight identities"
        )
    if global_record is not None:
        checks = {
            "schema_version": GRID_SCHEMA,
            "status": "completed",
            "formal_eligible": True,
            "formal_test_used": False,
            "query_pixels_read": False,
            "query_labels_read": False,
            "unit_count": UNIT_COUNT,
            "method": "mvrec_rgb",
        }
        for key, wanted in checks.items():
            if global_record.get(key) != wanted:
                raise P6BFailClosedError(f"global grid {key} mismatch")
        if global_record.get("units") != records:
            raise P6BFailClosedError("global grid unit index is incomplete or stale")
        if global_record.get("backbone_identities") != {
            "alpha_clip": {"weight_sha256": alpha_sha_value},
        }:
            raise P6BFailClosedError("global grid backbone identity index is stale")
    return {"execution_commit": expected_commit, "units": records,
            "backbone_identities": {"alpha_clip": {"weight_sha256": alpha_sha_value}}}
