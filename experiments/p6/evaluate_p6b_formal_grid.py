#!/usr/bin/env python3
"""Unique P6.B formal query outlet; no support training is implemented here."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# Make direct-file launch equivalent to ``python -m`` before package imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # deliberate before torch (Windows/OpenMP safety)
import torch
import torch.nn.functional as F
from PIL import Image

from experiments.p6.p6b_execution_common import P6BFailClosedError, atomic_write_json
from experiments.p6 import p6b_stage2_execution as stage2
from experiments.p6 import p6b_stage2_models as models
from experiments.p6 import run_p6b_stage2_support as support
from experiments.p6 import build_p6b_support_manifests as builder
from experiments.p6.audit_p6b_formal_grid import audit_files, read_prediction_rows
from experiments.p6.p6b_stage3_common import (
    EXPECTED_FORMAL_QUERY_COUNT, GRID_COMPLETION, GRID_SCHEMA, K_VALUES, METHODS, PRODUCTS,
    PREDICTION_MANIFEST_SCHEMA, PREDICTION_SCHEMA, SEEDS, expected_units,
    UNIT_COUNT, UNIT_PREDICTION_MANIFEST_SCHEMA, file_sha256, load_json, load_query_contract,
    validate_formal_grid,
)

PREDICTIONS_NAME = "p6b_formal_query_predictions.jsonl"
PREDICTION_MANIFEST_NAME = "p6b_formal_query_predictions_manifest.json"
AUDIT_NAME = "p6b_formal_query_audit.json"
RESUME_STATE_NAME = "p6b_formal_query_resume_state.json"
RESUME_STATE_SCHEMA = "p6b_formal_query_resume_v1"
METHOD_EXECUTION_ORDER = (
    "ours", "p30_frozen_proto", "linear_probe", "fusion_proto",
    "tip_adapter_f", "tip_adapter", "clip_proto", "dino_proto",
    "zero_shot_clip",
)


def _artifact(unit_dir: Path, name: str) -> Path:
    path = (Path(unit_dir) / name).resolve()
    root = Path(unit_dir).resolve()
    if root not in path.parents or not path.is_file():
        raise P6BFailClosedError(f"method artifact missing/escaping unit: {name}")
    return path


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise P6BFailClosedError(f"cannot load method artifact {path}: {exc}") from exc


def _validate_tensor(tensor, rows, cols, name):
    tensor = torch.as_tensor(tensor, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape != (rows, cols):
        raise P6BFailClosedError(f"{name} shape {tuple(tensor.shape)} != {(rows, cols)}")
    if not torch.isfinite(tensor).all():
        raise P6BFailClosedError(f"{name} contains non-finite values")
    return tensor


def _images(records):
    result = []
    for record in records:
        try:
            with Image.open(record["image_path"]) as handle:
                result.append(handle.convert("RGB").copy())
        except (OSError, ValueError) as exc:
            raise P6BFailClosedError(f"cannot decode query RGB {record['relative_path']}: {exc}") from exc
    return result


def _chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _build_adapted_clip(project_root: Path, device):
    from model.adapter import AdaptedCLIP
    locked = support.LOCKED_OURS
    clip_model, _ = models.build_clip_backbone(project_root, device)
    hub_repo = project_root / models.DINO_HUB_REPO_RELATIVE_PATH
    hub_checkpoint = project_root / models.DINO_HUB_CHECKPOINT_RELATIVE_PATH
    models._fail_closed_offline_dino()
    if not hub_checkpoint.is_file() or file_sha256(hub_checkpoint) != models.EXPECTED_DINO_SHA256:
        raise P6BFailClosedError("adapted-model DINO checkpoint missing or SHA mismatch")
    model = AdaptedCLIP(
        clip_model=clip_model, relu=True, dino_use_hub_first=True,
        gate_type=locked["gate_type"], img_size=support.IMG_SIZE,
        proto_text_refine=locked["proto_text_refine"],
        dino_mask_ratio=locked["dino_mask_ratio"], clip_pool=locked["clip_pool"],
        dino_pool=locked["dino_pool"], dino_patch_policy=locked["dino_patch_policy"],
        dino_strict_backend=True, dino_hub_repo=str(hub_repo),
        dino_hub_checkpoint=str(hub_checkpoint), rng_protocol=locked["rng_protocol"],
        adapter_train_mode=locked["adapter_train_mode"],
        normal_aware_head=locked["normal_aware_head"],
    ).to(device)
    model.eval()
    return model


class MethodPredictor:
    """One-method-at-a-time inference to bound accelerator memory."""

    def __init__(self, method, project_root, device, batch_size):
        self.method = method
        self.project_root = Path(project_root)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise P6BFailClosedError("query batch size must be positive")
        self.factory = None
        self.model = None
        self.model_seed = None
        self.feature_cache = {}
        ctx = SimpleNamespace(project_root=self.project_root, device=self.device)
        if method in {"clip_proto", "zero_shot_clip", "tip_adapter", "tip_adapter_f"}:
            self.factory = support._build_lazy_clip_only(ctx)
        elif method == "dino_proto":
            self.factory = support._build_lazy_dino_only(ctx)
        elif method in {"fusion_proto", "linear_probe"}:
            self.factory = support._build_lazy_clip_and_dino(ctx)
        elif method == "ours":
            self.model = _build_adapted_clip(self.project_root, self.device)
        elif method == "p30_frozen_proto":
            # R1 construction consumes its locked seed; build lazily per seed
            # and reuse across K=1/3/5 for that seed.
            pass
        else:
            raise P6BFailClosedError(f"unsupported formal method: {method}")

    def close(self):
        self.factory = None
        self.model = None
        self.feature_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _baseline_features(self, records):
        product = records[0]["product"]
        if product in self.feature_cache:
            return self.feature_cache[product]
        output = []
        for batch_records in _chunks(records, self.batch_size):
            images = _images(batch_records)
            with torch.no_grad():
                if self.method == "dino_proto":
                    features = self.factory.dino_extractor_interface.embed_images(images)
                elif self.method in {"fusion_proto", "linear_probe"}:
                    clip = self.factory.clip_extractor.embed_images(images)
                    dino = self.factory.dino_extractor_interface.embed_images(images)
                    features = self.factory.fusion.fuse(clip, dino)
                else:
                    features = self.factory.clip_extractor.embed_images(images)
            output.append(features.detach().cpu())
        result = torch.cat(output, dim=0)
        self.feature_cache[product] = result
        return result

    def _adapted_features(self, records, pre_terminal=False, seed=None):
        if pre_terminal:
            if seed is None:
                raise P6BFailClosedError("R1 query inference requires the locked seed")
            if self.model_seed != int(seed):
                from utils import setup_seed
                self.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                setup_seed(int(seed))
                self.model = _build_adapted_clip(self.project_root, self.device)
                for parameter in self.model.parameters():
                    parameter.requires_grad_(False)
                self.model_seed = int(seed)
                self.feature_cache.clear()
            cache_key = (int(seed), records[0]["product"])
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
        output = []
        for batch_records in _chunks(records, self.batch_size):
            images = _images(batch_records)
            batch = torch.cat([models.deterministic_eval_tensor(image) for image in images]).to(self.device)
            with torch.no_grad():
                if pre_terminal:
                    features, _tokens = support.extract_r1_pre_terminal_feature(self.model, batch)
                else:
                    features = self.model.forward_fusion(batch)
            output.append(features.detach().cpu())
        result = torch.cat(output, dim=0)
        if pre_terminal:
            self.feature_cache[(int(seed), records[0]["product"])] = result
        return result

    def predict(self, unit_dir: Path, class_order: list[str], records: list[dict], *, seed=None):
        n_classes = len(class_order)
        method = self.method
        if method in {"clip_proto", "dino_proto", "fusion_proto"}:
            data = _torch_load(_artifact(unit_dir, f"{method}_prototypes.pt"))
            if (data.get("schema_version") != support.RUN_CONFIG_SCHEMA or
                    data.get("method") != method or data.get("class_order") != class_order):
                raise P6BFailClosedError(f"{method}: artifact identity mismatch")
            query = self._baseline_features(records)
            prototypes = _validate_tensor(data.get("prototypes"), n_classes, query.shape[1], "prototypes")
            return F.normalize(query, dim=-1) @ F.normalize(prototypes, dim=-1).t()

        if method == "zero_shot_clip":
            data = _torch_load(_artifact(unit_dir, "zero_shot_clip_text_weights.pt"))
            expected_prompts = [stage2.format_zero_shot_prompt(name) for name in class_order]
            if (data.get("schema_version") != support.RUN_CONFIG_SCHEMA or
                    data.get("method") != method or data.get("class_order") != class_order or
                    data.get("logit_scale") != 100.0 or data.get("prompts") != expected_prompts):
                raise P6BFailClosedError("zero-shot artifact identity/constants mismatch")
            query = self._baseline_features(records)
            return stage2.zero_shot_clip_logits(query, data["text_embeddings"], data["logit_scale"])

        if method == "tip_adapter":
            data = _torch_load(_artifact(unit_dir, "tip_adapter_cache.pt"))
            if (data.get("schema_version") != support.RUN_CONFIG_SCHEMA or
                    data.get("method") != method or data.get("class_order") != class_order or
                    data.get("alpha") != 1.0 or data.get("beta") != 5.5):
                raise P6BFailClosedError("Tip-Adapter artifact identity/constants mismatch")
            query = self._baseline_features(records)
            prompts = [stage2.format_zero_shot_prompt(name) for name in class_order]
            text = self.factory.text_extractor.embed_text(prompts).cpu()
            clip_logits = stage2.zero_shot_clip_logits(query, text)
            cache = stage2.TipAdapterCache(data["cache_keys"], data["cache_labels"], n_classes)
            if not torch.equal(torch.as_tensor(data.get("cache_values")), cache.values):
                raise P6BFailClosedError("Tip-Adapter cache_values/labels mismatch")
            return cache.logits(clip_logits, query)

        if method == "linear_probe":
            data = _torch_load(_artifact(unit_dir, "linear_probe_head.pt"))
            locked_spec = support.dataclasses_asdict(stage2.validate_linear_probe_spec(stage2.LinearProbeSpec()))
            if (data.get("schema_version") != support.RUN_CONFIG_SCHEMA or
                    data.get("method") != method or data.get("class_order") != class_order or
                    data.get("num_classes") != n_classes or
                    data.get("fusion_dim") != models.EXPECTED_FUSION_DIM or
                    data.get("locked_spec") != locked_spec or data.get("final_epoch") != 19):
                raise P6BFailClosedError("Linear Probe artifact identity mismatch")
            query = self._baseline_features(records)
            state = data.get("head_state_dict") or {}
            weight = _validate_tensor(state.get("weight"), n_classes, query.shape[1], "linear weight")
            bias = torch.as_tensor(state.get("bias"), dtype=torch.float32)
            if bias.shape != (n_classes,) or not torch.isfinite(bias).all():
                raise P6BFailClosedError("invalid linear bias")
            return query @ weight.t() + bias

        if method == "p30_frozen_proto":
            data = _torch_load(_artifact(unit_dir, "p30_frozen_proto_prototypes.pt"))
            if (data.get("schema_version") != support.RUN_CONFIG_SCHEMA or
                    data.get("method") != method or data.get("class_order") != class_order or
                    data.get("terminal_adaptation") is not False or
                    data.get("trainable_param_count") != 0):
                raise P6BFailClosedError("R1 artifact identity mismatch")
            query = self._adapted_features(records, pre_terminal=True, seed=seed)
            prototypes = _validate_tensor(data.get("prototypes"), n_classes, query.shape[1], "R1 prototypes")
            return F.normalize(query, dim=-1) @ F.normalize(prototypes, dim=-1).t()

        if method == "tip_adapter_f":
            data = _torch_load(_artifact(unit_dir, "tip_adapter_f_cache.pt"))
            constants = (data.get("alpha"), data.get("beta"), data.get("logit_scale"))
            locked_spec = support.dataclasses_asdict(stage2.validate_tip_adapter_f_spec(stage2.TipAdapterFSpec()))
            if (data.get("schema_version") != support.RUN_CONFIG_SCHEMA or
                    data.get("method") != method or data.get("class_order") != class_order or
                    constants != (1.0, 5.5, 100.0) or data.get("locked_spec") != locked_spec or
                    data.get("final_epoch") != 19 or
                    data.get("checkpoint_selection") != "final_epoch_only" or
                    data.get("query_used_for_selection") is not False or
                    data.get("validation_used_for_selection") is not False):
                raise P6BFailClosedError("R2 artifact identity/constants mismatch")
            query = self._baseline_features(records)
            module = stage2.TipAdapterFModule(
                data["cache_keys"], data["cache_labels"], n_classes, data["text_weights"])
            with torch.no_grad():
                module.cache_keys.copy_(torch.as_tensor(data["cache_keys"]))
                if not torch.equal(torch.as_tensor(data.get("cache_values")), module.cache_values):
                    raise P6BFailClosedError("R2 cache_values/labels mismatch")
                return module.logits(query)

        if method == "ours":
            from model.checkpoint_io import load_model_checkpoint_state, restore_prototype_extra_state
            from model.faiss_memory import FaissPrototypeMemory
            checkpoint = _torch_load(_artifact(unit_dir, "multiclass_epoch_19.pth"))
            expected_class_to_id = {name: index for index, name in enumerate(class_order)}
            # Production train_multiclass stores the completed epoch count
            # (20) in the final epoch_19 filename.
            if (checkpoint.get("epoch") != support.LOCKED_OURS["num_epochs"] or
                    checkpoint.get("class_to_id") != expected_class_to_id or
                    (checkpoint.get("run_config") or {}).get("class_order") != class_order):
                raise P6BFailClosedError("Ours checkpoint identity/final-epoch mismatch")
            load_model_checkpoint_state(self.model, checkpoint)
            prototype = FaissPrototypeMemory(
                num_classes=n_classes, feat_dim=self.model.fusion_dim,
                orth_margin=0.3, init_logit_scale=support.LOCKED_OURS["init_logit_scale"],
                prototype_init_mode=support.LOCKED_OURS["prototype_init_mode"],
                prototypes_per_class=support.LOCKED_OURS["prototypes_per_class"],
                prototype_aggregation="max", prototype_mixture_temperature=0.2,
            )
            prototype.load_state_dict(checkpoint["prototype_mem_state_dict"], strict=True)
            restore_prototype_extra_state(prototype, checkpoint)
            prototype.eval()
            query = self._adapted_features(records, pre_terminal=False)
            with torch.no_grad():
                return prototype(query)
        raise AssertionError(method)


def _canonical_row(row):
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def _atomic_write_lines(path: Path, lines: list[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _unit_prediction_paths(staging: Path, unit):
    relative = Path("predictions") / unit.method / unit.product / f"k{unit.k_shot}_s{unit.seed}.jsonl"
    return relative, Path(staging) / relative, relative.with_suffix(".manifest.json")


def _validate_unit_prediction(staging: Path, grid_root: Path, unit, queries: list[dict]):
    """Validate one resumable unit without computing or exposing any metric."""
    unit_relative, unit_path, manifest_relative = _unit_prediction_paths(staging, unit)
    manifest_path = Path(staging) / manifest_relative
    if not unit_path.is_file() or not manifest_path.is_file():
        raise P6BFailClosedError(f"resumable unit artifact missing: {unit.key}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unit_dir = unit.directory(Path(grid_root))
    completion = json.loads(
        (unit_dir / "p6b_stage2_support_complete.json").read_text(encoding="utf-8"))
    class_order = completion["class_order"]
    product_queries = [query for query in queries if query["product"] == unit.product]
    exact = {
        "schema_version": UNIT_PREDICTION_MANIFEST_SCHEMA,
        "method": unit.method, "product": unit.product,
        "K": unit.k_shot, "seed": unit.seed,
        "class_order": class_order,
        "query_count": len(product_queries),
        "query_rgb_sha256": [query["rgb_sha256"] for query in product_queries],
        "support_completion_sha256": file_sha256(
            unit_dir / "p6b_stage2_support_complete.json"),
        "predictions_relative_path": unit_relative.as_posix(),
    }
    for key, wanted in exact.items():
        if manifest.get(key) != wanted:
            raise P6BFailClosedError(f"resumable unit manifest {key} mismatch: {unit.key}")
    if (manifest.get("predictions_size_bytes") != unit_path.stat().st_size or
            manifest.get("predictions_sha256") != file_sha256(unit_path)):
        raise P6BFailClosedError(f"resumable unit prediction hash mismatch: {unit.key}")
    rows, streamed_sha = read_prediction_rows(unit_path)
    if streamed_sha != manifest["predictions_sha256"] or len(rows) != len(product_queries):
        raise P6BFailClosedError(f"resumable unit prediction coverage mismatch: {unit.key}")
    for row, query in zip(rows, product_queries):
        wanted = {
            "schema_version": PREDICTION_SCHEMA,
            "method": unit.method, "product": unit.product,
            "K": unit.k_shot, "seed": unit.seed,
            "rgb_sha256": query["rgb_sha256"],
            "true_label": query["true_label"],
            "class_order": class_order,
        }
        if any(row.get(key) != value for key, value in wanted.items()):
            raise P6BFailClosedError(f"resumable prediction identity mismatch: {unit.key}")
        scores = row.get("scores")
        if (not isinstance(scores, list) or len(scores) != len(class_order) or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) for value in scores)):
            raise P6BFailClosedError(f"resumable prediction scores invalid: {unit.key}")
        predicted = class_order[max(range(len(scores)), key=lambda index: scores[index])]
        if row.get("predicted_label") != predicted:
            raise P6BFailClosedError(f"resumable prediction argmax mismatch: {unit.key}")
    return {
        "key": unit.key,
        "predictions_relative_path": unit_relative.as_posix(),
        "predictions_sha256": manifest["predictions_sha256"],
        "prediction_manifest_relative_path": manifest_relative.as_posix(),
        "prediction_manifest_sha256": file_sha256(manifest_path),
        "query_count": len(product_queries),
    }


def _resume_state(*, grid, evaluator_commit, records_path, stage1_bundle, batch_size):
    return {
        "schema_version": RESUME_STATE_SCHEMA,
        "status": "in_progress",
        "formal_test_used": False,
        "query_pixels_read": True,
        "query_labels_read": True,
        "support_grid_execution_commit": grid["execution_commit"],
        "evaluator_execution_commit": evaluator_commit,
        "query_records_sha256": file_sha256(Path(records_path)),
        "stage1_completion_sha256": file_sha256(
            Path(stage1_bundle) / "p6b_manifest_build_complete.json"),
        "formal_query_count": EXPECTED_FORMAL_QUERY_COUNT,
        "batch_size": int(batch_size),
    }


def _prepare_staging(*, staging, output_dir, resume, expected_state):
    staging, output_dir = Path(staging), Path(output_dir)
    if output_dir.exists():
        raise P6BFailClosedError("formal query output already exists")
    state_path = staging / RESUME_STATE_NAME
    if resume:
        if not staging.is_dir() or not state_path.is_file():
            raise P6BFailClosedError("--resume requires a valid existing staging directory")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key, wanted in expected_state.items():
            if state.get(key) != wanted:
                raise P6BFailClosedError(f"resume state mismatch at {key}")
        # These three artifacts are derived from canonical unit files and may
        # be safely regenerated after a late finalization interruption.
        for name in (PREDICTIONS_NAME, PREDICTION_MANIFEST_NAME, AUDIT_NAME):
            path = staging / name
            if path.exists():
                if not path.is_file():
                    raise P6BFailClosedError(f"invalid resumable finalization path: {path}")
                path.unlink()
    else:
        if staging.exists():
            raise P6BFailClosedError(
                "formal query staging exists; use --resume after verifying it belongs to this run")
        staging.mkdir(parents=True)
        atomic_write_json(state_path, expected_state)


def _write_canonical_global_predictions(staging: Path, grid_root: Path, queries):
    """Rebuild the global stream/index in canonical expected_units() order."""
    predictions_path = Path(staging) / PREDICTIONS_NAME
    temporary = predictions_path.with_name(f".{predictions_path.name}.tmp-{os.getpid()}")
    unit_records = []
    row_count = 0
    try:
        with temporary.open("wb") as target:
            for unit in expected_units():
                record = _validate_unit_prediction(staging, grid_root, unit, queries)
                unit_records.append(record)
                unit_path = Path(staging) / record["predictions_relative_path"]
                with unit_path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                row_count += record["query_count"]
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, predictions_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return predictions_path, unit_records, row_count


def _run_method_units(*, method, grid_root, staging, queries, device, batch_size):
    """Evaluate or resume one method; safe to execute in an isolated process."""
    if method not in METHODS:
        raise P6BFailClosedError(f"query worker method is outside locked set: {method}")
    predictor = None
    completed = 0
    method_units = [unit for unit in expected_units() if unit.method == method]
    # Operational order only: R1 rebuilds its frozen seeded model five times
    # instead of 195 times. Canonical publication is rebuilt independently.
    if method == "p30_frozen_proto":
        method_units.sort(key=lambda unit: (unit.seed, unit.product, unit.k_shot))
    try:
        for ordinal, unit in enumerate(method_units, 1):
            unit_relative, unit_path, unit_manifest_relative = _unit_prediction_paths(
                staging, unit)
            unit_manifest_path = Path(staging) / unit_manifest_relative
            if unit_path.is_file() and unit_manifest_path.is_file():
                _validate_unit_prediction(staging, grid_root, unit, queries)
                completed += 1
                print(
                    f"[P6.B query worker:{method}] resume-valid "
                    f"{ordinal}/{len(method_units)} {unit.key}", flush=True)
                continue
            if unit_path.exists() or unit_manifest_path.exists():
                for partial in (unit_path, unit_manifest_path):
                    if partial.exists():
                        if not partial.is_file():
                            raise P6BFailClosedError(
                                f"invalid partial unit artifact: {partial}")
                        partial.unlink()
            if predictor is None:
                predictor = MethodPredictor(method, PROJECT_ROOT, device, batch_size)
            unit_dir = unit.directory(Path(grid_root))
            completion = json.loads(
                (unit_dir / "p6b_stage2_support_complete.json").read_text(encoding="utf-8"))
            class_order = completion["class_order"]
            product_queries = [query for query in queries if query["product"] == unit.product]
            scores = predictor.predict(
                unit_dir, class_order, product_queries, seed=unit.seed)
            scores = _validate_tensor(
                scores, len(product_queries), len(class_order), "query logits")
            unit_lines = []
            for query, row_scores in zip(product_queries, scores.tolist()):
                predicted_index = max(
                    range(len(row_scores)), key=lambda index: row_scores[index])
                row = {
                    "schema_version": PREDICTION_SCHEMA,
                    "method": method, "product": unit.product,
                    "K": unit.k_shot, "seed": unit.seed,
                    "rgb_sha256": query["rgb_sha256"],
                    "true_label": query["true_label"],
                    "class_order": class_order,
                    "scores": [float(value) for value in row_scores],
                    "predicted_label": class_order[predicted_index],
                }
                unit_lines.append(_canonical_row(row))
            _atomic_write_lines(unit_path, unit_lines)
            unit_manifest = {
                "schema_version": UNIT_PREDICTION_MANIFEST_SCHEMA,
                "method": method, "product": unit.product,
                "K": unit.k_shot, "seed": unit.seed,
                "class_order": class_order,
                "query_count": len(product_queries),
                "query_rgb_sha256": [query["rgb_sha256"] for query in product_queries],
                "support_completion_sha256": file_sha256(
                    unit_dir / "p6b_stage2_support_complete.json"),
                "predictions_relative_path": unit_relative.as_posix(),
                "predictions_sha256": file_sha256(unit_path),
                "predictions_size_bytes": unit_path.stat().st_size,
            }
            atomic_write_json(unit_manifest_path, unit_manifest)
            _validate_unit_prediction(staging, grid_root, unit, queries)
            completed += 1
            print(
                f"[P6.B query worker:{method}] completed "
                f"{ordinal}/{len(method_units)} {unit.key}", flush=True)
    finally:
        if predictor is not None:
            predictor.close()
    return {"method": method, "completed_units": completed}


def _method_worker_command(*, method, grid_root, records_path, dataset_root,
                           stage1_bundle, output_dir, device, batch_size):
    command = [
        sys.executable, "-m", "experiments.p6.evaluate_p6b_formal_grid",
        "--grid-root", str(Path(grid_root).resolve()),
        "--records-path", str(Path(records_path).resolve()),
        "--dataset-root", str(Path(dataset_root).resolve()),
        "--stage1-bundle", str(Path(stage1_bundle).resolve()),
        "--output-dir", str(Path(output_dir).resolve()),
        "--batch-size", str(int(batch_size)),
        "--worker-method", method,
    ]
    if device:
        command.extend(["--device", str(device)])
    return command


def _run_method_subprocess(command, method):
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise P6BFailClosedError(
            f"query method worker failed ({completed.returncode}): {method}")
    return method


def _run_parallel_methods(*, methods, max_parallel_methods, grid_root,
                          records_path, dataset_root, stage1_bundle, output_dir,
                          device, batch_size):
    if max_parallel_methods < 1 or max_parallel_methods > len(METHODS):
        raise P6BFailClosedError(
            f"max-parallel-methods must be in [1, {len(METHODS)}]")
    commands = {
        method: _method_worker_command(
            method=method, grid_root=grid_root, records_path=records_path,
            dataset_root=dataset_root, stage1_bundle=stage1_bundle,
            output_dir=output_dir, device=device, batch_size=batch_size)
        for method in methods
    }
    failures = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_parallel_methods,
            thread_name_prefix="p6b-query-method") as pool:
        futures = {
            pool.submit(_run_method_subprocess, commands[method], method): method
            for method in methods
        }
        for future in concurrent.futures.as_completed(futures):
            method = futures[future]
            try:
                future.result()
            except BaseException as exc:
                failures.append((method, exc))
    if failures:
        detail = "; ".join(f"{method}: {exc}" for method, exc in failures)
        raise P6BFailClosedError(
            f"{len(failures)} query method worker(s) failed; {detail}")


def evaluate_method_worker(*, method, grid_root, records_path, dataset_root,
                           stage1_bundle, output_dir, device=None, batch_size=16):
    """Internal subprocess entry: write only one method's resumable units."""
    evaluator_commit, evaluator_dirty = builder.git_commit_and_dirty(PROJECT_ROOT)
    if evaluator_dirty:
        raise P6BFailClosedError("formal evaluator worker requires a clean git tree")
    output_dir = Path(output_dir).resolve()
    staging = output_dir.with_name(output_dir.name + ".p6b_staging")
    state_path = staging / RESUME_STATE_NAME
    if output_dir.exists() or not staging.is_dir() or not state_path.is_file():
        raise P6BFailClosedError("query worker requires parent-owned resumable staging")
    actual_state = load_json(state_path)
    # The parent deep-validates all 1,755 units before creating staging. Child
    # processes verify the immutable global record and parent-owned state, but
    # deliberately do not repeat the ~25GB artifact hash pass four times.
    grid_record = load_json(Path(grid_root) / GRID_COMPLETION)
    grid_checks = {
        "schema_version": GRID_SCHEMA, "status": "completed",
        "formal_eligible": True, "formal_test_used": False,
        "query_pixels_read": False, "query_labels_read": False,
        "unit_count": UNIT_COUNT,
        "execution_commit": actual_state.get("support_grid_execution_commit"),
    }
    for key, wanted in grid_checks.items():
        if grid_record.get(key) != wanted:
            raise P6BFailClosedError(f"query worker grid record mismatch at {key}")
    grid = {"execution_commit": grid_record["execution_commit"]}
    expected_state = _resume_state(
        grid=grid, evaluator_commit=evaluator_commit, records_path=records_path,
        stage1_bundle=stage1_bundle, batch_size=batch_size)
    for key, wanted in expected_state.items():
        if actual_state.get(key) != wanted:
            raise P6BFailClosedError(f"query worker resume state mismatch at {key}")
    queries, _query_audit = load_query_contract(
        records_path, dataset_root, stage1_bundle, verify_rgb=False)
    if len(queries) != EXPECTED_FORMAL_QUERY_COUNT:
        raise P6BFailClosedError("formal query worker contract count drift")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return _run_method_units(
        method=method, grid_root=grid_root, staging=staging,
        queries=queries, device=device, batch_size=batch_size)


def evaluate(*, grid_root, records_path, dataset_root, stage1_bundle, output_dir,
             device=None, batch_size=16, resume=False, max_parallel_methods=1):
    # Absolute firewall: all 1,755 support units and the global index pass
    # before the records JSONL (and therefore query labels/paths) is opened.
    grid = validate_formal_grid(grid_root, require_global_completion=True)
    evaluator_commit, evaluator_dirty = builder.git_commit_and_dirty(PROJECT_ROOT)
    if evaluator_dirty:
        raise P6BFailClosedError("formal evaluator requires a clean git tree")
    queries, query_audit = load_query_contract(records_path, dataset_root, stage1_bundle)
    if len(queries) != EXPECTED_FORMAL_QUERY_COUNT:
        raise P6BFailClosedError("formal query contract count drift")
    by_product = {product: [q for q in queries if q["product"] == product]
                  for product in PRODUCTS}
    output_dir = Path(output_dir).resolve()
    staging = output_dir.with_name(output_dir.name + ".p6b_staging")
    state = _resume_state(
        grid=grid, evaluator_commit=evaluator_commit, records_path=records_path,
        stage1_bundle=stage1_bundle, batch_size=batch_size)
    _prepare_staging(
        staging=staging, output_dir=output_dir, resume=resume, expected_state=state)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        total_units = len(list(expected_units()))
        operational_methods = [method for method in METHOD_EXECUTION_ORDER if method in METHODS]
        if set(operational_methods) != set(METHODS):
            raise P6BFailClosedError("operational method order does not cover locked methods")
        if max_parallel_methods == 1:
            for method in operational_methods:
                _run_method_units(
                    method=method, grid_root=grid_root, staging=staging,
                    queries=queries, device=device, batch_size=batch_size)
        else:
            _run_parallel_methods(
                methods=operational_methods,
                max_parallel_methods=max_parallel_methods,
                grid_root=grid_root, records_path=records_path,
                dataset_root=dataset_root, stage1_bundle=stage1_bundle,
                output_dir=output_dir, device=device, batch_size=batch_size)
        predictions_path, unit_prediction_artifacts, row_count = (
            _write_canonical_global_predictions(staging, grid_root, queries))
        expected_rows = len(METHODS) * len(K_VALUES) * len(SEEDS) * EXPECTED_FORMAL_QUERY_COUNT
        if row_count != expected_rows:
            raise P6BFailClosedError(f"prediction row count {row_count} != {expected_rows}")
        manifest = {
            "schema_version": PREDICTION_MANIFEST_SCHEMA,
            "status": "completed", "formal_test_used": False,
            "query_pixels_read": True, "query_labels_read": True,
            "support_grid_execution_commit": grid["execution_commit"],
            "evaluator_execution_commit": evaluator_commit,
            "operational_method_workers": int(max_parallel_methods),
            "formal_query_count": EXPECTED_FORMAL_QUERY_COUNT,
            "prediction_row_count": row_count,
            "predictions_relative_path": PREDICTIONS_NAME,
            "predictions_sha256": file_sha256(predictions_path),
            "query_contract": query_audit,
            "unit_predictions": unit_prediction_artifacts,
        }
        atomic_write_json(staging / PREDICTION_MANIFEST_NAME, manifest)
        # Independent code path recomputes every metric before publication.
        audit_files(
            grid_root=grid_root, predictions_path=predictions_path,
            prediction_manifest_path=staging / PREDICTION_MANIFEST_NAME,
            records_path=records_path, dataset_root=dataset_root,
            stage1_bundle=stage1_bundle, output_path=staging / AUDIT_NAME,
        )
        completed_state = dict(state)
        completed_state["status"] = "completed"
        completed_state["completed_unit_count"] = total_units
        completed_state["predictions_sha256"] = manifest["predictions_sha256"]
        atomic_write_json(staging / RESUME_STATE_NAME, completed_state)
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        if output_dir.exists():
            raise P6BFailClosedError("late evaluator failure left a published output")
        print(
            f"[P6.B formal evaluator] resumable staging preserved at: {staging}",
            flush=True)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", required=True)
    parser.add_argument("--records-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stage1-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-parallel-methods", type=int, default=1,
                        help="isolated method subprocesses sharing one GPU; default 1")
    parser.add_argument("--resume", action="store_true",
                        help="resume a validated staging run; never recompute completed units")
    parser.add_argument("--worker-method", choices=METHODS, default=None,
                        help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    try:
        args = vars(build_parser().parse_args(argv))
        worker_method = args.pop("worker_method")
        if worker_method is not None:
            args.pop("resume")
            args.pop("max_parallel_methods")
            evaluate_method_worker(method=worker_method, **args)
            print(f"[P6.B formal evaluator] method worker completed: {worker_method}")
        else:
            evaluate(**args)
            print("[P6.B formal evaluator] completed")
        return 0
    except (P6BFailClosedError, OSError, RuntimeError) as exc:
        print(f"[P6.B formal evaluator] INCONCLUSIVE / FAIL-CLOSED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
