"""MVREC-RGB formal support-side unit runner — one (product, K, seed) config.

Formal single-config CLI producing support-side artifacts ONLY for the
MVREC-RGB adaptation.  This runner:

- reuses the frozen P6.B support manifest for the unit (self-SHA verified);
- builds the official MVREC model (AlphaClip ViT-L/14 GRIT-20M +
  EchoClassfierF) and executes the official support pipeline
  ``set_img_prototype`` (ft_epo=500 AdamW lr=1e-4 eps=1e-6 CE(+triplet)
  CosineAnnealingLR) exactly as the semantic audit did;
- writes the trained head checkpoint plus a per-unit provenance/audit
  bundle, atomically published;
- never reads query pixels/labels/metrics and never writes accuracy/BA/F1.

Support-only smoke is the ONLY authorized execution mode until the main
agent authorizes the cloud 195-unit support grid.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# cole environment safe import order: numpy MUST be imported before torch to
# avoid the OpenMP runtime duplicate-library crash in this Windows env.
import numpy  # noqa: F401  (deliberately before torch)

import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import (  # noqa: E402
    P6BFailClosedError,
    atomic_write_json,
    atomic_write_text,
    sha256_bytes,
    sha256_file,
)
from experiments.p6.mvrec_rgb.protocol_lock import (  # noqa: E402
    compute_protocol_hashes,
    sha256_file as lock_sha256_file,
)
from experiments.p6.mvrec_rgb.semantic_smoke import (  # noqa: E402
    EXPECTED_TRAINABLE,
    _TrainingCapture,
    _fix_rng,
    _optimizer_ownership,
    check_trainable_whitelist,
)
from experiments.p6.mvrec_rgb_formal.execution_manifest import (  # noqa: E402
    MvrecRgbUnit,
    load_frozen_manifest,
)

RUN_CONFIG_SCHEMA = "mvrec_rgb_support_run_config_v1"
ARTIFACT_MANIFEST_SCHEMA = "mvrec_rgb_support_artifact_manifest_v1"
COMPLETION_SCHEMA = "mvrec_rgb_support_complete_v1"
STEP_AUDIT_SCHEMA = "mvrec_rgb_step_audit_v1"
STAGING_SUFFIX = ".mvrec_rgb_staging"

HEAD_CHECKPOINT = "mvrec_rgb_head.pt"
SUPPORT_PROTO = "mvrec_rgb_support_proto.pt"
STEP_AUDIT = "mvrec_rgb_step_audit.json"
RUN_CONFIG = "support_run_config.json"
META_JSONL = "p6b_support_meta.jsonl"
ARTIFACT_MANIFEST = "support_artifact_manifest.json"
COMPLETION = "mvrec_rgb_support_complete.json"

# Official locked training semantics (semantic audit constants).
FT_EPO = 500
ZIP_CONFIG_INDEX = 6
OPTIMIZER_SPEC = {"name": "AdamW", "lr": 1e-4, "eps": 1e-6}
SCHEDULER_SPEC = {"name": "CosineAnnealingLR", "T_max": FT_EPO,
                  "eta_min": 1e-5}


def git_commit_and_dirty(project_root: Path):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(project_root),
        capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(project_root),
        capture_output=True, text=True).stdout
    return commit, bool(status.strip())


def build_artifacts_dir_clean(output_dir: Path) -> Path:
    if output_dir.exists():
        raise P6BFailClosedError(
            f"output directory already exists: {output_dir} (resume via grid)"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_dir


def _capture_training(model, dataset, unit_key):
    """Run the official support pipeline under the step/optimizer capture.

    Returns the capture record from the exact official code path
    (``ClipModel.set_img_prototype``), never a re-implementation.

    Locked resource-measurement scope (decision rule section 5.1):
    the GPU-timing and peak-memory window covers exactly the official
    ``set_img_prototype`` call — support feature encoding plus the official
    head fine-tuning loop — surrounded by CUDA synchronize and a
    reset_peak_memory_stats immediately before the call.  Model
    construction and checkpoint loading are outside the window.  This scope
    is fixed and must not be changed after any result is seen.
    """
    capture = _TrainingCapture()
    try:
        _fix_rng(dataset.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model.set_img_prototype(dataset.k_shot, dataset)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_allocated = int(torch.cuda.max_memory_allocated())
        else:
            peak_allocated = None
        train_seconds = time.perf_counter() - t0
        trainable_names = sorted(
            n for n, p in model.named_parameters() if p.requires_grad
        )
        support_proto = getattr(model, "proto", None)
        if not isinstance(support_proto, tuple) or len(support_proto) != 2:
            raise P6BFailClosedError(
                f"{unit_key}: official support proto not captured"
            )
        support_proto = tuple(t.detach().cpu() for t in support_proto)
        actual_steps = None
        if capture.scheduler is not None:
            actual_steps = int(capture.scheduler.last_epoch) - int(
                getattr(capture.scheduler, "_capture_initial_last_epoch", 0)
            )
        ownership_ok, ownership_reason, ownership_entries = (
            _optimizer_ownership(capture.optimizer, FT_EPO)
            if capture.optimizer is not None
            else (False, "official optimizer instance not captured", [])
        )
        losses = capture.ce_losses
        return {
            "trainable_names": trainable_names,
            "actual_steps": actual_steps,
            "optimizer_ownership_ok": ownership_ok,
            "optimizer_ownership_reason": ownership_reason,
            "optimizer_ownership_entries": ownership_entries,
            "ce_loss_first": losses[0] if losses else None,
            "ce_loss_last": losses[-1] if losses else None,
            "ce_loss_finite": bool(losses and all(numpy.isfinite(x) for x in losses)),
            "head_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
                if k.startswith("head.")
            },
            "text_features_shape": list(model.text_features.shape),
            "support_proto": support_proto,
            "train_seconds": train_seconds,
            "peak_allocated_gpu_memory_bytes": peak_allocated,
        }
    finally:
        capture.restore()


def _resource_measurements(model, device):
    """Locked per-unit resource provenance (decision rule section 5.1).

    Scalar counts use the actual tensor ``numel()``: three trainable
    parameter tensors are reported as tensors=3 with their summed scalar
    count, never as three trainable parameters.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = list(model.parameters())
    measurements = {
        "device": device,
        "torch_version": torch.__version__,
        # Official training precision semantics: fp32 weights with the
        # official torch.cuda.amp.autocast forward inside train_compact.
        "precision": "fp32_weights_amp_autocast_forward",
        "trainable_parameter_tensor_count": len(trainable_params),
        "trainable_parameter_scalar_count": sum(
            int(p.numel()) for p in trainable_params
        ),
        "total_parameter_scalar_count": sum(int(p.numel()) for p in total_params),
        "gpu_name": None,
        "cuda_version": None,
    }
    if torch.cuda.is_available():
        measurements["gpu_name"] = torch.cuda.get_device_name(0)
        measurements["cuda_version"] = torch.version.cuda
    return measurements


def run_support_unit(
    unit: MvrecRgbUnit,
    manifest: dict,
    manifest_path: Path,
    image_root: Path,
    alpha_ckpt: Path,
    device: str,
    output_dir: Path,
    execution: dict,
    model_factory=None,
) -> dict:
    """Execute one formal support unit and atomically publish its artifacts.

    ``model_factory`` defaults to the real official model constructor; tests
    inject a fake with the same support-pipeline interface.
    """
    from experiments.p6.mvrec_rgb.rgb_dataset import MVRecRgbSupportDataset
    from experiments.p6.mvrec_rgb.rgb_model import build_mvrec_rgb_model

    if model_factory is None:
        def model_factory(class_names):
            model, _ = build_mvrec_rgb_model(
                class_names=class_names,
                alpha_ckpt_path=str(alpha_ckpt),
                k_shot=unit.k_shot,
                ft_epo=FT_EPO,
                zip_config_index=ZIP_CONFIG_INDEX,
                seed=unit.seed,
            )
            return model

    staging = output_dir.with_name(output_dir.name + STAGING_SUFFIX)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        dataset = MVRecRgbSupportDataset(
            str(manifest_path), str(image_root), device=device
        )
        identity_ok, mismatches = dataset.verify_support_identity()
        if not identity_ok:
            raise P6BFailClosedError(f"{unit.key}: support identity mismatch {mismatches}")

        class_order = dataset.class_order
        model = model_factory(class_order)
        capture = _capture_training(model, dataset, unit.key)
        resource = _resource_measurements(model, device)
        resource["support_training_gpu_seconds"] = capture["train_seconds"]
        resource["peak_allocated_gpu_memory_bytes"] = (
            capture["peak_allocated_gpu_memory_bytes"]
        )

        # ---- gate: gradient ownership + budget (same as the semantic smoke)
        if not check_trainable_whitelist(capture["trainable_names"]):
            raise P6BFailClosedError(
                f"{unit.key}: trainable whitelist drift "
                f"{set(capture['trainable_names']) ^ EXPECTED_TRAINABLE}"
            )
        if capture["actual_steps"] != FT_EPO:
            raise P6BFailClosedError(
                f"{unit.key}: step count {capture['actual_steps']} != {FT_EPO}"
            )
        if not capture["optimizer_ownership_ok"]:
            raise P6BFailClosedError(
                f"{unit.key}: optimizer ownership: "
                f"{capture['optimizer_ownership_reason']}"
            )
        if not capture["ce_loss_finite"]:
            raise P6BFailClosedError(f"{unit.key}: non-finite support loss")

        # ---- artifacts
        torch.save(capture["head_state_dict"], staging / HEAD_CHECKPOINT)
        torch.save(capture["support_proto"], staging / SUPPORT_PROTO)
        step_audit = {
            "schema_version": STEP_AUDIT_SCHEMA,
            "method": unit.method,
            "product": unit.product,
            "K": unit.k_shot,
            "seed": unit.seed,
            "expected_steps": FT_EPO,
            "actual_steps": capture["actual_steps"],
            "optimizer_spec": OPTIMIZER_SPEC,
            "scheduler_spec": SCHEDULER_SPEC,
            "zip_config_index": ZIP_CONFIG_INDEX,
            "trainable_parameter_names": capture["trainable_names"],
            "optimizer_ownership_ok": capture["optimizer_ownership_ok"],
            "optimizer_ownership_entries": capture["optimizer_ownership_entries"],
            "ce_loss_first": capture["ce_loss_first"],
            "ce_loss_last": capture["ce_loss_last"],
            "ce_loss_finite": capture["ce_loss_finite"],
        }
        atomic_write_json(staging / STEP_AUDIT, step_audit)

        run_config = {
            "schema_version": RUN_CONFIG_SCHEMA,
            "method": unit.method,
            "product": unit.product,
            "K": unit.k_shot,
            "seed": unit.seed,
            "class_order": list(class_order),
            "class_names": list(class_order),
            "input_shape": dataset.input_shape,
            "device": device,
            "ft_epo": FT_EPO,
            "zip_config_index": ZIP_CONFIG_INDEX,
            "alpha_ckpt_sha256": sha256_file(alpha_ckpt) if Path(alpha_ckpt).is_file() else None,
            "support_metadata": None,  # filled after meta written
        }
        meta_lines = []
        for sample in dataset.samples:
            meta_lines.append(json.dumps({
                "class_name": sample["class_name"],
                "label": sample["class_idx"],
                "rel_path": sample["rel_path"],
                "rgb_sha256": sample["rgb_sha256"],
            }, sort_keys=True, separators=(",", ":")))
        atomic_write_text(staging / META_JSONL, "\n".join(meta_lines) + "\n")
        run_config["support_metadata"] = {
            "records": len(meta_lines),
            "sha256": sha256_file(staging / META_JSONL),
        }
        atomic_write_json(staging / RUN_CONFIG, run_config)

        artifacts = []
        for name in (HEAD_CHECKPOINT, SUPPORT_PROTO, STEP_AUDIT, RUN_CONFIG, META_JSONL):
            artifacts.append({
                "relative_path": name,
                "size_bytes": (staging / name).stat().st_size,
                "sha256": sha256_file(staging / name),
            })
        artifact_manifest = {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA,
            "method": unit.method,
            "product": unit.product,
            "K": unit.k_shot,
            "seed": unit.seed,
            "artifacts": artifacts,
        }
        atomic_write_json(staging / ARTIFACT_MANIFEST, artifact_manifest)

        completion = {
            "schema_version": COMPLETION_SCHEMA,
            "status": "completed",
            "formal_eligible": bool(not execution.get("allow_dirty_diagnostic")),
            "formal_test_used": False,
            "query_pixels_read": False,
            "query_labels_read": False,
            "performance_metrics_generated": False,
            "method": unit.method,
            "product": unit.product,
            "K": unit.k_shot,
            "seed": unit.seed,
            "class_order": list(class_order),
            "execution": execution,
            "protocol_bundle": {
                "hash_mode": "canonical-lf",
                **compute_protocol_hashes(),
            },
            "manifest_sha256": manifest.get("manifest_sha256"),
            "model_provenance": {
                "backbone_family": "AlphaCLIP ViT-L/14",
                "alpha_ckpt_sha256": (
                    sha256_file(alpha_ckpt) if Path(alpha_ckpt).is_file() else None
                ),
                "classifier": "EchoClassfierF",
                "text_prior": "a photo of {label} defect on the image",
                "trainable_parameter_names": capture["trainable_names"],
                "text_features_shape": capture["text_features_shape"],
            },
            "resource_measurements": resource,
            "artifacts": artifacts + [{
                "relative_path": ARTIFACT_MANIFEST,
                "size_bytes": (staging / ARTIFACT_MANIFEST).stat().st_size,
                "sha256": sha256_file(staging / ARTIFACT_MANIFEST),
            }],
        }
        atomic_write_json(staging / COMPLETION, completion)

        # single atomic publish
        if output_dir.exists():
            raise P6BFailClosedError(f"{unit.key}: output dir appeared during run")
        os.replace(staging, output_dir)
        return completion
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True,
                        help="frozen Stage 1 bundle root (p6b_stage1_manifest_bundle)")
    parser.add_argument("--image-root", required=True,
                        help="mvtec-fs dataset root (data/mvtec-fs)")
    parser.add_argument("--alpha-ckpt", required=True,
                        help="AlphaCLIP ViT-L/14 GRIT-20M checkpoint")
    parser.add_argument("--product", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k8", action="store_true",
                        help="K8 extension (Protocol B section 4): load the "
                             "unit manifest from the K8 bundle "
                             "(p6b_k8_support_manifests/, schema "
                             "p6b_k8_support_manifest_v1) instead of the "
                             "frozen Stage 1 bundle")
    parser.add_argument("--allow-dirty-diagnostic", action="store_true",
                        help="downgrade to status=diagnostic, formal_eligible=false")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        # K8 extension: identical official support pipeline with the K8
        # manifest loaded from the K8 bundle (Protocol B section 4).
        if args.k8:
            from experiments.p6.mvrec_rgb_formal.k8_execution import (
                K8MethodUnit,
                load_k8_manifest,
            )

            k8_unit = K8MethodUnit("mvrec_rgb", args.product, args.seed)
            manifest = load_k8_manifest(Path(args.manifest_dir), k8_unit)
            manifest_path = Path(args.manifest_dir) / k8_unit.manifest_relative_path
            unit = MvrecRgbUnit(args.product, 8, args.seed)
        else:
            unit = MvrecRgbUnit(args.product, args.k, args.seed)
            manifest = load_frozen_manifest(Path(args.manifest_dir), unit)
            manifest_path = Path(args.manifest_dir) / unit.manifest_relative_path
        commit, dirty = git_commit_and_dirty(PROJECT_ROOT)
        if dirty and not args.allow_dirty_diagnostic:
            raise P6BFailClosedError(
                f"{unit.key}: formal execution requires a clean git tree")
        execution = {
            "commit": commit,
            "dirty": bool(dirty),
            "allow_dirty_diagnostic": bool(args.allow_dirty_diagnostic),
        }
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        completion = run_support_unit(
            unit=unit,
            manifest=manifest,
            manifest_path=manifest_path,
            image_root=Path(args.image_root),
            alpha_ckpt=Path(args.alpha_ckpt),
            device=device,
            output_dir=Path(args.output_dir),
            execution=execution,
        )
        print(f"[mvrec-rgb support] completed {unit.key} "
              f"formal_eligible={completion['formal_eligible']}", flush=True)
        return 0
    except (P6BFailClosedError, OSError) as exc:
        print(f"[mvrec-rgb support] FAIL-CLOSED: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
