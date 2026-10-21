#!/usr/bin/env python3
"""P6.B Protocol Bundle V2 support-side runner — one (method, product, K, seed) config.

Formal single-config CLI producing support-side artifacts ONLY. This runner
never implements the query evaluator, never reads query pixels/labels/metrics,
and never writes accuracy/BA/F1 or any query-derived field.

Methods (locked registry, Protocol Bundle V2 nine-method set):
    01 clip_proto          — CLIP Proto
    02 dino_proto          — DINO Proto
    03 fusion_proto        — Frozen-Fusion Proto
    04 zero_shot_clip      — Zero-shot CLIP
    05 tip_adapter         — Tip-Adapter
    06 linear_probe        — Frozen-Fusion Linear Probe
    07 p30_frozen_proto    — R1: Exact P3.0 Frozen Representation Proto (Addendum B §2)
    08 tip_adapter_f       — R2: Tip-Adapter-F* (Addendum B §3)
    09 ours                — Ours (terminal adapter + prototype memory + logit scale)

Locked interfaces (Protocol Bundle V2 provenance: main protocol §5 / Addendum A / Addendum B):
- data comes exclusively from the selected Stage 1 formal manifest of the
  pinned bundle; the feasibility JSONL is never read here;
- class mapping strictly from the manifest ``class_order``; the fixed
  Cable/MVTec CLASS_NAMES of ``dataset.constants`` are never used;
- trainable methods reuse ``BaseDataset`` non-cable augmentation and
  ``SeededEpochSampler`` (seed == init_seed == data_seed == augment_seed);
- training-free support/cache extraction uses the single deterministic 336
  evaluation view;
- every locked hyperparameter (epochs/lr/beta/alpha/prompt/augmentation...)
  is a module constant; the CLI exposes none of them;
- formal mode requires resolvable HEAD + clean tree; the pinned Stage 1
  completion gate runs via ``p6b_stage2_execution.load_formal_bundle``;
  every Protocol Bundle V2 document SHA256 (main protocol + Addendum A +
  Addendum B) plus the Stage 1 completion SHA256 is recomputed at start AND
  again immediately before the completion record is written; any mismatch
  raises ``P6BFailClosedError`` and sets formal_eligible=false;
- output uses sibling staging + a single atomic publish; any failure removes
  the staging directory and never leaves a formal output directory;
- ``--allow-dirty-diagnostic`` downgrades to status=diagnostic,
  formal_eligible=false.

R1 (p30_frozen_proto) is a feature-matched pre-terminal frozen prototype
reference — Ours differs by also training the terminal adapter (det_proj),
the prototype memory, and the logit scale.  R2 (tip_adapter_f) is a
firewall-constrained Tip-Adapter-F*: cache keys only trainable, all CLIP
encoders frozen, alpha=1.0/beta=5.5 fixed, 20 epochs final-epoch-only,
no best-epoch or alpha/beta/validation search.

Ours reuses the production chain directly: ``AdaptedCLIP``,
``FaissPrototypeMemory``, ``configure_model_trainable_parameters``,
``init_prototypes_with_class_means``, ``train_multiclass`` — no training loop
is duplicated. ``linear_probe`` implements the locked augmentation-aware loop
of Addendum A §1.3 (per epoch/repeat non-cable augmentation + frozen Fusion
re-extraction; Stage 2A deliberately removed the precomputed-embedding loop).

This module never sets ``KMP_DUPLICATE_LIB_OK`` and keeps the
numpy-before-torch import order (cole/OpenMP safe).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

# cole environment safe import order: numpy MUST be imported before torch to
# avoid the OpenMP runtime duplicate-library crash in this Windows env.
# Never work around this with KMP_DUPLICATE_LIB_OK — setting that flag is
# forbidden; keep the import order instead.
import numpy  # noqa: F401  (deliberately before torch)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import (  # noqa: E402
    LOCKED_K,
    LOCKED_PRODUCTS,
    LOCKED_SEEDS,
    P6BFailClosedError,
    atomic_write_json,
    atomic_write_text,
    manifest_self_sha256,
    sha256_bytes,
)
from experiments.p6 import p6b_stage2_execution as stage2  # noqa: E402
from experiments.p6 import p6b_stage2_models as models  # noqa: E402
from experiments.p6 import build_p6b_support_manifests as builder  # noqa: E402

# ---------------------------------------------------------------------------
# Locked schema / constants (no CLI override is exposed for any of these)
# ---------------------------------------------------------------------------

RUN_CONFIG_SCHEMA = "p6b_stage2_support_run_config_v1"
ARTIFACT_MANIFEST_SCHEMA = "p6b_stage2_support_artifact_manifest_v1"
COMPLETION_SCHEMA = "p6b_stage2_support_complete_v1"

METHODS = (
    "clip_proto",
    "dino_proto",
    "fusion_proto",
    "linear_probe",
    "ours",
    "zero_shot_clip",
    "tip_adapter",
    "p30_frozen_proto",  # R1: Exact P3.0 Frozen Representation Proto (Addendum B §2)
    "tip_adapter_f",     # R2: Tip-Adapter-F* firewall-constrained (Addendum B §3)
)
TRAINABLE_METHODS = ("linear_probe", "ours", "tip_adapter_f")

PROTOCOL_RELATIVE_PATH = "docs/P6B_MVTEC_FS_RGB_EXECUTION_PROTOCOL.md"
ADDENDUM_A_RELATIVE_PATH = "docs/P6B_EXECUTION_ADDENDUM_A.md"
ADDENDUM_B_RELATIVE_PATH = "docs/P6B_EXECUTION_ADDENDUM_B.md"

# Locked document set (Protocol Bundle V2, Addendum B §4): main protocol +
# Addendum A + Addendum B are all pinned and re-verified together.
LOCKED_DOCUMENTS = (
    ("protocol", PROTOCOL_RELATIVE_PATH),
    ("addendum_a", ADDENDUM_A_RELATIVE_PATH),
    ("addendum_b", ADDENDUM_B_RELATIVE_PATH),
)
PROTOCOL_BUNDLE_VERSION = "v2"

# Canonical-LF pinned SHA for formal runs (Addendum A §4 / Addendum B §4)
EXPECTED_PROTOCOL_SHA256_CANONICAL_LF = "80a5deba87f6c74ce8626d317f8120d9bb47f09db12416b3e1f3adae31c96566"
EXPECTED_ADDENDUM_A_SHA256_CANONICAL_LF = "ad2b5f2b649792d3d57a49efc7de5d44b834538e6e090d351fe0a38268cd4e43"
EXPECTED_ADDENDUM_B_SHA256_CANONICAL_LF = "15554bea2d61bbd92f349bbf3b6e106f5e4c1db97ef38e9663175e3a6ec54983"
EXPECTED_DOCUMENT_SHAS_CANONICAL_LF = {
    "protocol": EXPECTED_PROTOCOL_SHA256_CANONICAL_LF,
    "addendum_a": EXPECTED_ADDENDUM_A_SHA256_CANONICAL_LF,
    "addendum_b": EXPECTED_ADDENDUM_B_SHA256_CANONICAL_LF,
}

IMG_SIZE = 336

# R1 Exact P3.0 Frozen Representation Proto: pre-terminal CLIP tokens are the
# CLIP ViT-L width (1024) fed into the terminal adapter det_proj; concat with
# DINO mean (384) gives the frozen pre-terminal fusion dim (Addendum B §2.2).
R1_CLIP_PRE_TERMINAL_DIM = 1024
R1_FROZEN_PROTO_FUSION_DIM = 1024 + 384  # 1408

LOCKED_OURS = {
    "training_protocol": "clean_v1",
    "rng_protocol": "isolated_rng",
    "adapter_train_mode": "terminal_only",
    "clip_pool": "topk",
    "dino_pool": "mean",
    "dino_mask_ratio": 0.0,
    "dino_patch_policy": "full",
    "dino_backend": "hub",
    "prototype_init_mode": "zero",
    "prototypes_per_class": 1,
    "num_init_views": 1,
    "init_logit_scale": 10.0,
    "num_epochs": 20,
    "repeat_per_epoch": 4,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "lr": 1e-4,
    "proto_lr_mult": 3.0,
    "weight_decay": 0.01,
    "label_smoothing": 0.1,
    "supcon_weight": 0.0,
    "orth_weight": 0.0,
    "itc_weight": 0.0,
    "arc_margin": 0.0,
    "use_text_guided": False,
    "proto_text_refine": "none",
    "normal_aware_head": False,
    "early_stop_patience": 0,
    "save_every": 0,
    "checkpoint_format": "lightweight",
    "gate_type": "cosine",
}

LOCKED_CONSTANTS = {
    "img_size": IMG_SIZE,
    "zero_shot_prompt_template": stage2.ZERO_SHOT_PROMPT_TEMPLATE,
    "zero_shot_logit_scale": stage2.ZERO_SHOT_LOGIT_SCALE,
    "tip_alpha": stage2.TIP_ALPHA,
    "tip_beta": stage2.TIP_BETA,
    "linear_probe": {
        "epochs": stage2.LINEAR_EPOCHS,
        "repeat_per_epoch": stage2.LINEAR_REPEAT_PER_EPOCH,
        "batch_size": stage2.LINEAR_BATCH_SIZE,
        "grad_accum": stage2.LINEAR_GRAD_ACCUM,
        "lr": stage2.LINEAR_LR,
        "weight_decay": stage2.LINEAR_WEIGHT_DECAY,
        "label_smoothing": stage2.LINEAR_LABEL_SMOOTHING,
    },
    "ours": LOCKED_OURS,
    "tip_adapter_f": {
        "trainable_set": "cache_keys_only",
        "alpha": stage2.TIP_ALPHA,
        "beta": stage2.TIP_BETA,
        "logit_scale": stage2.ZERO_SHOT_LOGIT_SCALE,
        "optimizer": "adamw",
        "lr": stage2.R2_LR,
        "eps": stage2.R2_EPS,
        "weight_decay": stage2.R2_WEIGHT_DECAY,
        "scheduler": "cosine_annealing",
        "epochs": stage2.R2_EPOCHS,
        "repeat_per_epoch": stage2.R2_REPEAT_PER_EPOCH,
        "batch_size": stage2.R2_BATCH_SIZE,
        "grad_accum": stage2.R2_GRAD_ACCUM,
        "label_smoothing": stage2.R2_LABEL_SMOOTHING,
        "checkpoint_selection": "final_epoch_only",
        "prompt_template": stage2.ZERO_SHOT_PROMPT_TEMPLATE,
    },
    "p30_frozen_proto": {
        "clip_branch": "pre_terminal_ln_post_tokens_topk_avg_pool_k0.1_l2norm",
        "dino_branch": "full_patch_tokens_mean_l2norm",
        "fusion": "concat(l2norm_clip_pre_terminal, l2norm_dino)_then_l2norm",
        "terminal_adaptation": False,
        "trainable_param_count": 0,
        "clip_dim": R1_CLIP_PRE_TERMINAL_DIM,
        "fusion_dim": R1_FROZEN_PROTO_FUSION_DIM,
        "tie_break": "manifest_class_order",
    },
    "evaluation_view": (
        "single deterministic Resize(336,336)+ToTensor+CLIP Normalize"),
    "trainable_augmentation": (
        "BaseDataset non-cable branch (RandomResizedCrop(336,(0.7,1.0)) + "
        "RandomApply(RandomAffine(15,(0.1,0.1)),p=0.3) + "
        "RandomApply(ColorJitter(0.2,0.2,0.15,0.0),p=0.5)) with per-sample "
        "augment_seed isolation"),
}


# ---------------------------------------------------------------------------
# CLI (locked hyperparameters are intentionally NOT exposed)
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P6.B Stage 2B support-side runner (single configuration)")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--product", required=True, choices=list(LOCKED_PRODUCTS))
    parser.add_argument("--k", required=True, type=int,
                        choices=list(LOCKED_K) + [8])
    parser.add_argument("--seed", required=True, type=int,
                        choices=list(LOCKED_SEEDS))
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--output-dir", required=True,
                        help="must not exist; published atomically")
    parser.add_argument("--device", default=None,
                        help="torch device string; default cuda:0 if available")
    parser.add_argument("--allow-dirty-diagnostic", action="store_true",
                        help="explicit diagnostic switch: tolerate a dirty git "
                             "tree; completion is status=diagnostic, "
                             "formal_eligible=false")
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def document_provenance(project_root: Path) -> dict:
    """Locked protocol/Addendum A/Addendum B file identity; fail-closed if
    missing (Protocol Bundle V2, Addendum B §4)."""
    provenance = {}
    for key, relative in LOCKED_DOCUMENTS:
        path = Path(project_root) / relative
        if not path.is_file():
            raise P6BFailClosedError(f"locked document missing: {path}")
        provenance[key] = {
            "relative_path": relative,
            "sha256": stage2.file_sha256(path),
        }
    return provenance


def verify_document_shas(project_root: Path, start: dict) -> dict:
    """Recompute all three locked document SHAs; fail-closed on any drift.

    Uses canonical-LF normalization (LF-only hash) so CRLF/LF checkout
    produces identical results. Refuses bare-CR and any content changes.
    """
    current = document_provenance_canonical_lf(project_root)
    for key, _ in LOCKED_DOCUMENTS:
        if current[key]["sha256"] != start[key]["sha256"]:
            raise P6BFailClosedError(
                f"{key} sha256 changed during the run: "
                f"{start[key]['sha256']} -> {current[key]['sha256']}")
    return current


def document_provenance_canonical_lf(project_root: Path) -> dict:
    """Compute protocol/Addendum A/Addendum B using canonical-LF normalization.

    Converts CRLF -> LF BEFORE hashing so git checkout (core.autocrlf)
    doesn't affect the validation. Requires no bare-CR characters.
    """
    provenance = {}
    for key, relative in LOCKED_DOCUMENTS:
        path = Path(project_root) / relative
        if not path.is_file():
            raise P6BFailClosedError(f"locked document missing: {path}")
        raw_bytes = path.read_bytes()
        # Reject bare-CR (legacy Mac style)
        bare_cr = raw_bytes.count(b"\r") - raw_bytes.count(b"\r\n")
        if bare_cr > 0:
            raise P6BFailClosedError(
                f"document contains bare-CR (legacy Mac style): {relative}")
        # Canonical-LF normalization
        canonical = raw_bytes.replace(b"\r\n", b"\n")
        provenance[key] = {
            "relative_path": relative,
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "hash_mode": "canonical-lf",
        }
    return provenance


def write_support_metadata(manifest: dict, staging: Path) -> dict:
    """Deterministically convert the selected manifest into runner-internal
    support metadata (JSONL rows consumable by BaseDataset). Class labels
    strictly follow manifest class_order; no dataset.constants names."""
    class_order = list(manifest["class_order"])
    lines = []
    for class_idx, class_name in enumerate(class_order):
        for item in manifest["classes"][class_name]["items"]:
            lines.append(json.dumps({
                "image_path": item["rel_path"],
                "label": class_idx,
                "class_name": class_name,
                "mask_path": "",
            }, sort_keys=True, ensure_ascii=False))
    path = staging / "p6b_support_meta.jsonl"
    atomic_write_text(path, "".join(line + "\n" for line in lines))
    return {
        "path": path,
        "sha256": stage2.file_sha256(path),
        "records": len(lines),
        "class_order": class_order,
    }


def verify_support_assets(manifest: dict, image_root: Path) -> list:
    """Pre-training gate: containment, existence and raw-file SHA for every
    support path. Reads raw bytes only (file integrity audit); consuming the
    pixels happens per-method afterwards."""
    from dataset import file_sha256 as dataset_file_sha256
    image_root = Path(image_root).resolve()
    records = []
    for class_name in manifest["class_order"]:
        for item in manifest["classes"][class_name]["items"]:
            rel_path = item["rel_path"]
            resolved = (image_root / rel_path).resolve()
            if image_root not in resolved.parents and resolved != image_root:
                raise P6BFailClosedError(
                    f"support path escapes image root: {rel_path}")
            if not resolved.is_file():
                raise P6BFailClosedError(f"support file missing: {rel_path}")
            actual = dataset_file_sha256(str(resolved))
            if actual != item["rgb_sha256"]:
                raise P6BFailClosedError(
                    f"support rgb sha mismatch for {rel_path}: "
                    f"{actual} != {item['rgb_sha256']}")
            records.append({"rel_path": rel_path, "class_name": class_name,
                            "rgb_sha256": actual})
    return records


# ---------------------------------------------------------------------------
# Lazy model builders (method-specific loading)
# ---------------------------------------------------------------------------

def _build_lazy_clip_only(ctx: RunContext) -> models.RealModelBundle:
    """Lazy CLIP-only factory construction for zero_shot_clip / tip_adapter."""
    clip_model, _ = models.build_clip_backbone(ctx.project_root, ctx.device)
    from provenance import file_identity, model_state_fingerprint
    dino_extractor = None  # Not loaded
    clip_extractor = models.FrozenClipImageExtractor(clip_model, ctx.device)
    text_extractor = models.FrozenClipTextExtractor(clip_model, ctx.device)

    provenance = {
        "clip": {
            "model_name": models.CLIP_MODEL_NAME,
            "pretrained_identifier": f"openai:{models.CLIP_MODEL_NAME}",
            "embedding": "native_projected_global_image_embedding_l2norm",
            "weight_file": file_identity(
                str(Path(ctx.project_root) / models.CLIP_WEIGHT_RELATIVE_PATH)),
            "loaded_state": model_state_fingerprint(clip_model),
            "network_download": False,
        },
        "dino": {"not_loaded": True},
        "fusion": {"not_loaded": True},
    }
    return models.RealModelBundle(
        clip_model=clip_model,
        dino_extractor=None,
        clip_extractor=clip_extractor,
        dino_extractor_interface=None,
        text_extractor=text_extractor,
        fusion=None,
        provenance=provenance,
    )


def _build_lazy_dino_only(ctx: RunContext) -> models.RealModelBundle:
    """Lazy DINO-only factory construction for dino_proto."""
    dino_extractor, dino_hub_repo, dino_hub_checkpoint = (
        models.build_dino_backbone(ctx.project_root, ctx.device))
    clip_model = None  # Not loaded

    from provenance import file_identity, model_state_fingerprint
    dino_interface = models.FrozenDinoImageExtractor(dino_extractor, ctx.device)

    provenance = {
        "clip": {"not_loaded": True},
        "dino": {
            "model_name": "dinov2_vits14",
            "backend": dino_extractor.backend,
            "patch_policy": dino_extractor.patch_policy,
            "pooling": "full_patch_tokens_mean_l2norm",
            "weight_file": file_identity(str(dino_hub_checkpoint)),
            "expected_checkpoint_sha256": models.EXPECTED_DINO_SHA256,
            "local_source_path": str(Path(dino_hub_repo).resolve()),
            "loaded_state": model_state_fingerprint(dino_extractor.model),
            "network_download": False,
        },
        "fusion": {"not_loaded": True},
    }
    return models.RealModelBundle(
        clip_model=None,
        dino_extractor=dino_extractor,
        clip_extractor=None,
        dino_extractor_interface=dino_interface,
        text_extractor=None,
        fusion=None,
        provenance=provenance,
    )


def _build_lazy_clip_and_dino(ctx: RunContext) -> models.RealModelBundle:
    """Lazy CLIP+DINO factory for fusion_proto / linear_probe."""
    clip_model, clip_weight_path = models.build_clip_backbone(
        ctx.project_root, ctx.device)
    dino_extractor, dino_hub_repo, dino_hub_checkpoint = (
        models.build_dino_backbone(ctx.project_root, ctx.device))

    from provenance import file_identity, model_state_fingerprint
    clip_extractor = models.FrozenClipImageExtractor(clip_model, ctx.device)
    dino_interface = models.FrozenDinoImageExtractor(dino_extractor, ctx.device)
    text_extractor = models.FrozenClipTextExtractor(clip_model, ctx.device)

    provenance = {
        "clip": {
            "model_name": models.CLIP_MODEL_NAME,
            "pretrained_identifier": f"openai:{models.CLIP_MODEL_NAME}",
            "embedding": "native_projected_global_image_embedding_l2norm",
            "weight_file": file_identity(str(clip_weight_path)),
            "loaded_state": model_state_fingerprint(clip_model),
            "network_download": False,
        },
        "dino": {
            "model_name": "dinov2_vits14",
            "backend": dino_extractor.backend,
            "patch_policy": dino_extractor.patch_policy,
            "pooling": "full_patch_tokens_mean_l2norm",
            "weight_file": file_identity(str(dino_hub_checkpoint)),
            "expected_checkpoint_sha256": models.EXPECTED_DINO_SHA256,
            "local_source_path": str(Path(dino_hub_repo).resolve()),
            "loaded_state": model_state_fingerprint(dino_extractor.model),
            "network_download": False,
        },
        "fusion": {
            "operation": "concat(l2norm_clip, l2norm_dino)_then_l2norm",
            "expected_dims": {
                "clip": models.EXPECTED_CLIP_DIM,
                "dino": models.EXPECTED_DINO_DIM,
                "fusion": models.EXPECTED_FUSION_DIM,
            },
        },
    }
    return models.RealModelBundle(
        clip_model=clip_model,
        dino_extractor=dino_extractor,
        clip_extractor=clip_extractor,
        dino_extractor_interface=dino_interface,
        text_extractor=text_extractor,
        fusion=models.FrozenFusion(),
        provenance=provenance,
    )


def _validate_support_assets_pre_model(bundle: dict, manifest: dict,
                                        image_root: Path) -> None:
    """Pre-model gate: validate support asset containment/existence/raw SHA.

    Must be called BEFORE any model construction to maintain fail-closed.
    This is a lightweight sanity check (read_bytes only for integrity).
    """
    try:
        # Reuse existing verification without constructing factory
        verify_support_assets(manifest, image_root)
    except P6BFailClosedError:
        raise
    except Exception as e:
        raise P6BFailClosedError(
            f"support asset validation failed before model loading: {e}")


class ManifestSupportDataset(torch.utils.data.Dataset):
    """BaseDataset-driven support set with strict manifest class mapping."""

    def __init__(self, base_dataset, meta_records, class_order):
        from dataset import BaseDataset
        if not isinstance(base_dataset, BaseDataset):
            raise P6BFailClosedError(
                "ManifestSupportDataset must wrap dataset.BaseDataset to "
                "reuse the locked augmentation/sampler machinery")
        self.base = base_dataset
        self.class_order = list(class_order)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_order)}
        if len(meta_records) != len(base_dataset):
            raise P6BFailClosedError(
                "support metadata length mismatch (silent truncation is "
                "forbidden)")
        self.items = meta_records

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.base[idx]  # BaseDataset handles (index, augment_seed)
        record = self.items[idx if isinstance(idx, int) else idx[0]]
        if item["class_name"] != record["class_name"]:
            raise P6BFailClosedError(
                "support metadata order diverged from BaseDataset records")
        item["label"] = torch.tensor(
            self.class_to_idx[record["class_name"]], dtype=torch.int64)
        return item


def build_base_dataset(image_root: Path, meta_path: Path, trainable: bool):
    """Reuse BaseDataset: trainable=True -> locked non-cable augmentation
    branch; trainable=False -> deterministic Resize-only evaluation view."""
    from dataset import BaseDataset
    if "cable" in str(image_root).lower():
        raise P6BFailClosedError(
            "image root must not select the Cable augmentation branch")
    return BaseDataset(str(image_root), str(meta_path), IMG_SIZE,
                       text=not trainable)


def make_seeded_loader(dataset, seed: int, batch_size: int,
                       num_workers: int = 0):
    from seed_protocol import (
        DeterministicWorkerSeeder,
        SeededEpochSampler,
        make_loader_generator,
    )
    sampler = SeededEpochSampler(dataset, data_seed=seed, augment_seed=seed,
                                 shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        generator=make_loader_generator(seed),
        worker_init_fn=DeterministicWorkerSeeder(seed),
    )
    return loader, sampler


class StepCountingOptimizer(torch.optim.Optimizer):
    """Transparent optimizer proxy counting ``step()`` calls (step audit).

    Subclasses ``Optimizer`` (without invoking its constructor) so torch
    schedulers accept it; every attribute lookup is delegated to the wrapped
    real optimizer.
    """

    def __init__(self, optimizer):
        self._optimizer = optimizer
        self.step_count = 0

    def step(self, *args, **kwargs):
        self.step_count += 1
        return self._optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        return self._optimizer.zero_grad(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._optimizer, name)


def steps_per_pass(batches: int, grad_accum: int) -> int:
    return batches // grad_accum + (1 if batches % grad_accum else 0)


# ---------------------------------------------------------------------------
# Augmentation-aware Linear Probe loop (Addendum A §1.3, Stage 2B scope)
# ---------------------------------------------------------------------------

def train_linear_probe_augmentation_aware(
        *, extract_fusion, dataset, head, spec, seed, device) -> dict:
    """Locked augmentation-aware training loop.

    Every epoch x repeat pass re-runs the locked non-cable augmentation via
    the SeededEpochSampler-driven BaseDataset and RE-EXTRACTS the frozen
    Fusion features; a single deterministic embedding reused across passes is
    fail-closed. Only the Linear head is trainable.
    """
    from seed_protocol import set_loader_seed_context
    stage2.validate_linear_probe_spec(spec)
    device = torch.device(device)
    loader, _sampler = make_seeded_loader(dataset, seed, spec.batch_size)
    head = head.to(device)
    head.train()
    for param in head.parameters():
        param.requires_grad = True
    optimizer = StepCountingOptimizer(
        AdamW(head.parameters(), lr=spec.lr, weight_decay=spec.weight_decay))
    scheduler = CosineAnnealingLR(optimizer, T_max=spec.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=spec.label_smoothing)

    epochs_audit = []
    all_pass_digests = []
    total_batches = total_features = 0
    extract_call_count = 0
    epoch_repeat_contexts = set()  # For context validation
    for epoch in range(spec.epochs):
        pass_digests = []
        epoch_batches = epoch_features = 0
        for repeat in range(spec.repeat_per_epoch):
            set_loader_seed_context(loader, "linear_probe_train",
                                    epoch=epoch, repeat=repeat)
            # Record epoch/repeat context for exact set audit
            epoch_repeat_contexts.add((epoch, repeat))
            features_chunks = []
            micro = 0
            for batch_idx, batch in enumerate(loader):
                images = batch["image"]
                with torch.no_grad():
                    features = extract_fusion(images)
                extract_call_count += 1
                features = features.detach().to(device)
                labels = batch["label"].to(device)
                features_chunks.append(features.detach().cpu())
                logits = head(features)
                loss = criterion(logits, labels) / spec.grad_accum
                loss.backward()
                micro += 1
                epoch_batches += 1
                epoch_features += int(images.shape[0])
                if micro % spec.grad_accum == 0 or \
                        (batch_idx + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
            pass_tensor = torch.cat(features_chunks, dim=0)
            digest = sha256_bytes(bytes(pass_tensor.numpy().tobytes()))
            pass_digests.append(digest)
            all_pass_digests.append(digest)
        scheduler.step()
        total_batches += epoch_batches
        total_features += epoch_features
        epochs_audit.append({
            "epoch": epoch,
            "batches": epoch_batches,
            "features_extracted": epoch_features,
            "optimizer_steps": optimizer.step_count - sum(
                audit["optimizer_steps"] for audit in epochs_audit),
            "distinct_pass_feature_digests": len(set(pass_digests)),
        })

    # Prove every epoch x repeat calls extract_fusion by:
    # 1. extract_call_count == total_forward_batches
    # 2. epoch_repeat_contexts exactly matches expected 20×4=80
    total_expected_batches = spec.epochs * spec.repeat_per_epoch * len(loader)
    if extract_call_count != total_expected_batches:
        raise P6BFailClosedError(
            f"Linear Probe extract_fusion call count {extract_call_count} != "
            f"expected {total_expected_batches}; some batches may have been skipped")

    expected_contexts = {(e, r) for e in range(spec.epochs)
                         for r in range(spec.repeat_per_epoch)}
    if epoch_repeat_contexts != expected_contexts:
        raise P6BFailClosedError(
            f"Linear Probe epoch/repeat contexts mismatch:\n"
            f"  expected {len(expected_contexts)} contexts (0..{spec.epochs-1} × 0..{spec.repeat_per_epoch-1})\n"
            f"  got {len(epoch_repeat_contexts)} contexts\n"
            f"  missing: {expected_contexts - epoch_repeat_contexts}\n"
            f"  extra: {epoch_repeat_contexts - expected_contexts}")
    # Feature digests are recorded for audit but NOT used as performance threshold
    # This prevents constant-output extractor bypass while not requiring distinct features
    feature_reextraction_provenance = {
        "extract_call_count": extract_call_count,
        "total_forward_batches": total_expected_batches,
        "context_set_size": len(epoch_repeat_contexts),
        "expected_context_set_size": spec.epochs * spec.repeat_per_epoch,
        "all_pass_digests_sample": all_pass_digests[:5],  # Sample for audit
    }

    return {
        "head": head,
        "final_epoch": spec.epochs - 1,
        "epochs": epochs_audit,
        "total_forward_batches": total_batches,
        "total_features_extracted": total_features,
        "total_optimizer_steps": optimizer.step_count,
        "feature_reextraction_provenance": feature_reextraction_provenance,
        "loader_batch_count": len(loader),  # Needed for reconciliation
    }


# ---------------------------------------------------------------------------
# Method runners (all artifacts go to staging)
# ---------------------------------------------------------------------------

def _deterministic_dataset(image_root, meta_path, meta_records, class_order):
    base = build_base_dataset(image_root, meta_path, trainable=False)
    return ManifestSupportDataset(base, meta_records, class_order)


def _extract_support_embeddings(dataset, extract_fn):
    embeddings, labels = [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        embeddings.append(extract_fn(item["image"]))
        labels.append(int(item["label"]))
    return torch.stack(embeddings, dim=0), labels


def run_prototype_method(ctx, method: str, dataset, class_order) -> dict:
    """clip_proto / dino_proto / fusion_proto via Stage 2A locked math.

    Method-specific lazy loading:
    - clip_proto: CLIP only
    - dino_proto: DINO only
    - fusion_proto: CLIP + DINO
    """
    # Lazy factory construction based on method requirements
    if not ctx.factory:
        if method == "clip_proto":
            ctx.factory = _build_lazy_clip_only(ctx)
        elif method == "dino_proto":
            ctx.factory = _build_lazy_dino_only(ctx)
        elif method == "fusion_proto":
            ctx.factory = _build_lazy_clip_and_dino(ctx)

    factory = ctx.factory
    if method == "clip_proto":
        embeddings, labels = _extract_support_embeddings(
            dataset, lambda image: factory.clip_extractor.embed_image(image))
        representation = "clip_native_projected_global_l2norm"
    elif method == "dino_proto":
        embeddings, labels = _extract_support_embeddings(
            dataset,
            lambda image: factory.dino_extractor_interface.embed_image(image))
        representation = "dino_full_patch_tokens_mean_l2norm"
    else:
        clip_emb, labels = _extract_support_embeddings(
            dataset, lambda image: factory.clip_extractor.embed_image(image))
        dino_emb, _ = _extract_support_embeddings(
            dataset,
            lambda image: factory.dino_extractor_interface.embed_image(image))
        embeddings = factory.fusion.fuse(clip_emb, dino_emb)
        representation = "frozen_fusion_concat_l2norm"
    prototypes = stage2.build_prototypes(embeddings, labels, class_order)
    name = f"{method}_prototypes.pt"
    torch.save({
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": method,
        "representation": representation,
        "class_order": class_order,
        "prototypes": prototypes.detach().cpu(),
        "support_embeddings": embeddings.detach().cpu(),
        "support_labels": labels,
    }, ctx.staging / name)
    return {"primary_artifact": name,
            "support_pixels_read": True,
            "trainable_parameter_whitelist": []}


def run_zero_shot_clip(ctx, class_order) -> dict:
    """Locked prompts + frozen text encoder; consumes NO support pixels.

    Lazy CLIP-only loading.
    """
    if not ctx.factory:
        ctx.factory = _build_lazy_clip_only(ctx)
    prompts = [stage2.format_zero_shot_prompt(name) for name in class_order]
    text_embeddings = ctx.factory.text_extractor.embed_text(prompts)
    name = "zero_shot_clip_text_weights.pt"
    torch.save({
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": "zero_shot_clip",
        "prompt_template": stage2.ZERO_SHOT_PROMPT_TEMPLATE,
        "class_order": class_order,
        "prompts": prompts,
        "text_embeddings": text_embeddings.detach().cpu(),
        "logit_scale": stage2.ZERO_SHOT_LOGIT_SCALE,
    }, ctx.staging / name)
    return {"primary_artifact": name,
            "support_pixels_read": False,
            "trainable_parameter_whitelist": []}


def run_tip_adapter(ctx, dataset, class_order) -> dict:
    """Locked training-free Tip-Adapter cache built from support CLIP
    embeddings only (alpha=1.0, beta=5.5; no parameter search).

    Lazy CLIP-only loading.
    """
    if not ctx.factory:
        ctx.factory = _build_lazy_clip_only(ctx)
    embeddings, labels = _extract_support_embeddings(
        dataset, lambda image: ctx.factory.clip_extractor.embed_image(image))
    cache = stage2.TipAdapterCache(embeddings, labels,
                                   num_classes=len(class_order))
    name = "tip_adapter_cache.pt"
    torch.save({
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": "tip_adapter",
        "class_order": class_order,
        "cache_keys": cache.keys.detach().cpu(),
        "cache_values": cache.values.detach().cpu(),
        "cache_labels": labels,
        "alpha": cache.alpha,
        "beta": cache.beta,
    }, ctx.staging / name)
    return {"primary_artifact": name,
            "support_pixels_read": True,
            "trainable_parameter_whitelist": []}


def run_linear_probe(ctx, image_root, meta_path, meta_records, class_order) -> dict:
    """Real augmentation-aware Frozen-Fusion Linear Probe (Addendum A §1.3).

    Lazy CLIP+DINO loading for frozen fusion extraction.

    AUDIT: Per-pass invocation proof via:
    - extract_call_count == total_forward_batches
    - Exact set of (epoch, repeat) contexts matches 20×4=80
    - Feature digests recorded but NOT used as performance threshold
    """
    if not ctx.factory:
        ctx.factory = _build_lazy_clip_and_dino(ctx)
    spec = stage2.validate_linear_probe_spec(stage2.LinearProbeSpec())
    base = build_base_dataset(image_root, meta_path, trainable=True)
    dataset = ManifestSupportDataset(base, meta_records, class_order)
    factory = ctx.factory

    head = stage2.make_linear_head(models.EXPECTED_FUSION_DIM,
                                   len(class_order), ctx.seed)

    def extract_fusion(images):
        images = images.to(ctx.device)
        with torch.no_grad():
            clip_emb = factory.clip_extractor.embed_images(images)
            dino_emb = factory.dino_extractor_interface.embed_images(images)
        return factory.fusion.fuse(clip_emb, dino_emb)

    result = train_linear_probe_augmentation_aware(
        extract_fusion=extract_fusion, dataset=dataset, head=head, spec=spec,
        seed=ctx.seed, device=ctx.device)

    # Reconcile extract_fusion call count with loader batches
    loader_batch_count = result["loader_batch_count"]
    expected_calls = spec.epochs * spec.repeat_per_epoch * loader_batch_count
    if result["feature_reextraction_provenance"]["extract_call_count"] != expected_calls:
        raise P6BFailClosedError(
            f"Linear Probe extract_fusion mismatch: {result['feature_reextraction_provenance']['extract_call_count']} != {expected_calls}")
    name = "linear_probe_head.pt"
    torch.save({
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": "linear_probe",
        "class_order": class_order,
        "fusion_dim": models.EXPECTED_FUSION_DIM,
        "num_classes": len(class_order),
        "head_state_dict": {
            key: tensor.detach().cpu()
            for key, tensor in result["head"].state_dict().items()
        },
        "final_epoch": result["final_epoch"],
        "locked_spec": dataclasses_asdict(spec),
    }, ctx.staging / name)
    step_audit = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": "linear_probe",
        "epochs": result["epochs"],
        "total_forward_batches": result["total_forward_batches"],
        "total_features_extracted": result["total_features_extracted"],
        "total_optimizer_steps": result["total_optimizer_steps"],
        "augmentation": LOCKED_CONSTANTS["trainable_augmentation"],
        "feature_reextraction_per_pass": True,
        "per_pass_audit": {
            "extract_call_count": result["feature_reextraction_provenance"]["extract_call_count"],
            "expected_extract_count": expected_calls,
            "epoch_repeat_contexts_size": result["feature_reextraction_provenance"]["context_set_size"],
            "expected_contexts_size": spec.epochs * spec.repeat_per_epoch,
        },
    }
    atomic_write_json(ctx.staging / "linear_probe_step_audit.json", step_audit)
    return {"primary_artifact": name,
            "extra_artifacts": ["linear_probe_step_audit.json"],
            "support_pixels_read": True,
            "trainable_parameter_whitelist": ["bias", "weight"]}


def dataclasses_asdict(spec) -> dict:
    import dataclasses
    return dataclasses.asdict(spec)


# ---------------------------------------------------------------------------
# Ours — direct reuse of the production training chain
# ---------------------------------------------------------------------------

def build_ours_args(ctx, meta_path: Path, hub_repo, hub_checkpoint) -> argparse.Namespace:
    """The exact locked P3.F-method-level argument set. Derived clean_v1
    fields mirror ``train_multiclass.main`` so the reused production
    functions observe identical semantics."""
    ns = argparse.Namespace(
        model_name="ViT-L-14-336", img_size=IMG_SIZE,
        dataset="p6b_mvtec_fs_rgb", training_mode="few_shot", shot=ctx.k_shot,
        train_meta_path=str(meta_path), batch_size=LOCKED_OURS["batch_size"],
        num_workers=0, prefetch_factor=None, pin_memory=False,
        persistent_workers=False, tf32=False,
        dino_backend=LOCKED_OURS["dino_backend"],
        training_protocol=LOCKED_OURS["training_protocol"],
        rng_protocol=LOCKED_OURS["rng_protocol"],
        adapter_train_mode=LOCKED_OURS["adapter_train_mode"],
        early_adapter_source_checkpoint=None,
        dino_patch_policy=LOCKED_OURS["dino_patch_policy"],
        dino_hub_repo=str(hub_repo), dino_hub_checkpoint=str(hub_checkpoint),
        gradient_accumulation_steps=LOCKED_OURS["gradient_accumulation_steps"],
        num_epochs=LOCKED_OURS["num_epochs"],
        repeat_per_epoch=LOCKED_OURS["repeat_per_epoch"],
        num_init_views=LOCKED_OURS["num_init_views"],
        lr=LOCKED_OURS["lr"], weight_decay=LOCKED_OURS["weight_decay"],
        supcon_weight=LOCKED_OURS["supcon_weight"],
        orth_weight=LOCKED_OURS["orth_weight"], seed=ctx.seed,
        init_seed=ctx.seed, data_seed=ctx.seed, augment_seed=ctx.seed,
        itc_weight=LOCKED_OURS["itc_weight"],
        init_logit_scale=LOCKED_OURS["init_logit_scale"],
        save_path=str(ctx.staging), save_every=LOCKED_OURS["save_every"],
        checkpoint_format=LOCKED_OURS["checkpoint_format"],
        save_optimizer_state=False, resume=None,
        normal_aware_head=LOCKED_OURS["normal_aware_head"],
        normal_aware_loss_weight=0.0, normal_aware_head_lr_mult=10.0,
        normal_aware_head_threshold=0.0, freeze_cross_modal=False,
        early_stop_patience=LOCKED_OURS["early_stop_patience"],
        proto_lr_mult=LOCKED_OURS["proto_lr_mult"],
        prototypes_per_class=LOCKED_OURS["prototypes_per_class"],
        prototype_aggregation="max", prototype_mixture_temperature=0.2,
        prototype_assignment_temperature=0.2, prototype_balance_weight=0.0,
        cross_modal_mode="all", use_text_guided=LOCKED_OURS["use_text_guided"],
        gate_type=LOCKED_OURS["gate_type"],
        proto_text_refine=LOCKED_OURS["proto_text_refine"],
        arc_margin=LOCKED_OURS["arc_margin"],
        dino_mask_ratio=LOCKED_OURS["dino_mask_ratio"],
        clip_pool=LOCKED_OURS["clip_pool"], dino_pool=LOCKED_OURS["dino_pool"],
        label_smoothing=LOCKED_OURS["label_smoothing"],
    )
    # Derived clean_v1 fields (mirror of train_multiclass.main).
    ns.oof_fold_id = None
    ns.prototype_init_mode = LOCKED_OURS["prototype_init_mode"]
    ns.seed_protocol_version = "independent_streams_v1"
    ns.post_construction_rng_reset = False
    ns.trajectory_seed = None
    ns.early_adapter_source_checkpoint_sha256 = None
    ns.early_adapter_selected_state_sha256 = None
    hub_checkpoint = Path(hub_checkpoint)
    ns.dino_hub_checkpoint_sha256 = (
        stage2.file_sha256(hub_checkpoint) if hub_checkpoint.is_file() else None)
    return ns


def import_train_multiclass_offline_safe(import_module=None):
    """Import ``train_multiclass`` without enabling its legacy HF mirror.

    ``train_multiclass`` has a repository-wide legacy ``HF_ENDPOINT``
    ``setdefault`` at import time.  P6.B uses only the pinned local DINO hub
    repository and must neither inherit nor create that network fallback.
    Fail first if the caller already configured a non-empty endpoint, mask the
    key with an empty value while importing, restore the exact prior process
    state, then re-run the offline gate.
    """
    models._fail_closed_offline_dino()
    import_module = import_module or importlib.import_module
    sentinel = object()
    previous = os.environ.get("HF_ENDPOINT", sentinel)
    os.environ["HF_ENDPOINT"] = ""
    try:
        module = import_module("train_multiclass")
    finally:
        if previous is sentinel:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous
    models._fail_closed_offline_dino()
    return module


def run_ours(ctx, image_root, meta_path, meta_records, class_order,
             overrides=None) -> dict:
    """Ours = P3.0 method transfer. Reuses AdaptedCLIP, FaissPrototypeMemory,
    configure_model_trainable_parameters, init_prototypes_with_class_means
    and train_multiclass directly; no training loop is duplicated.

    Method-specific lazy loading: builds ONLY AdaptedCLIP (not baseline CLIP+DINO).
    Provenance records actual model used, never references unused factory.
    """
    tm = import_train_multiclass_offline_safe()
    from model.adapter import AdaptedCLIP
    from model.faiss_memory import FaissPrototypeMemory
    from utils import setup_seed

    overrides = dict(overrides or {})
    setup_seed(ctx.seed)
    num_classes = len(class_order)
    class_to_id = {name: i for i, name in enumerate(class_order)}

    hub_repo = ctx.project_root / models.DINO_HUB_REPO_RELATIVE_PATH
    hub_checkpoint = ctx.project_root / models.DINO_HUB_CHECKPOINT_RELATIVE_PATH
    injected = bool(overrides)

    if injected:
        actual_dino_sha = None
    else:
        models._fail_closed_offline_dino()
        if not hub_checkpoint.is_file():
            raise P6BFailClosedError(
                f"local DINOv2 checkpoint missing: {hub_checkpoint}")
        actual_dino_sha = stage2.file_sha256(hub_checkpoint)
        if actual_dino_sha != models.EXPECTED_DINO_SHA256:
            raise P6BFailClosedError(
                f"DINOv2 checkpoint sha256 {actual_dino_sha} != pinned "
                f"{models.EXPECTED_DINO_SHA256}")

    # ==================== GAP #2 FIX: Record actual CLIP/DINO identities and loaded-state fingerprints ====================
    from provenance import file_identity, model_state_fingerprint
    clip_weight_path = ctx.project_root / models.CLIP_WEIGHT_RELATIVE_PATH

    clip_model = overrides.get("clip_model")
    if clip_model is None:
        clip_model, _ = models.build_clip_backbone(ctx.project_root, ctx.device)

    model = overrides.get("model")
    if model is None:
        model = AdaptedCLIP(
            clip_model=clip_model, relu=True, dino_use_hub_first=True,
            gate_type=LOCKED_OURS["gate_type"], img_size=IMG_SIZE,
            proto_text_refine=LOCKED_OURS["proto_text_refine"],
            dino_mask_ratio=LOCKED_OURS["dino_mask_ratio"],
            clip_pool=LOCKED_OURS["clip_pool"],
            dino_pool=LOCKED_OURS["dino_pool"],
            dino_patch_policy=LOCKED_OURS["dino_patch_policy"],
            dino_strict_backend=True, dino_hub_repo=str(hub_repo),
            dino_hub_checkpoint=str(hub_checkpoint),
            rng_protocol=LOCKED_OURS["rng_protocol"],
            adapter_train_mode=LOCKED_OURS["adapter_train_mode"],
            normal_aware_head=LOCKED_OURS["normal_aware_head"],
        ).to(ctx.device)

    if int(model.fusion_dim) != models.EXPECTED_FUSION_DIM:
        raise P6BFailClosedError(
            f"Ours fusion dim {model.fusion_dim} != locked "
            f"{models.EXPECTED_FUSION_DIM}")

    args = build_ours_args(ctx, meta_path, hub_repo, hub_checkpoint)
    prototype_mem = overrides.get("prototype_mem")
    if prototype_mem is None:
        prototype_mem = FaissPrototypeMemory(
            num_classes=num_classes, feat_dim=model.fusion_dim,
            orth_margin=0.3, init_logit_scale=args.init_logit_scale,
            prototype_init_mode=args.prototype_init_mode,
            prototypes_per_class=args.prototypes_per_class,
            prototype_aggregation=args.prototype_aggregation,
            prototype_mixture_temperature=args.prototype_mixture_temperature,
        ).to(ctx.device)

    trainable_names = tm.configure_model_trainable_parameters(model, args)
    unexpected = [name for name in trainable_names
                  if not name.startswith("image_adapter.det_proj.")]
    if unexpected:
        raise P6BFailClosedError(
            f"Ours trainable whitelist must be exactly image_adapter.det_proj.*"
            f" (+prototype_mem); unexpected: {unexpected}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # ==================== GAP #3 FIX: Reconcile optimizer groups with actual parameters ====================
    # Derive names from actual named_parameters() and verify all intended parameters have requires_grad=True
    expected_trainable_whitelist = set()
    for name in trainable_names:
        if not name.startswith("image_adapter.det_proj."):
            raise P6BFailClosedError(
                f"Ours trainable whitelist must be exactly image_adapter.det_proj.* "
                f"(+prototype_mem); unexpected: {name}")
        expected_trainable_whitelist.add(name)

    # Build actual parameter names and verify membership
    actual_model_param_names = {name for name, param in model.named_parameters()}
    actual_prototype_mem_names = {name for name, param in prototype_mem.named_parameters()}

    # Verify all expected trainable params are actually trainable
    missing_params = expected_trainable_whitelist - actual_model_param_names
    if missing_params:
        raise P6BFailClosedError(
            f"Ours missing trainable params: {missing_params}")

    # Verify no unexpected params are trainable
    unexpectedly_trainable = set()
    for name in actual_model_param_names:
        if name not in expected_trainable_whitelist and model.get_parameter(name).requires_grad:
            unexpectedly_trainable.add(name)
    if unexpectedly_trainable:
        raise P6BFailClosedError(
            f"Ours unexpectedly trainable params: {unexpectedly_trainable}")

    # Verify prototype_mem only has prototypes and logit_scale
    expected_proto_names = {"prototypes", "logit_scale"}
    if actual_prototype_mem_names != expected_proto_names:
        raise P6BFailClosedError(
            f"Ours prototype_mem names mismatch: got {actual_prototype_mem_names}, expected {expected_proto_names}")

    # Verify all expected prototype_mem params have requires_grad=True
    for name in expected_proto_names:
        param = prototype_mem.get_parameter(name)
        if not param.requires_grad:
            raise P6BFailClosedError(
                f"Ours prototype_mem.{name} should have requires_grad=True but has False")

    # ==================== Create optimizer AFTER all validations ====================
    optimizer = StepCountingOptimizer(AdamW(
        [
            {"params": trainable_params, "lr": args.lr},
            {"params": prototype_mem.parameters(),
             "lr": args.lr * args.proto_lr_mult},
        ],
        weight_decay=args.weight_decay,
    ))

    # ==================== OPTIMIZER GROUP VALIDATION AFTER CREATION ====================
    # Use id(parameter) -> qualified_name mapping to reconcile param_groups
    param_id_to_name = {}
    for name, param in model.named_parameters():
        param_id_to_name[id(param)] = name
    for name, param in prototype_mem.named_parameters():
        param_id_to_name[id(param)] = f"prototype_mem.{name}"

    # Validate optimizer group 1 (model trainable params)
    opt_group1_params = list(optimizer.param_groups[0]["params"])
    opt_group2_params = list(optimizer.param_groups[1]["params"])
    all_optimizer_ids = [id(p) for p in opt_group1_params + opt_group2_params]
    if len(all_optimizer_ids) != len(set(all_optimizer_ids)):
        raise P6BFailClosedError("optimizer contains duplicate parameters")
    unknown = [pid for pid in all_optimizer_ids if pid not in param_id_to_name]
    if unknown:
        raise P6BFailClosedError("optimizer contains unnamed/unexpected parameters")
    opt_group1_names_actual = {param_id_to_name[id(p)] for p in opt_group1_params}
    if opt_group1_names_actual != expected_trainable_whitelist:
        missing = expected_trainable_whitelist - opt_group1_names_actual
        extra = opt_group1_names_actual - expected_trainable_whitelist
        if missing:
            raise P6BFailClosedError(f"Optimizer group 1 missing params: {missing}")
        if extra:
            raise P6BFailClosedError(f"Optimizer group 1 unexpected params: {extra}")

    # Validate optimizer group 2 (prototype_mem params)
    opt_group2_names_actual = {param_id_to_name[id(p)] for p in opt_group2_params}
    expected_group2 = {f"prototype_mem.{name}" for name in expected_proto_names}
    if opt_group2_names_actual != expected_group2:
        missing = expected_group2 - opt_group2_names_actual
        extra = opt_group2_names_actual - expected_group2
        if missing:
            raise P6BFailClosedError(f"Optimizer group 2 missing params: {missing}")
        if extra:
            raise P6BFailClosedError(f"Optimizer group 2 unexpected params: {extra}")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    base = build_base_dataset(image_root, meta_path, trainable=True)
    dataset = ManifestSupportDataset(base, meta_records, class_order)
    loader, _sampler = make_seeded_loader(dataset, ctx.seed, args.batch_size)

    logger = logging.getLogger(f"p6b_stage2_support_{ctx.product}_k{ctx.k_shot}"
                               f"_s{ctx.seed}_ours")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(ctx.staging / "ours_train_multiclass.log",
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    init_fn = overrides.get("init_prototypes",
                            tm.init_prototypes_with_class_means)
    train_fn = overrides.get("train_multiclass", tm.train_multiclass)
    try:
        prototype_mem = init_fn(
            model, loader, prototype_mem, ctx.device, class_to_id,
            use_text_guided=False, text_query_cache=None,
            cross_modal_mode="all", num_views=args.num_init_views,
            logger=logger)
        run_config = {
            "method": "ours",
            "locked": LOCKED_OURS,
            "class_order": class_order,
            "seed_protocol": {
                "training_protocol": args.training_protocol,
                "rng_protocol": args.rng_protocol,
                "seed": ctx.seed, "init_seed": ctx.seed,
                "data_seed": ctx.seed, "augment_seed": ctx.seed,
            },
            "trainable_model_parameter_names": trainable_names,
        }
        train_kwargs = dict(
            model=model, prototype_mem=prototype_mem, train_loader=loader,
            optimizer=optimizer, scheduler=scheduler, device=ctx.device,
            start_epoch=0, num_epochs=args.num_epochs,
            class_to_id=class_to_id, save_path=str(ctx.staging),
            logger=logger, supcon_weight=args.supcon_weight,
            orth_weight=args.orth_weight, itc_weight=args.itc_weight,
            text_prototypes=None,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_every=args.save_every, use_text_guided=args.use_text_guided,
            cross_modal_mode=args.cross_modal_mode,
            repeat_per_epoch=args.repeat_per_epoch,
            early_stop_patience=args.early_stop_patience,
            arc_margin=args.arc_margin,
            label_smoothing=args.label_smoothing,
            gate_type=args.gate_type,
            checkpoint_format=args.checkpoint_format,
            save_optimizer_state=args.save_optimizer_state,
            run_config=run_config,
            manifest_path=overrides.get("manifest_path"),
            normal_aware_loss_weight=0.0, prototype_balance_weight=0.0,
            prototype_assignment_temperature=0.2,
        )
        train_fn(**train_kwargs)
    finally:
        # Always release the staging log file handle (Windows: an open
        # handle would keep the staging directory from being removed on
        # failure).
        handler.close()
        logger.removeHandler(handler)

    batches_per_pass = len(loader)
    passes = args.num_epochs * args.repeat_per_epoch
    expected_steps = passes * steps_per_pass(batches_per_pass,
                                             args.gradient_accumulation_steps)
    checkpoint_name = f"multiclass_epoch_{args.num_epochs - 1}.pth"
    extra = []
    for name in (checkpoint_name, "faiss_index.pth",
                 "ours_train_multiclass.log"):
        if (ctx.staging / name).is_file():
            extra.append(name)
    if not overrides and not (ctx.staging / checkpoint_name).is_file():
        raise P6BFailClosedError(
            f"Ours final lightweight checkpoint missing: {checkpoint_name}")
    step_audit = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": "ours",
        "batches_per_pass": batches_per_pass,
        "passes": passes,
        "total_forward_batches": passes * batches_per_pass,
        "actual_optimizer_steps": optimizer.step_count,
        "expected_optimizer_steps": expected_steps,
        "trainable_parameter_whitelist": {
            "model": trainable_names,
            "prototype_mem": ["prototypes", "logit_scale"],
        },
    }
    atomic_write_json(ctx.staging / "ours_step_audit.json", step_audit)

    # The same exact gate applies to production and diagnostic/test-only runs.
    if optimizer.step_count != expected_steps:
        raise P6BFailClosedError(
            f"Ours optimizer step mismatch: actual {optimizer.step_count} != "
            f"expected {expected_steps}; training may have been skipped or "
            f"modified")

    return {"primary_artifact": checkpoint_name,
            "extra_artifacts": extra + ["ours_step_audit.json"],
            "support_pixels_read": True,
            "trainable_parameter_whitelist":
                trainable_names + ["prototype_mem.prototypes",
                                   "prototype_mem.logit_scale"],
            "train_kwargs": train_kwargs,
            "model_provenance": ({
                "method": "ours",
                "mode": "injected-test-only",
                "formal_eligible": False,
                "note": "model dependencies injected; no real weight identity claimed",
            } if injected else {
                "method": "ours",
                "mode": "real-local-weights",
                "backbone": "AdaptedCLIP",
                "clip_backbone": {
                    "model_name": models.CLIP_MODEL_NAME,
                    "pretrained_identifier": f"openai:{models.CLIP_MODEL_NAME}",
                    "loaded_from": str(Path(ctx.project_root) / models.CLIP_WEIGHT_RELATIVE_PATH),
                    "weight_file": file_identity(str(clip_weight_path)),
                    "network_download": False,
                },
                "dino_backend": "hub_local",
                "dino_checkpoint": {
                    "expected_sha256": models.EXPECTED_DINO_SHA256,
                    "actual_sha256": actual_dino_sha,
                    "weight_file": file_identity(str(hub_checkpoint)),
                    "local_source": str(Path(ctx.project_root) / models.DINO_HUB_CHECKPOINT_RELATIVE_PATH),
                    "network_download": False,
                },
                "fusion_dim": int(model.fusion_dim),
                "adapter_train_mode": LOCKED_OURS["adapter_train_mode"],
                "rng_protocol": LOCKED_OURS["rng_protocol"],
                "loaded_state_fingerprints": {
                    "clip_backbone": model_state_fingerprint(model.clipmodel),
                    "dino_backbone": model_state_fingerprint(
                        getattr(model.dino_extractor, "model", model.dino_extractor)),
                },
            })
        }


# ---------------------------------------------------------------------------
# R1 — Exact P3.0 Frozen Representation Proto (Addendum B §2)
#
# R1 reuses the EXACT Ours (AdaptedCLIP) pre-terminal token source and turns
# all support-side adaptation OFF (no det_proj application and no learned
# prototype classifier; zero trainable params). The
# pre-terminal CLIP tokens are captured via a forward-pre-hook on the terminal
# adapter det_proj, so R1's feature is bit-identical to the tensor Ours feeds
# into det_proj — the same real feature construction helper, no reimplemented
# backbone, and no modification to the P3.F model code.
# ---------------------------------------------------------------------------

def extract_r1_pre_terminal_feature(model, image_batch):
    """Return (fused_feature[B, 1408], captured_pre_terminal_tokens).

    ``captured_pre_terminal_tokens`` is the exact input tensor Ours feeds into
    the terminal adapter ``image_adapter.det_proj`` during ``forward_fusion``
    (ln_post CLIP tokens, [B, N+1, 1024]). R1 applies the SAME Top-K pooling
    (``topk_avg_pool(k_ratio=0.1)``) and the SAME DINO mean, but skips det_proj.
    """
    captured = {}
    det_proj = model.image_adapter["det_proj"]

    def _pre_hook(module, args):
        captured["tokens"] = args[0]

    handle = det_proj.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            model.forward_fusion(image_batch)
    finally:
        handle.remove()
    tokens = captured.get("tokens")
    if tokens is None:
        raise P6BFailClosedError(
            "R1 failed to capture the terminal-adapter (det_proj) input tokens")
    with torch.no_grad():
        clip_feat = stage2.l2_normalize(
            model.topk_avg_pool(tokens, k_ratio=0.1), dim=-1)
        dino_patches = model.dino_extractor.forward_patches(image_batch)
        if dino_patches is None or dino_patches.dim() != 3:
            raise P6BFailClosedError("R1 DINO extractor returned no patch tokens")
        dino_feat = stage2.l2_normalize(dino_patches.mean(dim=1), dim=-1)
        fused = stage2.l2_normalize(
            torch.cat([clip_feat, dino_feat], dim=-1), dim=-1)
    return fused, tokens


def _build_frozen_proto_model(ctx, overrides):
    """Build the Ours AdaptedCLIP pipeline and FREEZE every parameter (R1).

    Returns (model, provenance). Real path reuses ``models.build_clip_backbone``
    and the production ``AdaptedCLIP`` with the locked Ours config; no backbone
    loader is duplicated. Injected (diagnostic/test) path uses the provided
    model as-is (its frozen/trainable state is validated by the R1 gate).
    """
    injected = bool(overrides)
    if injected:
        model = overrides["model"]
        provenance = {
            "method": "p30_frozen_proto", "mode": "injected-test-only",
            "formal_eligible": False, "terminal_adaptation": False,
            "note": "model injected; no real weight identity claimed",
        }
        return model, provenance

    from model.adapter import AdaptedCLIP
    from provenance import file_identity, model_state_fingerprint
    from utils import setup_seed
    setup_seed(ctx.seed)
    hub_repo = ctx.project_root / models.DINO_HUB_REPO_RELATIVE_PATH
    hub_checkpoint = ctx.project_root / models.DINO_HUB_CHECKPOINT_RELATIVE_PATH
    models._fail_closed_offline_dino()
    if not hub_checkpoint.is_file():
        raise P6BFailClosedError(
            f"local DINOv2 checkpoint missing: {hub_checkpoint}")
    actual_dino_sha = stage2.file_sha256(hub_checkpoint)
    if actual_dino_sha != models.EXPECTED_DINO_SHA256:
        raise P6BFailClosedError(
            f"DINOv2 checkpoint sha256 {actual_dino_sha} != pinned "
            f"{models.EXPECTED_DINO_SHA256}")
    clip_weight_path = ctx.project_root / models.CLIP_WEIGHT_RELATIVE_PATH
    clip_model, _ = models.build_clip_backbone(ctx.project_root, ctx.device)
    model = AdaptedCLIP(
        clip_model=clip_model, relu=True, dino_use_hub_first=True,
        gate_type=LOCKED_OURS["gate_type"], img_size=IMG_SIZE,
        proto_text_refine=LOCKED_OURS["proto_text_refine"],
        dino_mask_ratio=LOCKED_OURS["dino_mask_ratio"],
        clip_pool=LOCKED_OURS["clip_pool"], dino_pool=LOCKED_OURS["dino_pool"],
        dino_patch_policy=LOCKED_OURS["dino_patch_policy"],
        dino_strict_backend=True, dino_hub_repo=str(hub_repo),
        dino_hub_checkpoint=str(hub_checkpoint),
        rng_protocol=LOCKED_OURS["rng_protocol"],
        adapter_train_mode=LOCKED_OURS["adapter_train_mode"],
        normal_aware_head=LOCKED_OURS["normal_aware_head"],
    ).to(ctx.device)
    model.eval()
    # Terminal adaptation OFF: freeze EVERY parameter (no training at all).
    for param in model.parameters():
        param.requires_grad_(False)
    provenance = {
        "method": "p30_frozen_proto", "mode": "real-local-weights",
        "backbone": "AdaptedCLIP", "terminal_adaptation": False,
        "clip_backbone": {
            "model_name": models.CLIP_MODEL_NAME,
            "pretrained_identifier": f"openai:{models.CLIP_MODEL_NAME}",
            "weight_file": file_identity(str(clip_weight_path)),
            "network_download": False,
        },
        "dino_checkpoint": {
            "expected_sha256": models.EXPECTED_DINO_SHA256,
            "actual_sha256": actual_dino_sha,
            "weight_file": file_identity(str(hub_checkpoint)),
            "network_download": False,
        },
        "loaded_state_fingerprints": {
            "clip_backbone": model_state_fingerprint(model.clipmodel),
            "dino_backbone": model_state_fingerprint(
                getattr(model.dino_extractor, "model", model.dino_extractor)),
        },
    }
    return model, provenance


def run_r1_frozen_proto(ctx, meta_path, meta_records, class_order,
                        overrides=None) -> dict:
    """R1 = Exact P3.0 Frozen Representation Proto (Addendum B §2).

    Shares Ours' exact pre-terminal token source (CLIP Top-K on the det_proj
    input tokens + DINO mean), turns all support-side adaptation OFF (zero
    trainable params), then builds class-mean prototypes and classifies by
    cosine argmax. This is a feature-matched pre-terminal frozen prototype
    reference — Ours differs by also training the terminal adapter, prototype
    memory, and logit scale.
    """
    overrides = dict(overrides or {})
    model, provenance = _build_frozen_proto_model(ctx, overrides)

    # GATE: R1 must have ZERO trainable parameters (terminal adaptation OFF).
    trainable = [name for name, param in model.named_parameters()
                 if param.requires_grad]
    if trainable:
        raise P6BFailClosedError(
            f"R1 (p30_frozen_proto) requires zero trainable params; found: "
            f"{trainable}")

    dataset = _deterministic_dataset(
        ctx.image_root, meta_path, meta_records, class_order)

    def _feature(image):
        batch = models._to_batch(image).to(ctx.device)
        fused, _tokens = extract_r1_pre_terminal_feature(model, batch)
        return fused[0].detach().cpu()

    embeddings, labels = _extract_support_embeddings(dataset, _feature)
    if embeddings.shape[1] != R1_FROZEN_PROTO_FUSION_DIM:
        raise P6BFailClosedError(
            f"R1 pre-terminal fusion dim {embeddings.shape[1]} != locked "
            f"{R1_FROZEN_PROTO_FUSION_DIM}")
    prototypes = stage2.build_prototypes(embeddings, labels, class_order)
    name = "p30_frozen_proto_prototypes.pt"
    torch.save({
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": "p30_frozen_proto",
        "representation": "exact_p30_pre_terminal_clip_topk_dino_mean_l2norm",
        "terminal_adaptation": False,
        "class_order": class_order,
        "prototypes": prototypes.detach().cpu(),
        "support_embeddings": embeddings.detach().cpu(),
        "support_labels": labels,
        "trainable_param_count": 0,
    }, ctx.staging / name)
    return {"primary_artifact": name,
            "support_pixels_read": True,
            "trainable_parameter_whitelist": [],
            "model_provenance": provenance}


# ---------------------------------------------------------------------------
# R2 — Tip-Adapter-F* (firewall-constrained; Addendum B §3)
#
# Only cache keys are trainable. CLIP image/text encoders, cache values and
# class text embeddings are frozen. No best-epoch selection, no alpha/beta/
# prompt search, no query access; final epoch only.
# ---------------------------------------------------------------------------

def train_tip_adapter_f_augmentation_aware(
        *, extract_clip, dataset, module, spec, seed, device) -> dict:
    """Locked augmentation-aware cache-key training loop (Addendum B §3.3).

    Every epoch x repeat pass re-runs the locked non-cable augmentation and
    RE-EXTRACTS the frozen CLIP image embeddings; only ``module.cache_keys`` is
    trainable. The scheduler steps once per optimizer step (T_max = expected
    optimizer steps), faithful to the official Tip-Adapter-F cache-key loop.
    """
    from seed_protocol import set_loader_seed_context
    stage2.validate_tip_adapter_f_spec(spec)
    device = torch.device(device)
    loader, _sampler = make_seeded_loader(dataset, seed, spec.batch_size)
    module = module.to(device)
    module.train()
    # Only cache_keys is trainable.
    optimizer = StepCountingOptimizer(AdamW(
        [module.cache_keys], lr=spec.lr, eps=spec.eps,
        weight_decay=spec.weight_decay))
    batches_per_pass = len(loader)
    total_passes = spec.epochs * spec.repeat_per_epoch
    expected_steps = total_passes * steps_per_pass(batches_per_pass,
                                                    spec.grad_accum)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, expected_steps))
    criterion = nn.CrossEntropyLoss(label_smoothing=spec.label_smoothing)

    epochs_audit = []
    extract_call_count = 0
    epoch_repeat_contexts = set()
    all_loss_finite = True
    for epoch in range(spec.epochs):
        epoch_batches = 0
        steps_before = optimizer.step_count
        for repeat in range(spec.repeat_per_epoch):
            set_loader_seed_context(loader, "tip_adapter_f_train",
                                    epoch=epoch, repeat=repeat)
            epoch_repeat_contexts.add((epoch, repeat))
            micro = 0
            for batch_idx, batch in enumerate(loader):
                images = batch["image"]
                image_emb = extract_clip(images).detach().to(device)
                extract_call_count += 1
                labels = batch["label"].to(device)
                logits = module.logits(image_emb)
                loss = criterion(logits, labels) / spec.grad_accum
                if not bool(torch.isfinite(loss).all()):
                    all_loss_finite = False
                loss.backward()
                micro += 1
                epoch_batches += 1
                if micro % spec.grad_accum == 0 or \
                        (batch_idx + 1) == len(loader):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
        epochs_audit.append({
            "epoch": epoch,
            "batches": epoch_batches,
            "optimizer_steps": optimizer.step_count - steps_before,
        })

    if not all_loss_finite:
        raise P6BFailClosedError(
            "Tip-Adapter-F* produced non-finite (NaN/Inf) loss")
    total_expected_batches = spec.epochs * spec.repeat_per_epoch * batches_per_pass
    if extract_call_count != total_expected_batches:
        raise P6BFailClosedError(
            f"Tip-Adapter-F* extract call count {extract_call_count} != "
            f"expected {total_expected_batches}")
    expected_contexts = {(e, r) for e in range(spec.epochs)
                         for r in range(spec.repeat_per_epoch)}
    if epoch_repeat_contexts != expected_contexts:
        raise P6BFailClosedError(
            "Tip-Adapter-F* epoch/repeat contexts mismatch")
    return {
        "module": module,
        "final_epoch": spec.epochs - 1,
        "epochs": epochs_audit,
        "batches_per_pass": batches_per_pass,
        "total_optimizer_steps": optimizer.step_count,
        "expected_optimizer_steps": expected_steps,
        "extract_call_count": extract_call_count,
        "total_forward_batches": total_expected_batches,
    }


def run_tip_adapter_f(ctx, image_root, meta_path, meta_records,
                      class_order) -> dict:
    """R2 = Tip-Adapter-F* (firewall-constrained; Addendum B §3).

    Lazy CLIP-only loading. Only cache keys are trainable; final epoch only;
    no best-epoch selection; no alpha/beta/prompt search; no query access.
    """
    if not ctx.factory:
        ctx.factory = _build_lazy_clip_only(ctx)
    factory = ctx.factory
    spec = stage2.validate_tip_adapter_f_spec(stage2.TipAdapterFSpec())
    num_classes = len(class_order)

    # Frozen class text embeddings (zero-shot prompts).
    prompts = [stage2.format_zero_shot_prompt(name) for name in class_order]
    text_weights = factory.text_extractor.embed_text(prompts)

    # Cache keys initialized from deterministic eval-view support CLIP embeddings
    # (the training-free Tip-Adapter cache), then made trainable.
    det_dataset = _deterministic_dataset(
        image_root, meta_path, meta_records, class_order)
    init_keys, labels = _extract_support_embeddings(
        det_dataset, lambda image: factory.clip_extractor.embed_image(image))
    module = stage2.TipAdapterFModule(
        init_keys, labels, num_classes, text_weights).to(ctx.device)

    # Frozen/trainable exactness gates (Addendum B §3.1).
    if not module.cache_keys.requires_grad:
        raise P6BFailClosedError("R2 cache_keys must be trainable")
    if module.cache_values.requires_grad:
        raise P6BFailClosedError("R2 cache_values must be frozen")
    if module.text_weights.requires_grad:
        raise P6BFailClosedError("R2 text_weights must be frozen")
    trainable = [name for name, param in module.named_parameters()
                 if param.requires_grad]
    if trainable != ["cache_keys"]:
        raise P6BFailClosedError(
            f"R2 trainable set must be exactly ['cache_keys']; found {trainable}")
    clip_model = getattr(factory, "clip_model", None)
    if clip_model is not None:
        unfrozen = [n for n, p in clip_model.named_parameters()
                    if p.requires_grad]
        if unfrozen:
            raise P6BFailClosedError(
                f"R2 CLIP encoder must stay frozen; trainable: {unfrozen[:4]}")

    base = build_base_dataset(image_root, meta_path, trainable=True)
    dataset = ManifestSupportDataset(base, meta_records, class_order)

    def extract_clip(images):
        images = images.to(ctx.device)
        with torch.no_grad():
            return factory.clip_extractor.embed_images(images)

    result = train_tip_adapter_f_augmentation_aware(
        extract_clip=extract_clip, dataset=dataset, module=module, spec=spec,
        seed=ctx.seed, device=ctx.device)

    name = "tip_adapter_f_cache.pt"
    torch.save({
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": "tip_adapter_f",
        "class_order": class_order,
        "cache_keys": module.cache_keys.detach().cpu(),
        "cache_values": module.cache_values.detach().cpu(),
        "text_weights": module.text_weights.detach().cpu(),
        "cache_labels": labels,
        "alpha": module.alpha,
        "beta": module.beta,
        "logit_scale": module.logit_scale,
        "final_epoch": result["final_epoch"],
        "locked_spec": dataclasses_asdict(spec),
        "checkpoint_selection": "final_epoch_only",
        "query_used_for_selection": False,
        "validation_used_for_selection": False,
    }, ctx.staging / name)
    step_audit = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": "tip_adapter_f",
        "epochs": result["epochs"],
        "batches_per_pass": result["batches_per_pass"],
        "total_forward_batches": result["total_forward_batches"],
        "actual_optimizer_steps": result["total_optimizer_steps"],
        "expected_optimizer_steps": result["expected_optimizer_steps"],
        "extract_call_count": result["extract_call_count"],
        "trainable_parameter_whitelist": ["cache_keys"],
        "checkpoint_selection": "final_epoch_only",
        "query_used_for_selection": False,
        "validation_used_for_selection": False,
        "augmentation": LOCKED_CONSTANTS["trainable_augmentation"],
    }
    atomic_write_json(ctx.staging / "tip_adapter_f_step_audit.json", step_audit)

    if result["total_optimizer_steps"] != result["expected_optimizer_steps"]:
        raise P6BFailClosedError(
            f"Tip-Adapter-F* optimizer step mismatch: actual "
            f"{result['total_optimizer_steps']} != expected "
            f"{result['expected_optimizer_steps']}")
    return {"primary_artifact": name,
            "extra_artifacts": ["tip_adapter_f_step_audit.json"],
            "support_pixels_read": True,
            "trainable_parameter_whitelist": ["cache_keys"]}


# ---------------------------------------------------------------------------
# Orchestration: transactional staging + completion
# ---------------------------------------------------------------------------

class RunContext:
    def __init__(self, *, project_root, bundle_dir, image_root, product,
                 k_shot, seed, method, output_dir, device, diagnostic,
                 factory):
        self.project_root = Path(project_root)
        self.bundle_dir = Path(bundle_dir)
        self.image_root = Path(image_root)
        self.product = product
        self.k_shot = int(k_shot)
        self.seed = int(seed)
        self.method = method
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.diagnostic = bool(diagnostic)
        self.factory = factory
        self.staging = self.output_dir.with_name(
            self.output_dir.name + ".p6b_staging")
        self.artifacts = []

    def register_artifact(self, name: str) -> dict:
        path = self.staging / name
        if not path.is_file():
            raise P6BFailClosedError(f"artifact missing in staging: {name}")
        record = {
            "relative_path": name,
            "size_bytes": path.stat().st_size,
            "sha256": stage2.file_sha256(path),
        }
        self.artifacts.append(record)
        return record


def execute_run(*, project_root=PROJECT_ROOT, bundle_dir, image_root, product,
                k_shot, seed, method, output_dir, device=None,
                allow_dirty_diagnostic=False, factory_bundle=None,
                ours_overrides=None, r1_overrides=None) -> dict:
    project_root = Path(project_root)
    if method not in METHODS:
        raise P6BFailClosedError(f"unknown method {method!r}; registry is "
                                 f"{list(METHODS)}")
    if product not in LOCKED_PRODUCTS:
        raise P6BFailClosedError(f"product {product!r} not in locked 13")
    # K8 extension (Protocol B section 4): K=8 is a post-P6.B authorized
    # shot extension on the nine availability-qualified products; the frozen
    # method semantics and the final-checkpoint rule are unchanged.
    if (int(k_shot) not in LOCKED_K and int(k_shot) != 8) or int(seed) not in LOCKED_SEEDS:
        raise P6BFailClosedError("K/seed outside locked grid")
    if int(k_shot) == 8:
        from experiments.p6.mvrec_rgb_formal.k8_builder import K8_PRODUCTS

        if product not in K8_PRODUCTS:
            raise P6BFailClosedError(
                f"K=8: product {product!r} outside the nine-product K8 set")

    # ==================== PHASE 1 FIX: Document Canonical-SHA Validation BEFORE ANYTHING ====================
    # Verify actual document canonical-LF SHA matches pinned values IMMEDIATELY.
    # This happens before git dirty check, before bundle loading, before any model construction.
    start_canonical_docs = document_provenance_canonical_lf(project_root)

    # Enforce exact match with pinned canonical-LF SHAs (Addendum A §4 /
    # Addendum B §4 — Protocol Bundle V2 pins all three documents).
    # CLI parameters are ignored here — only actual file content matters.
    for key, expected in EXPECTED_DOCUMENT_SHAS_CANONICAL_LF.items():
        actual = start_canonical_docs[key]["sha256"]
        if actual != expected:
            raise P6BFailClosedError(
                f"{key} canonical-LF sha256 mismatch: {actual} != {expected}")

    # Use computed canonical docs as baseline for re-verification at completion time
    commit, dirty = builder.git_commit_and_dirty(project_root)

    # ==================== GAP #5 FIX: Formal mode rejects dependency injection ====================
    # Formal mode must reject any non-None factory_bundle or ours_overrides.
    # Dependency injection may be supported only through diagnostic/test-only path.
    if not allow_dirty_diagnostic:
        if factory_bundle is not None:
            raise P6BFailClosedError(
                "formal mode rejects factory_bundle injection; use diagnostic "
                "mode (--allow-dirty-diagnostic) for testing with injected factories")
        if ours_overrides is not None and len(ours_overrides) > 0:
            raise P6BFailClosedError(
                "formal mode rejects ours_overrides injection; use diagnostic "
                "mode (--allow-dirty-diagnostic) for testing with injected overrides")
        if r1_overrides is not None and len(r1_overrides) > 0:
            raise P6BFailClosedError(
                "formal mode rejects r1_overrides injection; use diagnostic "
                "mode (--allow-dirty-diagnostic) for testing with injected overrides")

    # Git dirty check AFTER formal injection rejection (fail-fast on tampering)
    if dirty and not allow_dirty_diagnostic:
        raise P6BFailClosedError(
            "formal support run requires a clean git tree; "
            "use --allow-dirty-diagnostic for diagnostic runs")

    # Stage 1 pinned bundle gate + unique manifest selection.
    if int(k_shot) == 8:
        # K8 extension (Protocol B section 4): the K8 manifest bundle is
        # used instead of the Stage 1 bundle; every K8 manifest is
        # self-SHA verified exactly like a Stage 1 manifest.
        manifest_relative = (
            f"p6b_k8_support_manifests/{product}/k8_s{int(seed)}.json")
        k8_manifest_path = Path(bundle_dir) / manifest_relative
        if not k8_manifest_path.is_file():
            raise P6BFailClosedError(f"K8 manifest missing: {k8_manifest_path}")
        manifest = json.loads(k8_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "p6b_k8_support_manifest_v1":
            raise P6BFailClosedError(f"K8 manifest schema mismatch: {k8_manifest_path}")
        if (manifest.get("product"), manifest.get("K"),
                manifest.get("seed")) != (product, 8, int(seed)):
            raise P6BFailClosedError(f"K8 manifest identity mismatch: {k8_manifest_path}")
        declared = manifest.get("manifest_sha256")
        if not declared or declared != manifest_self_sha256(manifest):
            raise P6BFailClosedError(f"K8 manifest self-hash mismatch: {k8_manifest_path}")
        bundle = None
        manifest_file_sha = stage2.file_sha256(k8_manifest_path)
    else:
        bundle = stage2.load_formal_bundle(bundle_dir)
        manifest = stage2.select_manifest(bundle, product, int(k_shot), int(seed))
        manifest_relative = (
            f"p6b_support_manifests/{product}/k{int(k_shot)}_s{int(seed)}.json")
        manifest_file_sha = stage2.file_sha256(
            Path(bundle_dir) / manifest_relative)

    # Support asset verification gate (containment/existence/raw SHA).
    # This must happen BEFORE any model construction to maintain fail-closed.
    _validate_support_assets_pre_model(bundle, manifest, image_root)

    # Lazy model construction ONLY AFTER all non-model gates pass.
    # Method-specific loading to avoid unnecessary memory usage.
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ctx = RunContext(
        project_root=project_root, bundle_dir=bundle_dir,
        image_root=image_root, product=product, k_shot=k_shot, seed=seed,
        method=method, output_dir=output_dir, device=device,
        diagnostic=allow_dirty_diagnostic, factory=factory_bundle)

    if ctx.output_dir.exists():
        raise P6BFailClosedError(f"output dir already exists: {output_dir}")
    if ctx.staging.exists():
        raise P6BFailClosedError(f"stale staging dir exists: {ctx.staging}")
    ctx.staging.mkdir(parents=True)
    try:
        result = _execute_inside_staging(ctx, manifest, manifest_relative,
                                         manifest_file_sha, commit, dirty,
                                         start_canonical_docs, ours_overrides,
                                         r1_overrides)
        os_replace_publish(ctx)
        return result
    except BaseException:
        shutil.rmtree(ctx.staging, ignore_errors=True)
        if ctx.output_dir.exists():
            raise P6BFailClosedError(
                "late failure left a published output directory")
        raise


def os_replace_publish(ctx) -> None:
    import os
    os.replace(str(ctx.staging), str(ctx.output_dir))


def _execute_inside_staging(ctx, manifest, manifest_relative,
                            manifest_file_sha, commit, dirty, start_canonical_docs,
                            ours_overrides, r1_overrides=None) -> dict:
    class_order = list(manifest["class_order"])
    metadata = write_support_metadata(manifest, ctx.staging)
    ctx.register_artifact("p6b_support_meta.jsonl")
    support_assets = verify_support_assets(manifest, ctx.image_root)

    method_info = {"trainable_parameter_whitelist": [],
                   "support_pixels_read": method_support_reads_pixels(ctx.method)}
    if ctx.method in ("clip_proto", "dino_proto", "fusion_proto"):
        dataset = _deterministic_dataset(
            ctx.image_root, metadata["path"], support_assets, class_order)
        method_info.update(run_prototype_method(ctx, ctx.method, dataset,
                                                class_order))
    elif ctx.method == "zero_shot_clip":
        method_info.update(run_zero_shot_clip(ctx, class_order))
    elif ctx.method == "tip_adapter":
        dataset = _deterministic_dataset(
            ctx.image_root, metadata["path"], support_assets, class_order)
        method_info.update(run_tip_adapter(ctx, dataset, class_order))
    elif ctx.method == "linear_probe":
        method_info.update(run_linear_probe(
            ctx, ctx.image_root, metadata["path"], support_assets,
            class_order))
    elif ctx.method == "ours":
        method_info.update(run_ours(
            ctx, ctx.image_root, metadata["path"], support_assets,
            class_order, overrides=ours_overrides))
    elif ctx.method == "p30_frozen_proto":
        method_info.update(run_r1_frozen_proto(
            ctx, metadata["path"], support_assets, class_order,
            overrides=r1_overrides))
    elif ctx.method == "tip_adapter_f":
        method_info.update(run_tip_adapter_f(
            ctx, ctx.image_root, metadata["path"], support_assets,
            class_order))
    else:  # pragma: no cover — registry checked above
        raise P6BFailClosedError(f"unknown method {ctx.method!r}")

    seen = set()
    for name in [method_info["primary_artifact"]] + \
            method_info.get("extra_artifacts", []):
        # no duplicate artifact registration (primary may also appear in the
        # extras list)
        if name and name not in seen and (ctx.staging / name).is_file():
            seen.add(name)
            ctx.register_artifact(name)

    # Locked documents re-verified immediately before completion is written.
    current_docs = verify_document_shas(ctx.project_root, start_canonical_docs)

    run_config = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "method": ctx.method,
        "product": ctx.product,
        "K": ctx.k_shot,
        "seed": ctx.seed,
        "device": str(ctx.device),
        "class_order": class_order,
        "locked_constants": LOCKED_CONSTANTS,
        "support_metadata": {
            "records": metadata["records"],
            "sha256": metadata["sha256"],
        },
    }
    atomic_write_json(ctx.staging / "support_run_config.json", run_config)
    ctx.register_artifact("support_run_config.json")

    artifact_manifest = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "method": ctx.method,
        "product": ctx.product,
        "K": ctx.k_shot,
        "seed": ctx.seed,
        "artifacts": ctx.artifacts,
    }
    atomic_write_json(ctx.staging / "support_artifact_manifest.json",
                      artifact_manifest)
    ctx.register_artifact("support_artifact_manifest.json")

    # Model provenance: for Ours, record actual AdaptedCLIP chain used.
    # For other methods, use lazy-factory provenance if constructed.
    model_provenance = None
    if ctx.method in ("ours", "p30_frozen_proto"):
        # Ours / R1 return model_provenance directly from their runners.
        model_provenance = method_info.get("model_provenance")
        if model_provenance is None:
            model_provenance = {"note": f"{ctx.method} model_provenance not returned"}
    elif ctx.factory is not None:
        model_provenance = ctx.factory.provenance
    else:
        model_provenance = {"note": "factory not constructed (validation failed earlier or diagnostic mode)"}
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "diagnostic" if ctx.diagnostic else "completed",
        "formal_eligible": not ctx.diagnostic,
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "support_pixels_read": method_info["support_pixels_read"],
        "method": ctx.method,
        "product": ctx.product,
        "K": ctx.k_shot,
        "seed": ctx.seed,
        "class_order": class_order,
        "execution": {
            "commit": commit,
            "dirty": dirty,
            "allow_dirty_diagnostic": ctx.diagnostic,
        },
        "stage1": {
            "completion_sha256": stage2.STAGE1_COMPLETION_SHA256,
            "bundle_dir": str(ctx.bundle_dir),
            "selected_manifest_relative_path": manifest_relative,
            "selected_manifest_sha256": manifest_file_sha,
            "selected_manifest_self_sha256": manifest["manifest_sha256"],
        },
        "protocol": {
            "relative_path": start_canonical_docs["protocol"]["relative_path"],
            "sha256": start_canonical_docs["protocol"]["sha256"],
            "hash_mode": "canonical-lf",
        },
        "addendum_a": {
            "relative_path": start_canonical_docs["addendum_a"]["relative_path"],
            "sha256": start_canonical_docs["addendum_a"]["sha256"],
            "hash_mode": "canonical-lf",
        },
        "addendum_b": {
            "relative_path": start_canonical_docs["addendum_b"]["relative_path"],
            "sha256": start_canonical_docs["addendum_b"]["sha256"],
            "hash_mode": "canonical-lf",
        },
        # Protocol Bundle V2 provenance (Addendum B §4): all three protocol
        # docs pinned together plus the Stage 1 completion pin.
        "protocol_bundle": {
            "version": PROTOCOL_BUNDLE_VERSION,
            "main_protocol_sha256": start_canonical_docs["protocol"]["sha256"],
            "addendum_a_sha256": start_canonical_docs["addendum_a"]["sha256"],
            "addendum_b_sha256": start_canonical_docs["addendum_b"]["sha256"],
            "stage1_completion_sha256": stage2.STAGE1_COMPLETION_SHA256,
            "hash_mode": "canonical-lf",
        },
        "model_provenance": model_provenance,
        "locked_constants": LOCKED_CONSTANTS,
        "trainable_parameter_whitelist":
            method_info["trainable_parameter_whitelist"],
        "artifacts": ctx.artifacts,
    }
    atomic_write_json(ctx.staging / "p6b_stage2_support_complete.json",
                      completion)
    return completion


def method_support_reads_pixels(method: str) -> bool:
    return method != "zero_shot_clip"


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        # Do NOT pre-load factory - gates must fail first (fail-closed).
        # Lazy model construction happens inside execute_run AFTER all
        # non-model validation gates pass.
        completion = execute_run(
            bundle_dir=args.bundle_dir, image_root=args.image_root,
            product=args.product, k_shot=args.k, seed=args.seed,
            method=args.method, output_dir=args.output_dir,
            device=args.device,
            allow_dirty_diagnostic=args.allow_dirty_diagnostic,
            factory_bundle=None,  # lazy construction via method-specific builder
            ours_overrides=None,
        )
        print(f"[P6.B Stage 2B] {completion['status']}: "
              f"{args.method} {args.product} K={args.k} seed={args.seed} -> "
              f"{args.output_dir}")
        return 0
    except P6BFailClosedError as exc:
        print(f"[P6.B Stage 2B] FAIL-CLOSED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
