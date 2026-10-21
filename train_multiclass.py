"""
Multi-class training script: fused DINOv2 features + FaissPrototypeMemory prototype memory.

Training strategy:
1. CLIP backbone and DINOv2 are frozen.
2. The adapter layers (image_adapter) and prototype_mem (prototypes + logit_scale) are trained.
3. Losses: CrossEntropy + SupConLoss + OrthogonalLoss + ITC loss (optional).
4. Features come from the full adapter path of model.forward() (not from encode_image under torch.no_grad),
   so that gradients reach the adapters.
5. Multi-prompt ITC loss: detailed REAL_NAMES descriptions filled into several prompt templates give the
   text prototypes, aligning the CLIP visual features with the text space to separate the classes.
6. Text-guided features: --use_text_guided uses K-way residual-fused features instead of the globally averaged
   CLS token, so that the model attends to local defect regions rather than the normal surface.
"""

import os
from runtime_config import configure_torch_runtime, dataloader_kwargs

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import argparse
import re
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from model.faiss_memory import FaissPrototypeMemory
from model.checkpoint_io import (
    build_training_checkpoint,
    checkpoint_sha256,
    load_model_checkpoint_state,
    restore_prototype_extra_state,
)
from model.losses import SupConLoss
from dataset import get_dataset
from dataset.constants import CLASS_NAMES, REAL_NAMES
from model.prompts_config import get_prompt_templates
from model.tokenizer import tokenize
from provenance import (
    collect_backbone_provenance,
    collect_runtime_environment,
    initialize_training_manifest,
    jsonl_identity,
    parameter_fingerprint,
    record_artifact,
    state_dicts_fingerprint,
    update_training_manifest,
    utc_now,
)
from seed_protocol import (
    DeterministicWorkerSeeder,
    SeededEpochSampler,
    make_loader_generator,
    set_loader_seed_context,
)
import warnings
print("starting multi-class training...")
warnings.filterwarnings("ignore")


LEGACY_TRAINING_PROTOCOLS = {"legacy_9359", "legacy_seed_split_diag"}
EARLY_ADAPTER_STATE_PREFIX = "image_adapter.layer_adapters."


def is_legacy_training_protocol(training_protocol):
    """Return whether a protocol intentionally retains the legacy graph/loader."""
    return training_protocol in LEGACY_TRAINING_PROTOCOLS


def reset_split_training_rng(args):
    """Reset all global RNGs after construction for the split-seed diagnostic.

    The exact ``legacy_9359`` path must never call this function: its defining
    behavior is that construction and training consume one continuous global
    RNG stream. ``legacy_seed_split_diag`` retains every other legacy behavior
    but resets that stream immediately before dataset/DataLoader construction.
    """
    if args.training_protocol != "legacy_seed_split_diag":
        return False
    if args.data_seed != args.augment_seed:
        raise ValueError(
            "legacy_seed_split_diag requires data_seed == augment_seed because "
            "DataLoader order, worker augmentation and CLIP Patch Dropout still "
            "share one post-construction global RNG stream"
        )
    setup_seed(args.data_seed)
    return True


def load_early_adapter_override(model, checkpoint_path):
    """Load exactly the six frozen early adapters from a donor checkpoint.

    This diagnostic intentionally leaves det_proj, seg_proj, prototypes and all
    other construction-seed-dependent tensors untouched.  The donor's final
    early-adapter tensors equal its initialization because the legacy graph
    gives them grad=None throughout training.
    """
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Early-adapter donor checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            "Early-adapter donor checkpoint has no model_state_dict: "
            f"{checkpoint_path}"
        )
    donor_run_config = checkpoint.get("run_config")
    if not isinstance(donor_run_config, dict):
        raise ValueError(
            "Early-adapter donor checkpoint has no auditable run_config: "
            f"{checkpoint_path}"
        )
    if donor_run_config.get("adapter_train_mode") != "legacy_random_fixed":
        raise ValueError(
            "Early-adapter donor must use adapter_train_mode="
            f"legacy_random_fixed, got {donor_run_config.get('adapter_train_mode')!r}"
        )

    donor = {
        name: tensor
        for name, tensor in state_dict.items()
        if name.startswith(EARLY_ADAPTER_STATE_PREFIX)
    }
    expected = {
        name
        for name in model.state_dict()
        if name.startswith(EARLY_ADAPTER_STATE_PREFIX)
    }
    if set(donor) != expected:
        missing = sorted(expected - set(donor))
        unexpected = sorted(set(donor) - expected)
        raise ValueError(
            "Early-adapter donor keys do not match the target model: "
            f"missing={missing[:6]}, unexpected={unexpected[:6]}"
        )

    target_state = model.state_dict()
    for name, tensor in donor.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"Donor state entry is not a tensor: {name}")
        if tensor.shape != target_state[name].shape:
            raise ValueError(
                f"Early-adapter shape mismatch for {name}: "
                f"donor={tuple(tensor.shape)}, target={tuple(target_state[name].shape)}"
            )

    module_state = {
        name[len(EARLY_ADAPTER_STATE_PREFIX):]: tensor
        for name, tensor in donor.items()
    }
    model.image_adapter["layer_adapters"].load_state_dict(module_state, strict=True)
    selected_fingerprint = state_dicts_fingerprint(
        [("image_adapter.layer_adapters", module_state)]
    )
    return {
        "path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_training_protocol": donor_run_config.get("training_protocol"),
        "source_init_seed": donor_run_config.get(
            "init_seed", donor_run_config.get("seed")
        ),
        "source_trajectory_seed": donor_run_config.get("trajectory_seed"),
        "selected_state": selected_fingerprint,
        "tensor_count": len(donor),
    }


def configure_loader_seed_context(loader, stage, *, epoch=0, repeat=0):
    """Set clean sampler context or describe the untouched legacy RNG stream."""
    if isinstance(getattr(loader, "sampler", None), SeededEpochSampler):
        return set_loader_seed_context(
            loader, stage, epoch=epoch, repeat=repeat
        )
    return {
        "stage": stage,
        "epoch": int(epoch),
        "repeat": int(repeat),
        "protocol": "legacy_global_rng",
    }


def build_class_mapping(dataset_name):
    """Map class names to integer ids."""
    class_names = CLASS_NAMES[dataset_name]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    id_to_class = {idx: name for idx, name in enumerate(class_names)}
    return class_to_id, id_to_class


def deterministic_spherical_prototypes(features, count, iterations=20):
    """Create deterministic spherical clusters without consuming RNG state."""
    points = F.normalize(torch.as_tensor(features), dim=-1)
    if points.ndim != 2 or points.shape[0] < int(count):
        raise ValueError(
            f"Need at least {count} two-dimensional feature rows; got {tuple(points.shape)}"
        )
    count = int(count)
    if count == 1:
        center = F.normalize(points.mean(dim=0, keepdim=True), dim=-1)
        return center, torch.zeros(points.shape[0], dtype=torch.long)

    # Start from the globally farthest pair, rather than mean + farthest.
    # The latter systematically produced 6+1 "main cluster + outlier" splits
    # on the 7-shot folds, which does not represent two class modes fairly.
    pair_similarity = points @ points.T
    upper_triangle = torch.triu(
        torch.ones_like(pair_similarity, dtype=torch.bool), diagonal=1
    )
    masked_similarity = pair_similarity.masked_fill(~upper_triangle, float("inf"))
    flat_pair = int(masked_similarity.argmin().item())
    first_id = flat_pair // points.shape[0]
    second_id = flat_pair % points.shape[0]
    centers = [points[first_id].clone(), points[second_id].clone()]
    for _ in range(2, count):
        similarity = points @ torch.stack(centers, dim=0).T
        centers.append(points[similarity.max(dim=1).values.argmin()].clone())
    centers = torch.stack(centers, dim=0)

    assignments = None
    for _ in range(int(iterations)):
        similarity = points @ centers.T
        new_assignments = similarity.argmax(dim=1)
        if assignments is not None and torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        updated = []
        for cluster_id in range(count):
            members = points[assignments == cluster_id]
            if members.numel() == 0:
                updated.append(points[similarity.max(dim=1).values.argmin()].clone())
            else:
                updated.append(F.normalize(members.mean(dim=0), dim=0))
        centers = torch.stack(updated, dim=0)
    assignments = (points @ centers.T).argmax(dim=1)
    return centers, assignments


@torch.no_grad()
def init_prototypes_with_class_means(model, train_loader, prototype_mem, device, class_to_id,
                                     use_text_guided=False, text_query_cache=None, cross_modal_mode='all',
                                     num_views=5, logger=None):
    """Initialise the prototypes with the mean feature of each class.

    With use_text_guided=True the fused features [B, 1920] (residual added to clip_feat) are used,
    so that the feature width matches prototype_mem.feat_dim.

    Args:
        use_text_guided: use text-guided features (1920-d)
        text_query_cache: [K, 768] text prototypes, required with use_text_guided
        cross_modal_mode: cross-modal fusion path
        num_views: number of augmented views per image; each image passes num_views random augmentations,
                   e.g. 8 shots x 5 views = 40 features per mean, which lowers the initialisation variance
    """
    model.eval()
    class_feats = {cid: [] for cid in range(len(class_to_id))}

    print(f"\nExtracting Multi-View features for prototype initialization ({num_views} views)...")
    for view_idx in range(num_views):
        seed_context = configure_loader_seed_context(
            train_loader, "prototype_init", epoch=0, repeat=view_idx
        )
        seed_message = f"Seed context: {seed_context}"
        print(seed_message)
        if logger:
            logger.info(seed_message)
        # every pass over train_loader yields different random augmentations of the same images
        for input_data in tqdm(train_loader, desc=f"View {view_idx+1}/{num_views}"):
            images = input_data["image"].to(
                device, non_blocking=device.type == "cuda"
            )
            class_names = input_data["class_name"]

            if use_text_guided and text_query_cache is not None:
                labels = torch.tensor(
                    [class_to_id[name] for name in class_names],
                    dtype=torch.long, device=device,
                )
                vis_feat_kway, _, _ = model.forward_hybrid_features_kway(
                    images, text_prototypes=text_query_cache, mode=cross_modal_mode
                )
                batch_indices = torch.arange(images.size(0), device=device)
                feat = vis_feat_kway[batch_indices, labels]  # [B, 1920]
            else:
                feat = model.forward_fusion(images)  # [B, 1920]

            for i, name in enumerate(class_names):
                cid = class_to_id[name]
                class_feats[cid].append(feat[i].cpu())

    empty_class_ids = [cid for cid, feats in class_feats.items() if not feats]
    if empty_class_ids:
        empty_class_names = [
            name for name, cid in class_to_id.items() if cid in empty_class_ids
        ]
        raise RuntimeError(
            "Class-mean prototype initialization requires at least one sample "
            f"for every class; empty classes: {empty_class_names}"
        )

    init_protos = []
    cluster_sizes = {}
    for cid in sorted(class_feats.keys()):
        class_stack = torch.stack(class_feats[cid], dim=0)
        centers, assignments = deterministic_spherical_prototypes(
            class_stack, prototype_mem.prototypes_per_class
        )
        init_protos.append(centers)
        cluster_sizes[cid] = [
            int((assignments == cluster_id).sum().item())
            for cluster_id in range(prototype_mem.prototypes_per_class)
        ]

    init_protos = F.normalize(torch.stack(init_protos, dim=0), dim=-1)  # [K, M, D]
    copy_value = init_protos[:, 0] if prototype_mem.prototypes_per_class == 1 else init_protos
    prototype_mem.prototypes.data.copy_(copy_value.to(device))

    # ========== Per-class manifold radius (class-aware adaptive radius) ==========
    class_radii = {}
    print("\nComputing class-specific manifold radii...")
    for cid in sorted(class_feats.keys()):
        # all features of the class [N, D]
        class_stack = torch.stack(class_feats[cid], dim=0)  # [N, D]
        class_stack = F.normalize(class_stack, dim=-1)

        # class prototype [D]
        proto = init_protos[cid]  # [M, D]

        # cosine distance of every feature to the prototype (1 - cosine similarity)
        cos_sim = (class_stack @ proto.T).max(dim=1).values  # [N]
        cos_dist = 1.0 - cos_sim  # [N]

        # mean cosine distance = manifold radius
        radius = cos_dist.mean().item()
        class_radii[cid] = radius

        class_name = [k for k, v in class_to_id.items() if v == cid][0]
        print(
            f"  Class {cid:2d} ({class_name:20s}): radius={radius:.4f}, "
            f"n_samples={len(class_feats[cid])}, cluster_sizes={cluster_sizes[cid]}"
        )

    # stored on prototype_mem (additional attribute)
    prototype_mem.class_radii = class_radii
    prototype_mem.prototype_cluster_sizes = cluster_sizes

    print(
        "Prototypes initialized with deterministic equal-capacity spherical "
        f"sets (M={prototype_mem.prototypes_per_class}) + adaptive radii."
    )
    return prototype_mem


@torch.no_grad()
def compute_text_prototypes(model, class_to_id, real_names, prompt_templates, device, dataset_name=None):
    """Text prototype of every class from multiple prompts.

    For each class the REAL_NAMES description is filled into several prompt templates;
    the mean of the encoded prompts is the text prototype of the class.
    
    Class-specific prompt groups (CLASS_PROMPT_GROUPS) are supported:
    - if dataset_name has an entry for the class in CLASS_PROMPT_GROUPS,
      only the listed prompt groups are used (e.g. strip_aware for MT_Crack);
    - normal samples (e.g. MT_Free) use only normal_sample + clip_style,
      so that they carry no defect semantics;
    - without an entry, the earlier behaviour applies (normal or defect templates).

    Args:
        model: AdaptedCLIP model
        class_to_id: {class_name: class_id}
        real_names: {class_name: detailed_description}
        prompt_templates: list[str] or dict of prompt groups
        device: torch device
        dataset_name: dataset name (optional, for dataset-specific logic)

    Returns:
        text_prototypes: [num_classes, 768] normalised text prototype of every class
    """
    from model.prompts_config import CLASS_PROMPT_GROUPS, DATASET_PROMPTS, PROMPTS

    num_classes = len(class_to_id)
    text_prototypes_list = []
    
    # are the prompt templates grouped (dict)?
    is_grouped = isinstance(prompt_templates, dict)
    
    # for a dict, collect the names of all available groups
    if is_grouped:
        group_map = prompt_templates  # {group_key: [templates]}

    for class_name, class_id in sorted(class_to_id.items(), key=lambda x: x[1]):
        # detailed description of the class
        desc = real_names.get(class_name, class_name)  # fallback to class_name
        
        # ==========================================================
        # class-specific prompt selection
        # ==========================================================
        # does CLASS_PROMPT_GROUPS have an entry for this class?
        class_group_keys = None
        if dataset_name and dataset_name in CLASS_PROMPT_GROUPS:
            cls_groups = CLASS_PROMPT_GROUPS[dataset_name]
            if class_name in cls_groups:
                class_group_keys = cls_groups[class_name]

        if class_group_keys is not None and is_grouped:
            # class-specific prompt groups
            selected_templates = []
            for key in class_group_keys:
                if key in group_map:
                    selected_templates.extend(group_map[key])
            desc_prefix = "CUSTOM"
        else:
            # fallback: normal or defect templates (earlier behaviour)
            if is_grouped:
                # merge all templates
                all_templates = []
                for key, templates in group_map.items():
                    if isinstance(templates, list):
                        all_templates.extend(templates)
                
                is_normal = 'Free' in class_name or 'Normal' in class_name or 'good' in class_name.lower()
                if is_normal and 'normal_sample' in group_map:
                    selected_templates = group_map['normal_sample']
                else:
                    selected_templates = all_templates
            else:
                selected_templates = prompt_templates
            desc_prefix = "NORMAL" if ('Free' in class_name or 'Normal' in class_name or 'good' in class_name.lower()) else "DEFECT"
        
        # prompts from the selected templates
        all_prompts = []
        for tmpl in selected_templates:
            all_prompts.append(tmpl.format(desc))
        
        # tokenize and encode
        tokens = tokenize(all_prompts).to(device)  # [N, 77]
        text_feats = model.encode_text(tokens)  # [N, 768], normalised
        
        # mean over prompts
        proto = text_feats.mean(dim=0)  # [768]
        proto = F.normalize(proto, dim=-1)  # normalise again to unit length
        text_prototypes_list.append(proto)
        
        # debug output (which groups were used)
        group_keys_str = str(class_group_keys) if class_group_keys else desc_prefix
        print(f"  [{class_name}] groups={group_keys_str} ({len(selected_templates)} templates)")

    text_prototypes = torch.stack(text_prototypes_list, dim=0)  # [K, 768]
    return text_prototypes



def _optimizer_named_parameters(model, prototype_mem, optimizer):
    names_by_id = {
        id(param): f"model.{name}" for name, param in model.named_parameters()
    }
    names_by_id.update(
        {
            id(param): f"prototype_mem.{name}"
            for name, param in prototype_mem.named_parameters()
        }
    )
    records = []
    for group_index, group in enumerate(optimizer.param_groups):
        for param in group["params"]:
            records.append(
                {
                    "name": names_by_id.get(id(param), f"unnamed.{id(param)}"),
                    "parameter": param,
                    "group": group_index,
                    "numel": param.numel(),
                }
            )
    return records


def _gradient_graph_audit(records):
    tensors = []
    module_squared_norms = {}
    for record in records:
        param = record["parameter"]
        grad = param.grad
        grad_norm = None if grad is None else float(grad.detach().float().norm())
        nonzero = False if grad is None else bool(torch.count_nonzero(grad).item())
        name = record["name"]
        parts = name.split(".")
        if "layer_adapters" in parts:
            adapter_index = parts.index("layer_adapters")
            module_name = ".".join(parts[: adapter_index + 2])
        elif len(parts) >= 3:
            module_name = ".".join(parts[:3])
        else:
            module_name = name
        if grad_norm is not None:
            module_squared_norms[module_name] = (
                module_squared_norms.get(module_name, 0.0) + grad_norm ** 2
            )
        tensors.append(
            {
                "name": name,
                "optimizer_group": record["group"],
                "numel": record["numel"],
                "grad_is_none": grad is None,
                "grad_nonzero": nonzero,
                "grad_norm": grad_norm,
            }
        )
    return {
        "optimizer_tensor_count": len(records),
        "optimizer_parameter_count": sum(r["numel"] for r in records),
        "grad_none_names": [t["name"] for t in tensors if t["grad_is_none"]],
        "zero_grad_names": [
            t["name"]
            for t in tensors
            if not t["grad_is_none"] and not t["grad_nonzero"]
        ],
        "nonzero_grad_names": [t["name"] for t in tensors if t["grad_nonzero"]],
        "module_grad_norms": {
            name: value ** 0.5 for name, value in module_squared_norms.items()
        },
        "tensors": tensors,
    }


def balanced_normal_aware_bce(logits, multiclass_labels, good_id, num_classes):
    """Class-balanced good/defect BCE without changing the data sampler.

    Cable has one good class and eight defect classes with equal shots per
    class.  Weighting good by K/2 and every defect sample by K/(2*(K-1)) gives
    both binary groups equal total weight in expectation while preserving the
    exact multiclass sampling trajectory.
    """
    logits = logits.reshape(-1)
    multiclass_labels = multiclass_labels.reshape(-1)
    if logits.shape[0] != multiclass_labels.shape[0]:
        raise ValueError("normal-aware logits and labels must have equal length")
    if not 0 <= int(good_id) < int(num_classes) or int(num_classes) < 2:
        raise ValueError("normal-aware BCE requires one valid good class and a defect class")
    targets = (multiclass_labels != int(good_id)).to(logits.dtype)
    good_weight = float(num_classes) / 2.0
    defect_weight = float(num_classes) / (2.0 * float(num_classes - 1))
    weights = torch.where(
        targets > 0,
        torch.full_like(targets, defect_weight),
        torch.full_like(targets, good_weight),
    )
    per_sample = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (per_sample * weights).mean()


def train_multiclass(
    model: nn.Module,
    prototype_mem: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    num_epochs: int,
    class_to_id: dict,
    save_path: str,
    logger: logging.Logger,
    supcon_weight: float = 0.1,
    orth_weight: float = 0.05,
    itc_weight: float = 0.0,
    text_prototypes: torch.Tensor = None,
    gradient_accumulation_steps: int = 4,
    save_every: int = 10,
    use_text_guided: bool = False,
    cross_modal_mode: str = 'all',
    repeat_per_epoch: int = 1,
    early_stop_patience: int = 0,
    arc_margin: float = 0.0,
    label_smoothing: float = 0.1,
    gate_type: str = 'cosine',
    checkpoint_format: str = 'lightweight',
    save_optimizer_state: bool = False,
    run_config: dict = None,
    manifest_path: str = None,
    normal_aware_loss_weight: float = 0.0,
    prototype_balance_weight: float = 0.0,
    prototype_assignment_temperature: float = 0.2,
):

    """Main training loop for multi-class classification.

    Args:
        use_text_guided: use text-refined features instead of mean-pooled CLS tokens
        cross_modal_mode: path mode of the text guidance (all/direct/strip/scale/ds/dsc/ss)
        text_prototypes: [K, 768] normalised multi-prompt text prototypes, used by the ITC loss and as text queries
    """
    optimizer_records = _optimizer_named_parameters(
        model, prototype_mem, optimizer
    )
    optimizer_parameter_names = [record["name"] for record in optimizer_records]
    first_step_snapshot = {
        id(record["parameter"]): record["parameter"].detach().cpu().clone()
        for record in optimizer_records
    }
    first_backward_audit = None
    first_update_recorded = False
    optimizer_message = (
        f"Gradient audit optimizer parameters ({len(optimizer_parameter_names)} tensors): "
        f"{optimizer_parameter_names}"
    )
    logger.info(optimizer_message)
    print(optimizer_message)
    if manifest_path:
        update_training_manifest(
            manifest_path,
            {
                "gradient_graph_audit": {
                    "status": "awaiting_first_backward",
                    "adapter_train_mode": getattr(
                        model, "adapter_train_mode", "unknown"
                    ),
                    "optimizer_parameter_names": optimizer_parameter_names,
                }
            },
        )

    supcon_loss_fn = SupConLoss(temperature=0.07).to(device)
    
    # text queries per batch (the text prototypes indexed by class);
    # during training the text feature of each label serves as its text query
    text_query_cache = text_prototypes  # [K, 768], normalised
    
    # early stopping
    best_cls_loss = float('inf')
    best_epoch = 0
    best_state = None
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print(f"{'='*60}")
        logger.info(f"Epoch {epoch}/{num_epochs - 1}")

        model.train()
        prototype_mem.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        loss_cls_list = []
        loss_supcon_list = []
        loss_orth_list = []
        loss_itc_list = []
        loss_normal_aware_list = []
        loss_prototype_balance_list = []
        loss_total_list = []
        max_sim_list = []
        track_prototype_usage = prototype_balance_weight > 0
        if track_prototype_usage:
            epoch_prototype_hard_counts = torch.zeros(
                prototype_mem.num_classes,
                prototype_mem.prototypes_per_class,
                dtype=torch.long,
            )
            epoch_prototype_soft_mass = torch.zeros(
                prototype_mem.num_classes,
                prototype_mem.prototypes_per_class,
                dtype=torch.float64,
            )
            epoch_class_sample_counts = torch.zeros(
                prototype_mem.num_classes, dtype=torch.long
            )

        # reset the gradient-accumulation state
        optimizer.zero_grad()
        accumulated_loss = 0.0

        # repeat_per_epoch: several passes over the few-shot data per epoch
        for repeat_idx in range(repeat_per_epoch):
            seed_context = configure_loader_seed_context(
                train_loader, "train", epoch=epoch, repeat=repeat_idx
            )
            seed_message = f"Seed context: {seed_context}"
            logger.info(seed_message)
            print(seed_message)
            desc = f"Epoch {epoch}" + (f" (r{repeat_idx+1}/{repeat_per_epoch})" if repeat_per_epoch > 1 else "")
            for batch_idx, input_data in enumerate(tqdm(train_loader, desc=desc)):
                images = input_data["image"].to(
                    device, non_blocking=device.type == "cuda"
                )
                class_names = input_data["class_name"]

                # class names -> integer labels
                labels = torch.tensor(
                    [class_to_id[name] for name in class_names],
                    dtype=torch.long,
                    device=device,
                )

                # ============================================================
                # feature extraction: unified K-way training and inference
                # ============================================================
                if use_text_guided and text_query_cache is not None and model.proto_text_refine == 'none':
                    # 1. feed the texts of all K classes and obtain a [B, K, 1920] feature tensor,
                    # so that the model also sees the wrong texts during training and learns to reject them
                    vis_feat_kway, purified_kway, kway_aux_list = model.forward_hybrid_features_kway(
                        images, text_prototypes=text_query_cache, mode=cross_modal_mode
                    )

                    # 2. K-way classification logits
                    # the K prototypes of the prototype memory [K, 1920]
                    protos = F.normalize(prototype_mem.prototypes, dim=-1)

                    # the visual feature guided by text k must match prototype k only:
                    # element-wise product of [B, K, 1920] and [1, K, 1920] summed over the last axis -> [B, K] diagonal scores
                    cos_sim = (vis_feat_kway * protos.unsqueeze(0)).sum(dim=-1)

                    logit_scale = prototype_mem.logit_scale.exp()
                    vis_logits = cos_sim * logit_scale

                    # 3. the feature of the ground-truth text (oracle) for the SupCon loss
                    batch_indices = torch.arange(images.size(0), device=device)
                    vis_feat = vis_feat_kway[batch_indices, labels]       # [B, 1920]
                    clip_purified = purified_kway[batch_indices, labels]  # [B, 768]
                elif model.proto_text_refine != 'none' and text_query_cache is not None:
                    # text refinement outside the visual features (the visual features are never modified by text)
                    fusion_feat = model.forward_fusion(images)  # [B, 1920], visual only
                    vis_feat = fusion_feat
                    clip_purified = fusion_feat[:, :768]  # [B, 768]

                    if model.proto_text_refine == 'logit_boost':
                        # temperature softmax in logit space (winner-take-all)
                        vis_logits = prototype_mem(vis_feat)  # [B, K], visual logits
                        logit_scale = prototype_mem.logit_scale.exp()
                        vis_logits = model.compute_text_boosted_logits(
                            vis_logits, text_query_cache, logit_scale
                        )
                    elif model.proto_text_refine == 'proto_refine':
                        # prototype refinement in feature space, computed inline (no set/clear);
                        # refined_protos stay in the graph, so text_proto_proj receives gradients
                        refined_protos = model.refine_prototypes_with_text(
                            prototype_mem.prototypes, text_query_cache
                        )  # [K, 1920], L2-normalised
                        clip_feat_norm = F.normalize(vis_feat, dim=-1)  # [B, 1920]
                        logit_scale = prototype_mem.logit_scale.exp()
                        vis_logits = logit_scale * (clip_feat_norm @ refined_protos.T)  # [B, K]
                else:
                    # base mode: forward_fusion returns [B, 1920]
                    fusion_feat = model.forward_fusion(images)  # [B, 1920]
                    vis_feat = fusion_feat  # [B, 1920]
                    clip_purified = fusion_feat[:, :768]  # [B, 768] clip_avg, for the ITC loss

                    # base mode: logits from the prototype memory directly
                    vis_logits = prototype_mem(vis_feat)  # [B, num_classes]

                loss_prototype_balance = torch.tensor(0.0, device=device)
                if track_prototype_usage:
                    loss_prototype_balance = prototype_mem.prototype_balance_loss(
                        vis_feat,
                        labels,
                        assignment_temperature=prototype_assignment_temperature,
                    )
                    with torch.no_grad():
                        responsibilities = prototype_mem.true_class_responsibilities(
                            vis_feat,
                            labels,
                            assignment_temperature=prototype_assignment_temperature,
                        )
                        winners = responsibilities.argmax(dim=-1)
                        labels_cpu = labels.detach().cpu()
                        winners_cpu = winners.detach().cpu()
                        responsibilities_cpu = responsibilities.detach().double().cpu()
                        for sample_index in range(labels_cpu.numel()):
                            class_id = int(labels_cpu[sample_index])
                            winner_id = int(winners_cpu[sample_index])
                            epoch_prototype_hard_counts[class_id, winner_id] += 1
                            epoch_prototype_soft_mass[class_id] += responsibilities_cpu[
                                sample_index
                            ]
                            epoch_class_sample_counts[class_id] += 1

                loss_normal_aware = torch.tensor(0.0, device=device)
                if normal_aware_loss_weight > 0:
                    good_id = class_to_id.get("good")
                    if good_id is None:
                        raise ValueError(
                            "--normal_aware_head requires a class named 'good'"
                        )
                    binary_logits = model.forward_normal_aware(vis_feat)
                    loss_normal_aware = balanced_normal_aware_bce(
                        binary_logits, labels, good_id, len(class_to_id)
                    )

                # 3. losses (scaled for gradient accumulation)
                if arc_margin > 0:
                    # --- asymmetric margin penalty ---
                    # the margin applies to defect classes only, not to the good class
                    scale = prototype_mem.logit_scale.exp()
                    cos_sim = vis_logits / scale
                    # per-sample margin: arc_margin for defects, 0 for good
                    margin_tensor = torch.full_like(labels, arc_margin, dtype=torch.float)
                    good_id = class_to_id.get("good", -1)
                    margin_tensor[labels == good_id] = 0.0
                    # subtract the margin from the target-class cosine
                    target_cos = cos_sim[torch.arange(cos_sim.size(0)), labels]
                    cos_sim[torch.arange(cos_sim.size(0)), labels] = target_cos - margin_tensor
                    # rescale and compute the cross-entropy
                    margin_logits = cos_sim * scale
                    loss_cls = F.cross_entropy(margin_logits, labels, label_smoothing=label_smoothing)
                else:
                    loss_cls = F.cross_entropy(vis_logits, labels, label_smoothing=label_smoothing)
                loss_supcon = supcon_loss_fn(vis_feat, labels, exclude_labels=[class_to_id.get("good", -1)])
                loss_orth = prototype_mem.orthogonal_loss()
                total_loss = loss_cls + supcon_weight * loss_supcon + orth_weight * loss_orth
                total_loss = total_loss + normal_aware_loss_weight * loss_normal_aware
                if track_prototype_usage:
                    total_loss = total_loss + prototype_balance_weight * loss_prototype_balance

                # 3b. image-text contrastive (ITC) loss
                # purified_kway is detached before the ITC loss:
                # clip_patches/clip_cls are already detached and the classification gradient is scaled by gamma,
                # so the ITC loss would otherwise be the only gradient source of text_guided_fusion
                # and would push it towards features that fit the text but leave the visual manifold.
                # The ITC loss is therefore only monitored and sends no gradient to text_guided_fusion;
                # the text module is trained by the classification loss alone (through gamma).
                loss_itc = torch.tensor(0.0, device=device)
                if itc_weight > 0 and text_prototypes is not None:
                    logit_scale = prototype_mem.logit_scale.exp()

                    if use_text_guided:
                        # K-way ITC loss; detach blocks the gradient into text_guided_fusion
                        purified_detached = purified_kway.detach()
                        itc_cos_sim = (purified_detached * text_prototypes.unsqueeze(0)).sum(dim=-1)
                        itc_logits = logit_scale * itc_cos_sim
                        loss_itc = F.cross_entropy(itc_logits, labels)
                    else:
                        clip_feat = clip_purified  # [B, 768]
                        itc_logits = logit_scale * clip_feat @ text_prototypes.T  # [B, K]
                        loss_itc = F.cross_entropy(itc_logits, labels)

                    total_loss = total_loss + itc_weight * loss_itc

                # scale for gradient accumulation
                total_loss = total_loss / gradient_accumulation_steps

                # 4. backward pass (accumulated gradients)
                total_loss.backward()
                accumulated_loss += total_loss.item()

                if first_backward_audit is None:
                    first_backward_audit = _gradient_graph_audit(optimizer_records)
                    first_backward_audit.update(
                        {
                            "status": "first_backward_complete",
                            "adapter_train_mode": getattr(
                                model, "adapter_train_mode", "unknown"
                            ),
                            "epoch": epoch,
                            "repeat": repeat_idx,
                            "batch": batch_idx,
                        }
                    )
                    audit_message = (
                        "First-backward gradient audit: "
                        f"nonzero={first_backward_audit['nonzero_grad_names']} "
                        f"grad_none={first_backward_audit['grad_none_names']} "
                        f"zero_grad={first_backward_audit['zero_grad_names']} "
                        f"module_norms={first_backward_audit['module_grad_norms']}"
                    )
                    logger.info(audit_message)
                    print(audit_message)
                    if manifest_path:
                        update_training_manifest(
                            manifest_path,
                            {"gradient_graph_audit": first_backward_audit},
                        )
                    if (
                        getattr(model, "adapter_train_mode", None)
                        != "legacy_random_fixed"
                        and (
                            first_backward_audit["grad_none_names"]
                            or first_backward_audit["zero_grad_names"]
                        )
                    ):
                        raise RuntimeError(
                            "clean adapter graph contains optimizer parameters "
                            "without a nonzero first-backward gradient: "
                            + ", ".join(
                                first_backward_audit["grad_none_names"]
                                + first_backward_audit["zero_grad_names"]
                            )
                        )

                # update the parameters every gradient_accumulation_steps steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(prototype_mem.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    if not first_update_recorded:
                        updated = []
                        updated_parameter_count = 0
                        for record in optimizer_records:
                            before = first_step_snapshot[id(record["parameter"])]
                            after = record["parameter"].detach().cpu()
                            if not torch.equal(before, after):
                                delta = float((after.float() - before.float()).abs().max())
                                updated.append(
                                    {
                                        "name": record["name"],
                                        "numel": record["numel"],
                                        "max_abs_delta": delta,
                                    }
                                )
                                updated_parameter_count += record["numel"]
                        first_backward_audit["status"] = "first_update_complete"
                        first_backward_audit["updated_tensors"] = updated
                        first_backward_audit["actual_updated_tensor_count"] = len(updated)
                        first_backward_audit[
                            "actual_updated_parameter_count"
                        ] = updated_parameter_count
                        update_message = (
                            "First optimizer update audit: "
                            f"updated={[item['name'] for item in updated]} "
                            f"actual_updated_parameters={updated_parameter_count}"
                        )
                        logger.info(update_message)
                        print(update_message)
                        if manifest_path:
                            update_training_manifest(
                                manifest_path,
                                {"gradient_graph_audit": first_backward_audit},
                            )
                        first_update_recorded = True
                        first_step_snapshot.clear()
                    optimizer.zero_grad()
                    accumulated_loss = 0.0

                # 5. logging
                loss_cls_list.append(loss_cls.item())
                loss_supcon_list.append(loss_supcon.item())
                loss_orth_list.append(loss_orth.item())
                loss_itc_list.append(loss_itc.item())
                loss_normal_aware_list.append(loss_normal_aware.item())
                loss_prototype_balance_list.append(loss_prototype_balance.item())
                loss_total_list.append(total_loss.item())
                max_sim = compute_max_proto_similarity(prototype_mem)
                max_sim_list.append(max_sim)

                # print the losses of every batch
                mode_tag = "TG" if use_text_guided else "CLS"
                prototype_balance_info = (
                    f"proto_balance={loss_prototype_balance.item():.4f} | "
                    if track_prototype_usage
                    else ""
                )
                if itc_weight > 0:
                    msg = (
                        f"  [{mode_tag}] Batch {batch_idx + 1:2d}/{len(train_loader):2d} | "
                        f"cls={loss_cls.item():.4f} | "
                        f"supcon={loss_supcon.item():.4f} | "
                        f"orth={loss_orth.item():.4f} | "
                        f"normal={loss_normal_aware.item():.4f} | "
                        f"{prototype_balance_info}"
                        f"itc={loss_itc.item():.4f} | "
                        f"total={total_loss.item():.4f} | "
                        f"sim={max_sim:.4f}"
                    )
                else:
                    msg = (
                        f"  [{mode_tag}] Batch {batch_idx + 1:2d}/{len(train_loader):2d} | "
                        f"cls={loss_cls.item():.4f} | "
                        f"supcon={loss_supcon.item():.4f} | "
                        f"orth={loss_orth.item():.4f} | "
                        f"normal={loss_normal_aware.item():.4f} | "
                        f"{prototype_balance_info}"
                        f"total={total_loss.item():.4f} | "
                        f"sim={max_sim:.4f}"
                    )
                logger.info(msg)
                print(msg)

        # scheduler steps once per epoch
        scheduler.step()

        # epoch summary
        avg_loss_cls = np.mean(loss_cls_list)
        avg_loss_supcon = np.mean(loss_supcon_list)
        avg_loss_orth = np.mean(loss_orth_list)
        avg_loss_normal_aware = np.mean(loss_normal_aware_list)
        avg_loss_prototype_balance = np.mean(loss_prototype_balance_list)
        avg_loss_total = np.mean(loss_total_list)
        avg_max_sim = np.mean(max_sim_list)
        prototype_balance_summary = ""
        if track_prototype_usage:
            prototype_mem.prototype_training_usage = {
                "epoch": int(epoch),
                "assignment_temperature": float(prototype_assignment_temperature),
                "hard_counts": epoch_prototype_hard_counts.tolist(),
                "soft_mass": epoch_prototype_soft_mass.tolist(),
                "class_sample_counts": epoch_class_sample_counts.tolist(),
            }
            starved_slots = int((epoch_prototype_hard_counts == 0).sum().item())
            prototype_balance_summary = (
                f"avg_proto_balance={avg_loss_prototype_balance:.4f} | "
                f"proto_starved_slots={starved_slots} | "
            )

        # log the residual gate (gamma) to monitor how far the gate opens;
        # skipped in the text-refinement modes, which have no residual_gate
        gamma_val = model.residual_gate.item() if hasattr(model, 'residual_gate') else 0.0

        # log the mean path gate of the cosine gate, to monitor whether the gates open
        gate_info = ""
        if use_text_guided and gate_type == 'cosine':
            # path_gate from kway_aux_list of the last batch;
            # kway_aux_list: list of K aux dicts, each with aux['path_gate'] = [B, 3]
            try:
                last_aux = kway_aux_list  # from the training loop
                # aux of the ground-truth class
                if isinstance(last_aux, list) and len(last_aux) > 0:
                    gt_aux = last_aux[labels[0].item() if labels.numel() > 0 else 0]
                    if 'path_gate' in gt_aux:
                        pg = gt_aux['path_gate'].detach()  # [B, 3]
                        gate_info = f" | gate_mean={pg.mean().item():.4f} gate_max={pg.max().item():.4f}"
            except (NameError, IndexError, KeyError):
                pass

        summary_msg = (
            f">>> Epoch {epoch} Summary: "
            f"avg_cls={avg_loss_cls:.4f} | "
            f"avg_supcon={avg_loss_supcon:.4f} | "
            f"avg_orth={avg_loss_orth:.4f} | "
            f"avg_normal={avg_loss_normal_aware:.4f} | "
            f"{prototype_balance_summary}"
            f"avg_total={avg_loss_total:.4f} | "
            f"avg_sim={avg_max_sim:.4f} | "
            f"gamma={gamma_val:.6f}"
            f"{gate_info}"
        )
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            summary_msg += (
                f" | cuda_peak_alloc={peak_allocated:.2f}GiB"
                f" cuda_peak_reserved={peak_reserved:.2f}GiB"
            )
        # text-refinement modes: log their parameters
        if model.proto_text_refine == 'logit_boost':
            beta_val = model.text_logit_gate.item()
            tau_val = model.text_temp.item()
            summary_msg += f" | beta={beta_val:.6f} | tau={tau_val:.6f}"
        elif model.proto_text_refine == 'proto_refine':
            beta_val = model.proto_text_gate.item()
            summary_msg += f" | beta={beta_val:.6f}"
        logger.info(summary_msg)
        print(summary_msg)

        # early stopping
        if early_stop_patience > 0:
            if avg_loss_cls < best_cls_loss - 1e-4:
                best_cls_loss = avg_loss_cls
                best_epoch = epoch
                best_state = build_training_checkpoint(
                    model=model,
                    prototype_mem=prototype_mem,
                    optimizer=optimizer if save_optimizer_state else None,
                    scheduler=scheduler if save_optimizer_state else None,
                    checkpoint_format=checkpoint_format,
                    metadata={
                        "epoch": epoch + 1,
                        "class_to_id": class_to_id,
                        "avg_loss_total": avg_loss_total,
                        "use_text_guided": use_text_guided,
                        "cross_modal_mode": cross_modal_mode,
                        "run_config": run_config or {},
                    },
                )
                patience_counter = 0
            else:
                patience_counter += 1
                msg = f"  Early stopping: no cls loss improvement for {patience_counter}/{early_stop_patience} epochs (best={best_cls_loss:.4f} @ epoch {best_epoch})"
                logger.info(msg)
                print(msg)
                if patience_counter >= early_stop_patience:
                    # save the best checkpoint
                    ckp_path = os.path.join(save_path, f"multiclass_epoch_{best_epoch}.pth")
                    torch.save(best_state, ckp_path)
                    if manifest_path:
                        record_artifact(
                            manifest_path,
                            ckp_path,
                            kind="checkpoint",
                            metadata={"epoch": best_epoch + 1, "selected_by": "early_stop"},
                        )
                    msg = f"  Early stopping triggered! Best cls loss: {best_cls_loss:.4f} @ epoch {best_epoch}, saved to {ckp_path}"
                    logger.info(msg)
                    print(msg)
                    # build and save the FAISS index before stopping
                    logger.info("Building FAISS index (early stop)...")
                    if model.proto_text_refine == 'proto_refine' and text_query_cache is not None:
                        prototype_mem.build_faiss_index(
                            use_text_refined=True, model=model, text_prototypes=text_query_cache
                        )
                    else:
                        prototype_mem.build_faiss_index()
                    faiss_path = os.path.join(save_path, "faiss_index.pth")
                    torch.save(
                        {
                            "prototype_mem_state_dict": prototype_mem.state_dict(),
                            "faiss_index": prototype_mem.faiss_index,
                            "class_to_id": class_to_id,
                        },
                        faiss_path,
                    )
                    if manifest_path:
                        record_artifact(manifest_path, faiss_path, kind="faiss_index")
                    logger.info(f"FAISS index saved to {faiss_path}")
                    break

        # save a checkpoint every save_every epochs and after the last epoch
        is_last_epoch = (epoch == num_epochs - 1)
        should_save = is_last_epoch or (
            save_every > 0 and (epoch + 1) % save_every == 0
        )
        if should_save:
            ckp_path = os.path.join(save_path, f"multiclass_epoch_{epoch}.pth")
            checkpoint = build_training_checkpoint(
                model=model,
                prototype_mem=prototype_mem,
                optimizer=optimizer if save_optimizer_state else None,
                scheduler=scheduler if save_optimizer_state else None,
                checkpoint_format=checkpoint_format,
                metadata={
                    "epoch": epoch + 1,
                    "class_to_id": class_to_id,
                    "avg_loss_total": avg_loss_total,
                    "use_text_guided": use_text_guided,  # training mode
                    "cross_modal_mode": cross_modal_mode,
                    "run_config": run_config or {},
                },
            )
            torch.save(checkpoint, ckp_path)
            size_mb = os.path.getsize(ckp_path) / (1024 ** 2)
            checkpoint_digest = checkpoint_sha256(ckp_path)
            logger.info(
                "Checkpoint saved to %s (format=%s, %.1f MiB, sha256=%s)",
                ckp_path,
                checkpoint["checkpoint_format"],
                size_mb,
                checkpoint_digest,
            )
            print(
                f"Checkpoint saved: {ckp_path} "
                f"({checkpoint['checkpoint_format']}, {size_mb:.1f} MiB, "
                f"sha256={checkpoint_digest[:16]}...)"
            )
            if manifest_path:
                record_artifact(
                    manifest_path,
                    ckp_path,
                    kind="checkpoint",
                    metadata={
                        "epoch": epoch + 1,
                        "checkpoint_format": checkpoint["checkpoint_format"],
                    },
                )

    # build the FAISS index after training
    logger.info("Building FAISS index...")
    if model.proto_text_refine == 'proto_refine' and text_query_cache is not None:
        prototype_mem.build_faiss_index(
            use_text_refined=True, model=model, text_prototypes=text_query_cache
        )
    else:
        prototype_mem.build_faiss_index()
    faiss_path = os.path.join(save_path, "faiss_index.pth")
    torch.save(
        {
            "prototype_mem_state_dict": prototype_mem.state_dict(),
            "faiss_index": prototype_mem.faiss_index,
            "class_to_id": class_to_id,
        },
        faiss_path,
    )
    if manifest_path:
        record_artifact(manifest_path, faiss_path, kind="faiss_index")
    logger.info(f"FAISS index saved to {faiss_path}")

    return model, prototype_mem


@torch.no_grad()
def compute_max_proto_similarity(prototype_mem):
    """Largest similarity between prototypes (monitors prototype collapse)."""
    protos = prototype_mem.class_representatives()
    sim = protos @ protos.T
    mask = ~torch.eye(protos.shape[0], device=sim.device).bool()
    max_sim = sim[mask].max().item()
    return max_sim


def _active_text_guided_parameter(name, args):
    prefix = "text_guided_fusion."
    if not name.startswith(prefix):
        return False
    relative = name[len(prefix):]
    mode = args.cross_modal_mode
    direct_active = mode in {"all", "direct", "ds", "dsc"}
    strip_active = mode in {"all", "strip", "ds", "ss"}
    scale_active = mode in {"all", "scale", "dsc", "ss"}
    dual_mode = mode in {"ds", "dsc", "ss"}

    is_gate = False
    if strip_active and relative.startswith("strip_path.hv_weights"):
        is_gate = True
    if scale_active and relative.startswith("scale_path.scale_weights"):
        is_gate = True
    if mode == "all":
        if args.gate_type == "cosine" and relative.startswith(
            ("path_scales", "path_biases")
        ):
            is_gate = True
        elif args.gate_type == "scalar" and relative.startswith("path_weights"):
            is_gate = True
        elif args.gate_type == "heavy" and relative.startswith("path_gate."):
            is_gate = True
    elif dual_mode:
        if args.gate_type == "cosine" and relative.startswith(
            ("path_scales", "path_biases")
        ):
            is_gate = True
        elif args.gate_type == "scalar" and relative.startswith("dual_weights"):
            is_gate = True
        elif args.gate_type == "heavy" and relative.startswith("dual_path_gate."):
            is_gate = True

    if args.freeze_cross_modal:
        return is_gate
    return (
        relative.startswith(("final_proj.", "final_norm."))
        or (direct_active and relative.startswith("direct_attn."))
        or (strip_active and relative.startswith("strip_path."))
        or (scale_active and relative.startswith("scale_path."))
        or is_gate
    )


def configure_model_trainable_parameters(model, args):
    """Make optimizer membership match the selected forward gradient graph."""
    if is_legacy_training_protocol(args.training_protocol):
        # Exact historical selection. Several of these tensors intentionally
        # receive grad=None because the old graph was cut by no_grad().
        for name, param in model.named_parameters():
            if "clipmodel" in name or "dino_extractor" in name:
                param.requires_grad = False
            elif args.freeze_cross_modal and "text_guided_fusion" in name:
                param.requires_grad = any(
                    token in name
                    for token in (
                        "path_weights", "dual_weight", "hv_weight", "scale_weights"
                    )
                )
            elif not args.use_text_guided and any(
                token in name
                for token in (
                    "text_guided_fusion", "text_adapter", "clip_patch_proj",
                    "patch_fusion_proj", "residual_gate",
                )
            ):
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            selected = name.startswith("image_adapter.det_proj.")
            if args.adapter_train_mode == "learned_early":
                selected = selected or name.startswith(
                    "image_adapter.layer_adapters."
                )

            if args.use_text_guided and args.proto_text_refine == "none":
                selected = selected or any(
                    name.startswith(prefix)
                    for prefix in (
                        "clip_patch_proj.",
                        "patch_fusion_proj.",
                    )
                ) or name == "residual_gate" or _active_text_guided_parameter(
                    name, args
                )
            elif args.proto_text_refine == "logit_boost":
                selected = selected or name in {"text_temp", "text_logit_gate"}
            elif args.proto_text_refine == "proto_refine":
                selected = (
                    selected
                    or name.startswith("text_proto_proj.")
                    or name == "proto_text_gate"
                )
            if getattr(args, "normal_aware_head", False):
                selected = selected or name.startswith("normal_aware_head.")
            param.requires_grad = selected

    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not any(name.startswith("image_adapter.det_proj.") for name in names):
        raise RuntimeError("det_proj is missing from the trainable parameter graph")
    if any(name.startswith("image_adapter.seg_proj.") for name in names) and (
        not is_legacy_training_protocol(args.training_protocol)
    ):
        raise RuntimeError("seg_proj must not be optimized by clean_v1")
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Multi-class classification training with DINOv2 + FaissPrototypeMemory"
    )
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--dataset", "--data", type=str, default="MT_Dataset")
    parser.add_argument("--training_mode", type=str, default="few_shot",
                        choices=["few_shot", "full_shot"])
    parser.add_argument("--shot", type=int, default=8)
    parser.add_argument(
        "--train_meta_path",
        type=str,
        default=None,
        help=(
            "Explicit training JSONL metadata. This is used by the auditable "
            "OOF protocol; when omitted, the historical shot-based path is unchanged."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size. Use gradient_accumulation_steps to increase effective batch size.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers. Linux server profile uses 4 per process.")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Batches prefetched by each DataLoader worker.")
    parser.add_argument("--pin_memory", action="store_true",
                        help="Use page-locked host memory for asynchronous CUDA transfers.")
    parser.add_argument("--persistent_workers", action="store_true",
                        help="Keep DataLoader workers alive across repeat_per_epoch passes.")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 CUDA matrix multiplication for faster fp32 training.")
    parser.add_argument("--dino_backend", choices=["auto", "hub", "transformers"],
                        default="auto", help="DINOv2 loader backend; transformers avoids torch.hub network checks.")
    parser.add_argument(
        "--training_protocol",
        choices=["clean_v1", "legacy_9359", "legacy_seed_split_diag"],
        default="clean_v1",
        help=(
            "clean_v1 uses independent RNG streams, zero prototype construction, "
            "and an explicit adapter graph. legacy_9359 preserves the historical "
            "93.59/93.67 training semantics for reproduction only. "
            "legacy_seed_split_diag keeps the legacy graph/loader but resets the "
            "global RNG after model construction to diagnose initialization versus "
            "training-trajectory effects."
        ),
    )
    parser.add_argument(
        "--rng_protocol",
        choices=["isolated_rng", "legacy_rng"],
        default=None,
        help=(
            "isolated_rng prevents DINO backend construction from changing later "
            "adapter initialization. legacy_rng is only for historical reproduction."
        ),
    )
    parser.add_argument(
        "--adapter_train_mode",
        choices=[
            "legacy_random_fixed", "terminal_only", "frozen_random_early",
            "learned_early",
        ],
        default=None,
        help=(
            "legacy_random_fixed reproduces the historical graph cut; "
            "terminal_only disables early adapters and trains det_proj; "
            "frozen_random_early applies isolated-RNG early adapters as a "
            "frozen feature map under clean_v1; "
            "learned_early retains autograd through frozen CLIP blocks."
        ),
    )
    parser.add_argument(
        "--early_adapter_source_checkpoint",
        type=str,
        default=None,
        help=(
            "P1.10 diagnostic only: after construction, replace exactly "
            "image_adapter.layer_adapters from this checkpoint while retaining "
            "all other initialization tensors. Requires legacy_seed_split_diag."
        ),
    )
    parser.add_argument(
        "--dino_patch_policy",
        choices=["full", "legacy_drop_first"],
        default="full",
        help=(
            "DINO spatial-token protocol. legacy_drop_first reproduces the "
            "historical 575-token implementation and is diagnostic only."
        ),
    )
    parser.add_argument(
        "--dino_hub_repo",
        type=str,
        default=None,
        help="Offline local DINOv2 repository containing hubconf.py.",
    )
    parser.add_argument(
        "--dino_hub_checkpoint",
        type=str,
        default=None,
        help="Offline local dinov2_vits14_pretrain.pth path.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--repeat_per_epoch", type=int, default=1,
                        help="Repeat data loading N times per epoch. For few-shot (e.g. 8-shot), "
                             "set to 4-8 to get more gradient steps per epoch without increasing epochs.")
    parser.add_argument("--num_init_views", type=int, default=5,
                        help="Multi-View sampling times for prototype initialization. "
                             "8-shot x 5 views = 40 feature points per class, reducing init variance. "
                             "Set to 1 to disable (original single-view behavior).")

    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--supcon_weight", type=float, default=0.0)
    parser.add_argument("--orth_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--init_seed", type=int, default=None,
        help="Model/parameter initialization seed; defaults to --seed.",
    )
    parser.add_argument(
        "--data_seed", type=int, default=None,
        help="Sampler and DataLoader ordering seed; defaults to --seed.",
    )
    parser.add_argument(
        "--augment_seed", type=int, default=None,
        help="Per-sample augmentation seed stream; defaults to --seed.",
    )
    parser.add_argument("--itc_weight", type=float, default=0.0,
                        help="Image-Text Contrastive loss weight (0=disable ITC). Uses multi-prompt fused text prototypes from REAL_NAMES + PROMPTS.")
    parser.add_argument(
        "--init_logit_scale",
        type=float,
        default=None,
        help=(
            "Initial prototype logit scale. Defaults to historical 100 for "
            "legacy_9359 and gradient-active 10 for clean_v1."
        ),
    )
    parser.add_argument("--save_path", type=str, default="ckpt/multiclass")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs (0=only save last + faiss). Default 10 saves disk space.")
    parser.add_argument(
        "--checkpoint_format",
        choices=["lightweight", "full"],
        default="lightweight",
        help=(
            "lightweight excludes frozen CLIP/DINO weights and is recommended for remote runs; "
            "full keeps the legacy complete model state."
        ),
    )
    parser.add_argument(
        "--save_optimizer_state",
        action="store_true",
        help=(
            "Also save optimizer/scheduler state for exact resume. Disabled by default "
            "to minimize ablation checkpoint size."
        ),
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--normal_aware_head",
        action="store_true",
        help=(
            "P2.3: train a zero-initialized binary good/defect head on the "
            "terminal fused feature. Requires clean_v1 terminal_only."
        ),
    )
    parser.add_argument(
        "--normal_aware_loss_weight",
        type=float,
        default=0.0,
        help="P2.3 balanced binary auxiliary-loss weight; 0 keeps the historical graph.",
    )
    parser.add_argument(
        "--normal_aware_head_lr_mult",
        type=float,
        default=10.0,
        help="Learning-rate multiplier for the 1,921-parameter binary head.",
    )
    parser.add_argument(
        "--normal_aware_head_threshold",
        type=float,
        default=0.0,
        help="Fixed binary-logit decision threshold. P2.3 preregisters 0.0.",
    )
    parser.add_argument("--freeze_cross_modal", action="store_true",
                        help="Freeze cross_modal projection layers, only train scalar gates + adapter + prototypes. "
                             "Reduces trainable params from ~32M to ~14M for few-shot.")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop training if cls loss doesn't improve for N epochs (0=disabled). "
                             "Recommended: 5 for few-shot to prevent overfitting.")
    parser.add_argument("--proto_lr_mult", type=float, default=3.0,
                        help="Prototype learning rate multiplier relative to --lr. "
                             "E.g. --lr 5e-5 --proto_lr_mult 5 → prototype lr=2.5e-4")

    # ================================================================
    # Cross-modal purification mode
    # ================================================================
    # Selects the paths that forward_text_guided() uses to aggregate the text-refined visual features.
    # 
    # Modes:
    #   all     - default; all 3 paths with learnable gated fusion
    #   direct  - direct cross-attention only (text query x all patches)
    #   strip   - strip path only (horizontal + vertical strips; thin elongated scratches)
    #   scale   - scale path only (multi-scale; defects of different sizes)
    #   ds      - Direct + Strip
    #   dsc     - Direct + Scale
    #   ss      - Strip + Scale
    #   none    - plain mean pooling (no text guidance; ablation control)
    # ================================================================
    parser.add_argument(
        "--prototypes_per_class",
        type=int,
        default=1,
        help=(
            "Equal number of visual prototypes allocated to every class. "
            "P5.0 preregisters exactly 2; 1 preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--prototype_aggregation",
        choices=["max", "logmeanexp"],
        default="max",
        help="Aggregate equal-capacity within-class prototype scores.",
    )
    parser.add_argument(
        "--prototype_mixture_temperature",
        type=float,
        default=0.2,
        help="Temperature for normalized log-mean-exp prototype aggregation.",
    )
    parser.add_argument(
        "--prototype_balance_weight",
        type=float,
        default=0.0,
        help="Weight of the class-wise prototype responsibility balance loss.",
    )
    parser.add_argument(
        "--prototype_assignment_temperature",
        type=float,
        default=0.2,
        help="Temperature used by the prototype responsibility balance loss.",
    )
    parser.add_argument("--cross_modal_mode", type=str, default="all",
                        choices=["all", "direct", "strip", "scale", "ds", "dsc", "ss", "none"],
                        help="Cross-modal fusion mode for TextGuidedFusion. Controls which paths are used.")

    # ================================================================
    # text-guided training switch
    # ================================================================
    parser.add_argument("--use_text_guided", action="store_true", default=False,
                        help="Enable text-guided feature purification during training. "
                             "Uses K-way text-conditioned residual hybrid features with pre-computed text prototypes. "
                             "Combines CLIP_TopKAvg + γ*text-purified residual + DINOv2_Gram for [B, 1920] hybrid features.")

    # ================================================================
    # Gate type
    # ================================================================
    # Gating of the path fusion in TextGuidedFusion:
    #   heavy  - nn.Sequential gate network (many parameters, prone to over-fitting)
    #   scalar - nn.Parameter scalar weights (a few parameters; suited to few-shot)
    #   cosine - gate activated by cosine similarity (recommended; 6 parameters, uses the CLIP pre-trained alignment)
    # ================================================================
    parser.add_argument("--gate_type", type=str, default="cosine",
                        choices=["heavy", "scalar", "cosine"],
                        help="Gate type for path fusion: 'cosine'(recommended for few-shot), 'heavy'(nn.Sequential), 'scalar'(nn.Parameter)")

    # ================================================================
    # Text refinement (two modes; the visual features are never modified by text)
    # ================================================================
    # logit_boost: temperature softmax in logit space, winner-take-all (recommended)
    #   final_logits = vision_logits + β * logit_scale * softmax(text_sim / τ)
    #   image-text similarity from raw_clip_cls, which keeps it calibrated
    #   only 2 learnable scalars (beta, tau), so no over-fitting with 8 shots
    #
    # proto_refine: prototype refinement in feature space
    #   refined_proto = L2_norm(visual_proto + β * text_proj(text_proto))
    #   text is projected into the visual space by an MLP and refines the class prototypes
    #   (every class receives some residual, but the prototype positions are optimised directly)
    #
    # Mutually exclusive with --use_text_guided: the K-way feature fusion is skipped when enabled
    # ================================================================
    parser.add_argument("--proto_text_refine", type=str, default="none",
                        choices=["none", "logit_boost", "proto_refine"],
                        help="Direction B: Text enhancement mode. "
                             "'logit_boost': temperature softmax winner-takes-all on logits (recommended). "
                             "'proto_refine': text-refined prototypes in feature space. "
                             "'none': no text refinement.")

    # ================================================================
    # ArcMargin penalty
    # ================================================================
    # Subtracts a margin from the cosine similarity of the correct class in the classification loss,
    # pulling features towards their prototype and widening the margins between classes.
    # margin = 0 gives the plain cross-entropy; 0.2-0.4 is recommended
    # ================================================================
    parser.add_argument("--arc_margin", type=float, default=0.0,
                        help="ArcMargin penalty for classification loss. "
                             "0.0 = standard CE, 0.2~0.4 = margin penalty (recommended for few-shot)")

    parser.add_argument("--dino_mask_ratio", type=float, default=0.0,
                        help="Ratio of Top-K spatial tokens to keep for DINOv2 Gram Matrix calculation. "
                             "0.0 = disable (use all tokens). Recommend 0.1 for cable datasets.")
    parser.add_argument("--clip_pool", choices=["mean", "topk"], default="topk",
                        help="CLIP patch aggregation: mean is the baseline, topk is the proposed pooling.")
    parser.add_argument("--dino_pool", choices=["cls", "mean", "gram"], default="gram",
                        help="DINO representation: cls baseline, patch mean, or mean plus Gram statistics.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Cross-entropy label smoothing. Set 0.0 for the no-smoothing ablation.")

    args = parser.parse_args()
    if args.normal_aware_head:
        if args.dataset != "MVTecAD-Cable":
            parser.error("P2.3 screening is currently registered only for MVTecAD-Cable")
        if args.training_protocol != "clean_v1":
            parser.error("--normal_aware_head requires --training_protocol clean_v1")
        if args.adapter_train_mode not in (None, "terminal_only"):
            parser.error("--normal_aware_head currently requires terminal_only")
        if args.use_text_guided or args.proto_text_refine != "none":
            parser.error("--normal_aware_head P2.3 screening requires the pure-vision path")
        if (
            args.img_size != 336
            or args.clip_pool != "topk"
            or args.dino_pool != "gram"
            or args.dino_mask_ratio != 0.0
            or args.dino_patch_policy != "full"
        ):
            parser.error(
                "P2.3 must retain the P2.0 Hub576 Top-K+Gram visual protocol"
            )
        if args.normal_aware_loss_weight <= 0:
            parser.error("--normal_aware_head requires --normal_aware_loss_weight > 0")
        if args.normal_aware_head_lr_mult <= 0:
            parser.error("--normal_aware_head_lr_mult must be positive")
        if args.normal_aware_head_threshold != 0.0:
            parser.error("P2.3 preregisters --normal_aware_head_threshold 0.0")
    elif args.normal_aware_loss_weight != 0.0:
        parser.error("--normal_aware_loss_weight requires --normal_aware_head")
    if args.prototypes_per_class < 1:
        parser.error("--prototypes_per_class must be positive")
    if args.prototype_mixture_temperature <= 0:
        parser.error("--prototype_mixture_temperature must be positive")
    if args.prototype_assignment_temperature <= 0:
        parser.error("--prototype_assignment_temperature must be positive")
    if args.prototype_balance_weight < 0:
        parser.error("--prototype_balance_weight must be non-negative")
    if args.prototype_aggregation == "logmeanexp":
        if args.prototypes_per_class != 2:
            parser.error("P5.1 logmeanexp preregisters exactly 2 prototypes per class")
        if args.prototype_mixture_temperature != 0.2:
            parser.error("P5.1 preregisters --prototype_mixture_temperature 0.2")
        if args.prototype_assignment_temperature != 0.2:
            parser.error("P5.1 preregisters --prototype_assignment_temperature 0.2")
        if args.prototype_balance_weight != 0.1:
            parser.error("P5.1 preregisters --prototype_balance_weight 0.1")
    elif args.prototype_balance_weight != 0.0:
        parser.error("--prototype_balance_weight requires --prototype_aggregation logmeanexp")
    if args.prototypes_per_class > 1:
        if args.training_protocol != "clean_v1":
            parser.error("Multi-prototype P5.x requires --training_protocol clean_v1")
        if args.adapter_train_mode not in (None, "terminal_only"):
            parser.error("Multi-prototype P5.x requires terminal_only adapters")
        if args.use_text_guided or args.proto_text_refine != "none":
            parser.error("Multi-prototype P5.x is registered only for pure visual training")
        if args.normal_aware_head:
            parser.error("P5.x and the P2.3/P2.4 normal-aware head are mutually exclusive")
    args.oof_fold_id = None
    if args.train_meta_path:
        fold_match = re.fullmatch(
            r"fold_(\d{2})_train\.jsonl", os.path.basename(args.train_meta_path)
        )
        if fold_match:
            args.oof_fold_id = int(fold_match.group(1))
            if not 0 <= args.oof_fold_id < 8:
                parser.error("OOF fold inferred from --train_meta_path must be 00-07")
    if (
        args.early_adapter_source_checkpoint is not None
        and args.training_protocol != "legacy_seed_split_diag"
    ):
        parser.error(
            "--early_adapter_source_checkpoint is diagnostic-only and requires "
            "--training_protocol legacy_seed_split_diag"
        )
    if is_legacy_training_protocol(args.training_protocol):
        if args.rng_protocol not in (None, "legacy_rng"):
            parser.error(
                f"{args.training_protocol} requires --rng_protocol legacy_rng"
            )
        if args.adapter_train_mode not in (None, "legacy_random_fixed"):
            parser.error(
                f"{args.training_protocol} requires "
                "--adapter_train_mode legacy_random_fixed"
            )
        if args.init_logit_scale not in (None, 100.0):
            parser.error(
                f"{args.training_protocol} requires --init_logit_scale 100"
            )
        args.rng_protocol = "legacy_rng"
        args.adapter_train_mode = "legacy_random_fixed"
        args.prototype_init_mode = "legacy_random"
        args.init_logit_scale = 100.0

        if args.training_protocol == "legacy_9359":
            for seed_name in ("init_seed", "data_seed", "augment_seed"):
                value = getattr(args, seed_name)
                if value not in (None, args.seed):
                    parser.error(
                        f"legacy_9359 requires --{seed_name} to equal --seed"
                    )
            args.init_seed = args.seed
            args.data_seed = args.seed
            args.augment_seed = args.seed
            args.seed_protocol_version = "legacy_global_rng_v1"
        else:
            if args.init_seed not in (None, args.seed):
                parser.error(
                    "legacy_seed_split_diag requires --init_seed to equal "
                    "--seed so test-time reconstruction uses the initialization seed"
                )
            if args.data_seed is None or args.augment_seed is None:
                parser.error(
                    "legacy_seed_split_diag requires explicit --data_seed and "
                    "--augment_seed"
                )
            if args.data_seed != args.augment_seed:
                parser.error(
                    "legacy_seed_split_diag requires --data_seed == "
                    "--augment_seed; both name the post-construction training RNG"
                )
            args.init_seed = args.seed
            args.seed_protocol_version = "legacy_split_after_construction_v1"
    else:
        if args.rng_protocol == "legacy_rng":
            parser.error(
                "clean_v1 requires isolated_rng; use legacy_9359 for legacy_rng"
            )
        if args.adapter_train_mode == "legacy_random_fixed":
            parser.error(
                "clean_v1 requires terminal_only, frozen_random_early, or "
                "learned_early; "
                "use legacy_9359 for legacy_random_fixed"
            )
        args.rng_protocol = args.rng_protocol or "isolated_rng"
        args.adapter_train_mode = args.adapter_train_mode or "terminal_only"
        args.prototype_init_mode = "zero"
        args.seed_protocol_version = "independent_streams_v1"
        args.init_logit_scale = (
            10.0 if args.init_logit_scale is None else args.init_logit_scale
        )
        if not 0.0 < args.init_logit_scale < 33.0:
            parser.error(
                "clean_v1 requires 0 < --init_logit_scale < 33 so the "
                "clamped scale retains a nonzero classification gradient"
            )
    args.dino_hub_repo = args.dino_hub_repo or os.environ.get("DINOV2_HUB_REPO")
    args.dino_hub_checkpoint = (
        args.dino_hub_checkpoint or os.environ.get("DINOV2_HUB_CHECKPOINT")
    )
    args.dino_hub_checkpoint_sha256 = (
        checkpoint_sha256(args.dino_hub_checkpoint)
        if args.dino_hub_checkpoint and os.path.isfile(args.dino_hub_checkpoint)
        else None
    )
    args.init_seed = args.seed if args.init_seed is None else args.init_seed
    args.data_seed = args.seed if args.data_seed is None else args.data_seed
    args.augment_seed = args.seed if args.augment_seed is None else args.augment_seed
    args.post_construction_rng_reset = (
        args.training_protocol == "legacy_seed_split_diag"
    )
    args.trajectory_seed = (
        args.data_seed if args.post_construction_rng_reset else None
    )
    if args.img_size % 14 != 0:
        parser.error("--img_size must be divisible by the CLIP/DINO patch size 14")


    # ========================================================
    # Setup
    setup_seed(args.init_seed)
    runtime_info = configure_torch_runtime(torch, enable_tf32=args.tf32)
    os.makedirs(args.save_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train_multiclass.log"),
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Args: %s", vars(args))
    print("Args:", vars(args))
    logger.info("Runtime: %s", runtime_info)
    print("Runtime:", runtime_info)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")

    # ========================================================
    # Build class mapping
    class_to_id, id_to_class = build_class_mapping(args.dataset)
    num_classes = len(class_to_id)
    logger.info(f"Number of classes: {num_classes}")
    print(f"Number of classes: {num_classes}")
    logger.info(f"Class mapping: {class_to_id}")

    # ========================================================
    # Load CLIP model
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    # Build AdaptedCLIP with DINOv2 branch
    dino_use_hub_first = {
        "auto": None,
        "hub": True,
        "transformers": False,
    }[args.dino_backend]
    model = AdaptedCLIP(
        clip_model=clip_model,
        relu=True,
        dino_use_hub_first=dino_use_hub_first,
        gate_type=args.gate_type,
        img_size=args.img_size,   # img_hw is computed from it
        proto_text_refine=args.proto_text_refine,  # text refines the prototypes only
        dino_mask_ratio=args.dino_mask_ratio,      # share of patches for masked pooling
        clip_pool=args.clip_pool,
        dino_pool=args.dino_pool,
        dino_patch_policy=args.dino_patch_policy,
        dino_strict_backend=args.dino_backend == "hub",
        dino_hub_repo=args.dino_hub_repo,
        dino_hub_checkpoint=args.dino_hub_checkpoint,
        rng_protocol=args.rng_protocol,
        adapter_train_mode=args.adapter_train_mode,
        normal_aware_head=args.normal_aware_head,
    ).to(device)

    early_adapter_override = None
    if args.early_adapter_source_checkpoint is not None:
        early_adapter_override = load_early_adapter_override(
            model, args.early_adapter_source_checkpoint
        )
        args.early_adapter_source_checkpoint_sha256 = (
            early_adapter_override["checkpoint_sha256"]
        )
        args.early_adapter_selected_state_sha256 = (
            early_adapter_override["selected_state"]["sha256"]
        )
        logger.info("Early-adapter override: %s", early_adapter_override)
        print("Early-adapter override:", early_adapter_override)
    else:
        args.early_adapter_source_checkpoint_sha256 = None
        args.early_adapter_selected_state_sha256 = None

    logger.info(f"Model vis_dim: {model.vis_dim}")
    print(f"Model vis_dim: {model.vis_dim}")

    # ========================================================
    # Initialize FaissPrototypeMemory
    if args.use_text_guided:
        # text-guided mode: CLIP_TopKAvg (768) + gamma * purified_residual (added) + DINOv2_Gram (1152) = 1920
        # the residual is added to clip_feat (768-d) rather than concatenated into a 2688-d vector
        feat_dim = 1920
        logger.info("Using text-guided mode: feat_dim=1920 (CLIP_TopKAvg 768 + γ*residual(add) + DINOv2_Gram 1152)")
        print("Using text-guided mode: feat_dim=1920 (CLIP_TopKAvg 768 + γ*residual(add) + DINOv2_Gram 1152)")
    else:
        # base mode: CLIP_TopKAvg (768) + DINOv2_Gram (1152) = 1920
        feat_dim = model.fusion_dim
        logger.info("Using Base mode: feat_dim=%d (clip_pool=%s, dino_pool=%s)", feat_dim, args.clip_pool, args.dino_pool)
        print(f"Using Base mode: feat_dim={feat_dim} (clip_pool={args.clip_pool}, dino_pool={args.dino_pool})")
    
    prototype_mem = FaissPrototypeMemory(
        num_classes=num_classes,
        feat_dim=feat_dim,
        orth_margin=0.3,
        init_logit_scale=args.init_logit_scale,
        prototype_init_mode=args.prototype_init_mode,
        prototypes_per_class=args.prototypes_per_class,
        prototype_aggregation=args.prototype_aggregation,
        prototype_mixture_temperature=args.prototype_mixture_temperature,
    ).to(device)

    logger.info(
        "PrototypeMemory: %d classes x %d prototypes, feat_dim=%d, aggregation=%s, mixture_temperature=%.3f",
        num_classes, args.prototypes_per_class, feat_dim,
        args.prototype_aggregation, args.prototype_mixture_temperature,
    )
    print(
        f"PrototypeMemory: {num_classes} classes x {args.prototypes_per_class} "
        f"prototypes, feat_dim={feat_dim}, aggregation={args.prototype_aggregation}, "
        f"mixture_temperature={args.prototype_mixture_temperature}"
    )

    # ========================================================
    # Match optimizer membership to the selected forward/gradient protocol.
    trainable_model_names = configure_model_trainable_parameters(model, args)
    normal_aware_head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("normal_aware_head.")
    ]
    trainable_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("normal_aware_head.")
    ]
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    train_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Params: trainable={train_count:,} frozen={frozen_count:,} total={total_count:,}"
    )
    print(
        f"Params: trainable={train_count:,} frozen={frozen_count:,} total={total_count:,}"
    )
    logger.info("Optimizer model parameters: %s", trainable_model_names)
    print("Optimizer model parameters:", trainable_model_names)

    # optimizer: adapter parameters use lr, the prototype memory 10 x lr
    optimizer_groups = [
        {"params": trainable_params, "lr": args.lr},
        {"params": prototype_mem.parameters(), "lr": args.lr * args.proto_lr_mult},
    ]
    if normal_aware_head_params:
        optimizer_groups.append(
            {
                "params": normal_aware_head_params,
                "lr": args.lr * args.normal_aware_head_lr_mult,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    construction_trainable_fingerprint = parameter_fingerprint(
        [("model", model), ("prototype_mem", prototype_mem)],
        trainable_only=True,
    )
    prototype_train_count = sum(
        parameter.numel()
        for parameter in prototype_mem.parameters()
        if parameter.requires_grad
    )
    optimizer_trainable_count = train_count + prototype_train_count

    # ========================================================
    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.exists(args.resume):
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            load_model_checkpoint_state(model, checkpoint, strict_full=True)
            prototype_mem.load_state_dict(checkpoint["prototype_mem_state_dict"])
            restore_prototype_extra_state(prototype_mem, checkpoint)
            if "optimizer_state_dict" not in checkpoint or "scheduler_state_dict" not in checkpoint:
                raise ValueError(
                    "Resume checkpoint does not contain optimizer/scheduler state. "
                    "Use a training checkpoint rather than an inference-only export."
                )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            logger.info(
                "Resumed from epoch %s (format=%s)",
                start_epoch,
                checkpoint.get("checkpoint_format", "legacy_full"),
            )
        else:
            logger.warning(f"Checkpoint {args.resume} not found, training from scratch.")

    rng_was_reset = reset_split_training_rng(args)
    if rng_was_reset:
        logger.info(
            "Legacy split diagnostic: reset global RNG after construction to "
            "trajectory_seed=%d (init_seed=%d)",
            args.trajectory_seed,
            args.init_seed,
        )
        print(
            "Legacy split diagnostic: reset global RNG after construction to "
            f"trajectory_seed={args.trajectory_seed} "
            f"(init_seed={args.init_seed})"
        )

    # ========================================================
    # Load dataset
    if args.training_mode == "full_shot":
        args.shot = -1

    kwargs = dataloader_kwargs(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        use_cuda=use_cuda,
    )
    logger.info("Loading dataset ...")
    _, image_dataset = get_dataset(
        args.dataset,
        args.img_size,
        args.training_mode,
        args.shot,
        "train",
        logger,
        meta_path_override=args.train_meta_path,
    )
    if is_legacy_training_protocol(args.training_protocol):
        # Keep the historical shuffle/worker/augmentation mechanism. Exact
        # legacy continues the construction RNG stream; the split diagnostic
        # starts the same mechanism from the explicit trajectory seed.
        train_loader = DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs,
        )
    else:
        train_sampler = SeededEpochSampler(
            image_dataset,
            data_seed=args.data_seed,
            augment_seed=args.augment_seed,
            shuffle=True,
        )
        train_loader = DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            generator=make_loader_generator(args.data_seed),
            worker_init_fn=DeterministicWorkerSeeder(args.augment_seed),
            **kwargs,
        )

    logger.info("DataLoader: %s", kwargs)
    print("DataLoader:", kwargs)
    seed_protocol = {
        "training_protocol": args.training_protocol,
        "version": args.seed_protocol_version,
        "compatibility_seed": args.seed,
        "init_seed": args.init_seed,
        "data_seed": args.data_seed,
        "augment_seed": args.augment_seed,
        "post_construction_rng_reset": args.post_construction_rng_reset,
        "trajectory_seed": args.trajectory_seed,
    }
    logger.info("Seed protocol: %s", seed_protocol)
    print("Seed protocol:", seed_protocol)
    logger.info(f"Total batches per epoch: {len(train_loader)}")
    print(f"Total batches per epoch: {len(train_loader)}")

    # ========================================================
    # with text guidance or the ITC loss, precompute the multi-prompt text prototypes;
    # text_prototypes serve as 1) the targets of the ITC loss and 2) the text queries
    # ========================================================
    text_prototypes = None
    if args.itc_weight > 0 or args.use_text_guided or args.proto_text_refine != 'none':
        logger.info("Computing multi-prompt fused text prototypes for ITC Loss / text-guided training...")
        print("\nComputing multi-prompt fused text prototypes for ITC Loss / text-guided training...")
        
        # dataset-specific prompt configuration (kept as a dict, not merged)
        from model.prompts_config import DATASET_PROMPTS, PROMPTS
        if args.dataset in DATASET_PROMPTS:
            prompt_templates_dict = DATASET_PROMPTS[args.dataset]
            print(f"  Using dataset-specific prompts for {args.dataset}")
        else:
            prompt_templates_dict = PROMPTS
            print(f"  Using default prompts")
        
        # number of templates
        total_templates = sum(len(v) for v in prompt_templates_dict.values() if isinstance(v, list))
        has_normal_specific = 'normal_sample' in prompt_templates_dict
        
        print(f"  Total templates: {total_templates}")
        if has_normal_specific:
            print(f"  - Normal samples use dedicated 'normal_sample' templates")
            print(f"  - Defect samples use all other templates")
        else:
            print(f"  All classes use the same templates")
        
        logger.info(f"  Prompt templates groups: {list(prompt_templates_dict.keys())}")
        
        text_prototypes = compute_text_prototypes(
            model, class_to_id, REAL_NAMES, prompt_templates_dict, device, dataset_name=args.dataset
        )
        logger.info(f"Text prototypes shape: {text_prototypes.shape}")
        print(f"  Text prototypes shape: {text_prototypes.shape}")

    # ========================================================
    # initialise the prototypes with the class means (not randomly);
    # with use_text_guided, text_query_cache keeps the width at 1920
    # ========================================================
    logger.info("Initializing prototypes with class mean features...")
    print("\nInitializing prototypes with class mean features...")
    init_text_query = text_prototypes if args.use_text_guided else None
    # text-refinement modes need no K-way initialisation and use visual features only
    # (the text_guided_fusion module does not exist, so forward_hybrid_features_kway cannot be called)
    if args.proto_text_refine != 'none':
        init_text_query = None
    prototype_mem = init_prototypes_with_class_means(
        model, train_loader, prototype_mem, device, class_to_id,
        use_text_guided=args.use_text_guided,
        text_query_cache=init_text_query,
        cross_modal_mode=args.cross_modal_mode,
        num_views=args.num_init_views,
        logger=logger,
    )

    # ========================================================
    # Training
    logger.info("=" * 60)
    logger.info("Starting multi-class classification training")
    logger.info("=" * 60)
    print("=" * 60)
    print("Starting multi-class classification training")
    print("=" * 60)
    
    if args.use_text_guided:
        print(f"  [TG] Mode: TEXT-GUIDED (mode={args.cross_modal_mode}, gate={args.gate_type})")
        print(f"     Hybrid feature: [CLIP_TopKAvg(768) + γ*residual(add) + DINOv2_Gram(1152)] = [1920]")
    else:
        print(f"  Mode: clip_pool={args.clip_pool}, dino_pool={args.dino_pool} (pure vision)")
        print(f"     Fusion feature dimension: {model.fusion_dim}")
    print(f"  Gate type: {args.gate_type}")
    print(f"  ITC Loss: {'enabled' if args.itc_weight > 0 else 'disabled'} (weight={args.itc_weight})")

    run_config = dict(vars(args))
    run_config["actual_dino_backend"] = model.dino_extractor.backend
    run_config["train_metadata_path"] = image_dataset.meta_path
    run_config["train_metadata_filename"] = os.path.basename(image_dataset.meta_path)
    run_config["train_metadata_sha256"] = image_dataset.metadata_sha256
    run_config["train_metadata_records"] = image_dataset.metadata_record_count
    run_config["seed_protocol_version"] = args.seed_protocol_version
    run_config["early_adapter_override"] = early_adapter_override

    manifest_path = os.path.join(args.save_path, "run_manifest.json")
    training_start_fingerprint = parameter_fingerprint(
        [("model", model), ("prototype_mem", prototype_mem)],
        trainable_only=True,
    )
    clip_weight_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model", "ViT-L-14-336px.pt"
    )
    metadata_manifest = {
        "train": {
            "path": image_dataset.meta_path,
            "filename": os.path.basename(image_dataset.meta_path),
            "sha256": image_dataset.metadata_sha256,
            "records": image_dataset.metadata_record_count,
        },
    }
    if args.train_meta_path:
        validation_candidate = os.path.join(
            os.path.dirname(image_dataset.meta_path),
            os.path.basename(image_dataset.meta_path).replace("_train.jsonl", "_val.jsonl"),
        )
        if validation_candidate != image_dataset.meta_path and os.path.isfile(validation_candidate):
            metadata_manifest["paired_validation"] = jsonl_identity(validation_candidate)
    else:
        metadata_manifest["test"] = jsonl_identity(
            os.path.join(os.path.dirname(image_dataset.meta_path), "test-shot.jsonl")
        )

    training_manifest = {
        "status": "started",
        "environment": collect_runtime_environment(),
        "cli_args": vars(args),
        "runtime_configuration": runtime_info,
        "seed_protocol": {
            **seed_protocol,
            "derivation": (
                "sha256(base_seed, stage, epoch, repeat, sample_index)"
                if args.training_protocol == "clean_v1"
                else (
                    "global RNG reset after construction to trajectory_seed"
                    if args.training_protocol == "legacy_seed_split_diag"
                    else "single continuous process-global RNG stream"
                )
            ),
            "sampler": (
                "SeededEpochSampler"
                if args.training_protocol == "clean_v1"
                else "DataLoader(shuffle=True)"
            ),
            "per_sample_augmentation": args.training_protocol == "clean_v1",
        },
        "protocol": run_config,
        "actual_dino_backend": model.dino_extractor.backend,
        "backbones": collect_backbone_provenance(
            model=model,
            model_name=args.model_name,
            actual_dino_backend=model.dino_extractor.backend,
            dino_hub_repo=args.dino_hub_repo,
            dino_hub_checkpoint=args.dino_hub_checkpoint,
            clip_weight_path=clip_weight_path,
        ),
        "metadata": metadata_manifest,
        "parameters": {
            "after_construction_before_resume_or_class_mean": construction_trainable_fingerprint,
            "training_start_after_class_mean_initialization": training_start_fingerprint,
            "reported_model_trainable_count": train_count,
            "prototype_trainable_count": prototype_train_count,
            "optimizer_trainable_count": optimizer_trainable_count,
            "optimizer_model_parameter_names": trainable_model_names,
            "reported_model_frozen_count": frozen_count,
            "reported_model_total_count": total_count,
        },
        "resume_checkpoint": args.resume,
    }
    initialize_training_manifest(manifest_path, training_manifest)
    logger.info("Run manifest initialized: %s", manifest_path)
    print(f"Run manifest initialized: {manifest_path}")

    model, prototype_mem = train_multiclass(
        model=model,
        prototype_mem=prototype_mem,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        start_epoch=start_epoch,
        num_epochs=args.num_epochs,
        class_to_id=class_to_id,
        save_path=args.save_path,
        logger=logger,
        supcon_weight=args.supcon_weight,
        orth_weight=args.orth_weight,
        itc_weight=args.itc_weight,
        text_prototypes=text_prototypes,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_every=args.save_every,
        use_text_guided=args.use_text_guided,
        cross_modal_mode=args.cross_modal_mode,
        repeat_per_epoch=args.repeat_per_epoch,
        early_stop_patience=args.early_stop_patience,
        arc_margin=args.arc_margin,
        label_smoothing=args.label_smoothing,
        gate_type=args.gate_type,
        checkpoint_format=args.checkpoint_format,
        save_optimizer_state=args.save_optimizer_state,
        run_config=run_config,
        manifest_path=manifest_path,
        normal_aware_loss_weight=args.normal_aware_loss_weight,
        prototype_balance_weight=args.prototype_balance_weight,
        prototype_assignment_temperature=args.prototype_assignment_temperature,
    )
    update_training_manifest(
        manifest_path,
        {
            "status": "completed",
            "completed_utc": utc_now(),
            "final_trainable_parameters": parameter_fingerprint(
                [("model", model), ("prototype_mem", prototype_mem)],
                trainable_only=True,
            ),
        },
    )
    print("Training completed!")


if __name__ == "__main__":
    main()
