"""Checkpoint helpers that avoid duplicating frozen pretrained backbones.

The CLIP and DINO weights are reconstructed from their original pretrained
sources before a checkpoint is loaded.  A lightweight checkpoint therefore
only needs the project-specific modules, prototype memory, and (for resume)
optimizer/scheduler states.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch


CHECKPOINT_VERSION = 2
LIGHTWEIGHT_FORMAT = "lightweight_v1"
FULL_FORMAT = "full_v1"

# AdaptedCLIP registers CLIP twice (clipmodel and image_encoder point to the
# same visual tower), so both prefixes must be excluded.  DINO is also frozen
# and is reconstructed from its pretrained source.
PRETRAINED_BACKBONE_PREFIXES: Tuple[str, ...] = (
    "clipmodel.",
    "image_encoder.",
    "dino_extractor.",
)


def checkpoint_sha256(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a stable content fingerprint for checkpoint/log provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pretrained_backbone_key(key: str) -> bool:
    return key.startswith(PRETRAINED_BACKBONE_PREFIXES)


def is_lightweight_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    return str(checkpoint.get("checkpoint_format", "")).startswith("lightweight")


def _clone_to_cpu(value: Any) -> Any:
    """Create a stable CPU snapshot, including nested optimizer states."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value


def extract_model_state_dict(model, checkpoint_format: str = "lightweight") -> Dict[str, torch.Tensor]:
    """Return a detached model snapshot in lightweight or legacy full form."""
    if checkpoint_format not in {"lightweight", "full"}:
        raise ValueError(f"Unsupported checkpoint_format: {checkpoint_format}")

    state_dict = model.state_dict()
    if checkpoint_format == "lightweight":
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not is_pretrained_backbone_key(key)
        }
    return _clone_to_cpu(state_dict)


def build_training_checkpoint(
    *,
    model,
    prototype_mem,
    optimizer,
    scheduler,
    checkpoint_format: str = "lightweight",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an evaluation checkpoint, optionally with exact-resume state."""
    checkpoint: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_format": (
            LIGHTWEIGHT_FORMAT if checkpoint_format == "lightweight" else FULL_FORMAT
        ),
        "excluded_backbone_prefixes": (
            list(PRETRAINED_BACKBONE_PREFIXES)
            if checkpoint_format == "lightweight"
            else []
        ),
        "model_state_dict": extract_model_state_dict(model, checkpoint_format),
        "prototype_mem_state_dict": _clone_to_cpu(prototype_mem.state_dict()),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = _clone_to_cpu(optimizer.state_dict())
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = _clone_to_cpu(scheduler.state_dict())
    prototype_extra_state = {}
    if hasattr(prototype_mem, "class_radii"):
        prototype_extra_state["class_radii"] = _clone_to_cpu(
            prototype_mem.class_radii
        )
    if hasattr(prototype_mem, "prototype_cluster_sizes"):
        prototype_extra_state["prototype_cluster_sizes"] = _clone_to_cpu(
            prototype_mem.prototype_cluster_sizes
        )
    if hasattr(prototype_mem, "prototype_training_usage"):
        prototype_extra_state["prototype_training_usage"] = _clone_to_cpu(
            prototype_mem.prototype_training_usage
        )
    if prototype_extra_state:
        checkpoint["prototype_extra_state"] = prototype_extra_state
    if metadata:
        checkpoint.update(dict(metadata))
    return checkpoint


def load_model_checkpoint_state(
    model,
    checkpoint: Mapping[str, Any],
    *,
    strict_full: bool = False,
) -> Tuple[Iterable[str], Iterable[str]]:
    """Load either a new lightweight checkpoint or an old full checkpoint.

    Lightweight checkpoints are validated against every non-backbone state
    entry.  Missing pretrained backbone keys are intentional and are not
    returned as warnings.
    """
    state_dict = checkpoint["model_state_dict"]
    if not is_lightweight_checkpoint(checkpoint):
        incompatible = model.load_state_dict(state_dict, strict=strict_full)
        return incompatible.missing_keys, incompatible.unexpected_keys

    expected_project_keys = {
        key for key in model.state_dict() if not is_pretrained_backbone_key(key)
    }
    saved_keys = set(state_dict)
    missing_project_keys = sorted(expected_project_keys - saved_keys)
    if missing_project_keys:
        preview = ", ".join(missing_project_keys[:8])
        raise RuntimeError(
            "Lightweight checkpoint is missing project-specific parameters: "
            f"{preview}"
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise RuntimeError(
            f"Lightweight checkpoint has unexpected parameters: {preview}"
        )

    # Only excluded pretrained weights should remain missing.
    invalid_missing = [
        key for key in incompatible.missing_keys if not is_pretrained_backbone_key(key)
    ]
    if invalid_missing:
        preview = ", ".join(invalid_missing[:8])
        raise RuntimeError(f"Unexpected missing parameters while loading: {preview}")
    return [], []


def restore_prototype_extra_state(prototype_mem, checkpoint: Mapping[str, Any]) -> None:
    extra_state = checkpoint.get("prototype_extra_state", {})
    if "class_radii" in extra_state:
        prototype_mem.class_radii = extra_state["class_radii"]
    if "prototype_cluster_sizes" in extra_state:
        prototype_mem.prototype_cluster_sizes = extra_state[
            "prototype_cluster_sizes"
        ]
    if "prototype_training_usage" in extra_state:
        prototype_mem.prototype_training_usage = extra_state[
            "prototype_training_usage"
        ]
