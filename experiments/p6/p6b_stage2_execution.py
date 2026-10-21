#!/usr/bin/env python3
"""P6.B Stage 2A shared core — preregistered semantics (Addendum A).

Implements the locked baseline mathematical semantics of
``docs/P6B_EXECUTION_ADDENDUM_A.md`` on top of the formal Stage 1 manifest
bundle:

- formal bundle gate: completion ``completed`` / ``formal_eligible`` /
  manifest membership + self-hash verification;
- manifest-driven support dataset with path containment, file existence and
  RGB SHA256 verification;
- injectable frozen CLIP/DINO feature interfaces (ABCs; this module never
  constructs real backbones);
- pure functions/components: prototype, frozen fusion, linear head training,
  zero-shot CLIP, Tip-Adapter (training-free).

Firewall: the training/support interfaces consume ONLY the bundle directory
plus an image root. They never accept query paths, query metadata, or the
feasibility JSONL. All backbones are frozen by contract
(``requires_grad=False``); embeddings flow through ``torch.no_grad()``
extraction and detached frozen tensors.
"""

from __future__ import annotations

import abc
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

# cole environment safe import order: numpy MUST be imported before torch to
# avoid the OpenMP runtime duplicate-library crash in this Windows env.
# Never work around this with KMP_DUPLICATE_LIB_OK — setting that flag is
# forbidden; keep the import order instead.
import numpy  # noqa: F401  (deliberately before torch)

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from experiments.p6.p6b_execution_common import (
    EXPECTED_MANIFEST_COUNT,
    LOCKED_K,
    LOCKED_PRODUCTS,
    LOCKED_SEEDS,
    MANIFEST_SCHEMA_VERSION,
    P6BFailClosedError,
    manifest_self_sha256,
)
from experiments.p6 import p6b_execution_common as common

# ---------------------------------------------------------------------------
# Locked constants (Addendum A)
# ---------------------------------------------------------------------------

ZERO_SHOT_PROMPT_TEMPLATE = "a photo of a {}."
ZERO_SHOT_LOGIT_SCALE = 100.0
TIP_BETA = 5.5
TIP_ALPHA = 1.0
LINEAR_LABEL_SMOOTHING = 0.1

# R2 Tip-Adapter-F* locked training constants (Addendum B §3.3). Faithful to
# the official Tip-Adapter-F cache-key training where compatible with the P6.B
# query firewall; every field is a locked constant, not a tunable hyperparameter.
R2_LR = 1e-3
R2_EPS = 1e-4
R2_WEIGHT_DECAY = 1e-2
R2_EPOCHS = 20
R2_REPEAT_PER_EPOCH = 1
R2_BATCH_SIZE = 8
R2_GRAD_ACCUM = 1
R2_LABEL_SMOOTHING = 0.0
LINEAR_LR = 1e-4
LINEAR_WEIGHT_DECAY = 0.01
LINEAR_EPOCHS = 20
LINEAR_REPEAT_PER_EPOCH = 4
LINEAR_BATCH_SIZE = 8
LINEAR_GRAD_ACCUM = 2

# Pinned SHA256 of the CONFIRMED Stage 1 formal bundle completion
# (Addendum A §2). Locked: any mismatch is fail-closed.
STAGE1_COMPLETION_SHA256 = (
    "5da9eebb5463ffba11a92ba6f7ac4125fcf4f85d0b8fa2dcd4cbc79d2d901a73"
)
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

COMPLETION_FILENAME = "p6b_manifest_build_complete.json"
COMPLETION_SCHEMA = "p6b_support_manifest_build_v1"


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    return F.normalize(x, p=2, dim=dim)


def format_zero_shot_prompt(class_name: str) -> str:
    """Locked prompt: underscores in class names become spaces."""
    return ZERO_SHOT_PROMPT_TEMPLATE.format(class_name.replace("_", " "))


# ---------------------------------------------------------------------------
# Input validation helpers (fail-closed; no silent truncation)
# ---------------------------------------------------------------------------

def _coerce_embedding_input(value, name: str) -> torch.Tensor:
    """Accept tensors, ndarray-like sequences, or sequences of 1-D tensors
    (stacked, never silently truncated)."""
    if isinstance(value, (list, tuple)) and value and \
            isinstance(value[0], torch.Tensor):
        try:
            return torch.stack([torch.as_tensor(v, dtype=torch.float32)
                                for v in value])
        except RuntimeError as exc:
            raise P6BFailClosedError(
                f"{name}: ragged embedding rows cannot be stacked") from exc
    return torch.as_tensor(value, dtype=torch.float32)


def _as_embedding_matrix(value, name: str) -> torch.Tensor:
    tensor = _coerce_embedding_input(value, name)
    if tensor.dim() != 2:
        raise P6BFailClosedError(f"{name} must be a 2-D embedding matrix")
    if not torch.isfinite(tensor).all():
        raise P6BFailClosedError(f"{name} contains non-finite values")
    norms = tensor.norm(dim=1)
    if bool((norms == 0).any()):
        raise P6BFailClosedError(f"{name} contains zero-norm rows")
    return tensor


def _as_embedding_vector(value, name: str) -> torch.Tensor:
    tensor = _coerce_embedding_input(value, name)
    if tensor.dim() != 1:
        raise P6BFailClosedError(f"{name} must be a 1-D embedding vector")
    if not torch.isfinite(tensor).all():
        raise P6BFailClosedError(f"{name} contains non-finite values")
    if float(tensor.norm().item()) == 0.0:
        raise P6BFailClosedError(f"{name} is a zero-norm vector")
    return tensor


def _validate_label_list(labels, expected_len: int, num_classes: int, name: str) -> list:
    labels = list(labels)
    if len(labels) != expected_len:
        raise P6BFailClosedError(
            f"{name} length {len(labels)} != expected {expected_len} "
            "(silent truncation is forbidden)")
    validated = []
    for label in labels:
        if isinstance(label, bool):
            raise P6BFailClosedError(f"{name} label {label!r} is not a class index")
        try:
            value = int(label)
        except (TypeError, ValueError) as exc:
            raise P6BFailClosedError(
                f"{name} label {label!r} is not an integer class index") from exc
        if not 0 <= value < num_classes:
            raise P6BFailClosedError(
                f"{name} label {value} outside [0, {num_classes})")
        validated.append(value)
    return validated


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Formal Stage 1 bundle loading (fail-closed)
# ---------------------------------------------------------------------------

def load_formal_bundle(bundle_dir) -> dict:
    """Load the Stage 1 bundle and verify the formal gate.

    Checks: completion schema/status/formal flags, manifest count, every
    per-product manifest listed in the completion exists, parses, matches its
    recorded sha256 and passes self-hash recomputation.
    """
    bundle_dir = Path(bundle_dir)
    completion_path = bundle_dir / COMPLETION_FILENAME
    if not completion_path.is_file():
        raise P6BFailClosedError(f"missing completion: {completion_path}")
    # Locked provenance pin (Addendum A §2): the CONFIRMED Stage 1 completion
    # file must hash to the pinned value before anything else is trusted.
    completion_sha = file_sha256(completion_path)
    if completion_sha != STAGE1_COMPLETION_SHA256:
        raise P6BFailClosedError(
            f"completion sha256 {completion_sha} != pinned "
            f"{STAGE1_COMPLETION_SHA256}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("schema_version") != COMPLETION_SCHEMA:
        raise P6BFailClosedError("completion schema_version mismatch")
    if completion.get("status") != "completed":
        raise P6BFailClosedError(
            f"completion status {completion.get('status')!r} != 'completed'")
    if completion.get("formal_eligible") is not True:
        raise P6BFailClosedError("completion formal_eligible must be true")
    if completion.get("formal_test_used") is not False:
        raise P6BFailClosedError("completion formal_test_used must be false")
    if completion.get("manifest_count") != EXPECTED_MANIFEST_COUNT:
        raise P6BFailClosedError("completion manifest_count mismatch")
    if completion.get("products") != list(LOCKED_PRODUCTS):
        raise P6BFailClosedError("completion products != locked 13 products")
    if completion.get("K_values") != list(LOCKED_K):
        raise P6BFailClosedError("completion K_values != locked [1, 3, 5]")
    if completion.get("seeds") != list(LOCKED_SEEDS):
        raise P6BFailClosedError("completion seeds != locked [111..555]")
    provenance = completion.get("project_provenance") or {}
    if provenance.get("dirty") is not False:
        raise P6BFailClosedError("completion provenance dirty must be false")
    commit = provenance.get("commit")
    if not isinstance(commit, str) or not _COMMIT_SHA_RE.fullmatch(commit):
        raise P6BFailClosedError("completion provenance commit must be a 40-hex SHA")

    bundle_root = bundle_dir.resolve()
    manifests = {}
    for product in LOCKED_PRODUCTS:
        audit = (completion.get("per_product") or {}).get(product)
        if not audit:
            raise P6BFailClosedError(f"completion lacks audit for {product!r}")
        for k_shot in LOCKED_K:
            for seed in LOCKED_SEEDS:
                key = f"k{k_shot}_s{seed}"
                entry = (audit.get("manifests") or {}).get(key)
                if not entry:
                    raise P6BFailClosedError(f"manifest {product}/{key} not listed")
                manifest_path = bundle_dir / entry["relative_path"]
                resolved = manifest_path.resolve()
                if bundle_root not in resolved.parents:
                    raise P6BFailClosedError(
                        f"manifest path escapes bundle dir: {entry['relative_path']!r}")
                if not resolved.is_file():
                    raise P6BFailClosedError(f"manifest missing on disk: {manifest_path}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                    raise P6BFailClosedError(f"manifest schema mismatch: {manifest_path}")
                # the completion records each manifest's canonical self-hash;
                # membership, recorded-hash and recomputed-hash must all agree
                if manifest.get("manifest_sha256") != entry["manifest_sha256"]:
                    raise P6BFailClosedError(f"manifest sha mismatch: {manifest_path}")
                if manifest["manifest_sha256"] != manifest_self_sha256(manifest):
                    raise P6BFailClosedError(f"manifest self-hash mismatch: {manifest_path}")
                # byte-level pin: the file must equal the canonical serialization
                if common.canonical_json_bytes(manifest) != manifest_path.read_bytes():
                    raise P6BFailClosedError(f"manifest bytes not canonical: {manifest_path}")
                if (manifest["product"], manifest["K"], manifest["seed"]) != (product, k_shot, seed):
                    raise P6BFailClosedError(f"manifest identity mismatch: {manifest_path}")
                if manifest.get("class_order") != manifest.get("class_universe"):
                    raise P6BFailClosedError(f"class_order/class_universe mismatch: {manifest_path}")
                manifests[(product, k_shot, seed)] = manifest
    if len(manifests) != EXPECTED_MANIFEST_COUNT:
        raise P6BFailClosedError("loaded manifest count mismatch")
    return {"completion": completion, "manifests": manifests}


def select_manifest(bundle: dict, product: str, k_shot: int, seed: int) -> dict:
    """Select one locked manifest; fail-closed on unknown identity."""
    if product not in LOCKED_PRODUCTS:
        raise P6BFailClosedError(f"product {product!r} outside locked scope")
    if k_shot not in LOCKED_K:
        raise P6BFailClosedError(f"K={k_shot} outside locked set")
    if seed not in LOCKED_SEEDS:
        raise P6BFailClosedError(f"seed={seed} outside locked set")
    return bundle["manifests"][(product, k_shot, seed)]


# ---------------------------------------------------------------------------
# Manifest-driven support dataset (firewall: support-side only)
# ---------------------------------------------------------------------------

class P6BSupportDataset(Dataset):
    """Support samples driven exclusively by one locked manifest.

    Items are returned in manifest order (fixed class_order); each image is
    verified for path containment under ``image_root``, existence and RGB
    SHA256 equality before being handed out. No query path/metadata is ever
    accepted by this class.
    """

    def __init__(self, manifest: dict, image_root):
        image_root = Path(image_root)
        if not image_root.is_dir():
            raise P6BFailClosedError(f"image root missing: {image_root}")
        self.image_root = image_root.resolve()
        self.class_order = list(manifest["class_order"])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_order)}
        self.items = []
        for class_name in self.class_order:
            for item in manifest["classes"][class_name]["items"]:
                self.items.append((class_name, item["rel_path"], item["rgb_sha256"]))

    def _resolve_contained(self, rel_path: str) -> Path:
        if not rel_path or rel_path.startswith(("/", "\\")) or ".." in Path(rel_path).parts:
            raise P6BFailClosedError(f"support path escapes image root: {rel_path!r}")
        resolved = (self.image_root / rel_path).resolve()
        if self.image_root not in resolved.parents:
            raise P6BFailClosedError(f"support path escapes image root: {rel_path!r}")
        return resolved

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        class_name, rel_path, expected_sha = self.items[idx]
        path = self._resolve_contained(rel_path)
        if not path.is_file():
            raise P6BFailClosedError(f"support image missing: {rel_path!r}")
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            raise P6BFailClosedError(
                f"support RGB sha mismatch for {rel_path!r}: "
                f"{actual_sha} != {expected_sha}")
        with Image.open(path) as handle:
            image = handle.convert("RGB").copy()
        return {
            "image": image,
            "label": self.class_to_idx[class_name],
            "class_name": class_name,
            "rel_path": rel_path,
            "rgb_sha256": actual_sha,
        }


# ---------------------------------------------------------------------------
# Injectable frozen feature interfaces
# ---------------------------------------------------------------------------

class ImageFeatureExtractor(abc.ABC):
    """Frozen image encoder contract. Implementations MUST keep every
    backbone parameter at ``requires_grad=False`` and extract under
    ``torch.no_grad()``; outputs are detached CPU tensors."""

    @abc.abstractmethod
    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """Return one L2-normalized embedding vector for a single RGB image."""


class TextFeatureExtractor(abc.ABC):
    """Frozen text encoder contract (zero-shot CLIP prompts)."""

    @abc.abstractmethod
    def embed_text(self, text: str) -> torch.Tensor:
        """Return one L2-normalized embedding vector for one prompt."""


# ---------------------------------------------------------------------------
# Prototype semantics (Addendum A §1.2)
# ---------------------------------------------------------------------------

def build_prototypes(support_embeddings, support_labels, class_order) -> torch.Tensor:
    """Per class: L2-normalize each support embedding, mean, L2-normalize.

    Prototype rows follow ``class_order``. Fail-closed on invalid shapes,
    length mismatch (no silent zip truncation), label range violations,
    non-finite or zero-norm embeddings, and classes without support.
    """
    embeddings = _as_embedding_matrix(support_embeddings, "support_embeddings")
    labels = _validate_label_list(support_labels, embeddings.shape[0],
                                  len(class_order), "support_labels")
    prototypes = []
    for class_idx, _ in enumerate(class_order):
        rows = [l2_normalize(embeddings[row])
                for row, label in enumerate(labels) if label == class_idx]
        if not rows:
            raise P6BFailClosedError(
                f"class index {class_idx} ({class_order[class_idx]!r}) has no support embeddings")
        prototypes.append(l2_normalize(torch.stack(rows).mean(dim=0)))
    return torch.stack(prototypes)


def classify_by_prototypes(query_embeddings, prototypes) -> list:
    """Cosine argmax over normalized prototypes; ties resolve to the lowest
    class index, i.e. manifest class_order (first occurrence wins)."""
    queries = _as_embedding_matrix(query_embeddings, "query_embeddings")
    prototype_matrix = _as_embedding_matrix(prototypes, "prototypes")
    if queries.shape[1] != prototype_matrix.shape[1]:
        raise P6BFailClosedError(
            f"query dim {queries.shape[1]} != prototype dim {prototype_matrix.shape[1]}")
    scores = l2_normalize(queries, dim=-1) @ l2_normalize(prototype_matrix, dim=-1).t()
    return [int(row.argmax().item()) for row in scores]


# ---------------------------------------------------------------------------
# Frozen fusion (Addendum A §1.1)
# ---------------------------------------------------------------------------

def fuse_frozen_embeddings(clip_embeddings, dino_embeddings) -> torch.Tensor:
    """concat(L2-normalized CLIP, L2-normalized DINO), then L2-normalize."""
    clip = _as_embedding_matrix(clip_embeddings, "clip_embeddings")
    dino = _as_embedding_matrix(dino_embeddings, "dino_embeddings")
    if clip.shape[0] != dino.shape[0]:
        raise P6BFailClosedError(
            f"clip/dino row counts differ: {clip.shape[0]} != {dino.shape[0]} "
            "(silent truncation is forbidden)")
    return l2_normalize(torch.cat([l2_normalize(clip, dim=-1),
                                   l2_normalize(dino, dim=-1)], dim=-1), dim=-1)


# ---------------------------------------------------------------------------
# Zero-shot CLIP (Addendum A §1.5)
# ---------------------------------------------------------------------------

def zero_shot_clip_logits(query_embeddings, text_embeddings,
                          logit_scale: float = ZERO_SHOT_LOGIT_SCALE) -> torch.Tensor:
    """Locked logit scale: any non-exact override is fail-closed."""
    if logit_scale != ZERO_SHOT_LOGIT_SCALE:
        raise P6BFailClosedError(
            f"zero-shot logit_scale {logit_scale!r} != locked {ZERO_SHOT_LOGIT_SCALE}")
    queries = _as_embedding_matrix(query_embeddings, "query_embeddings")
    texts = _as_embedding_matrix(text_embeddings, "text_embeddings")
    if queries.shape[1] != texts.shape[1]:
        raise P6BFailClosedError(
            f"query dim {queries.shape[1]} != text dim {texts.shape[1]}")
    return ZERO_SHOT_LOGIT_SCALE * (
        l2_normalize(queries, dim=-1) @ l2_normalize(texts, dim=-1).t())


# ---------------------------------------------------------------------------
# Tip-Adapter, training-free (Addendum A §1.6)
# ---------------------------------------------------------------------------

class TipAdapterCache:
    """Locked training-free Tip-Adapter: affinity = exp(-beta*(1-q@keys.T)),
    logits = clip_logits + alpha*(affinity @ one-hot values). alpha/beta are
    locked constants: any non-exact override is fail-closed; no parameter
    search is permitted."""

    def __init__(self, cache_keys, cache_labels, num_classes: int,
                 beta: float = TIP_BETA, alpha: float = TIP_ALPHA):
        if beta != TIP_BETA:
            raise P6BFailClosedError(f"Tip beta {beta!r} != locked {TIP_BETA}")
        if alpha != TIP_ALPHA:
            raise P6BFailClosedError(f"Tip alpha {alpha!r} != locked {TIP_ALPHA}")
        keys = _as_embedding_matrix(cache_keys, "cache_keys")
        labels = _validate_label_list(cache_labels, keys.shape[0],
                                      num_classes, "cache_labels")
        values = torch.zeros(keys.shape[0], num_classes)
        for row, label in enumerate(labels):
            values[row, label] = 1.0
        self.keys = l2_normalize(keys, dim=-1)
        self.values = values
        self.beta = TIP_BETA
        self.alpha = TIP_ALPHA

    def affinity(self, query_embeddings) -> torch.Tensor:
        queries = _as_embedding_matrix(query_embeddings, "query_embeddings")
        if queries.shape[1] != self.keys.shape[1]:
            raise P6BFailClosedError(
                f"query dim {queries.shape[1]} != cache key dim {self.keys.shape[1]}")
        return torch.exp(-self.beta * (
            1.0 - l2_normalize(queries, dim=-1) @ self.keys.t()))

    def logits(self, clip_logits, query_embeddings) -> torch.Tensor:
        clip = torch.as_tensor(clip_logits, dtype=torch.float32)
        if clip.dim() != 2 or not torch.isfinite(clip).all():
            raise P6BFailClosedError("clip_logits must be a finite 2-D tensor")
        if clip.shape[1] != self.values.shape[1]:
            raise P6BFailClosedError(
                f"clip_logits width {clip.shape[1]} != class universe "
                f"{self.values.shape[1]}")
        # Explicit row-count gate: clip_logits rows must equal query rows
        # exactly; PyTorch broadcasting is forbidden.
        queries = _as_embedding_matrix(query_embeddings, "query_embeddings")
        if clip.shape[0] != queries.shape[0]:
            raise P6BFailClosedError(
                f"clip_logits rows {clip.shape[0]} != query rows "
                f"{queries.shape[0]} (broadcasting is forbidden)")
        return clip + self.alpha * (self.affinity(queries) @ self.values)


# ---------------------------------------------------------------------------
# Frozen-Fusion Linear Probe (Addendum A §1.3)
#
# Stage 2A scope: deterministic head construction + locked training spec
# validation ONLY. The augmentation-aware training loop (per epoch/repeat
# non-cable augmentation + frozen Fusion re-extraction) is implemented by the
# Stage 2B runner; a precomputed-embedding loop is deliberately absent because
# it cannot express the locked augmentation semantics.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# R2 Tip-Adapter-F* (trainable cache keys ONLY; Addendum B §3)
# ---------------------------------------------------------------------------

class TipAdapterFModule(nn.Module):
    """Firewall-constrained Tip-Adapter-F* (Addendum B §3).

    The ONLY trainable parameter is ``cache_keys`` (initialized from the
    training-free Tip-Adapter support cache keys, i.e. L2-normalized support
    CLIP embeddings). ``cache_values``, ``text_weights`` and the CLIP encoders
    are frozen. Classification keeps the locked Tip-Adapter semantics:

        affinity     = exp(-beta * (1 - image_emb @ cache_keys.T))
        cache_logits = affinity @ cache_values
        clip_logits  = logit_scale * (image_emb @ text_weights.T)
        logits       = clip_logits + alpha * cache_logits

    ``alpha=1.0``, ``beta=5.5`` and ``logit_scale=100`` are locked constants:
    any non-exact override is fail-closed; no alpha/beta/prompt search is
    permitted.
    """

    def __init__(self, init_cache_keys, cache_labels, num_classes: int,
                 text_weights, beta: float = TIP_BETA, alpha: float = TIP_ALPHA,
                 logit_scale: float = ZERO_SHOT_LOGIT_SCALE):
        super().__init__()
        if beta != TIP_BETA:
            raise P6BFailClosedError(f"Tip-Adapter-F* beta {beta!r} != locked {TIP_BETA}")
        if alpha != TIP_ALPHA:
            raise P6BFailClosedError(f"Tip-Adapter-F* alpha {alpha!r} != locked {TIP_ALPHA}")
        if logit_scale != ZERO_SHOT_LOGIT_SCALE:
            raise P6BFailClosedError(
                f"Tip-Adapter-F* logit_scale {logit_scale!r} != locked "
                f"{ZERO_SHOT_LOGIT_SCALE}")
        keys = _as_embedding_matrix(init_cache_keys, "cache_keys")
        labels = _validate_label_list(cache_labels, keys.shape[0],
                                      num_classes, "cache_labels")
        text = _as_embedding_matrix(text_weights, "text_weights")
        if text.shape[0] != num_classes:
            raise P6BFailClosedError(
                f"text_weights rows {text.shape[0]} != num_classes {num_classes}")
        if text.shape[1] != keys.shape[1]:
            raise P6BFailClosedError(
                f"text_weights dim {text.shape[1]} != cache key dim {keys.shape[1]}")
        values = torch.zeros(keys.shape[0], num_classes)
        for row, label in enumerate(labels):
            values[row, label] = 1.0
        # cache_keys is the ONLY trainable parameter (initialized normalized).
        self.cache_keys = nn.Parameter(l2_normalize(keys, dim=-1).clone())
        # Everything else is frozen (registered buffers, requires_grad=False).
        self.register_buffer("cache_values", values)
        self.register_buffer("text_weights", l2_normalize(text, dim=-1))
        self.beta = TIP_BETA
        self.alpha = TIP_ALPHA
        self.logit_scale = ZERO_SHOT_LOGIT_SCALE

    def logits(self, image_embeddings) -> torch.Tensor:
        queries = _as_embedding_matrix(image_embeddings, "image_embeddings")
        if queries.shape[1] != self.cache_keys.shape[1]:
            raise P6BFailClosedError(
                f"image dim {queries.shape[1]} != cache key dim "
                f"{self.cache_keys.shape[1]}")
        queries = F.normalize(queries, p=2, dim=-1)
        # cache_keys drift during training and are NOT renormalized (faithful
        # to the official Tip-Adapter-F Linear-adapter cache-key mechanism).
        affinity = torch.exp(-self.beta * (1.0 - queries @ self.cache_keys.t()))
        cache_logits = affinity @ self.cache_values
        clip_logits = self.logit_scale * (queries @ self.text_weights.t())
        return clip_logits + self.alpha * cache_logits

    def forward(self, image_embeddings) -> torch.Tensor:
        return self.logits(image_embeddings)


@dataclass(frozen=True)
class TipAdapterFSpec:
    """Locked Tip-Adapter-F* training specification (Addendum B §3.3). Every
    field is a locked constant, not a tunable hyperparameter."""

    epochs: int = R2_EPOCHS
    repeat_per_epoch: int = R2_REPEAT_PER_EPOCH
    batch_size: int = R2_BATCH_SIZE
    grad_accum: int = R2_GRAD_ACCUM
    lr: float = R2_LR
    eps: float = R2_EPS
    weight_decay: float = R2_WEIGHT_DECAY
    label_smoothing: float = R2_LABEL_SMOOTHING


def validate_tip_adapter_f_spec(spec: TipAdapterFSpec) -> TipAdapterFSpec:
    """Fail-closed unless every field equals its locked value exactly."""
    locked = TipAdapterFSpec()
    if spec != locked:
        mismatched = [
            field for field in (
                "epochs", "repeat_per_epoch", "batch_size", "grad_accum",
                "lr", "eps", "weight_decay", "label_smoothing")
            if getattr(spec, field) != getattr(locked, field)
        ]
        raise P6BFailClosedError(
            f"Tip-Adapter-F* spec deviates from locked constants: {mismatched}")
    return spec


def make_linear_head(fusion_dim: int, num_classes: int, seed: int) -> nn.Linear:
    """Deterministically initialized Linear(dim, N, bias=True) — the ONLY
    trainable module of the Linear Probe baseline."""
    if int(fusion_dim) <= 0 or int(num_classes) <= 0:
        raise P6BFailClosedError("fusion_dim/num_classes must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    head = nn.Linear(int(fusion_dim), int(num_classes), bias=True)
    nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5), generator=generator)
    fan_in = head.weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
    nn.init.uniform_(head.bias, -bound, bound, generator=generator)
    return head


@dataclass(frozen=True)
class LinearProbeSpec:
    """Locked Linear Probe training specification (Addendum A §1.3). Every
    field is a locked constant, not a tunable hyperparameter."""

    epochs: int = LINEAR_EPOCHS
    repeat_per_epoch: int = LINEAR_REPEAT_PER_EPOCH
    batch_size: int = LINEAR_BATCH_SIZE
    grad_accum: int = LINEAR_GRAD_ACCUM
    lr: float = LINEAR_LR
    weight_decay: float = LINEAR_WEIGHT_DECAY
    label_smoothing: float = LINEAR_LABEL_SMOOTHING


def validate_linear_probe_spec(spec: LinearProbeSpec) -> LinearProbeSpec:
    """Fail-closed unless every field equals its locked value exactly."""
    locked = LinearProbeSpec()
    if spec != locked:
        mismatched = [
            field for field in (
                "epochs", "repeat_per_epoch", "batch_size", "grad_accum",
                "lr", "weight_decay", "label_smoothing")
            if getattr(spec, field) != getattr(locked, field)
        ]
        raise P6BFailClosedError(
            f"linear probe spec deviates from locked constants: {mismatched}")
    return spec
