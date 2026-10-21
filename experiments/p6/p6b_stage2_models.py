#!/usr/bin/env python3
"""P6.B Stage 2B real-model factory — frozen CLIP/DINO/Fusion backbones.

Builds the frozen backbone feature extractors for the Stage 2 support-side
runner by REUSING the project's existing model loading code (no backbone is
duplicated here):

- CLIP: ``model.clip.create_model`` (ViT-L-14-336 / pretrained=openai, local
  ``model/ViT-L-14-336px.pt`` only — a missing local weight file raises,
  there is no network download path);
- DINO: ``model.dino_extractor.Dinov2Extractor`` (vit-s14, hub backend,
  offline local repo ``third_party/dinov2`` + checkpoint
  ``pretrained/dinov2_vits14_pretrain.pth``, pinned by
  ``EXPECTED_DINO_SHA256``);
- Frozen Fusion: ``concat(L2norm CLIP, L2norm DINO)`` then L2-normalize again
  (Addendum A §1.1), i.e. the same math as
  ``experiments.p6.p6b_stage2_execution.fuse_frozen_embeddings``.

Locked dimensions: CLIP=768, DINO=384, Fusion=1152; any deviation is
fail-closed. All backbones are frozen by contract (``eval()``,
``requires_grad=False``, ``torch.no_grad()`` extraction); outputs are
detached, finite, non-zero and dimension-checked.

This module never sets ``KMP_DUPLICATE_LIB_OK`` and keeps the
numpy-before-torch import order (cole/OpenMP safe).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# cole environment safe import order: numpy MUST be imported before torch to
# avoid the OpenMP runtime duplicate-library crash in this Windows env.
# Never work around this with KMP_DUPLICATE_LIB_OK — setting that flag is
# forbidden; keep the import order instead.
import numpy  # noqa: F401  (deliberately before torch)

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from experiments.p6.p6b_execution_common import P6BFailClosedError
from experiments.p6 import p6b_stage2_execution as stage2

# ---------------------------------------------------------------------------
# Locked backbone configuration (protocol §3.1 / Addendum A §1.1)
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME = "ViT-L-14-336"
CLIP_WEIGHT_RELATIVE_PATH = Path("model") / "ViT-L-14-336px.pt"
DINO_HUB_REPO_RELATIVE_PATH = Path("third_party") / "dinov2"
DINO_HUB_CHECKPOINT_RELATIVE_PATH = Path("pretrained") / "dinov2_vits14_pretrain.pth"

# Existing pin shared with the P3.x OOF runners (experiments/p2 source).
EXPECTED_DINO_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)

EXPECTED_CLIP_DIM = 768
EXPECTED_DINO_DIM = 384
EXPECTED_FUSION_DIM = 1152
P6B_IMG_SIZE = 336

# Locked deterministic 336 evaluation view (Addendum A §1.4): single
# Resize(336,336) -> ToTensor -> CLIP Normalize; no random augmentation.
EVAL_VIEW = transforms.Compose([
    transforms.Resize((P6B_IMG_SIZE, P6B_IMG_SIZE), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


def deterministic_eval_tensor(image: Image.Image) -> torch.Tensor:
    """Locked single deterministic 336 evaluation view -> [1, 3, 336, 336]."""
    if not isinstance(image, Image.Image):
        raise P6BFailClosedError("deterministic_eval_tensor expects a PIL image")
    return EVAL_VIEW(image.convert("RGB")).unsqueeze(0)


@dataclass(frozen=True)
class RealModelBundle:
    """Constructed frozen backbones + their provenance record."""

    clip_model: object = None
    dino_extractor: object = None
    clip_extractor: "ImageFeatureExtractor" = None
    dino_extractor_interface: "ImageFeatureExtractor" = None
    text_extractor: "TextFeatureExtractor" = None
    fusion: "FrozenFusion" = None
    provenance: dict = None


# ---------------------------------------------------------------------------
# Output contract helpers
# ---------------------------------------------------------------------------

def _validate_embedding_rows(tensor: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    """Detached, finite, non-zero rows of the locked dimension (contract)."""
    tensor = tensor.detach().float()
    if tensor.dim() != 2 or tensor.shape[1] != dim:
        raise P6BFailClosedError(
            f"{name}: expected [B, {dim}] embeddings, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise P6BFailClosedError(f"{name}: non-finite embedding values")
    if bool((tensor.norm(dim=1) == 0).any()):
        raise P6BFailClosedError(f"{name}: zero-norm embedding rows")
    return tensor


def _to_batch(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        batch = image
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
    else:
        if not isinstance(image, Image.Image):
            raise P6BFailClosedError(
                "embed_image expects a PIL image or a [3,H,W]/[B,3,H,W] tensor")
        batch = deterministic_eval_tensor(image)
    if batch.dim() != 4 or batch.shape[1] != 3:
        raise P6BFailClosedError(
            f"image batch must be [B, 3, H, W], got {tuple(batch.shape)}")
    return batch


# ---------------------------------------------------------------------------
# Frozen extractor implementations (Stage 2A interfaces)
# ---------------------------------------------------------------------------

class FrozenClipImageExtractor(stage2.ImageFeatureExtractor):
    """Native projected global image embedding of frozen CLIP, L2-normalized."""

    def __init__(self, clip_model, device, dim: int = EXPECTED_CLIP_DIM):
        self.clip_model = clip_model
        self.device = torch.device(device)
        self.dim = int(dim)

    def embed_image(self, image):
        batch = _to_batch(image).to(self.device)
        with torch.no_grad():
            projected, _ = self.clip_model.visual(batch, [])
        return _validate_embedding_rows(
            l2_rows(projected), self.dim, "CLIP embedding")[0]

    def embed_images(self, images):
        batch = torch.cat([_to_batch(img) for img in images], dim=0)
        batch = batch.to(self.device)
        with torch.no_grad():
            projected, _ = self.clip_model.visual(batch, [])
        return _validate_embedding_rows(l2_rows(projected), self.dim,
                                        "CLIP embedding")


class FrozenDinoImageExtractor(stage2.ImageFeatureExtractor):
    """Mean pooling over the full DINOv2 patch tokens, L2-normalized."""

    def __init__(self, dino_extractor, device, dim: int = EXPECTED_DINO_DIM):
        self.dino_extractor = dino_extractor
        self.device = torch.device(device)
        self.dim = int(dim)

    def embed_image(self, image):
        return self.embed_images([image])[0]

    def embed_images(self, images):
        batch = torch.cat([_to_batch(img) for img in images], dim=0)
        batch = batch.to(self.device)
        with torch.no_grad():
            patches = self.dino_extractor.forward_patches(batch)
        if patches is None or patches.dim() != 3:
            raise P6BFailClosedError(
                "DINOv2 extractor returned no patch tokens")
        return _validate_embedding_rows(
            l2_rows(patches.mean(dim=1)), self.dim, "DINO embedding")


class FrozenClipTextExtractor(stage2.TextFeatureExtractor):
    """Locked-prompt text embeddings from the frozen CLIP text encoder."""

    def __init__(self, clip_model, device):
        self.clip_model = clip_model
        self.device = torch.device(device)

    def embed_text(self, prompts):
        from model.tokenizer import tokenize
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if not prompts:
            raise P6BFailClosedError("embed_text received no prompts")
        tokens = tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
        return _validate_embedding_rows(
            l2_rows(text_features), EXPECTED_CLIP_DIM, "CLIP text embedding")


class FrozenFusion:
    """concat(normalized CLIP, normalized DINO) -> L2-normalize (locked)."""

    def fuse(self, clip_embeddings, dino_embeddings):
        fused = stage2.fuse_frozen_embeddings(clip_embeddings, dino_embeddings)
        if fused.shape[1] != EXPECTED_FUSION_DIM:
            raise P6BFailClosedError(
                f"fusion dim {fused.shape[1]} != locked {EXPECTED_FUSION_DIM}")
        return fused


def l2_rows(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.detach().float(), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Real factory (never called from unit tests)
# ---------------------------------------------------------------------------

def _fail_closed_offline_dino() -> None:
    """DINO must load from the offline local hub repo; the transformers/HF
    fallback would touch the network, so it is forbidden here."""
    if os.environ.get("HF_ENDPOINT"):
        raise P6BFailClosedError(
            "HF_ENDPOINT is set; the P6.B DINO branch must load offline from "
            "third_party/dinov2 without any HuggingFace mirror")


def build_clip_backbone(project_root: Path, device) -> tuple:
    """Load frozen CLIP ViT-L-14-336 from the local weight file only."""
    from model.clip import create_model
    weight_path = Path(project_root) / CLIP_WEIGHT_RELATIVE_PATH
    if not weight_path.is_file():
        raise P6BFailClosedError(
            f"local CLIP weights missing: {weight_path} "
            "(network download is forbidden; fail-closed)")
    clip_model = create_model(
        model_name=CLIP_MODEL_NAME,
        img_size=P6B_IMG_SIZE,
        pretrained="openai",
        require_pretrained=True,
        device=torch.device(device),
    )
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    proj = getattr(clip_model.visual, "proj", None)
    if proj is None or int(proj.shape[1]) != EXPECTED_CLIP_DIM:
        raise P6BFailClosedError(
            "CLIP visual projection does not produce the locked 768-dim "
            "projected global embedding")
    return clip_model, weight_path


def build_dino_backbone(project_root: Path, device) -> tuple:
    """Load frozen DINOv2 vit-s14 from the offline local hub repo."""
    from model.dino_extractor import Dinov2Extractor
    _fail_closed_offline_dino()
    hub_repo = Path(project_root) / DINO_HUB_REPO_RELATIVE_PATH
    hub_checkpoint = Path(project_root) / DINO_HUB_CHECKPOINT_RELATIVE_PATH
    if not hub_checkpoint.is_file():
        raise P6BFailClosedError(
            f"local DINOv2 checkpoint missing: {hub_checkpoint} "
            "(network download is forbidden; fail-closed)")
    actual_sha = stage2.file_sha256(hub_checkpoint)
    if actual_sha != EXPECTED_DINO_SHA256:
        raise P6BFailClosedError(
            f"DINOv2 checkpoint sha256 {actual_sha} != pinned "
            f"{EXPECTED_DINO_SHA256}")
    if not (hub_repo / "hubconf.py").is_file():
        raise P6BFailClosedError(
            f"offline DINOv2 hub repo missing hubconf.py: {hub_repo}")
    # isolated_rng semantics: DINO construction must not consume the caller's
    # RNG stream (protocol §3.1 Initialization / RNG).
    with torch.random.fork_rng(devices=[]):
        dino_extractor = Dinov2Extractor(
            model_name="dinov2_vits14",
            freeze=True,
            use_hub_first=True,
            patch_policy="full",
            strict_backend=True,
            hub_repo_path=str(hub_repo),
            hub_checkpoint_path=str(hub_checkpoint),
        )
    if getattr(dino_extractor, "backend", None) not in {"hub", "hub_local"}:
        raise P6BFailClosedError(
            f"DINO backend {dino_extractor.backend!r} != locked hub backend")
    if int(getattr(dino_extractor, "feature_dim", -1)) != EXPECTED_DINO_DIM:
        raise P6BFailClosedError(
            f"DINO feature dim {dino_extractor.feature_dim} != locked "
            f"{EXPECTED_DINO_DIM}")
    dino_extractor.eval()
    dino_extractor.to(torch.device(device))
    for param in dino_extractor.parameters():
        param.requires_grad = False
    return dino_extractor, hub_repo, hub_checkpoint


def build_real_factory(project_root, device=None) -> RealModelBundle:
    """Construct all frozen backbones; fail-closed on any missing local
    weight or dimension mismatch. Never downloads anything."""
    from provenance import file_identity, model_state_fingerprint
    project_root = Path(project_root)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, clip_weight_path = build_clip_backbone(project_root, device)
    dino_extractor, dino_hub_repo, dino_hub_checkpoint = build_dino_backbone(
        project_root, device)
    clip_extractor = FrozenClipImageExtractor(clip_model, device)
    dino_interface = FrozenDinoImageExtractor(dino_extractor, device)
    text_extractor = FrozenClipTextExtractor(clip_model, device)
    provenance = {
        "clip": {
            "model_name": CLIP_MODEL_NAME,
            "pretrained_identifier": f"openai:{CLIP_MODEL_NAME}",
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
            "expected_checkpoint_sha256": EXPECTED_DINO_SHA256,
            "local_source_path": str(Path(dino_hub_repo).resolve()),
            "loaded_state": model_state_fingerprint(dino_extractor.model),
            "network_download": False,
        },
        "fusion": {
            "operation": "concat(l2norm_clip, l2norm_dino)_then_l2norm",
            "expected_dims": {
                "clip": EXPECTED_CLIP_DIM,
                "dino": EXPECTED_DINO_DIM,
                "fusion": EXPECTED_FUSION_DIM,
            },
        },
        "frozen_contract": {
            "eval_mode": True,
            "requires_grad": False,
            "extraction_mode": "torch.no_grad",
            "outputs": "detached/finite/non-zero/dimension-checked",
        },
    }
    return RealModelBundle(
        clip_model=clip_model,
        dino_extractor=dino_extractor,
        clip_extractor=clip_extractor,
        dino_extractor_interface=dino_interface,
        text_extractor=text_extractor,
        fusion=FrozenFusion(),
        provenance=provenance,
    )
