"""P18 alternative frozen backbones for the report-only backbone swap (protocol §K).

Both backbones give a global embedding and patch tokens at several layers from ONE forward pass.
Nothing here touches the main CLIP ViT-L/14@336 + DINOv2-S pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from experiments.p6 import p6b_stage2_models as models
from experiments.revision import revision_common as rc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAYERS = (6, 8, 9, 10, 11, 12)
CLIP_B_RES = (224, 336)
DINO_B_CKPT = PROJECT_ROOT / "pretrained" / "dinov2_vitb14_pretrain.pth"
DINO_HUB_REPO = PROJECT_ROOT / "third_party" / "dinov2"
CLIP_B_CKPT = PROJECT_ROOT / "model" / "ViT-B-16.pt"
CLIP_B_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def eval_tensor(image: Image.Image, res: int) -> torch.Tensor:
    """The main deterministic view at 336; the same recipe (squash resize, bicubic, CLIP norm) otherwise."""
    if res == models.P6B_IMG_SIZE:
        return models.deterministic_eval_tensor(image)
    view = transforms.Compose([transforms.Resize((res, res), Image.BICUBIC), transforms.ToTensor(),
                               transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    return view(image.convert("RGB")).unsqueeze(0)


class DinoB:
    """DINOv2 ViT-B/14 at the main 336 view; global = final normalised CLS token."""

    name = "dinov2_b"
    res = models.P6B_IMG_SIZE

    def __init__(self, device, expected_sha256: str | None = None):
        if not DINO_B_CKPT.is_file():
            raise rc.P6BFailClosedError(f"missing {DINO_B_CKPT} (download the official DINOv2 ViT-B/14 weights first)")
        self.sha256 = file_sha256(DINO_B_CKPT)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise rc.P6BFailClosedError(f"DINOv2-B sha256 {self.sha256} != dev-stage {expected_sha256}")
        with torch.random.fork_rng(devices=[]):
            model = torch.hub.load(str(DINO_HUB_REPO), "dinov2_vitb14", source="local", pretrained=False)
        model.load_state_dict(torch.load(DINO_B_CKPT, map_location="cpu"), strict=True)
        self.model = model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def features(self, batch: torch.Tensor, layers) -> tuple[torch.Tensor, dict]:
        layers = sorted(set(layers) | {12})
        outs = self.model.get_intermediate_layers(batch, n=[L - 1 for L in layers], norm=True, return_class_token=True)
        by_layer = dict(zip(layers, outs))
        glob = F.normalize(by_layer[12][1].float(), dim=-1)
        return glob, {L: F.normalize(by_layer[L][0].float(), dim=-1) for L in layers}


class ClipB:
    """OpenAI CLIP ViT-B/16 at r x r; global = projected image embedding."""

    name = "clip_b16"

    def __init__(self, device, res: int):
        if file_sha256(CLIP_B_CKPT) != CLIP_B_SHA256:
            raise rc.P6BFailClosedError(f"{CLIP_B_CKPT} does not match the OpenAI ViT-B/16 sha256")
        from model import clip as clip_module
        clip_module._MODEL_CKPT_PATHS.setdefault("ViT-B-16", CLIP_B_CKPT)  # runtime registration only
        self.res = res
        self.sha256 = CLIP_B_SHA256
        model = clip_module.create_model(model_name="ViT-B-16", img_size=res, pretrained="openai",
                                         require_pretrained=True, device=torch.device(device))
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def features(self, batch: torch.Tensor, layers) -> tuple[torch.Tensor, dict]:
        layers = sorted(set(layers))
        pooled, tokens = self.model.visual(batch, layers)  # tokens come in increasing block order
        return F.normalize(pooled.float(), dim=-1), {L: F.normalize(t[:, 1:, :].float(), dim=-1)
                                                    for L, t in zip(layers, tokens)}


def extract(backbone, images, device, batch_size: int, layers=LAYERS) -> tuple[torch.Tensor, dict]:
    """images: PIL images or paths -> (global [N, d] float cpu, {layer: patches [N, P, d] half cpu})."""
    glob, patches = [], {}
    for start in range(0, len(images), batch_size):
        chunk = []
        for item in images[start:start + batch_size]:
            if isinstance(item, Image.Image):
                chunk.append(item)
            else:
                with Image.open(item) as handle:
                    chunk.append(handle.convert("RGB").copy())
        batch = torch.cat([eval_tensor(im, backbone.res) for im in chunk]).to(device)
        g, per = backbone.features(batch, layers)
        glob.append(g.cpu())
        for L, p in per.items():
            patches.setdefault(L, []).append(p.half().cpu())
    return torch.cat(glob), {L: torch.cat(v) for L, v in patches.items() if L in layers}


def build_all(device, dino_sha: str | None = None, clip_res=CLIP_B_RES) -> dict:
    out = {"dinov2_b": DinoB(device, dino_sha)}
    for r in clip_res:
        out[f"clip_b16_r{r}"] = ClipB(device, r)
    return out
