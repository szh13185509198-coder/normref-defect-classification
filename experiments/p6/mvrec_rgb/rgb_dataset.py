"""Whole-image RGB support dataset for the MVREC-RGB adaptation.

This is the ONLY data-interface change allowed by Protocol A section 5:
"replacing bbox/ROI loaders with whole-image loading" and "replacing
annotation-derived views with a fixed, predeclared global-view list".

The dataset reads exclusively:
- the frozen P6.B support manifest (``rel_path`` + ``rgb_sha256`` per class),
- the original whole RGB image bytes,

and produces the same sample schema the official ``ClipModel.get_mvrec``
consumes: ``x`` (B,V,C,H,W image views), ``y`` (image-level class index) and a
runtime-generated constant full-frame mask placeholder for the unchanged
AlphaCLIP backbone API.

Forbidden inputs (never read, directly or indirectly):
- CSV annotation columns (x1/x2/y1/y2/points/label/...),
- GT mask / bbox / ROI / crop center / scale / offset,
- query manifests or query pixels.
"""

import hashlib
import json
import os

import torch
from PIL import Image
from torchvision import transforms

from experiments.p6.mvrec_rgb.rgb_param import (
    OFFICIAL_INPUT_SHAPE,
    OFFICIAL_TEXT_TEMPLATE,
)

# Official CLIP normalization constants (modules/clip_preprocess.py).
_RGB_MEAN = (0.48145466, 0.4578275, 0.40821073)
_RGB_STD = (0.26862954, 0.26130258, 0.27577711)

# Official mask normalization constants (modules/clip_preprocess.py
# mask_transform: Normalize(0.5, 0.26)).
_MASK_MEAN = 0.5
_MASK_STD = 0.26

# The predeclared fixed global-view list: exactly one view, the whole image
# resized deterministically to the official input resolution.  Identical rule
# for every image; computed from no annotation and no model prediction.
FIXED_GLOBAL_VIEWS = ("whole_image",)


class MVRecRgbSupportDataset(torch.utils.data.Dataset):
    """Support-only whole-RGB dataset backed by a frozen P6.B manifest."""

    def __init__(self, manifest_path, image_root, device=None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.product = self.manifest["product"]
        self.seed = int(self.manifest["seed"])
        self.k_shot = int(self.manifest["K"])
        self.class_order = list(self.manifest["class_order"])
        self.image_root = os.path.abspath(image_root)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = OFFICIAL_INPUT_SHAPE

        # Deterministic, label-independent whole-image transform.  Same rule
        # for every image (fixed global view); no augmentation is applied in
        # the support-only audit path.
        self._rgb_transform = transforms.Compose(
            [
                self._convert_rgb,
                transforms.ToTensor(),
                transforms.Normalize(_RGB_MEAN, _RGB_STD),
            ]
        )

        # Build the flattened sample list in class_order sequence.
        self.samples = []
        for class_idx, class_name in enumerate(self.class_order):
            items = self.manifest["classes"][class_name]["items"]
            assert len(items) == self.k_shot, (
                f"manifest K={self.k_shot} but class {class_name} has "
                f"{len(items)} items"
            )
            for item in items:
                self.samples.append(
                    {
                        "rel_path": item["rel_path"],
                        "rgb_sha256": item["rgb_sha256"],
                        "class_name": class_name,
                        "class_idx": class_idx,
                    }
                )

    @staticmethod
    def _convert_rgb(img):
        return img.convert("RGB")

    @staticmethod
    def sha256_file(path, chunk=1 << 20):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    def verify_support_identity(self):
        """Recompute RGB SHA256 of every sample and compare with the manifest.

        Returns (ok, mismatches).  Called by the A4 smoke; no model involved.
        """
        mismatches = []
        for s in self.samples:
            full = os.path.join(self.image_root, s["rel_path"].replace("/", os.sep))
            actual = self.sha256_file(full)
            if actual != s["rgb_sha256"]:
                mismatches.append((s["rel_path"], s["rgb_sha256"], actual))
        return len(mismatches) == 0, mismatches

    def _make_placeholder_mask(self):
        """Runtime-generated constant full-frame mask (all ones).

        Generated independently of every annotation and every model output;
        recorded as an API placeholder for the unchanged AlphaCLIP backbone.
        Normalized with the official mask constants so the placeholder value
        is exactly (1 - 0.5) / 0.26.
        """
        ones = torch.ones(1, self.input_shape, self.input_shape)
        return ((ones - _MASK_MEAN) / _MASK_STD).to(self.device).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        full = os.path.join(self.image_root, s["rel_path"].replace("/", os.sep))
        img = Image.open(full)
        img = img.resize((self.input_shape, self.input_shape))
        x = self._rgb_transform(img).to(self.device)
        x = x.unsqueeze(0)  # V = 1 fixed whole-image view
        masks = self._make_placeholder_mask().unsqueeze(0)  # [1, 1, H, W]
        return {
            "x": x,                    # [V=1, C, H, W]
            "y": torch.tensor([s["class_idx"]], device=self.device),
            "masks": masks,            # [V=1, 1, H, W] constant placeholder
            "rel_path": s["rel_path"],
            "rgb_sha256": s["rgb_sha256"],
            "class_name": s["class_name"],
        }

    def get_collate_fn(self):
        """Collate into the official get_mvrec schema: x/masks [B,V,...], y [B]."""
        ds = self

        def collate(batch):
            x = torch.stack([b["x"] for b in batch], dim=0)          # [B,V,C,H,W]
            masks = torch.stack([b["masks"] for b in batch], dim=0)  # [B,V,1,H,W]
            y = torch.stack([b["y"] for b in batch], dim=0).view(-1)  # [B]
            out = {"x": x, "y": y, "masks": masks}
            for k in ("rel_path", "rgb_sha256", "class_name"):
                out[k] = [b[k] for b in batch]
            return out

        return collate
