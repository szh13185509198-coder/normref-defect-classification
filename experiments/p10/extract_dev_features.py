#!/usr/bin/env python3
"""P10 step 1: extract dev features for the MVTec-FS **support pool only**.

Per image: global fusion feature (formal fusion_proto definition), DINOv2 patch tokens,
CLIP layer-18 patch tokens, and a 24x24 mask-area grid (oracle use only). Records whose
pool is not exactly ["support"] are never opened.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.extract_dev_features --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from experiments.p6 import p6b_stage2_models as models  # noqa: E402
from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

CLIP_LAYER = 18
OUT_DIR = PROJECT_ROOT / "ckpt" / "p10_pilot" / "dev_features"


def support_pool_records() -> dict:
    """{product: [record]} for single-type support-pool records only."""
    by_product = {p: [] for p in rc.PRODUCTS}
    with rc.RECORDS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("product") not in by_product:
                continue
            if record.get("source_pools") != ["support"]:
                continue  # firewall: formal queries and mixed pools are never touched
            if record.get("multi_type_removed") is True:
                continue
            labels = record.get("unique_defect_types")
            if not isinstance(labels, list) or len(labels) != 1:
                raise rc.P6BFailClosedError(f"support record without single label: {record}")
            by_product[record["product"]].append(record)
    for product, records in by_product.items():
        records.sort(key=lambda r: r["canonical_relative_path"])
        if not records:
            raise rc.P6BFailClosedError(f"empty support pool for {product}")
    return by_product


def mask_grid(record: dict, size: tuple[int, int], grid: int) -> torch.Tensor:
    """Area fraction of annotated defect per patch cell (oracle only)."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for rel in record.get("annotation_paths") or []:
        path = rc.DATASET_ROOT / rel
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for shape in data.get("shapes", []):
            points = [tuple(map(float, pt)) for pt in shape.get("points", [])]
            if shape.get("shape_type") == "rectangle" and len(points) == 2:
                (x0, y0), (x1, y1) = points
                draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
            elif len(points) >= 3:
                draw.polygon(points, fill=255)
    small = mask.resize((grid, grid), Image.BOX)
    return torch.from_numpy(numpy.asarray(small, dtype=numpy.float32) / 255.0).flatten()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    factory = support._build_lazy_clip_and_dino(
        SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pools = support_pool_records()
    for product in rc.PRODUCTS:
        records = pools[product]
        glob, dino_p, clip_p, masks = [], [], [], []
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start + args.batch_size]
            images, sizes = [], []
            for record in chunk:
                with Image.open(rc.DATASET_ROOT / record["canonical_relative_path"]) as handle:
                    img = handle.convert("RGB").copy()
                images.append(img)
                sizes.append(img.size)
            batch = torch.cat([models.deterministic_eval_tensor(im) for im in images]).to(device)
            with torch.no_grad():
                pooled, layer_tokens = factory.clip_model.visual(batch, [CLIP_LAYER])
                clip_global = models.l2_rows(pooled)
                dino_patches = factory.dino_extractor.forward_patches(batch)
                dino_global = models.l2_rows(dino_patches.mean(dim=1))
                fused = factory.fusion.fuse(clip_global, dino_global)
            tokens18 = layer_tokens[0][:, 1:, :]
            n_dino, n_clip = dino_patches.shape[1], tokens18.shape[1]
            if n_dino != n_clip or int(round(n_dino ** 0.5)) ** 2 != n_dino:
                raise rc.P6BFailClosedError(f"patch grids differ: dino {n_dino}, clip {n_clip}")
            grid = int(round(n_dino ** 0.5))
            glob.append(fused.float().cpu())
            dino_p.append(F.normalize(dino_patches.float(), dim=-1).half().cpu())
            clip_p.append(F.normalize(tokens18.float(), dim=-1).half().cpu())
            masks.extend(mask_grid(r, s, grid) for r, s in zip(chunk, sizes))
        torch.save({
            "schema_version": "p10_dev_features_v1",
            "product": product,
            "rel_paths": [r["canonical_relative_path"] for r in records],
            "labels": [r["unique_defect_types"][0] for r in records],
            "global_fusion": torch.cat(glob),
            "dino_patches": torch.cat(dino_p),
            "clip18_patches": torch.cat(clip_p),
            "mask_grid": torch.stack(masks),
            "clip_layer": CLIP_LAYER,
            "provenance": rc.git_provenance(),
        }, OUT_DIR / f"{product}.pt")
        print(f"[p10] {product}: {len(records)} support-pool images, grid {grid}x{grid}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
