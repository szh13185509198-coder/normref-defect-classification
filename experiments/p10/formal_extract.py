#!/usr/bin/env python3
"""P10 formal step 1: features for the 530 formal queries and the formal-seed normal banks.

Writes features only (no metric). Query masks are rasterized for the oracle rows only.
Support features are not re-extracted: the formal manifests draw from the support pool,
whose features already exist in ``ckpt/p10_pilot/dev_features``.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.formal_extract --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p6 import p6b_stage2_models as models  # noqa: E402
from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p10.extract_dev_features import CLIP_LAYER, mask_grid  # noqa: E402
from experiments.p10.extract_normal_features import BANK_SIZE  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT_DIR = PROJECT_ROOT / "ckpt" / "p10_formal" / "features"


def formal_bank_draws(normal_dir: Path, product: str) -> dict:
    names = sorted(f.name for f in normal_dir.glob("*.png"))
    return {seed: random.Random(f"p10A|{seed}|{product}").sample(names, BANK_SIZE)
            for seed in rc.SEEDS}


def _extract(factory, images, device, with_global: bool):
    batch = torch.cat([models.deterministic_eval_tensor(im) for im in images]).to(device)
    with torch.no_grad():
        pooled, layer_tokens = factory.clip_model.visual(batch, [CLIP_LAYER])
        dino = factory.dino_extractor.forward_patches(batch)
        glob = (factory.fusion.fuse(models.l2_rows(pooled), models.l2_rows(dino.mean(dim=1)))
                if with_global else None)
    return (None if glob is None else glob.float().cpu(),
            F.normalize(dino.float(), dim=-1).half().cpu(),
            F.normalize(layer_tokens[0][:, 1:, :].float(), dim=-1).half().cpu())


def query_annotation_records() -> dict:
    """{(product, rgb_sha256): raw record} for query-pool records (annotation paths)."""
    out = {}
    with rc.RECORDS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("source_pools") == ["query"]:
                out[(record["product"], record["rgb_sha256"])] = record
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    factory = support._build_lazy_clip_and_dino(
        SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    queries, query_audit = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE)
    raw = query_annotation_records()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(args.mvtec_root)
    for product in rc.PRODUCTS:
        records = [q for q in queries if q["product"] == product]
        glob, dino_p, clip_p, masks = [], [], [], []
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start + args.batch_size]
            images, sizes = [], []
            for record in chunk:
                with Image.open(record["image_path"]) as handle:
                    img = handle.convert("RGB").copy()
                images.append(img)
                sizes.append(img.size)
            g, d, c = _extract(factory, images, device, with_global=True)
            glob.append(g)
            dino_p.append(d)
            clip_p.append(c)
            grid = int(round(d.shape[1] ** 0.5))
            masks.extend(mask_grid(raw[(product, r["rgb_sha256"])], s, grid) for r, s in zip(chunk, sizes))
        torch.save({
            "schema_version": "p10_formal_query_features_v1", "product": product,
            "rgb_sha256": [r["rgb_sha256"] for r in records],
            "global_fusion": torch.cat(glob), "dino_patches": torch.cat(dino_p),
            "clip18_patches": torch.cat(clip_p), "mask_grid_oracle_only": torch.stack(masks),
            "formal_query_count_total": query_audit["formal_query_count"],
            "provenance": rc.git_provenance(),
        }, OUT_DIR / f"queries_{product}.pt")

        normal_dir = root / product / "train" / "good"
        draws = formal_bank_draws(normal_dir, product)
        names = sorted({n for draw in draws.values() for n in draw})
        dino_n, clip_n = [], []
        for start in range(0, len(names), args.batch_size):
            images = []
            for name in names[start:start + args.batch_size]:
                with Image.open(normal_dir / name) as handle:
                    images.append(handle.convert("RGB").copy())
            _, d, c = _extract(factory, images, device, with_global=False)
            dino_n.append(d)
            clip_n.append(c)
        torch.save({
            "schema_version": "p10_formal_normal_features_v1", "product": product,
            "source": f"{product}/train/good", "names": names,
            "draws": {str(s): d for s, d in draws.items()},
            "dino_patches": torch.cat(dino_n), "clip18_patches": torch.cat(clip_n),
            "provenance": rc.git_provenance(),
        }, OUT_DIR / f"normals_{product}.pt")
        print(f"[p10F] {product}: {len(records)} queries, {len(names)} normal images", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
