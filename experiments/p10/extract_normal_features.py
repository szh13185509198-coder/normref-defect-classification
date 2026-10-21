#!/usr/bin/env python3
"""P10-A step 1: patch features for the drawn MVTec AD ``train/good`` normal banks.

Only ``<mvtec-root>/<product>/train/good`` is read. Per product the union of the five
dev-seed draws (M=20, nested M=5) is extracted once.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.extract_normal_features --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
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
from experiments.p10.extract_dev_features import CLIP_LAYER  # noqa: E402
from experiments.p10.run_pilot import DEV_SEEDS  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

BANK_SIZE = 20
OUT_DIR = PROJECT_ROOT / "ckpt" / "p10_pilot" / "normal_features"


def bank_draws(normal_dir: Path, product: str) -> dict:
    names = sorted(f.name for f in normal_dir.glob("*.png"))
    if len(names) < BANK_SIZE:
        raise rc.P6BFailClosedError(f"{product}: only {len(names)} normal images")
    return {seed: random.Random(f"p10A|{seed}|{product}").sample(names, BANK_SIZE)
            for seed in DEV_SEEDS}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    root = Path(args.mvtec_root)
    device = torch.device(args.device)
    factory = support._build_lazy_clip_and_dino(
        SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for product in rc.PRODUCTS:
        normal_dir = root / product / "train" / "good"
        if not normal_dir.is_dir():
            raise rc.P6BFailClosedError(f"missing {normal_dir}")
        draws = bank_draws(normal_dir, product)
        names = sorted({n for draw in draws.values() for n in draw})
        dino_p, clip_p = [], []
        for start in range(0, len(names), args.batch_size):
            images = []
            for name in names[start:start + args.batch_size]:
                with Image.open(normal_dir / name) as handle:
                    images.append(handle.convert("RGB").copy())
            batch = torch.cat([models.deterministic_eval_tensor(im) for im in images]).to(device)
            with torch.no_grad():
                _, layer_tokens = factory.clip_model.visual(batch, [CLIP_LAYER])
                dino = factory.dino_extractor.forward_patches(batch)
            dino_p.append(F.normalize(dino.float(), dim=-1).half().cpu())
            clip_p.append(F.normalize(layer_tokens[0][:, 1:, :].float(), dim=-1).half().cpu())
        torch.save({
            "schema_version": "p10_normal_features_v1",
            "product": product,
            "source": f"{product}/train/good",
            "names": names,
            "draws": {str(s): d for s, d in draws.items()},
            "dino_patches": torch.cat(dino_p),
            "clip18_patches": torch.cat(clip_p),
            "provenance": rc.git_provenance(),
        }, OUT_DIR / f"{product}.pt")
        print(f"[p10A] {product}: {len(names)} normal images extracted "
              f"(union of {len(DEV_SEEDS)} draws of {BANK_SIZE})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
