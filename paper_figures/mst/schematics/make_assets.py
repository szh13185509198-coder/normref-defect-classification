#!/usr/bin/env python3
"""Image material for the colour schematics, computed from the official MVTec AD images.

Product: bottle. Query: test/broken_small/014.png, the published success case of the qualitative figure (file SHA-256
c74fb6a0..., predicted broken_small by the proposed method at K = 5, seed 111). Bank: the first M = 20 images of
train/good (sorted names). Chosen among the published success cases because its evidence lies on the defect: 87 % of
the weight w falls inside the ground-truth mask (transistor cut_lead/007: 12 %, a missing-part defect).

The maps are recomputed for illustration with the same definitions as the paper (CLIP ViT-L/14 at 336 px, layer-18
patch tokens, L2-normalised; a(t) = 1 - max cosine to the bank; tau = 0.99 quantile of leave-one-out bank-patch
scores; w(t) = max(a(t) - tau, 0)) but with the Hugging Face weights openai/clip-vit-large-patch14-336 and a plain
336 x 336 resize, not the frozen feature caches; they illustrate the mechanism and are not used for any number.

    python paper_figures/mst/schematics/make_assets.py --mvtec /home/user/mvtec_ad
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "bottle"
SIDE, LAYER, M = 336, 18, 20
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
PRODUCT, QUERY, QUERY_SHA = "bottle", ("broken_small", "014"), "c74fb6a0"
SUPPORT = {"broken_large": "000", "broken_small": "000", "contamination": "000"}


def tensor(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((SIDE, SIDE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im).astype(np.float32) / 255).permute(2, 0, 1)
    return ((x - MEAN) / STD).unsqueeze(0)


def view(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((SIDE, SIDE), Image.BICUBIC)).astype(float) / 255


def save(arr: np.ndarray, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(p, optimize=True)
    return p.name


def upsample(v: torch.Tensor) -> np.ndarray:
    g = int(round(v.numel() ** 0.5))
    return F.interpolate(v.float().reshape(1, 1, g, g), size=(SIDE, SIDE), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mvtec", type=Path, required=True)
    a = ap.parse_args(argv)
    from transformers import CLIPVisionModel
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").eval()
    root = a.mvtec / PRODUCT
    query = root / "test" / QUERY[0] / f"{QUERY[1]}.png"
    sha = hashlib.sha256(query.read_bytes()).hexdigest()
    if not sha.startswith(QUERY_SHA):
        raise SystemExit(f"unexpected query file {sha}")
    goods = sorted((root / "train" / "good").glob("*.png"))[:M]

    @torch.no_grad()
    def tokens(paths):
        out = []
        for p in paths:
            h = model(pixel_values=tensor(p), output_hidden_states=True).hidden_states[LAYER][0, 1:]
            out.append(F.normalize(h, dim=-1))
        return torch.stack(out)

    bank = tokens(goods)                       # [M, 576, d]
    q = tokens([query])[0]                     # [576, d]
    flat = bank.reshape(-1, bank.shape[-1])
    a_q = 1 - (q @ flat.T).max(dim=1).values   # a(t) of the query
    loo = []
    for i in range(M):
        others = torch.cat([bank[:i], bank[i + 1:]]).reshape(-1, bank.shape[-1])
        loo.append(1 - (bank[i] @ others.T).max(dim=1).values)
    loo = torch.cat(loo)
    tau = float(torch.quantile(loo, 0.99))
    w = (a_q - tau).clamp(min=0)
    n_exceed = int((w > 0).sum())

    inferno = colormaps["inferno"]
    v = view(query)
    an = upsample((a_q - a_q.min()) / (a_q.max() - a_q.min()))
    wn = upsample(w / w.max())
    files = {"query": save(v, "query.png"),
             "anomaly_map": save(0.45 * v + 0.55 * inferno(an)[..., :3], "query_anomaly_map.png"),
             "weight_map": save(0.35 * v * (0.25 + 0.75 * (wn > 0.02)[..., None]) + 0.65 * inferno(wn)[..., :3],
                                "query_weight_map.png")}
    for j in range(3):
        files[f"normal_{j}"] = save(view(goods[j]), f"normal_{j}.png")
    for t, n in SUPPORT.items():
        files[f"support_{t}"] = save(view(root / "test" / t / f"{n}.png"), f"support_{t}.png")
    hist = {"loo_scores": np.histogram(loo.numpy(), bins=40)[0].tolist(),
            "loo_edges": np.histogram(loo.numpy(), bins=40)[1].round(5).tolist()}
    mask = np.asarray(Image.open(root / "ground_truth" / QUERY[0] / f"{QUERY[1]}_mask.png").convert("L")
                      .resize((24, 24), Image.BOX)).flatten() > 0
    in_mask = float((w.numpy() * mask).sum() / w.sum())
    manifest = {"product": PRODUCT, "query": str(query.relative_to(a.mvtec)), "query_sha256": sha,
                "published_prediction": "broken_small (proposed method, K = 5, seed 111; success case of the "
                                        "qualitative figure)", "weight_share_inside_gt_mask": round(in_mask, 3),
                "bank": [str(g.relative_to(a.mvtec)) for g in goods],
                "support_examples": {t: f"{PRODUCT}/test/{t}/{n}.png" for t, n in SUPPORT.items()},
                "features": f"Hugging Face openai/clip-vit-large-patch14-336, hidden state {LAYER}, patch tokens, "
                            "L2-normalised, 336 x 336 bicubic resize (illustration only)",
                "tau_0.99_loo": tau, "patches_above_tau": n_exceed, "patches": int(a_q.numel()),
                "files": files, **hist}
    (OUT / "ASSETS.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"tau = {tau:.4f}; {n_exceed} of {a_q.numel()} query patches above tau; wrote {len(files)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
