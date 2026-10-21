#!/usr/bin/env python3
"""Assets for the draw.io method figure (paper Figure 3, V4): one MVTec-FS query, read from cached features only.

The query is the metal_nut example of fig_region_comparison (K = 5, seed 111, M = 20 bank). Everything shown in the
figure is recomputed from the P10 formal feature caches with the P14 code path (nothing is trained or re-extracted):
the patch anomaly map a(t), the soft weights w(t), the leave-one-out normal scores behind tau, and the three z-scored
class-score vectors that are summed for the decision. The recomputed predictions must equal the frozen P14 predictions,
otherwise nothing is written.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p20.make_fig3_assets
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402  (deliberately before torch)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p14.final_formal import ARR, SS, c1_descriptors  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

PRODUCT, SEED, K = "metal_nut", 111, 5
QUERY_SHA = "58cf9edeae910864755fefddad1208bb15bf7661959baf0cd76bad89c58182e9"
MVTEC_AD = PROJECT_ROOT.parent / "MVTEC"
P14_PRED = PROJECT_ROOT / "ckpt" / "p14_formal" / "mvtec" / "p14_mvtec_predictions.jsonl"
OUT = PROJECT_ROOT / "docs" / "paper" / "figures" / "assets" / "figure3" / "v4"
SIDE = 336


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def square(path: Path, size: int = SIDE) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB").resize((size, size), Image.BICUBIC)).astype(float) / 255


def upsample(values: torch.Tensor) -> np.ndarray:
    side = int(round(values.numel() ** 0.5))
    v = values.float().reshape(1, 1, side, side)
    return F.interpolate(v, size=(SIDE, SIDE), mode="bilinear", align_corners=False)[0, 0].numpy()


def save(array: np.ndarray, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8)).save(path, optimize=True)
    return path


def frozen_predictions() -> dict:
    preds = {}
    with P14_PRED.open(encoding="utf-8") as handle:
        for line in handle:
            u = json.loads(line)
            if u["product"] == PRODUCT and u["K"] == K and u["seed"] == SEED and u["method"] in (ARR, SS):
                for sha, p in zip(u["rgb_sha256"], u["pred"]):
                    preds[(u["method"], sha)] = p
    return preds


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.load(rp.FEATURE_DIR / f"{PRODUCT}.pt", map_location="cpu", weights_only=False)
    qf = torch.load(FORMAL_FEATURES / f"queries_{PRODUCT}.pt", map_location="cpu", weights_only=False)
    nf = torch.load(FORMAL_FEATURES / f"normals_{PRODUCT}.pt", map_location="cpu", weights_only=False)
    path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
    n_pool = len(dev["rel_paths"])
    g = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
    clip = torch.cat([dev["clip18_patches"], qf["clip18_patches"]]).to(device).float()
    name_index = {x: i for i, x in enumerate(nf["names"])}
    draw = nf["draws"][str(SEED)][:ra.M_PRIMARY]
    bank_imgs = nf["clip18_patches"].to(device).float()[torch.as_tensor([name_index[x] for x in draw], device=device)]
    manifest = rc.load_json(MANIFEST_DIR / PRODUCT / f"k{K}_s{SEED}.json")
    order = list(manifest["classes"]) if "class_order" not in manifest else list(manifest["class_order"])
    s_idx, s_lab, first_support = [], [], {}
    for c, cls in enumerate(order):
        for item in manifest["classes"][cls]["items"]:
            s_idx.append(path_index[item["rel_path"]])
            s_lab.append(c)
            first_support.setdefault(cls, item["rel_path"])
    si = torch.as_tensor(s_idx, device=device)
    qi = torch.arange(n_pool, n_pool + len(qf["rgb_sha256"]), device=device)
    j = list(qf["rgb_sha256"]).index(QUERY_SHA)
    with torch.no_grad():
        tau = fa.tau99(bank_imgs)
        loo = ab.loo_anomalies(bank_imgs).cpu().numpy()
        _, soft, rr = c1_descriptors(clip, bank_imgs, tau)
        n = len(order)
        z = {"global": rp._zscore(rp.simpleshot(g[si], s_lab, g[qi], n))[j],
             "appearance": rp._zscore(rp.simpleshot(soft[si], s_lab, soft[qi], n))[j],
             "deviation": rp._zscore(rp.simpleshot(rr[si], s_lab, rr[qi], n))[j]}
        fused = z["global"] + z["appearance"] + z["deviation"]
        a = ra.patch_anomaly(clip[n_pool + j:n_pool + j + 1], bank_imgs.reshape(-1, bank_imgs.shape[-1]))
        w = ab.region_weights(a, tau, soft=True)[0].cpu()
        a = a[0].cpu()
    frozen = frozen_predictions()
    recomputed = {ARR: order[int(fused.argmax())], SS: order[int(z["global"].argmax())]}
    if any(recomputed[m] != frozen[(m, QUERY_SHA)] for m in recomputed):
        raise SystemExit(f"recomputed {recomputed} != frozen {[frozen[(m, QUERY_SHA)] for m in recomputed]}")

    records = {}
    with rc.RECORDS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("rgb_sha256") == QUERY_SHA:
                records = r
    query_path = rc.DATASET_ROOT / records["canonical_relative_path"]
    view = square(query_path)
    inferno = colormaps["inferno"]
    a_norm = (upsample(a) - float(a.min())) / float(a.max() - a.min())
    w_norm = upsample(w) / float(w.max())
    assets = [("query", query_path, save(view, "query.png"), "336x336 bicubic resize"),
              ("anomaly_map", query_path, save(0.45 * view + 0.55 * inferno(a_norm)[..., :3], "query_anomaly_map.png"),
               "a(t) min-max normalised, bilinear upsampling to 336, inferno at alpha 0.55 (as fig_qualitative)"),
              ("weight_map", query_path, save(0.35 * view + 0.65 * inferno(w_norm)[..., :3], "query_weight_map.png"),
               "w(t) / max w, bilinear upsampling to 336, inferno at alpha 0.65 (zero weight renders dark)")]
    for i, name in enumerate(draw[:3]):
        src = MVTEC_AD / PRODUCT / "train" / "good" / name
        assets.append((f"normal_{i}", src, save(square(src, 300), f"normal_{i}.png"), "300x300 bicubic resize"))
    for cls in order:
        src = rc.DATASET_ROOT / first_support[cls]
        assets.append((f"support_{cls}", src, save(square(src, 300), f"support_{cls}.png"), "300x300 bicubic resize"))

    edges = np.linspace(0.0, max(float(loo.max()), float(a.max())) + 1e-6, 41)
    manifest_out = {
        "schema": "figure3_v4_asset_manifest_v1",
        "source": "P10 formal feature caches; P14 code path (experiments.p14.final_formal.c1_descriptors)",
        "product": PRODUCT, "K": K, "seed": SEED, "bank_size_M": ra.M_PRIMARY, "bank_draw": list(draw),
        "query_rgb_sha256": QUERY_SHA, "query_relative_path": records["canonical_relative_path"],
        "class_order": order, "frozen_predictions": {m: frozen[(m, QUERY_SHA)] for m in (SS, ARR)},
        "tau_q99": tau, "patches_with_positive_weight": int((w > 0).sum()), "patches_total": int(w.numel()),
        "anomaly_query_min_max": [float(a.min()), float(a.max())],
        "z_scores": {k: [round(float(v), 4) for v in z[k]] for k in z},
        "fused": [round(float(v), 4) for v in fused],
        "loo_histogram": {"edges": [round(float(e), 5) for e in edges],
                          "counts": np.histogram(loo, bins=edges)[0].tolist(), "n": int(loo.size)},
        "query_histogram": {"counts": np.histogram(a.numpy(), bins=edges)[0].tolist(), "n": int(a.numel())},
        "query_scores_above_tau": [round(float(v), 5) for v in sorted(a[a > tau].tolist())],
        "assets": [{"role": role, "source_path": str(src.relative_to(src.anchor)), "source_sha256": sha256(src),
                    "derived_path": str(dst.relative_to(PROJECT_ROOT)).replace("\\", "/"), "derived_sha256": sha256(dst),
                    "operation": op} for role, src, dst, op in assets],
    }
    rc.write_json(OUT / "ASSET_MANIFEST_V4.json", manifest_out)
    print(json.dumps({k: manifest_out[k] for k in ("class_order", "frozen_predictions", "tau_q99",
                                                  "patches_with_positive_weight", "z_scores", "fused")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
