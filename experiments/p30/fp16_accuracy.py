#!/usr/bin/env python3
"""P30 part B (report only): does CUDA-autocast FP16 change our CLIP-L/14@336-only method (P17 `a_r`) on MVTec-FS?

Protocol: docs/P30_CLIPL14_ONLY_COST_AND_FP16_PROTOCOL.md. Re-extracts the CLIP ViT-L/14@336 layer-18 patch tokens of
every MVTec-FS formal image (support pool, queries, normal images) with the P10 extraction path (same order, batch 4),
twice per batch: FP32 (gate 1: bitwise equal to the P10 caches) and FP16 autocast. Scores `a_r` = z(SS[h_A]) +
z(SS[h_R]) with the P14 code path for every product / K / seed (M = 20) from the cached and from the FP16 features
(gate 2: the cached features reproduce P17 `a_r` exactly). Only CLIP is loaded. Writes ckpt/p30/fp16/ once.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p30.fp16_accuracy --mvtec-root D:\\intergration\\MVTEC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p6 import p6b_stage2_models as models  # noqa: E402
from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.extract_dev_features import CLIP_LAYER, support_pool_records  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p12.threshold_controls import exact_equal  # noqa: E402
from experiments.p14.final_formal import c1_descriptors  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p30" / "fp16"
P17_MVTEC = PROJECT_ROOT / "ckpt" / "p17" / "mvtec" / "p17_mvtec_results.json"
AR, AR16 = "a_r", "a_r_fp16"
BATCH = 4   # = P10; the features depend bitwise on the batch composition


def open_rgb(path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB").copy()


def extract(clip_model, paths, device):
    """[N, 576, 1024] float16 CPU tensors: FP32 forward (as P10) and FP16-autocast forward, batch by batch."""
    fp32, fp16 = [], []
    for start in range(0, len(paths), BATCH):
        batch = torch.cat([models.deterministic_eval_tensor(open_rgb(p)) for p in paths[start:start + BATCH]]).to(device)
        with torch.no_grad():
            _, tokens = clip_model.visual(batch, [CLIP_LAYER])
            fp32.append(F.normalize(tokens[0][:, 1:, :].float(), dim=-1).half().cpu())
            with torch.autocast("cuda", dtype=torch.float16):
                _, tokens16 = clip_model.visual(batch, [CLIP_LAYER])
            fp16.append(F.normalize(tokens16[0][:, 1:, :].float(), dim=-1).half().cpu())
    return torch.cat(fp32), torch.cat(fp16)


def score_product(product, dev_clip, q_clip, normal_clip, nf, q_true, shas, method, device, sink):
    rows = []
    n_pool = dev_clip.shape[0]
    clip = torch.cat([dev_clip, q_clip]).to(device).float()
    normals = normal_clip.to(device).float()
    name_index = {x: i for i, x in enumerate(nf["names"])}
    path_index = {p: i for i, p in enumerate(DEV_PATHS[product])}
    qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
    with torch.no_grad():
        for seed in rc.SEEDS:
            draw = nf["draws"][str(seed)][:ra.M_PRIMARY]
            bank_imgs = normals[torch.as_tensor([name_index[x] for x in draw], device=device)]
            _, soft, rr = c1_descriptors(clip, bank_imgs, fa.tau99(bank_imgs))
            for k_shot in rc.K_VALUES:
                manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                order = list(manifest["class_order"])
                s_idx, s_lab = [], []
                for c, cls in enumerate(order):
                    for item in manifest["classes"][cls]["items"]:
                        s_idx.append(path_index[item["rel_path"]])
                        s_lab.append(c)
                si = torch.as_tensor(s_idx, device=device)
                n = len(order)
                sc = rp._zscore(rp.simpleshot(soft[si], s_lab, soft[qi], n)) + \
                    rp._zscore(rp.simpleshot(rr[si], s_lab, rr[qi], n))
                rows += fa.score_rows({method: sc}, order, q_true, {"product": product, "K": k_shot, "seed": seed},
                                      sink, shas)
    return rows


DEV_PATHS: dict = {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if OUT.exists():
        raise rc.P6BFailClosedError(f"{OUT} exists; P30 part B is single-shot")
    device = torch.device(args.device)
    clip_model, weight_path = models.build_clip_backbone(PROJECT_ROOT, device)
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    pools = support_pool_records()
    gate1, rows32, rows16, sink32, sink16 = {}, [], [], [], []
    for product in rc.PRODUCTS:
        dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
        nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
        dev_records = pools[product]
        if [r["canonical_relative_path"] for r in dev_records] != list(dev["rel_paths"]):
            raise rc.P6BFailClosedError(f"{product}: support-pool order differs from the P10 cache")
        q_records = [q for q in queries if q["product"] == product]
        if [q["rgb_sha256"] for q in q_records] != list(qf["rgb_sha256"]):
            raise rc.P6BFailClosedError(f"{product}: query order differs from the P10 cache")
        DEV_PATHS[product] = list(dev["rel_paths"])
        normal_dir = Path(args.mvtec_root) / product / "train" / "good"
        d32, d16 = extract(clip_model, [rc.DATASET_ROOT / r["canonical_relative_path"] for r in dev_records], device)
        q32, q16 = extract(clip_model, [q["image_path"] for q in q_records], device)
        n32, n16 = extract(clip_model, [normal_dir / x for x in nf["names"]], device)
        gate1[product] = {name: float((new.float() - old.float()).abs().max())
                          for name, new, old in (("support_pool", d32, dev["clip18_patches"]),
                                                 ("queries", q32, qf["clip18_patches"]),
                                                 ("normals", n32, nf["clip18_patches"]))}
        if any(v != 0.0 for v in gate1[product].values()):
            raise rc.P6BFailClosedError(f"gate 1 (FP32 re-extraction == P10 cache) failed for {product}: {gate1[product]}")
        q_true = [q["true_label"] for q in q_records]
        shas = list(qf["rgb_sha256"])
        rows32 += score_product(product, dev["clip18_patches"], qf["clip18_patches"], nf["clip18_patches"], nf,
                                q_true, shas, AR, device, sink32)
        rows16 += score_product(product, d16, q16, n16, nf, q_true, shas, AR16, device, sink16)
        print(f"[p30-fp16] {product}: gate 1 OK; {len(q_records)} queries", flush=True)

    p17_rows = [r for r in rc.load_json(P17_MVTEC)["raw_metrics"] if r["method"] == AR]
    gate2 = exact_equal(rows32, p17_rows, (AR,), ("product", "K", "seed"))
    if gate2 >= 1e-9:
        raise rc.P6BFailClosedError(f"gate 2 (cached features reproduce P17 a_r) failed: max |d| = {gate2}")
    grid = dict(rc.metric_grid_from_rows(rows32))
    grid.update(rc.metric_grid_from_rows(rows16))
    c = contrast(grid, AR, "balanced_accuracy", target=AR16)
    agree = total = 0
    by_key = {(u["product"], u["K"], u["seed"]): u for u in sink32}
    for u in sink16:
        ref = by_key[(u["product"], u["K"], u["seed"])]
        agree += sum(a == b for a, b in zip(u["pred"], ref["pred"]))
        total += len(u["pred"])
    levels = {m: {str(k): 100 * s[str(k)]["balanced_accuracy"]["mean"] for k in rc.K_VALUES}
              for m, s in ((m, rc.product_macro_summary(grid, m)) for m in (AR, AR16))}
    result = {"schema_version": "p30_fp16_v1", "protocol": "docs/P30_CLIPL14_ONLY_COST_AND_FP16_PROTOCOL.md",
              "label": "report only", "provenance": rc.git_provenance(), "clip_weights": str(weight_path),
              "gate1_fp32_reextraction_max_abs_diff": gate1, "gate2_vs_p17_a_r_max_abs_diff": gate2,
              "levels_ba_pct": levels,
              "fp16_minus_fp32_ba_pp": {"per_K": {k: 100 * v["D"] for k, v in c["per_K"].items()},
                                        "Dbar": 100 * c["Dbar"], "ci95": [100 * x for x in c["Dbar_ci95"]],
                                        "positive_products": c["Dbar_positive_products"]},
              "prediction_agreement": {"same": agree, "total": total, "rate": agree / total},
              "raw_metrics": rows32 + rows16}
    OUT.mkdir(parents=True)
    rc.write_json(OUT / "p30_fp16_results.json", result)
    rc.write_jsonl(OUT / "p30_fp16_predictions.jsonl", sink32 + sink16)
    print(json.dumps({k: result[k] for k in ("gate2_vs_p17_a_r_max_abs_diff", "levels_ba_pct", "fp16_minus_fp32_ba_pp",
                                             "prediction_agreement")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
