#!/usr/bin/env python3
"""P34 (report only): does FP16 autocast change our method on the single alternative backbones?

Protocol: docs/P34_FP16_ALT_BACKBONES_PROTOCOL.md. For DINOv2-B@336 (layer 11, P18), CLIP-B/16@336 (layer 8, P18)
and CLIP-B/16@224 (P24 layer) on MVTec-FS: re-extract support pool, queries and normal images with the P18 / P24 code
path (``experiments.p18.backbones.extract``, batch 4), once in FP32 and once under CUDA autocast FP16, and score ARR
(= ``experiments.p14.final_formal.episode_scores``) for every product / K / seed (M = 20).
Gate: the FP32 rows must reproduce P18 / P24 ``arr_*`` exactly. Then FP16 − FP32 paired product-cluster bootstrap,
per-query agreement and the P30 decision rule. Writes ckpt/p34/ once.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p34.fp16_alt --mvtec-root D:\\intergration\\MVTEC
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

from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.extract_dev_features import support_pool_records  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p12.threshold_controls import exact_equal  # noqa: E402
from experiments.p14 import final_formal as ff  # noqa: E402
from experiments.p18 import backbones as bb  # noqa: E402
from experiments.p18.run import c1_parts, dev_selection, open_rgb, parse_config  # noqa: E402
from experiments.p24.clipb224 import layer_224  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT = PROJECT_ROOT / "ckpt" / "p34"
P18 = PROJECT_ROOT / "ckpt" / "p18" / "mvtec" / "p18_mvtec_results.json"
P24 = PROJECT_ROOT / "ckpt" / "p24" / "mvtec" / "p24_results.json"
BATCH = 4   # = P18 / P24


def extract(backbone, images, device, layer, precision):
    if precision == "fp16":
        with torch.autocast("cuda", dtype=torch.float16):
            return bb.extract(backbone, images, device, BATCH, layers=(layer,))
    return bb.extract(backbone, images, device, BATCH, layers=(layer,))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if OUT.exists():
        raise rc.P6BFailClosedError(f"{OUT} exists; P34 is single-shot")
    device, mvtec_root = torch.device(a.device), Path(a.mvtec_root)
    sel = dev_selection()
    _, res336, layer336 = parse_config(sel["clip_b16"])
    _, _, layer_b = parse_config(sel["dinov2_b"])
    configs = {"dinov2_b": (bb.DinoB(device, sel["dinov2_b_sha256"]), layer_b, P18),
               "clip_b16": (bb.ClipB(device, res336), layer336, P18),
               "clip_b16_r224": (bb.ClipB(device, 224), layer_224(), P24)}
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    by_key = {(q["product"], q["rgb_sha256"]): q for q in queries}
    pools = support_pool_records()
    rows, sink = [], []
    for product in rc.PRODUCTS:
        dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
        nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
        if [r["canonical_relative_path"] for r in pools[product]] != list(dev["rel_paths"]):
            raise rc.P6BFailClosedError(f"{product}: support-pool records differ from the cached features")
        path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
        n_pool = len(dev["rel_paths"])
        images = ([open_rgb(rc.DATASET_ROOT / rel) for rel in dev["rel_paths"]] +
                  [open_rgb(by_key[(product, sha)]["image_path"]) for sha in qf["rgb_sha256"]])
        normals = [open_rgb(mvtec_root / product / "train" / "good" / x) for x in nf["names"]]
        name_index = {x: i for i, x in enumerate(nf["names"])}
        q_true = [by_key[(product, sha)]["true_label"] for sha in qf["rgb_sha256"]]
        shas = list(qf["rgb_sha256"])
        qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
        for tag, (backbone, layer, _) in configs.items():
            for precision in ("fp32", "fp16"):
                gb, pb = extract(backbone, images, device, layer, precision)
                _, pn = extract(backbone, normals, device, layer, precision)
                g, patches = gb.to(device), pb[layer].to(device).float()
                method = f"arr_{tag}" + ("_fp16" if precision == "fp16" else "")
                with torch.no_grad():
                    for seed in rc.SEEDS:
                        idx = torch.as_tensor([name_index[x] for x in nf["draws"][str(seed)][:ra.M_PRIMARY]])
                        bank = pn[layer][idx].to(device).float()
                        parts = c1_parts(patches, bank, fa.tau99(bank))[0]
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
                            dummy = torch.zeros(len(qi), n, device=device)
                            sc = ff.episode_scores(g, {"": parts}, si, s_lab, qi, n, dummy, dummy)[ff.ARR]
                            rows += fa.score_rows({method: sc}, order, q_true,
                                                  {"product": product, "K": k_shot, "seed": seed}, sink, shas)
        print(f"[p34] {product} done", flush=True)

    gates = {}
    for tag, (_, _, ref) in configs.items():
        ref_rows = [r for r in rc.load_json(ref)["raw_metrics"] if r["method"] == f"arr_{tag}"]
        gates[tag] = exact_equal(rows, ref_rows, (f"arr_{tag}",), ("product", "K", "seed"))
    if any(v >= 1e-9 for v in gates.values()):
        raise rc.P6BFailClosedError(f"gate failed (FP32 must reproduce P18 / P24 exactly): {gates}")
    grid = dict(rc.metric_grid_from_rows(rows))
    preds = {(u["method"], u["product"], u["K"], u["seed"]): u["pred"] for u in sink}
    comparisons, levels = {}, {}
    for tag in configs:
        m32, m16 = f"arr_{tag}", f"arr_{tag}_fp16"
        for m in (m32, m16):
            s = rc.product_macro_summary(grid, m)
            levels[m] = {str(k): 100 * s[str(k)]["balanced_accuracy"]["mean"] for k in rc.K_VALUES}
        c = contrast(grid, m32, "balanced_accuracy", target=m16)
        same = total = 0
        for (meth, p, k, sd), pred in preds.items():
            if meth == m16:
                ref = preds[(m32, p, k, sd)]
                same += sum(x == y for x, y in zip(pred, ref))
                total += len(pred)
        d, ci = 100 * c["Dbar"], [100 * x for x in c["Dbar_ci95"]]
        comparisons[f"{m16}_minus_{m32}"] = {"per_K_pp": {k: 100 * v["D"] for k, v in c["per_K"].items()}, "Dbar_pp": d,
                                             "ci95_pp": ci, "positive_products": c["Dbar_positive_products"],
                                             "agreement": same / total,
                                             "decision": "keeps accuracy" if abs(d) < 0.5 and ci[0] <= 0 <= ci[1]
                                             else "report FP16 accuracy"}
    result = {"schema_version": "p34_fp16_alt_v1", "protocol": "docs/P34_FP16_ALT_BACKBONES_PROTOCOL.md",
              "label": "report only", "provenance": rc.git_provenance(), "gates_fp32_vs_reference": gates,
              "levels_ba_pct": levels, "comparisons": comparisons, "raw_metrics": rows}
    OUT.mkdir(parents=True)
    rc.write_json(OUT / "p34_results.json", result)
    rc.write_jsonl(OUT / "p34_predictions.jsonl", sink)
    print(json.dumps({"gates": gates, "levels_ba_pct": levels, "comparisons": comparisons}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
