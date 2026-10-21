#!/usr/bin/env python3
"""P32 (report only): score the device-extracted features against the cached GPU features.

Protocol: docs/P32_DEVICE_ACCURACY_PROTOCOL.md. For each precision in ``ckpt/p32/<tag>/`` (written by
``experiments.p32.device_extract`` on the device):
  - order check against the P10 caches and feature-level differences (max |d|, mean cosine);
  - ARR (main configuration, M = 20) and P17 ``a_r`` (CLIP-L/14@336 alone) for every product / K / seed, scored with the
    unchanged code path from the cached features (gate: reproduce P14 ARR and P17 a_r exactly) and from the device
    features;
  - device - cache paired product-cluster bootstrap (pooled over K), per-query agreement, and the locked decision rule.
Writes ``ckpt/p32/<tag>/p32_scores.json`` and the predictions once.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p32.score_device --tag jetson_orin_nano
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

from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p12.threshold_controls import exact_equal  # noqa: E402
from experiments.p14.final_formal import ARR, c1_descriptors  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

P32 = PROJECT_ROOT / "ckpt" / "p32"
P14_MVTEC = PROJECT_ROOT / "ckpt" / "p14_formal" / "mvtec" / "p14_mvtec_results.json"
P17_MVTEC = PROJECT_ROOT / "ckpt" / "p17" / "mvtec" / "p17_mvtec_results.json"
AR = "a_r"
SOURCES = ("cache", "fp32", "fp16")


def feature_diff(new, old) -> dict:
    new, old = new.float(), old.float()
    cos = F.cosine_similarity(new.reshape(-1, new.shape[-1]), old.reshape(-1, old.shape[-1]), dim=-1)
    return {"max_abs": float((new - old).abs().max()), "mean_cos": float(cos.mean()), "min_cos": float(cos.min())}


def score_product(product, feats, nf, dev_paths, q_true, shas, src, device, sink) -> list:
    """ARR (as experiments.p14.final_formal, M = 20) and a_r (as experiments.p30.fp16_accuracy) from one source."""
    rows = []
    path_index = {p: i for i, p in enumerate(dev_paths)}
    n_pool = len(dev_paths)
    g = torch.cat([feats["support_global"], feats["query_global"]]).to(device)
    clip = torch.cat([feats["support_clip18"], feats["query_clip18"]]).to(device).float()
    normals = feats["normal_clip18"].to(device).float()
    name_index = {x: i for i, x in enumerate(nf["names"])}
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
                zb = rp._zscore(rp.simpleshot(g[si], s_lab, g[qi], n))
                za = rp._zscore(rp.simpleshot(soft[si], s_lab, soft[qi], n))
                zr = rp._zscore(rp.simpleshot(rr[si], s_lab, rr[qi], n))
                scores = {f"{ARR}|{src}": zb + za + zr, f"{AR}|{src}": za + zr}
                rows += fa.score_rows(scores, order, q_true, {"product": product, "K": k_shot, "seed": seed}, sink, shas)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="one tag, or several comma-separated (a product is read from the last tag that holds both of its precision files)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precisions", default="fp32,fp16", help="feature files <precision>_<product>.pt to score "
                    "(P38: trt)")
    a = ap.parse_args(argv)
    precs = tuple(x for x in a.precisions.split(",") if x)
    sources_all = ("cache", *precs)
    tags = [t for t in a.tag.split(",") if t]
    root = P32 / "+".join(tags)
    out = root / "p32_scores.json"
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; P32 scoring is single-shot")
    device = torch.device(a.device)
    root.mkdir(parents=True, exist_ok=True)

    def source_dir(product):
        held = [t for t in tags if all((P32 / t / f"{pr}_{product}.pt").is_file() for pr in precs)]
        if not held:
            raise rc.P6BFailClosedError(f"{product}: no tag holds both precision files")
        return P32 / held[-1]
    sources_used = {}
    queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
    truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
    rows, sink, diffs = [], [], {}
    for product in rc.PRODUCTS:
        dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
        qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
        nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
        cache = {"support_global": dev["global_fusion"], "support_clip18": dev["clip18_patches"],
                 "query_global": qf["global_fusion"], "query_clip18": qf["clip18_patches"],
                 "normal_clip18": nf["clip18_patches"]}
        dev_paths, shas = list(dev["rel_paths"]), list(qf["rgb_sha256"])
        q_true = [truth[(product, s)] for s in shas]
        sources = {"cache": cache}
        for prec in precs:
            src = source_dir(product)
            sources_used[product] = src.name
            d = torch.load(src / f"{prec}_{product}.pt", map_location="cpu", weights_only=False)
            if (d["support_rel_paths"] != dev_paths or d["query_rgb_sha256"] != shas
                    or d["normal_names"] != list(nf["names"])):
                raise rc.P6BFailClosedError(f"{product} {prec}: image order differs from the P10 caches")
            sources[prec] = d
            diffs.setdefault(product, {})[prec] = {k: feature_diff(d[k], cache[k]) for k in cache}
        for src in sources_all:
            rows += score_product(product, sources[src], nf, dev_paths, q_true, shas, src, device, sink)
        print(f"[p32] {product} scored", flush=True)

    rename = lambda rs, m, src: [{**r, "method": m} for r in rs if r["method"] == f"{m}|{src}"]  # noqa: E731
    gates = {"cache_ARR_vs_P14": exact_equal(rename(rows, ARR, "cache"), rc.load_json(P14_MVTEC)["raw_metrics"], (ARR,),
                                             ("product", "K", "seed")),
             "cache_a_r_vs_P17": exact_equal(rename(rows, AR, "cache"), rc.load_json(P17_MVTEC)["raw_metrics"], (AR,),
                                             ("product", "K", "seed"))}
    if any(v >= 1e-9 for v in gates.values()):
        raise rc.P6BFailClosedError(f"gate failed (cached features must reproduce P14 / P17 exactly): {gates}")
    grid = dict(rc.metric_grid_from_rows(rows))
    by_key = {(u["method"], u["product"], u["K"], u["seed"]): u["pred"] for u in sink}
    comparisons, levels = {}, {}
    for m in (ARR, AR):
        for src in sources_all:
            s = rc.product_macro_summary(grid, f"{m}|{src}")
            levels[f"{m}|{src}"] = {str(k): 100 * s[str(k)]["balanced_accuracy"]["mean"] for k in rc.K_VALUES}
        for prec in precs:
            c = contrast(grid, f"{m}|cache", "balanced_accuracy", target=f"{m}|{prec}")
            same = total = 0
            for (meth, p, k, sd), pred in by_key.items():
                if meth == f"{m}|{prec}":
                    ref = by_key[(f"{m}|cache", p, k, sd)]
                    same += sum(x == y for x, y in zip(pred, ref))
                    total += len(pred)
            d, ci = 100 * c["Dbar"], [100 * x for x in c["Dbar_ci95"]]
            comparisons[f"{m}|{prec}_minus_cache"] = {
                "per_K_pp": {k: 100 * v["D"] for k, v in c["per_K"].items()}, "Dbar_pp": d, "ci95_pp": ci,
                "positive_products": c["Dbar_positive_products"], "agreement": same / total,
                "decision": "keeps accuracy" if abs(d) < 0.5 and ci[0] <= 0 <= ci[1] else "report device accuracy"}
    result = {"schema_version": "p32_scores_v1", "protocol": "docs/P32_DEVICE_ACCURACY_PROTOCOL.md", "label": "report only",
              "tag": a.tag, "sources": sources_used, "provenance": rc.git_provenance(), "gates": gates, "feature_diff": diffs, "levels_ba_pct": levels,
              "comparisons": comparisons, "raw_metrics": rows}
    rc.write_json(out, result)
    rc.write_jsonl(root / "p32_predictions.jsonl", sink)
    print(json.dumps({"gates": gates, "levels_ba_pct": levels, "comparisons": comparisons}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
