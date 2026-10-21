#!/usr/bin/env python3
"""P22 formal addendum: DINOv2-S / 336 (FP32 and FP16) on MVTec-FS, VisA and MTD (report-only).

Protocol: docs/P22_LIGHTWEIGHT_PROTOCOL.md, section "Formal addendum". The layer comes from the P22
dev selection. Integrity: DINOv2-B / 336 / FP32 at the P18-selected layer is recomputed by the same
code and must reproduce the P18 formal rows exactly (MVTec-FS and VisA/MTD). Every cell is reported.

Usage (each stage once; a stopped stage continues only with --resume, after authorization):
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p22.formal --stage mvtec --mvtec-root D:\\intergration\\MVTEC
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p22.formal --stage external
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.extract_dev_features import mask_grid, support_pool_records  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES, query_annotation_records  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p12.threshold_controls import exact_equal  # noqa: E402
from experiments.p18 import backbones as bb  # noqa: E402
from experiments.p18.run import (SHORT, c1_parts, c1_parts_chunked, chunked_scores, dev_selection,  # noqa: E402
                                 open_rgb, parse_config, renamed, whole_scores)
from experiments.p20.stats_holm import exact_signflip  # noqa: E402
from experiments.p22 import lightweight as lw  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p22" / "formal" / s for s in ("mvtec", "external")}
P14 = {"mvtec": PROJECT_ROOT / "ckpt/p14_formal/mvtec/p14_mvtec_results.json",
       "external": PROJECT_ROOT / "ckpt/p14_formal/external/p14_external_results.json"}
P18 = {"mvtec": PROJECT_ROOT / "ckpt/p18/mvtec/p18_mvtec_results.json",
       "external": PROJECT_ROOT / "ckpt/p18/external/p18_external_results.json"}
S32, S16, B = "dinov2_s_r336_fp32", "dinov2_s_r336_fp16", "dinov2_b"   # B = the P18 names (integrity)
ARR, SS = "fused_adareg_regres_soft_q99", "simpleshot_cl2n"
DECLARED = {"S32 ARR - S32 SimpleShot": (f"arr_{S32}", f"ss_{S32}"),
            "S32 ARR - main ARR": (f"arr_{S32}", ARR),
            "S32 ARR - DINOv2-B ARR": (f"arr_{S32}", f"arr_{B}"),
            "S16 ARR - S32 ARR": (f"arr_{S16}", f"arr_{S32}"),
            "S32 A - S32 fixed fraction": (f"a_{S32}", f"nr_{S32}")}


def configs(device) -> dict:
    """tag -> (backbone, layer). The P22 dev layer for DINOv2-S; the P18 dev layer for DINOv2-B."""
    layer = lw.rc.load_json(lw.OUT["dev"] / "p22_dev_results.json")["selection"]
    name, _, layer_b = parse_config(dev_selection()[B])
    return {S32: (lw.build(S32, device), layer[S32]), S16: (lw.build(S16, device), layer[S16]),
            B: (lw.build("dinov2_b_r336_fp32", device), layer_b)}


def run_mvtec(mvtec_root: Path, device, batch_size: int, resume: bool) -> int:
    with lw.single_shot(OUT["mvtec"], resume) as staging:
        ckpt = staging / "products"
        ckpt.mkdir(exist_ok=True)
        cfg = configs(device)
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
        by_key = {(q["product"], q["rgb_sha256"]): q for q in queries}
        raw_q, pools = query_annotation_records(), support_pool_records()
        rows = []
        for product in rc.PRODUCTS:
            path = ckpt / f"{product}.json"
            if path.exists():
                rows += json.loads(path.read_text(encoding="utf-8"))
                continue
            t0 = time.time()
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            if [r["canonical_relative_path"] for r in pools[product]] != list(dev["rel_paths"]):
                raise rc.P6BFailClosedError(f"{product}: support-pool records differ from the cached features")
            path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
            n_pool = len(dev["rel_paths"])
            images = ([open_rgb(rc.DATASET_ROOT / rel) for rel in dev["rel_paths"]] +
                      [open_rgb(by_key[(product, sha)]["image_path"]) for sha in qf["rgb_sha256"]])
            records = pools[product] + [raw_q[(product, sha)] for sha in qf["rgb_sha256"]]
            normals = [open_rgb(mvtec_root / product / "train" / "good" / x) for x in nf["names"]]
            name_index = {x: i for i, x in enumerate(nf["names"])}
            q_true = [by_key[(product, sha)]["true_label"] for sha in qf["rgb_sha256"]]
            qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
            feats = {}
            for tag, (backbone, layer) in cfg.items():
                gb, pb = bb.extract(backbone, images, device, batch_size, layers=(layer,))
                _, pn = bb.extract(backbone, normals, device, batch_size, layers=(layer,))
                side = int(round(pb[layer].shape[1] ** 0.5))
                mb = torch.stack([mask_grid(r, im.size, side) for r, im in zip(records, images)])
                feats[tag] = (gb.to(device), pb[layer].to(device).float(), pn[layer], mb.to(device))
            prod_rows = []
            with torch.no_grad():
                for seed in rc.SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in nf["draws"][str(seed)][:ra.M_PRIMARY]])
                    parts = {}
                    for tag, (gb, pb, pn, mb) in feats.items():
                        bank = pn[idx].to(device).float()
                        parts[tag] = c1_parts(pb, bank, fa.tau99(bank))[0]
                    for k_shot in rc.K_VALUES:
                        manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                        order = list(manifest["class_order"])
                        s_idx, s_lab = [], []
                        for c, cls in enumerate(order):
                            for item in manifest["classes"][cls]["items"]:
                                s_idx.append(path_index[item["rel_path"]])
                                s_lab.append(c)
                        si, n = torch.as_tensor(s_idx, device=device), len(order)
                        sc = {}
                        for tag, (gb, pb, pn, mb) in feats.items():
                            sc.update(renamed(whole_scores(gb, pb, parts[tag], si, s_lab, qi, n), tag))
                            sc[f"oracle_{tag}"] = rp.oracle_maskpool(pb[si], mb[si], s_lab, pb[qi], mb[qi], n)
                        prod_rows += fa.score_rows(sc, order, q_true, {"product": product, "K": k_shot, "seed": seed})
            rc.write_json(path, prod_rows)
            rows += prod_rows
            print(f"[p22-formal mvtec] {product} done ({time.time() - t0:.0f} s)", flush=True)
        methods_b = {f"{s}_{B}" for s in SHORT.values()} | {f"oracle_{B}"}
        integrity = exact_equal(rows, rc.load_json(P18["mvtec"])["raw_metrics"], methods_b, ("product", "K", "seed"))
        if integrity >= 1e-9:
            raise rc.P6BFailClosedError(f"integrity vs P18 formal DINOv2-B failed: {integrity}")
        grid = rc.metric_grid_from_rows(rows)
        grid.update({m: v for m, v in rc.metric_grid_from_rows(rc.load_json(P14["mvtec"])["raw_metrics"]).items()
                     if m in (ARR, SS)})
        report = {}
        for name, (t, r) in DECLARED.items():
            c = contrast(grid, r, "balanced_accuracy", target=t)
            per = [100 * v for v in c["Dbar_per_product_mean_difference"].values()]
            report[name] = {"per_K_pp": {k: 100 * v["D"] for k, v in c["per_K"].items()}, "Dbar_pp": 100 * c["Dbar"],
                            "ci95_pp": [100 * x for x in c["Dbar_ci95"]], "positive": sum(v > 0 for v in per),
                            "signflip_p": exact_signflip(per)}
        levels = {m: {str(k): {"mean": 100 * s[str(k)]["balanced_accuracy"]["mean"],
                               "sd": 100 * s[str(k)]["balanced_accuracy"]["std_ddof1"],
                               "macro_f1": 100 * s[str(k)]["macro_f1"]["mean"]} for k in rc.K_VALUES}
                  for m in sorted(grid) for s in [rc.product_macro_summary(grid, m)]}
        finish(staging, "mvtec", integrity, cfg, levels, report, rows)
    return 0


def external_task(task, order, cfg, device, batch_size) -> list:
    dataset = task.split("__")[0]
    records = [json.loads(line) for line in (ee.BUNDLE / f"{task}.images.jsonl").open(encoding="utf-8")]
    draws = json.load((ee.BUNDLE / f"{task}.support_draws.json").open(encoding="utf-8"))
    by_sha = {r["rgb_sha256"]: r for r in records}
    queries = sorted((r for r in records if r["partition"] == "query"), key=lambda r: r["rgb_sha256"])
    normal = ee.NORMAL_CLASS[dataset]
    normal_candidates = [r["rgb_sha256"] for r in records
                         if r["partition"] == "support_candidate" and r["class_label"] == normal]
    query_shas = {r["rgb_sha256"] for r in queries}
    banks, needed = {}, set()
    for seed in rc.SEEDS:
        banks[seed] = ee.bank_draw(task, seed, normal_candidates, draws[normal][str(seed)]["K5"])
        support_all = {s for cls in order for s in draws[cls][str(seed)]["K5"]}
        if set(banks[seed]) & (support_all | query_shas) or support_all & query_shas:
            raise rc.P6BFailClosedError(f"{task}/s{seed}: bank/support/query overlap")
        needed |= support_all | set(banks[seed])
    shas = sorted(needed) + [r["rgb_sha256"] for r in queries]
    index = {s: i for i, s in enumerate(shas)}
    paths = [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas]
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    feats = {}
    for tag, (backbone, layer) in cfg.items():
        gb, pb = bb.extract(backbone, paths, device, batch_size, layers=(layer,))
        feats[tag] = (gb.to(device), pb[layer])
    rows = []
    for seed in rc.SEEDS:
        bidx = torch.as_tensor([index[s] for s in banks[seed]])
        parts = {}
        for tag, (gb, pb) in feats.items():
            bank = pb[bidx].to(device).float()
            parts[tag] = c1_parts_chunked(pb, bank, fa.tau99(bank), device)[0]
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            si, n = torch.as_tensor(s_idx, device=device), len(order)
            sc = {}
            for tag, (gb, pb) in feats.items():
                sc.update(renamed(chunked_scores(gb, pb, parts[tag], si, s_lab, qi, pb[qi.cpu()], n, device), tag))
            rows += fa.score_rows(sc, order, q_true, {"task": task, "K": k_shot, "seed": seed})
    return rows


def run_external(device, batch_size: int, resume: bool) -> int:
    with lw.single_shot(OUT["external"], resume) as staging:
        ckpt = staging / "tasks"
        ckpt.mkdir(exist_ok=True)
        orders, _ = ee.frozen_artifacts()
        cfg = configs(device)
        rows = []
        for task in ee.TASKS:
            path = ckpt / f"{task}.json"
            if path.exists():
                rows += json.loads(path.read_text(encoding="utf-8"))
                continue
            t0 = time.time()
            with torch.no_grad():
                task_rows = external_task(task, orders[task], cfg, device, batch_size)
            rc.write_json(path, task_rows)
            rows += task_rows
            print(f"[p22-formal external] {task} done ({time.time() - t0:.0f} s)", flush=True)
        methods_b = {f"{s}_{B}" for s in SHORT.values()}
        integrity = exact_equal(rows, rc.load_json(P18["external"])["raw_metrics"], methods_b, ("task", "K", "seed"))
        if integrity >= 1e-9:
            raise rc.P6BFailClosedError(f"integrity vs P18 external DINOv2-B failed: {integrity}")
        grid = ee.grid_of(rows)
        grid.update({m: v for m, v in ee.grid_of(rc.load_json(P14["external"])["raw_metrics"]).items() if m in (ARR, SS)})
        report = {}
        for name, (t, r) in DECLARED.items():
            c = ee.visa_contrast(grid, t, r, "balanced_accuracy")
            objs = c["per_K"]["1"]["per_object"].keys()
            per = [100 * statistics.fmean(c["per_K"][str(k)]["per_object"][o] for k in rc.K_VALUES) for o in objs]
            mc = ee.mtd_contrast(grid, t, r, "balanced_accuracy")
            report[name] = {"visa_Dbar_pp": 100 * c["Dbar"], "visa_ci95_pp": [100 * x for x in c["Dbar_ci95"]],
                            "visa_positive": c["Dbar_positive_objects"], "visa_signflip_p": exact_signflip(per),
                            "mtd_per_K_pp": [100 * mc[str(k)]["mean"] for k in rc.K_VALUES]}
        levels = {m: {"visa": [100 * ee.visa_macro(grid, m)[str(k)]["mean"] for k in rc.K_VALUES],
                      "mtd": [100 * statistics.fmean(grid[m][(ee.MTD_TASK, k, s)]["balanced_accuracy"] for s in rc.SEEDS)
                              for k in rc.K_VALUES]} for m in sorted(grid)}
        finish(staging, "external", integrity, cfg, levels, report, rows)
    return 0


def finish(staging, stage, integrity, cfg, levels, report, rows) -> None:
    result = {"schema_version": f"p22_formal_{stage}_v1", "protocol": lw.PROTOCOL, "provenance": rc.git_provenance(),
              "layers": {t: L for t, (_, L) in cfg.items()}, "integrity_max_abs_diff_vs_p18": integrity,
              "levels": levels, "declared_contrasts": report, "raw_metrics": rows}
    rc.write_json(staging / f"p22_formal_{stage}_results.json", result)
    print(f"[p22-formal {stage}] integrity vs P18 {integrity:.1e}; layers {result['layers']}")
    for name, v in report.items():
        print(f"  {name}: " + json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()}))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("mvtec", "external"))
    parser.add_argument("--mvtec-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)   # as P18 (integrity depends on it)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.stage == "mvtec":
        if not args.mvtec_root:
            raise SystemExit("--mvtec-root is required for the mvtec stage")
        return run_mvtec(Path(args.mvtec_root), device, args.batch_size, args.resume)
    return run_external(device, args.batch_size, args.resume)


if __name__ == "__main__":
    sys.exit(main())
