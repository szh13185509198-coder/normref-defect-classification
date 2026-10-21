#!/usr/bin/env python3
"""P12 T: threshold-source controls for A (dev, MVTec-FS formal, VisA/MTD), each stage once.

Protocol: docs/P12_THRESHOLD_CONTROLS_RELIABILITY_FIGURES_PROTOCOL.md (T).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p12.threshold_controls --stage dev --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p12.threshold_controls --stage mvtec --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p12.threshold_controls --stage external --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy  # noqa: F401,E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p6.p6b_stage3_common import classification_metrics, load_query_contract  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR, RESULTS_DIR as P10_FORMAL  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p12.thresholds import CONTROLS  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p12" / f"threshold_{s}" for s in ("dev", "mvtec", "external")}
P11_DEV = PROJECT_ROOT / "ckpt" / "p11_pilot" / "pilot_ab_results.json"
P11_MVTEC = PROJECT_ROOT / "ckpt" / "p11_formal" / "mvtec" / "p11_formal_mvtec_results.json"
P11_EXTERNAL = PROJECT_ROOT / "ckpt" / "p11_formal" / "external" / "p11_formal_external_results.json"
GUARDED = ("experiments/p10", "experiments/p11", "experiments/p12", "experiments/revision",
           "docs/P12_THRESHOLD_CONTROLS_RELIABILITY_FIGURES_PROTOCOL.md")
A, NR, SS = fa.A, fa.NR, fa.SS
METHODS = (SS, NR, A, *CONTROLS)


@contextlib.contextmanager
def single_shot(out: Path):
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; this P12 stage is single-shot")
    lock = out.with_name(out.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()  # left in place as a single-shot marker
    except FileExistsError:
        raise rc.P6BFailClosedError(f"{lock} exists: another run is active or was interrupted")
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    yield staging
    staging.rename(out)


def control_weights(a: torch.Tensor) -> dict:
    """Soft weights of every control on patch anomalies a [N, P] (same rule as A, other tau)."""
    return {name: ab.region_weights(a, fn(a), soft=True) for name, fn in CONTROLS.items()}


def scores_for(g, desc: dict, si, s_lab, qi, n):
    """desc: method -> per-image descriptor (fixed-region pooled for NR, soft for A / controls)."""
    base = rp.simpleshot(g[si], s_lab, g[qi], n)
    zb = rp._zscore(base)
    out = {SS: base}
    for m, h in desc.items():
        out[m] = zb + rp._zscore(rp.simpleshot(h[si], s_lab, h[qi], n))
    return out


def rows_of(scores, order, truth, meta, sink=None, shas=None):
    return fa.score_rows(scores, order, truth, meta, sink, shas)


def exact_equal(new_rows, old_rows, methods, key_fields):
    old = {(r["method"], *(r[f] for f in key_fields)): r for r in old_rows if r["method"] in methods}
    diffs = [abs(r[m] - old[(r["method"], *(r[f] for f in key_fields))][m])
             for r in new_rows if r["method"] in methods for m in rc.METRICS]
    n = sum(1 for r in new_rows if r["method"] in methods)
    if n != len(old):
        raise rc.P6BFailClosedError(f"integrity: {n} new cells vs {len(old)} reference cells for {methods}")
    return max(diffs)


# --------------------------------------------------------------------------- dev

def run_dev(device) -> int:
    with single_shot(OUT["dev"]) as staging:
        rows = []
        diag = {m: [] for m in (A, *CONTROLS, "fixed_top58_hard")}
        sizes = {m: [] for m in (A, *CONTROLS)}
        fallback = {m: [] for m in (A, *CONTROLS)}
        for product in rc.PRODUCTS:
            data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            g = data["global_fusion"].to(device)
            clip = data["clip18_patches"].to(device).float()
            masks = data["mask_grid"].to(device)
            name_index = {x: i for i, x in enumerate(normal["names"])}
            normal_clip = normal["clip18_patches"].to(device).float()
            with torch.no_grad():
                for seed in rp.DEV_SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]], device=device)
                    bank_imgs = normal_clip[idx]
                    a = ra.patch_anomaly(clip, bank_imgs.reshape(-1, clip.shape[-1]))
                    tau = float(torch.quantile(ab.loo_anomalies(bank_imgs), 0.99))  # exactly the P11 pilot path
                    weights = {A: ab.region_weights(a, tau, soft=True), **control_weights(a)}
                    desc = {NR: ra.anomaly_pool(clip, a), **{m: ab.weighted_pool(clip, w) for m, w in weights.items()}}
                    for m, w in weights.items():
                        diag[m] += ab.region_agreement(w > 0, masks)
                        sizes[m] += (w > 0).sum(dim=1).float().tolist()
                        raw = (a > tau) if m == A else (a > CONTROLS[m](a))
                        fallback[m] += (raw.sum(dim=1) < ab.MIN_REGION).tolist()
                    fixed = torch.zeros_like(a).scatter_(1, a.topk(math.ceil(rp.RHO * a.shape[1]), dim=1).indices, 1.0)
                    diag["fixed_top58_hard"] += ab.region_agreement(fixed > 0, masks)
                    for k_shot, order, s_idx, s_lab, q_idx, q_lab in rp.episodes(data["labels"], seed, product):
                        si = torch.as_tensor(s_idx, device=device)
                        qi = torch.as_tensor(q_idx, device=device)
                        sc = scores_for(g, desc, si, s_lab, qi, len(order))
                        rows += rows_of(sc, order, [order[c] for c in q_lab], {"product": product, "K": k_shot, "seed": seed})
            print(f"[p12-T-dev] {product} done", flush=True)

        integrity = exact_equal(rows, rc.load_json(P11_DEV)["rows"], (A, NR, SS), ("product", "K", "seed"))
        if integrity >= 1e-12:
            raise rc.P6BFailClosedError(f"dev integrity vs P11 pilot failed: max|Δ|={integrity}")
        summary = rp.summarize(rows)
        for v in summary["methods"].values():
            v.pop("gate_pass", None)
        cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows}
        vs_a = {}
        for m in CONTROLS:
            per_product = {p: 100 * statistics.fmean(cell[(m, p, k, s)] - cell[(A, p, k, s)]
                                                     for k in rp.DEV_K for s in rp.DEV_SEEDS) for p in rc.PRODUCTS}
            vs_a[m] = {"mean_delta_pp": statistics.fmean(per_product.values()),
                       "positive_products": sum(v > 0 for v in per_product.values()),
                       "per_product_delta_pp": per_product}
        region = {m: {**{s: statistics.fmean(x[s] for x in v) for s in ("precision", "recall", "f1")},
                      "images": len(v),
                      "mean_size": statistics.fmean(sizes[m]) if m in sizes else 58.0,
                      "fallback_fraction": statistics.fmean(float(x) for x in fallback[m]) if m in fallback else 0.0}
                  for m, v in diag.items()}
        result = {"schema_version": "p12_threshold_dev_v1", "protocol": "docs/P12_THRESHOLD_CONTROLS_RELIABILITY_FIGURES_PROTOCOL.md#t",
                  "provenance": rc.git_provenance(), "integrity_max_abs_diff_vs_p11_pilot": integrity,
                  "vs_simpleshot": summary, "controls_minus_A": vs_a, "region_diagnostics": region, "rows": rows}
        lines = ["# P12 T-dev — threshold-source controls (dev only)", "",
                 f"Integrity vs P11 dev pilot (A, normref, SimpleShot): max|Δ| = {integrity:.1e}", "",
                 "| Method | BA K=1 | BA K=3 | Δ vs SimpleShot (+products) | Δ vs A (+products) |", "|---|---|---|---|---|"]
        for m in METHODS:
            v = summary["methods"][m]
            c = vs_a.get(m)
            lines.append(f"| {m} | {100 * v['ba_by_K']['1']:.2f} | {100 * v['ba_by_K']['3']:.2f} | "
                         f"{v['mean_delta_pp']:+.2f} ({v['positive_products']}/13) | "
                         + (f"{c['mean_delta_pp']:+.2f} ({c['positive_products']}/13)" if c else "—") + " |")
        lines += ["", "Region vs mask (precision / recall / F1, mean size, fallback):"]
        for m, v in region.items():
            lines.append(f"- {m}: {v['precision']:.3f} / {v['recall']:.3f} / {v['f1']:.3f}, size {v['mean_size']:.1f}, "
                         f"fallback {v['fallback_fraction']:.3f} (n={v['images']})")
        text = "\n".join(lines)
        rc.write_json(staging / "threshold_dev_results.json", result)
        (staging / "threshold_dev_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- MVTec-FS formal

def run_mvtec(device) -> int:
    with single_shot(OUT["mvtec"]) as staging:
        provenance = rc.git_provenance()
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
        truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
        rows, sink = [], []
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
            n_pool = len(dev["rel_paths"])
            g = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
            clip = torch.cat([dev["clip18_patches"], qf["clip18_patches"]]).to(device).float()
            normal_clip = nf["clip18_patches"].to(device).float()
            name_index = {x: i for i, x in enumerate(nf["names"])}
            q_true = [truth[(product, sha)] for sha in qf["rgb_sha256"]]
            qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
            with torch.no_grad():
                for seed in rc.SEEDS:
                    draw = nf["draws"][str(seed)]
                    bank_imgs = normal_clip[torch.as_tensor([name_index[x] for x in draw[:ra.M_PRIMARY]], device=device)]
                    tau = fa.tau99(bank_imgs)
                    pooled, soft, _ = fa.descriptors(clip, bank_imgs, tau)  # whole tensor, as P10 / P11 formal
                    a = ra.patch_anomaly(clip, bank_imgs.reshape(-1, clip.shape[-1]))
                    desc = {NR: pooled, A: soft, **{m: ab.weighted_pool(clip, w) for m, w in control_weights(a).items()}}
                    for k_shot in rc.K_VALUES:
                        manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                        order = list(manifest["class_order"])
                        s_idx, s_lab = [], []
                        for c, cls in enumerate(order):
                            for item in manifest["classes"][cls]["items"]:
                                s_idx.append(path_index[item["rel_path"]])
                                s_lab.append(c)
                        si = torch.as_tensor(s_idx, device=device)
                        sc = scores_for(g, desc, si, s_lab, qi, len(order))
                        rows += rows_of(sc, order, q_true, {"product": product, "K": k_shot, "seed": seed},
                                        sink, list(qf["rgb_sha256"]))
            print(f"[p12-T-mvtec] {product} done", flush=True)

        integrity = {"vs_p10_formal": exact_equal(rows, rc.load_json(P10_FORMAL / "formal_results.json")["raw_metrics"],
                                                  (SS, NR), ("product", "K", "seed")),
                     "vs_p11_formal": exact_equal(rows, rc.load_json(P11_MVTEC)["raw_metrics"], (A,), ("product", "K", "seed"))}
        if any(v >= 1e-9 for v in integrity.values()):
            raise rc.P6BFailClosedError(f"integrity check failed: {integrity}")
        grid = rc.metric_grid_from_rows(rows)
        contrasts = {f"{A}_minus_{m}": {metric: contrast(grid, m, metric, target=A) for metric in rc.METRICS}
                     for m in CONTROLS}
        contrasts.update({f"{m}_minus_{NR}": {"balanced_accuracy": contrast(grid, NR, "balanced_accuracy", target=m)}
                          for m in CONTROLS})
        summary = {m: rc.product_macro_summary(grid, m) for m in METHODS}
        a_minus = {m: contrasts[f"{A}_minus_{m}"]["balanced_accuracy"]["Dbar_ci95"][0] > 0 for m in CONTROLS}
        reading = {"normal_calibration_beyond_image_only_thresholds": a_minus["fused_ctrl_soft_otsu"] and a_minus["fused_ctrl_soft_mebin"],
                   "adaptive_region_beyond_soft_fixed_fraction": a_minus["fused_ctrl_soft_top10"]}
        result = {"schema_version": "p12_threshold_mvtec_v1", "protocol": "docs/P12_THRESHOLD_CONTROLS_RELIABILITY_FIGURES_PROTOCOL.md#t",
                  "label": "control rows; A unchanged; single run", "provenance": provenance,
                  "integrity_max_abs_diff": integrity, "contrasts": contrasts, "reading_rule": reading,
                  "product_macro_summary": summary, "raw_metrics": rows}
        lines = ["# P12 T-mvtec — threshold-source controls on MVTec-FS formal (single run)", "",
                 f"Commit `{provenance['head']}`. Integrity: " + ", ".join(f"{k} max|Δ|={v:.1e}" for k, v in integrity.items()), "",
                 "| Contrast (BA) | K=1 | K=3 | K=5 | Pooled over K |", "|---|---|---|---|---|"]
        for key, v in contrasts.items():
            c = v["balanced_accuracy"]
            lines.append(f"| {key} | " + " | ".join(
                f"{100 * c['per_K'][str(k)]['D']:+.2f} [{100 * c['per_K'][str(k)]['ci95'][0]:+.2f}, {100 * c['per_K'][str(k)]['ci95'][1]:+.2f}]"
                for k in rc.K_VALUES) + f" | {fa.fmt(c)} |")
        lines += ["", "Reading rule: " + ", ".join(f"{k} = {v}" for k, v in reading.items()), "",
                  "| Method | K=1 | K=3 | K=5 |", "|---|---|---|---|"]
        for m, s in summary.items():
            lines.append(f"| {m} | " + " | ".join(
                f"{100 * s[str(k)]['balanced_accuracy']['mean']:.2f} / {100 * s[str(k)]['macro_f1']['mean']:.2f}" for k in rc.K_VALUES) + " |")
        text = "\n".join(lines)
        rc.write_json(staging / "threshold_mvtec_results.json", result)
        rc.write_jsonl(staging / "threshold_mvtec_predictions.jsonl", sink)
        (staging / "threshold_mvtec_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- VisA / MTD

def external_task(task, order, factory, device, batch_size):
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
    g, clip = ee.extract(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas], device, batch_size)
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    g = g.to(device)
    rows = []
    for seed in rc.SEEDS:
        bank_imgs = clip[torch.as_tensor([index[s] for s in banks[seed]])].to(device).float()
        bank = bank_imgs.reshape(-1, clip.shape[-1])
        tau = fa.tau99(bank_imgs)
        parts = {m: [] for m in (NR, A, *CONTROLS)}
        for chunk in clip.split(ee.Q_CHUNK):  # identical chunking to the P10 / P11 external runs
            c = chunk.to(device).float()
            pooled, soft, _ = fa.descriptors(c, bank_imgs, tau)
            parts[NR].append(pooled)
            parts[A].append(soft)
            for m, w in control_weights(ra.patch_anomaly(c, bank)).items():
                parts[m].append(ab.weighted_pool(c, w))
        desc = {m: torch.cat(v) for m, v in parts.items()}
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            si = torch.as_tensor(s_idx, device=device)
            sc = scores_for(g, desc, si, s_lab, qi, len(order))
            rows += rows_of(sc, order, q_true, {"task": task, "K": k_shot, "seed": seed})
    return rows, {"queries": len(queries), "extracted_images": len(shas)}


def run_external(device, batch_size) -> int:
    with single_shot(OUT["external"]) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        rows, info = [], {}
        for task in ee.TASKS:
            with torch.no_grad():
                r, i = external_task(task, orders[task], factory, device, batch_size)
            rows += r
            info[task] = i
            print(f"[p12-T-ext] {task}: {i['queries']} queries, {i['extracted_images']} images", flush=True)
        integrity = {"vs_p10_external": exact_equal(rows, rc.load_json(ee.RESULTS_DIR / "external_results.json")["raw_metrics"],
                                                    (SS, NR), ("task", "K", "seed")),
                     "vs_p11_external": exact_equal(rows, rc.load_json(P11_EXTERNAL)["raw_metrics"], (A,), ("task", "K", "seed"))}
        if any(v >= 1e-9 for v in integrity.values()):
            raise rc.P6BFailClosedError(f"integrity check failed: {integrity}")
        grid = ee.grid_of(rows)
        contrasts = {f"{A}_minus_{m}": {"visa": {metric: ee.visa_contrast(grid, A, m, metric) for metric in rc.METRICS},
                                        "mtd_BA": ee.mtd_contrast(grid, A, m, "balanced_accuracy")} for m in CONTROLS}
        reading = {"normal_calibration_beyond_image_only_thresholds":
                   all(contrasts[f"{A}_minus_{m}"]["visa"]["balanced_accuracy"]["Dbar_ci95"][0] > 0
                       for m in ("fused_ctrl_soft_otsu", "fused_ctrl_soft_mebin")),
                   "adaptive_region_beyond_soft_fixed_fraction":
                   contrasts[f"{A}_minus_fused_ctrl_soft_top10"]["visa"]["balanced_accuracy"]["Dbar_ci95"][0] > 0}
        summary = {m: {"visa_object_macro_BA": ee.visa_macro(grid, m),
                       "mtd_BA": {str(k): statistics.fmean(grid[m][(ee.MTD_TASK, k, s)]["balanced_accuracy"] for s in rc.SEEDS)
                                  for k in rc.K_VALUES}} for m in METHODS}
        result = {"schema_version": "p12_threshold_external_v1", "protocol": "docs/P12_THRESHOLD_CONTROLS_RELIABILITY_FIGURES_PROTOCOL.md#t",
                  "label": "control rows; A unchanged; single run", "provenance": provenance, "task_info": info,
                  "integrity_max_abs_diff": integrity, "contrasts": contrasts, "reading_rule": reading,
                  "summary": summary, "raw_metrics": rows}
        lines = ["# P12 T-external — threshold-source controls on VisA / MTD (single run)", "",
                 f"Commit `{provenance['head']}`. Integrity: " + ", ".join(f"{k} max|Δ|={v:.1e}" for k, v in integrity.items()), "",
                 "| Contrast (BA) | VisA pooled over K (+objects/11) | MTD K=1 / 3 / 5 (seed mean ± SD) |", "|---|---|---|"]
        for key, v in contrasts.items():
            mtd = " / ".join(f"{100 * v['mtd_BA'][str(k)]['mean']:+.2f}±{100 * v['mtd_BA'][str(k)]['sd']:.2f}" for k in rc.K_VALUES)
            lines.append(f"| {key} | {fa.fmt(v['visa']['balanced_accuracy'], 'Dbar_positive_objects', 11)} | {mtd} |")
        lines += ["", "Reading rule (VisA): " + ", ".join(f"{k} = {v}" for k, v in reading.items()), "",
                  "| Method | VisA K=1 | VisA K=3 | VisA K=5 | MTD K=1 | MTD K=3 | MTD K=5 |", "|---|---|---|---|---|---|---|"]
        for m, s in summary.items():
            lines.append(f"| {m} | " + " | ".join(f"{100 * s['visa_object_macro_BA'][str(k)]['mean']:.2f}" for k in rc.K_VALUES)
                         + " | " + " | ".join(f"{100 * s['mtd_BA'][str(k)]:.2f}" for k in rc.K_VALUES) + " |")
        text = "\n".join(lines)
        rc.write_json(staging / "threshold_external_results.json", result)
        (staging / "threshold_external_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("dev", "mvtec", "external"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.stage == "dev":
        return run_dev(device)
    return run_mvtec(device) if args.stage == "mvtec" else run_external(device, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
