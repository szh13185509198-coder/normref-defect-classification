#!/usr/bin/env python3
"""P11 formal addendum: adaptive region (A) and guided matching (B) — single-shot stages.

Protocol: docs/P11_FORMAL_ADDENDUM.md.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p11.formal_ab --stage mvtec --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p11.formal_ab --stage external --device cuda
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
from experiments.p10.formal_evaluate import MANIFEST_DIR, RESULTS_DIR as P10_FORMAL, mvrec_rows  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT = {"mvtec": PROJECT_ROOT / "ckpt" / "p11_formal" / "mvtec",
       "external": PROJECT_ROOT / "ckpt" / "p11_formal" / "external"}
SUPPLEMENT_S = PROJECT_ROOT / "ckpt" / "p10_formal" / "supplement_s" / "supplement_s_results.json"
GUARDED_PATHS = ("experiments/p10", "experiments/p11", "experiments/revision",
                 "docs/P11_FORMAL_ADDENDUM.md", "docs/P11_ADAPTIVE_REGION_AND_GUIDED_MATCHING_PILOT_PROTOCOL.md")
A, B, NR, SS = "fused_adareg_soft_q99", "fused_guided_xclass_normref", "fused_normref_clip18", "simpleshot_cl2n"
HYPOTHESES = {"mvtec": {"H4_primary_A": (A, NR), "H5_secondary_B": (B, NR)},
              "external": {"E4_primary_A": (A, NR), "E5_secondary_B": (B, NR)}}


@contextlib.contextmanager
def single_shot(stage: str):
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED_PATHS],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    out = OUT[stage]
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; the P11 {stage} stage is single-shot")
    lock = out.with_name(out.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()  # atomic: a second concurrent invocation fails here; left in place as marker
    except FileExistsError:
        raise rc.P6BFailClosedError(f"{lock} exists: another run is active or was interrupted")
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    yield staging
    staging.rename(out)


def descriptors(clip: torch.Tensor, bank_imgs: torch.Tensor, tau: float):
    """P10 fixed-region pooled descriptor, P11 soft-region descriptor, top-58 region patches.

    ``clip`` is on the device; the fixed-region path is exactly P10's (patch_anomaly +
    anomaly_pool on the given tensor), so the integrity check can be bitwise."""
    a = ra.patch_anomaly(clip, bank_imgs.reshape(-1, bank_imgs.shape[-1]))
    idx = a.topk(math.ceil(rp.RHO * clip.shape[1]), dim=1).indices
    region = torch.gather(clip, 1, idx.unsqueeze(-1).expand(-1, -1, clip.shape[-1]))
    return (ra.anomaly_pool(clip, a), ab.weighted_pool(clip, ab.region_weights(a, tau, soft=True)),
            region)


def tau99(bank_imgs: torch.Tensor) -> float:
    return float(torch.quantile(ab.loo_anomalies(bank_imgs), 0.99))


def episode_scores(g, per_m, si, s_lab, qi, n):
    base = rp.simpleshot(g[si], s_lab, g[qi], n)
    zb = rp._zscore(base)
    scores = {SS: base}
    for suffix, (pooled, soft, region) in per_m.items():
        nr = rp.simpleshot(pooled[si], s_lab, pooled[qi], n)
        _, gx = ab.guided_scores(region, s_lab, si, qi, n)
        scores[NR + suffix] = zb + rp._zscore(nr)
        scores[A + suffix] = zb + rp._zscore(rp.simpleshot(soft[si], s_lab, soft[qi], n))
        scores[B + suffix] = zb + rp._zscore(nr) + rp._zscore(gx)
    return scores


def score_rows(scores, order, truth, meta, sink=None, shas=None):
    rows = []
    for method, sc in scores.items():
        pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
        rows.append({"method": method, **meta, **classification_metrics(truth, pred, order)})
        if sink is not None:
            sink.append({"method": method, **meta, "rgb_sha256": shas, "true": truth, "pred": pred})
    return rows


def fmt(c, key="Dbar_positive_products", n=13):
    return (f"{100 * c['Dbar']:+.2f} [{100 * c['Dbar_ci95'][0]:+.2f}, {100 * c['Dbar_ci95'][1]:+.2f}] "
            f"({c[key]}/{n})")


# --------------------------------------------------------------------------- F1 MVTec-FS

def run_mvtec(device) -> int:
    with single_shot("mvtec") as staging:
        provenance = rc.git_provenance()
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
        truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
        rows, taus = [], {}
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
            n_pool = len(dev["rel_paths"])
            g = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
            clip = torch.cat([dev["clip18_patches"], qf["clip18_patches"]]).to(device).float()
            masks = torch.cat([dev["mask_grid"], qf["mask_grid_oracle_only"]]).to(device)
            normal_clip = nf["clip18_patches"].to(device).float()
            name_index = {x: i for i, x in enumerate(nf["names"])}
            q_true = [truth[(product, sha)] for sha in qf["rgb_sha256"]]
            qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
            with torch.no_grad():
                for seed in rc.SEEDS:
                    draw = nf["draws"][str(seed)]
                    per_m = {}
                    for m, suffix in ((ra.M_PRIMARY, ""), (ra.M_SENS, "_m5")):
                        bank_imgs = normal_clip[torch.as_tensor([name_index[x] for x in draw[:m]], device=device)]
                        tau = tau99(bank_imgs)
                        per_m[suffix] = descriptors(clip, bank_imgs, tau)  # whole tensor, as P10 formal
                        taus[f"{product}/s{seed}/m{m}"] = tau
                    for k_shot in rc.K_VALUES:
                        manifest = rc.load_json(MANIFEST_DIR / product / f"k{k_shot}_s{seed}.json")
                        order = list(manifest["class_order"])
                        s_idx, s_lab = [], []
                        for c, cls in enumerate(order):
                            for item in manifest["classes"][cls]["items"]:
                                s_idx.append(path_index[item["rel_path"]])
                                s_lab.append(c)
                        if any(t not in order for t in q_true):
                            raise rc.P6BFailClosedError(f"query label outside class order: {product}")
                        si = torch.as_tensor(s_idx, device=device)
                        n = len(order)
                        scores = episode_scores(g, per_m, si, s_lab, qi, n)
                        scores["oracle_maskpool_clip18"] = rp.oracle_maskpool(
                            clip[si], masks[si], s_lab, clip[qi], masks[qi], n)
                        rows += score_rows(scores, order, q_true, {"product": product, "K": k_shot, "seed": seed})
            print(f"[p11-F1] {product} done", flush=True)

        grid = rc.metric_grid_from_rows(rows)
        p10 = rc.metric_grid_from_rows(rc.load_json(P10_FORMAL / "formal_results.json")["raw_metrics"])
        integrity = {}
        for m in (SS, NR, NR + "_m5", "oracle_maskpool_clip18"):
            integrity[m] = max(abs(grid[m][key][metric] - p10[m][key][metric])
                               for key in grid[m] for metric in rc.METRICS)
        if any(v >= 1e-9 for v in integrity.values()):
            raise rc.P6BFailClosedError(f"integrity check vs P10 formal failed: {integrity}")
        full = dict(grid)
        full.update({m: v for m, v in rc.load_formal_metric_grid().items() if m == "ours"})
        full.update(rc.metric_grid_from_rows(mvrec_rows()))
        full.update({m: v for m, v in rc.metric_grid_from_rows(rc.load_json(SUPPLEMENT_S)["raw_metrics"]).items()
                     if m in ("residual_mean_clip18", "fused_residual_mean_clip18")})

        hyp = {}
        for name, (target, ref) in HYPOTHESES["mvtec"].items():
            res = {metric: contrast(full, ref, metric, target=target) for metric in rc.METRICS}
            hyp[name] = {"target": target, "reference": ref, "results": res,
                         "verdict_BA": "CONFIRMED" if res["balanced_accuracy"]["Dbar_ci95"][0] > 0 else "NOT CONFIRMED"}
        desc_pairs = [(A, SS), (B, SS), (A + "_m5", NR + "_m5"), (B + "_m5", NR + "_m5"),
                      (A, "residual_mean_clip18"), (A, "fused_residual_mean_clip18"),
                      (A, "ours"), (A, "mvrec_rgb")]
        descriptive = {f"{t}_minus_{r}": {metric: contrast(full, r, metric, target=t) for metric in rc.METRICS}
                       for t, r in desc_pairs}
        levels = (SS, NR, A, B, NR + "_m5", A + "_m5", B + "_m5", "oracle_maskpool_clip18",
                  "residual_mean_clip18", "fused_residual_mean_clip18", "ours", "mvrec_rgb")
        summary = {m: rc.product_macro_summary(full, m) for m in levels}
        headroom = {}
        for k in map(str, rc.K_VALUES):
            gap = summary["oracle_maskpool_clip18"][k]["balanced_accuracy"]["mean"] - summary[SS][k]["balanced_accuracy"]["mean"]
            headroom[k] = {m: (summary[m][k]["balanced_accuracy"]["mean"] - summary[SS][k]["balanced_accuracy"]["mean"]) / gap
                           for m in (NR, A, B)}
        result = {"schema_version": "p11_formal_mvtec_v1", "protocol": "docs/P11_FORMAL_ADDENDUM.md#2",
                  "label": "post-hoc hypotheses added after the P10 formal run; single run",
                  "provenance": provenance, "integrity_max_abs_diff_vs_p10_formal": integrity,
                  "hypotheses": hyp, "descriptive_contrasts": descriptive, "product_macro_summary": summary,
                  "fraction_of_oracle_headroom": headroom, "tau_0.99": taus, "raw_metrics": rows}

        lines = ["# P11 formal F1 — MVTec-FS-RGB (single run, post-hoc hypotheses)", "",
                 f"Commit `{provenance['head']}`. Integrity vs P10 formal: " +
                 ", ".join(f"{k} max|Δ|={v:.1e}" for k, v in integrity.items()), "",
                 "| Hypothesis | Contrast | K=1 | K=3 | K=5 | Pooled over K | Verdict |", "|---|---|---|---|---|---|---|"]
        for name, h in hyp.items():
            c = h["results"]["balanced_accuracy"]
            per_k = " | ".join(f"{100 * c['per_K'][str(k)]['D']:+.2f} [{100 * c['per_K'][str(k)]['ci95'][0]:+.2f}, "
                               f"{100 * c['per_K'][str(k)]['ci95'][1]:+.2f}]" for k in rc.K_VALUES)
            lines.append(f"| {name} | {h['target']} − {h['reference']} | {per_k} | {fmt(c)} | {h['verdict_BA']} |")
        lines += ["", "Macro-F1 pooled: " + "; ".join(
            f"{n} {fmt(h['results']['macro_f1'])}" for n, h in hyp.items()), "",
            "## Descriptive contrasts (BA, pooled over K)", "", "| Contrast | Δ |", "|---|---|"]
        for key, v in descriptive.items():
            lines.append(f"| {key} | {fmt(v['balanced_accuracy'])} |")
        lines += ["", "## Levels (product-macro BA / Macro-F1 %)", "", "| Method | K=1 | K=3 | K=5 |", "|---|---|---|---|"]
        for m, s in summary.items():
            lines.append(f"| {m} | " + " | ".join(
                f"{100 * s[str(k)]['balanced_accuracy']['mean']:.2f} / {100 * s[str(k)]['macro_f1']['mean']:.2f}"
                for k in rc.K_VALUES) + " |")
        lines += ["", "Fraction of mask-oracle headroom over SimpleShot: " + "; ".join(
            f"K={k}: " + ", ".join(f"{m} {v:.2f}" for m, v in h.items()) for k, h in headroom.items())]
        text = "\n".join(lines)
        rc.write_json(staging / "p11_formal_mvtec_results.json", result)
        (staging / "p11_formal_mvtec_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- F2 VisA / MTD

def run_external_task(task, order, factory, device, batch_size, sink):
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
    q_shas = [r["rgb_sha256"] for r in queries]
    g = g.to(device)
    rows, taus = [], {}
    for seed in rc.SEEDS:
        bank_idx = torch.as_tensor([index[s] for s in banks[seed]])
        per_m = {}
        for m, suffix in ((ee.BANK, ""), (ra.M_SENS, "_m5")):
            bank_imgs = clip[bank_idx[:m]].to(device).float()
            tau = tau99(bank_imgs)
            parts = [descriptors(chunk.to(device).float(), bank_imgs, tau)
                     for chunk in clip.split(ee.Q_CHUNK)]  # identical chunking to the P10 external run
            per_m[suffix] = tuple(torch.cat(x) for x in zip(*parts))
            taus[f"s{seed}/m{m}"] = tau
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            si = torch.as_tensor(s_idx, device=device)
            scores = episode_scores(g, per_m, si, s_lab, qi, len(order))
            meta = {"task": task, "K": k_shot, "seed": seed}
            rows += score_rows(scores, order, q_true, meta, sink, q_shas)
    return rows, {"queries": len(queries), "extracted_images": len(shas), "tau_0.99": taus}


def run_external(device, batch_size) -> int:
    with single_shot("external") as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        rows, info, sink = [], {}, []
        for task in ee.TASKS:
            with torch.no_grad():
                r, i = run_external_task(task, orders[task], factory, device, batch_size, sink)
            rows += r
            info[task] = i
            print(f"[p11-F2] {task}: {i['queries']} queries, {i['extracted_images']} images", flush=True)
        grid = ee.grid_of(rows)
        p10 = ee.grid_of(rc.load_json(ee.RESULTS_DIR / "external_results.json")["raw_metrics"])
        integrity = {m: max(abs(grid[m][key][metric] - p10[m][key][metric])
                            for key in grid[m] for metric in rc.METRICS) for m in (SS, NR, NR + "_m5")}
        if any(v >= 1e-9 for v in integrity.values()):
            raise rc.P6BFailClosedError(f"integrity check vs P10 external failed: {integrity}")

        hyp = {}
        for name, (target, ref) in HYPOTHESES["external"].items():
            visa = {metric: ee.visa_contrast(grid, target, ref, metric) for metric in rc.METRICS}
            hyp[name] = {"target": target, "reference": ref, "visa": visa,
                         "mtd_descriptive": {metric: ee.mtd_contrast(grid, target, ref, metric) for metric in rc.METRICS},
                         "verdict_BA": "CONFIRMED" if visa["balanced_accuracy"]["Dbar_ci95"][0] > 0 else "NOT CONFIRMED"}
        descriptive = {f"{t}_minus_{r}": {"visa_BA": ee.visa_contrast(grid, t, r, "balanced_accuracy"),
                                          "mtd_BA": ee.mtd_contrast(grid, t, r, "balanced_accuracy")}
                       for t, r in ((A, SS), (B, SS), (A + "_m5", NR + "_m5"), (B + "_m5", NR + "_m5"))}
        methods = (SS, NR, A, B, NR + "_m5", A + "_m5", B + "_m5")
        summary = {m: {"visa_object_macro_BA": ee.visa_macro(grid, m),
                       "mtd_BA": {str(k): {"mean": statistics.fmean(grid[m][(ee.MTD_TASK, k, s)]["balanced_accuracy"] for s in rc.SEEDS),
                                           "sd": statistics.stdev(grid[m][(ee.MTD_TASK, k, s)]["balanced_accuracy"] for s in rc.SEEDS)}
                                  for k in rc.K_VALUES}} for m in methods}
        result = {"schema_version": "p11_formal_external_v1", "protocol": "docs/P11_FORMAL_ADDENDUM.md#3",
                  "label": "post-hoc hypotheses added after the P10 external run; single run",
                  "provenance": provenance, "device": str(device), "task_info": info,
                  "integrity_max_abs_diff_vs_p10_external": integrity, "hypotheses": hyp,
                  "descriptive_contrasts": descriptive, "summary": summary, "raw_metrics": rows}

        def ci(c, k=None):
            if k is None:
                return fmt(c, "Dbar_positive_objects", 11)
            r = c["per_K"][str(k)]
            return f"{100 * r['D']:+.2f} [{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}] ({r['positive_objects']}/11)"

        def mtd(c):
            return " / ".join(f"{100 * c[str(k)]['mean']:+.2f}±{100 * c[str(k)]['sd']:.2f}" for k in rc.K_VALUES)

        lines = ["# P11 formal F2 — VisA (confirmatory) and MTD (descriptive), single run, post-hoc hypotheses", "",
                 f"Commit `{provenance['head']}`. Integrity vs P10 external: " +
                 ", ".join(f"{k} max|Δ|={v:.1e}" for k, v in integrity.items()), "",
                 "| Hypothesis | Contrast | K=1 | K=3 | K=5 | Pooled over K | Verdict |", "|---|---|---|---|---|---|---|"]
        for name, h in hyp.items():
            c = h["visa"]["balanced_accuracy"]
            lines.append(f"| {name} | {h['target']} − {h['reference']} | " +
                         " | ".join(ci(c, k) for k in rc.K_VALUES) + f" | {ci(c)} | {h['verdict_BA']} |")
        lines += ["", "| MTD contrast | Δ BA K=1 / 3 / 5 (seed mean ± SD) |", "|---|---|"]
        for name, h in hyp.items():
            lines.append(f"| {h['target']} − {h['reference']} | {mtd(h['mtd_descriptive']['balanced_accuracy'])} |")
        lines += ["", "| Descriptive | VisA pooled | MTD K=1 / 3 / 5 |", "|---|---|---|"]
        for key, v in descriptive.items():
            lines.append(f"| {key} | {ci(v['visa_BA'])} | {mtd(v['mtd_BA'])} |")
        lines += ["", "| Method | VisA K=1 | VisA K=3 | VisA K=5 | MTD K=1 | MTD K=3 | MTD K=5 |", "|---|---|---|---|---|---|---|"]
        for m, s in summary.items():
            lines.append(f"| {m} | " + " | ".join(
                f"{100 * s['visa_object_macro_BA'][str(k)]['mean']:.2f}" for k in rc.K_VALUES) + " | " + " | ".join(
                f"{100 * s['mtd_BA'][str(k)]['mean']:.2f}" for k in rc.K_VALUES) + " |")
        text = "\n".join(lines)
        rc.write_json(staging / "p11_formal_external_results.json", result)
        rc.write_jsonl(staging / "p11_formal_external_predictions.jsonl", sink)
        (staging / "p11_formal_external_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("mvtec", "external"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    return run_mvtec(device) if args.stage == "mvtec" else run_external(device, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
