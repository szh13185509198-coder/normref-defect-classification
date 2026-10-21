#!/usr/bin/env python3
"""P14 final formal addendum: ARR (A + region residual, C1) and CONS (support consensus, C0).

Protocol: docs/P14_FINAL_FORMAL_ADDENDUM.md. Each stage runs once.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p14.final_formal --stage mvtec --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p14.final_formal --stage external --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p14.final_formal --stage cable --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
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
from experiments.p6.p6b_stage3_common import load_query_contract  # noqa: E402
from experiments.p10 import cable_oof as co  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR, RESULTS_DIR as P10_FORMAL, mvrec_rows  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p10.industrial_metrics import mean_sd, normal_aware, per_thousand  # noqa: E402
from experiments.p10.external_normal_aware import METRIC_KEYS  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p12 import cable_ab  # noqa: E402
from experiments.p12.threshold_controls import P11_EXTERNAL, P11_MVTEC, exact_equal  # noqa: E402
from experiments.p13 import pilot as p13  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p14_formal" / s for s in ("mvtec", "external", "cable")}
GUARDED = ("experiments/p10", "experiments/p11", "experiments/p12", "experiments/p13", "experiments/p14",
           "experiments/revision", "docs/P14_FINAL_FORMAL_ADDENDUM.md")
A, NR, SS, XC = fa.A, fa.NR, fa.SS, "fused_xclass_clip18"
ARR, CONS = "fused_adareg_regres_soft_q99", "fused_cons_soft_supq90"
HYP = {"mvtec": {"H6_ARR_minus_A": (ARR, A), "H7_CONS_minus_xclass": (CONS, XC)},
       "external": {"E6_ARR_minus_A": (ARR, A), "E7_CONS_minus_xclass": (CONS, XC)}}


@contextlib.contextmanager
def single_shot(out: Path):
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; the P14 stage is single-shot")
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


def c1_descriptors(clip, bank_imgs, tau):
    """A's soft descriptor, the P10 fixed-fraction pooled descriptor and the region residual.

    ``tau`` = fa.tau99(bank_imgs), computed once per bank by the caller (the P11 path)."""
    bank = bank_imgs.reshape(-1, bank_imgs.shape[-1])
    pooled, soft, _ = fa.descriptors(clip, bank_imgs, tau)
    w = ab.region_weights(ra.patch_anomaly(clip, bank), tau, soft=True)
    return pooled, soft, p13.region_residual(clip, bank, w)


def episode_scores(g, per_m, si, s_lab, qi, n, xc, cons):
    base = rp.simpleshot(g[si], s_lab, g[qi], n)
    zb = rp._zscore(base)
    out = {SS: base, XC: zb + rp._zscore(xc), CONS: zb + rp._zscore(cons)}
    for suffix, (pooled, soft, rr) in per_m.items():
        sa = rp.simpleshot(soft[si], s_lab, soft[qi], n)
        srr = rp.simpleshot(rr[si], s_lab, rr[qi], n)
        if suffix == "":
            out[NR] = zb + rp._zscore(rp.simpleshot(pooled[si], s_lab, pooled[qi], n))
        out[A + suffix] = zb + rp._zscore(sa)
        out[ARR + suffix] = zb + rp._zscore(sa) + rp._zscore(srr)
    return out


# --------------------------------------------------------------------------- MVTec-FS

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
                    per_m = {}
                    for m, suffix in ((ra.M_PRIMARY, ""), (ra.M_SENS, "_m5")):
                        bank_imgs = normal_clip[torch.as_tensor([name_index[x] for x in draw[:m]], device=device)]
                        per_m[suffix] = c1_descriptors(clip, bank_imgs, fa.tau99(bank_imgs))  # whole tensor, as P11
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
                        xc = rp.local_xclass(clip[si], s_lab, clip[qi], n)  # the P10 formal path
                        cons = p13.consensus_scores(clip, si, s_lab, qi, n)[CONS]
                        sc = episode_scores(g, per_m, si, s_lab, qi, n, xc, cons)
                        rows += fa.score_rows(sc, order, q_true, {"product": product, "K": k_shot, "seed": seed},
                                              sink, list(qf["rgb_sha256"]))
            print(f"[p14-mvtec] {product} done", flush=True)

        integrity = {"vs_p10_formal": exact_equal(rows, rc.load_json(P10_FORMAL / "formal_results.json")["raw_metrics"],
                                                  (SS, NR, XC), ("product", "K", "seed")),
                     "vs_p11_formal": exact_equal(rows, rc.load_json(P11_MVTEC)["raw_metrics"], (A, A + "_m5"),
                                                  ("product", "K", "seed"))}
        if any(v >= 1e-9 for v in integrity.values()):
            raise rc.P6BFailClosedError(f"integrity check failed: {integrity}")
        full = dict(rc.metric_grid_from_rows(rows))
        full.update({m: v for m, v in rc.load_formal_metric_grid().items() if m == "ours"})
        full.update(rc.metric_grid_from_rows(mvrec_rows()))
        hyp = {}
        for name, (t, r) in HYP["mvtec"].items():
            res = {metric: contrast(full, r, metric, target=t) for metric in rc.METRICS}
            hyp[name] = {"target": t, "reference": r, "results": res,
                         "verdict_BA": "CONFIRMED" if res["balanced_accuracy"]["Dbar_ci95"][0] > 0 else "NOT CONFIRMED"}
        pairs = [(ARR, SS), (CONS, SS), (CONS, "ours"), (CONS, "mvrec_rgb"), (ARR, "ours"), (ARR, "mvrec_rgb"),
                 (ARR + "_m5", A + "_m5")]
        descriptive = {f"{t}_minus_{r}": {"balanced_accuracy": contrast(full, r, "balanced_accuracy", target=t)} for t, r in pairs}
        methods = (SS, XC, CONS, NR, A, ARR, A + "_m5", ARR + "_m5", "ours", "mvrec_rgb")
        summary = {m: rc.product_macro_summary(full, m) for m in methods}
        result = {"schema_version": "p14_final_mvtec_v1", "protocol": "docs/P14_FINAL_FORMAL_ADDENDUM.md",
                  "label": "final formal round; post-hoc hypotheses; single run", "provenance": provenance,
                  "integrity_max_abs_diff": integrity, "hypotheses": hyp, "descriptive_contrasts": descriptive,
                  "product_macro_summary": summary, "raw_metrics": rows}
        lines = ["# P14 final formal — MVTec-FS-RGB (single run)", "",
                 f"Commit `{provenance['head']}`. Integrity: " + ", ".join(f"{k} max|Δ|={v:.1e}" for k, v in integrity.items()), "",
                 "| Hypothesis | Contrast | K=1 | K=3 | K=5 | Pooled over K | Verdict |", "|---|---|---|---|---|---|---|"]
        for name, h in hyp.items():
            c = h["results"]["balanced_accuracy"]
            lines.append(f"| {name} | {h['target']} − {h['reference']} | " + " | ".join(
                f"{100 * c['per_K'][str(k)]['D']:+.2f} [{100 * c['per_K'][str(k)]['ci95'][0]:+.2f}, {100 * c['per_K'][str(k)]['ci95'][1]:+.2f}]"
                for k in rc.K_VALUES) + f" | {fa.fmt(c)} | {h['verdict_BA']} |")
        lines += ["", "Macro-F1 pooled: " + "; ".join(f"{n} {fa.fmt(h['results']['macro_f1'])}" for n, h in hyp.items()), "",
                  "| Descriptive (BA pooled) | Δ |", "|---|---|"]
        lines += [f"| {k} | {fa.fmt(v['balanced_accuracy'])} |" for k, v in descriptive.items()]
        lines += ["", "| Method | K=1 | K=3 | K=5 |", "|---|---|---|---|"]
        for m, s in summary.items():
            lines.append(f"| {m} | " + " | ".join(
                f"{100 * s[str(k)]['balanced_accuracy']['mean']:.2f} / {100 * s[str(k)]['macro_f1']['mean']:.2f}" for k in rc.K_VALUES) + " |")
        text = "\n".join(lines)
        rc.write_json(staging / "p14_mvtec_results.json", result)
        rc.write_jsonl(staging / "p14_mvtec_predictions.jsonl", sink)
        (staging / "p14_mvtec_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- VisA / MTD

def external_task(task, order, factory, device, batch_size, sink):
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
    q_patches = clip[qi.cpu()]
    g = g.to(device)
    rows = []
    for seed in rc.SEEDS:
        bank_idx = torch.as_tensor([index[s] for s in banks[seed]])
        per_m = {}
        for m, suffix in ((ee.BANK, ""), (ra.M_SENS, "_m5")):
            bank_imgs = clip[bank_idx[:m]].to(device).float()
            tau = fa.tau99(bank_imgs)
            parts = [c1_descriptors(chunk.to(device).float(), bank_imgs, tau) for chunk in clip.split(ee.Q_CHUNK)]
            per_m[suffix] = tuple(torch.cat(x) for x in zip(*parts))
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            n = len(order)
            si = torch.as_tensor(s_idx, device=device)
            sup = clip[si.cpu()].to(device).float()
            xc = ee.xclass_chunked(sup, s_lab, q_patches, n, device)  # the P10 external path
            s_range = torch.arange(len(s_idx), device=device)
            cons = []
            for chunk in q_patches.split(ee.Q_CHUNK):
                both = torch.cat([sup, chunk.to(device).float()])
                q_range = torch.arange(len(s_idx), both.shape[0], device=device)
                cons.append(p13.consensus_scores(both, s_range, s_lab, q_range, n)[CONS])
            sc = episode_scores(g, per_m, si, s_lab, qi, n, xc, torch.cat(cons))
            rows += fa.score_rows(sc, order, q_true, {"task": task, "K": k_shot, "seed": seed}, sink,
                                  [r["rgb_sha256"] for r in queries])
    return rows, {"queries": len(queries), "extracted_images": len(shas)}


def run_external(device, batch_size) -> int:
    with single_shot(OUT["external"]) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        rows, info, sink = [], {}, []
        for task in ee.TASKS:
            with torch.no_grad():
                r, i = external_task(task, orders[task], factory, device, batch_size, sink)
            rows += r
            info[task] = i
            print(f"[p14-ext] {task}: {i['queries']} queries, {i['extracted_images']} images", flush=True)
        integrity = {"vs_p10_external": exact_equal(rows, rc.load_json(ee.RESULTS_DIR / "external_results.json")["raw_metrics"],
                                                    (SS, NR, XC), ("task", "K", "seed")),
                     "vs_p11_external": exact_equal(rows, rc.load_json(P11_EXTERNAL)["raw_metrics"], (A, A + "_m5"),
                                                    ("task", "K", "seed"))}
        if any(v >= 1e-9 for v in integrity.values()):
            raise rc.P6BFailClosedError(f"integrity check failed: {integrity}")
        grid = ee.grid_of(rows)
        hyp = {}
        for name, (t, r) in HYP["external"].items():
            visa = {metric: ee.visa_contrast(grid, t, r, metric) for metric in rc.METRICS}
            hyp[name] = {"target": t, "reference": r, "visa": visa,
                         "mtd_descriptive": {metric: ee.mtd_contrast(grid, t, r, metric) for metric in rc.METRICS},
                         "verdict_BA": "CONFIRMED" if visa["balanced_accuracy"]["Dbar_ci95"][0] > 0 else "NOT CONFIRMED"}
        descriptive = {f"{t}_minus_{r}": {"visa_BA": ee.visa_contrast(grid, t, r, "balanced_accuracy"),
                                          "mtd_BA": ee.mtd_contrast(grid, t, r, "balanced_accuracy")}
                       for t, r in ((ARR, SS), (CONS, SS), (ARR + "_m5", A + "_m5"))}
        methods = (SS, XC, CONS, NR, A, ARR, A + "_m5", ARR + "_m5")
        summary = {m: {"visa_object_macro_BA": ee.visa_macro(grid, m),
                       "mtd_BA": {str(k): statistics.fmean(grid[m][(ee.MTD_TASK, k, s)]["balanced_accuracy"] for s in rc.SEEDS)
                                  for k in rc.K_VALUES}} for m in methods}
        cells = {(u["method"], u["task"], u["K"], u["seed"]): normal_aware(u["true"], u["pred"], ee.NORMAL_CLASS[u["task"].split("__")[0]])
                 for u in sink}
        industrial = {}
        for m in methods:
            industrial[m] = {}
            for k in rc.K_VALUES:
                visa = {x: mean_sd(statistics.fmean(cells[(m, t, k, s)][x] for t in ee.VISA_TASKS) for s in rc.SEEDS) for x in METRIC_KEYS}
                mtd = {x: mean_sd(cells[(m, ee.MTD_TASK, k, s)][x] for s in rc.SEEDS) for x in METRIC_KEYS}
                industrial[m][str(k)] = {"visa_object_macro": visa, "mtd": mtd,
                                         "visa_per_1000_at_95pct_good": per_thousand(visa["false_reject_rate"]["mean"], visa["escape_rate"]["mean"]),
                                         "mtd_per_1000_at_95pct_good": per_thousand(mtd["false_reject_rate"]["mean"], mtd["escape_rate"]["mean"])}
        result = {"schema_version": "p14_final_external_v1", "protocol": "docs/P14_FINAL_FORMAL_ADDENDUM.md",
                  "label": "final formal round; post-hoc hypotheses; single run", "provenance": provenance,
                  "task_info": info, "integrity_max_abs_diff": integrity, "hypotheses": hyp,
                  "descriptive_contrasts": descriptive, "summary": summary, "normal_aware": industrial, "raw_metrics": rows}

        def ci(c, k=None):
            if k is None:
                return fa.fmt(c, "Dbar_positive_objects", 11)
            r = c["per_K"][str(k)]
            return f"{100 * r['D']:+.2f} [{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}] ({r['positive_objects']}/11)"

        def mtd(c):
            return " / ".join(f"{100 * c[str(k)]['mean']:+.2f}±{100 * c[str(k)]['sd']:.2f}" for k in rc.K_VALUES)
        lines = ["# P14 final formal — VisA (confirmatory) and MTD (descriptive), single run", "",
                 f"Commit `{provenance['head']}`. Integrity: " + ", ".join(f"{k} max|Δ|={v:.1e}" for k, v in integrity.items()), "",
                 "| Hypothesis | Contrast | K=1 | K=3 | K=5 | Pooled over K | Verdict |", "|---|---|---|---|---|---|---|"]
        for name, h in hyp.items():
            c = h["visa"]["balanced_accuracy"]
            lines.append(f"| {name} | {h['target']} − {h['reference']} | " + " | ".join(ci(c, k) for k in rc.K_VALUES)
                         + f" | {ci(c)} | {h['verdict_BA']} |")
        lines += ["", "| MTD contrast | Δ BA K=1 / 3 / 5 |", "|---|---|"]
        lines += [f"| {h['target']} − {h['reference']} | {mtd(h['mtd_descriptive']['balanced_accuracy'])} |" for h in hyp.values()]
        lines += ["", "| Descriptive | VisA pooled | MTD K=1 / 3 / 5 |", "|---|---|---|"]
        lines += [f"| {k} | {ci(v['visa_BA'])} | {mtd(v['mtd_BA'])} |" for k, v in descriptive.items()]
        lines += ["", "| Method | VisA K=1 | VisA K=3 | VisA K=5 | MTD K=1 | MTD K=3 | MTD K=5 | VisA FR/1000 K=5 | MTD FR/1000 K=5 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m, s in summary.items():
            lines.append(f"| {m} | " + " | ".join(f"{100 * s['visa_object_macro_BA'][str(k)]['mean']:.2f}" for k in rc.K_VALUES)
                         + " | " + " | ".join(f"{100 * s['mtd_BA'][str(k)]:.2f}" for k in rc.K_VALUES)
                         + f" | {industrial[m]['5']['visa_per_1000_at_95pct_good']['false_rejects_per_1000']:.0f}"
                         + f" | {industrial[m]['5']['mtd_per_1000_at_95pct_good']['false_rejects_per_1000']:.0f} |")
        text = "\n".join(lines)
        rc.write_json(staging / "p14_external_results.json", result)
        rc.write_jsonl(staging / "p14_external_predictions.jsonl", sink)
        (staging / "p14_external_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- Cable

def run_cable(mvtec_root: Path, device, batch_size) -> int:
    from experiments.p10.external_evaluate import extract
    methods = (SS, XC, CONS, NR, A, ARR)
    with single_shot(OUT["cable"]) as staging:
        provenance = rc.git_provenance()
        folds = {f: (co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_train.jsonl"),
                     co.read_jsonl(co.FOLD_DIR / f"fold_{f:02d}_val.jsonl")) for f in co.FOLDS}
        truth = {r["image_path"]: r["class_name"] for f in co.FOLDS for r in folds[f][1]}
        order = co.load_v1(111)[next(iter(truth))]["class_names"]
        pool = sorted(truth)
        normal_dir = mvtec_root / "cable" / "train" / "good"
        normal_names = sorted(p.name for p in normal_dir.glob("*.png"))
        draws = {s: random.Random(f"p10cable|{s}").sample(normal_names, co.BANK) for s in rc.SEEDS}
        published = rc.load_json(co.OUT / "cable_results.json")
        if {str(s): v for s, v in draws.items()} != published["normal_bank_draws"]:
            raise rc.P6BFailClosedError("normal-bank draws differ from P10 R1")
        bank_names = sorted({n for d in draws.values() for n in d})
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        with torch.no_grad():
            g_all, clip_all = extract(factory, [co.IMAGE_ROOT / p for p in pool] + [normal_dir / n for n in bank_names],
                                      device, batch_size)
        index = {p: i for i, p in enumerate(pool)}
        bank_index = {n: len(pool) + i for i, n in enumerate(bank_names)}
        g = g_all[:len(pool)].to(device)
        clip = clip_all[:len(pool)].to(device).float()
        preds = {m: {} for m in methods}
        with torch.no_grad():
            for seed in rc.SEEDS:
                bank_imgs = clip_all[[bank_index[n] for n in draws[seed]]].to(device).float()
                per_m = {"": c1_descriptors(clip, bank_imgs, fa.tau99(bank_imgs))}
                seed_preds = {m: {} for m in methods}
                for f in co.FOLDS:
                    train, val = folds[f]
                    si = torch.as_tensor([index[r["image_path"]] for r in train], device=device)
                    sl = [order.index(r["class_name"]) for r in train]
                    qi = torch.as_tensor([index[r["image_path"]] for r in val], device=device)
                    n = len(order)
                    xc = rp.local_xclass(clip[si], sl, clip[qi], n)
                    cons = p13.consensus_scores(clip, si, sl, qi, n)[CONS]
                    sc = episode_scores(g, per_m, si, sl, qi, n, xc, cons)
                    for m in methods:
                        for r, c in zip(val, sc[m].argmax(dim=1).cpu().tolist()):
                            seed_preds[m][r["image_path"]] = order[c]
                for m in methods:
                    preds[m][seed] = seed_preds[m]
        old = {}
        for path, names in ((co.OUT / "cable_predictions.jsonl", (SS, NR, XC)), (cable_ab.OUT / "cable_ab_predictions.jsonl", (A,))):
            for line in path.open(encoding="utf-8"):
                r = json.loads(line)
                if r["method"] in names:
                    old[(r["method"], int(r["seed"]), r["image_path"])] = r["pred"]
        mismatch = [(m, s, i) for m in (SS, NR, XC, A) for s in rc.SEEDS for i in pool if preds[m][s][i] != old[(m, s, i)]]
        if mismatch:
            raise rc.P6BFailClosedError(f"integrity vs P10 R1 / P12 R1' failed: {len(mismatch)}, e.g. {mismatch[:3]}")
        summary = co.summarize(preds, truth, order)
        for m in ("v1_ours_tau", "residual_mean_clip18"):
            summary[m] = published["summary"][m]
        persistent = {m: sorted(i for i in pool if all(preds[m][s][i] != truth[i] for s in rc.SEEDS)) for m in methods}
        result = {"schema_version": "p14_final_cable_v1", "protocol": "docs/P14_FINAL_FORMAL_ADDENDUM.md",
                  "provenance": provenance, "integrity": {"predictions_rechecked": 4 * len(rc.SEEDS) * len(pool), "mismatches": 0},
                  "summary": summary, "persistent_errors_all_seeds": persistent}

        def cell(m, k):
            v = summary[m]["mean_sd"][k]
            return f"{100 * v['mean']:.2f} ± {100 * v['sd']:.2f}"
        lines = ["# P14 — Cable 8-fold OOF (72 images, 5 seeds; descriptive)", "",
                 f"Commit `{provenance['head']}`. SimpleShot / normref / cross-class equal P10 R1; A equals P12 R1'.", "",
                 "| Method | 9-class BA | Good recall | Defect recall | Binary BA | Type acc. (detected) | Persistent errors |",
                 "|---|---|---|---|---|---|---|"]
        for m in (*methods, "residual_mean_clip18", "v1_ours_tau"):
            pe = persistent.get(m)
            lines.append(f"| {m} | {cell(m, 'balanced_accuracy')} | {cell(m, 'good_recall')} | {cell(m, 'defect_recall')} | "
                         f"{cell(m, 'binary_ba')} | {cell(m, 'defect_type_acc_detected')} | {len(pe) if pe is not None else '—'} |")
        text = "\n".join(lines)
        rc.write_json(staging / "p14_cable_results.json", result)
        rc.write_jsonl(staging / "p14_cable_predictions.jsonl",
                       [{"method": m, "seed": s, "image_path": i, "true": truth[i], "pred": p}
                        for m in methods for s in rc.SEEDS for i, p in preds[m][s].items()])
        (staging / "p14_cable_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("mvtec", "external", "cable"))
    parser.add_argument("--mvtec-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.stage == "mvtec":
        return run_mvtec(device)
    if args.stage == "external":
        return run_external(device, args.batch_size)
    if not args.mvtec_root:
        raise SystemExit("--mvtec-root is required for the cable stage")
    return run_cable(Path(args.mvtec_root), device, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
