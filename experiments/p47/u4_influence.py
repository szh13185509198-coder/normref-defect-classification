#!/usr/bin/env python3
"""P47 U4: influence quantities - acquisition variation, repeatability, drift and recalibration.

Protocol: docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, "U4". Three stages, each write-once:

1. mvtec     - MVTec-FS formal, classification, K = 5, 5 seeds. -> ckpt/p47/u4_mvtec/
2. external  - VisA (11 tasks) and MTD, classification and screening (P15 statistic and calibration draws), K = 5,
               5 seeds. -> ckpt/p47/u4_external/
3. analyze   - paired deltas with cluster bootstrap, flip rates, FRR / AUROC, attribute agreement for J, sensitivity
               coefficients, drift tolerance. CPU only, from the two prediction files. -> ckpt/p47/u4_analysis/

Conditions (experiments/p47/common.py): "none", P1-P7 (20 levels) and J1-J5 (re-acquisition jitter). Perturbations act
on the decoded RGB image at native resolution, before the locked 336 px evaluation view, seeded per image and condition
with random.Random(f"p47pert|{sha}|{cond}"); sha = the manifest rgb_sha256 (MVTec-FS / VisA / MTD records) or the
SHA-256 of the file (MVTec AD train/good bank images).

Scenarios:
- D  (drift, no recalibration): queries perturbed; supports, bank and calibration normals unperturbed. All conditions.
- RC (recalibrated): queries, bank and calibration normals perturbed alike; supports unperturbed. "none" and P1-P7.
  The RC bank and calibration are re-extracted through the U4 path for "none" too, so every delta is taken against a
  reference computed in exactly the same way.

Reference and integrity:
- the unperturbed reference computed exactly as published (cached MVTec-FS features; the P14/P15 extraction call for
  VisA/MTD) must reproduce the published P14 BA rows / predictions and the P15 AUROC and FRR at alpha = 5 % at K = 5;
  otherwise the stage stops before writing anything;
- the "none" condition through the U4 path (process-pool decoding, query-only batches) is compared with that
  reference (max feature difference, prediction agreement) and reported; all deltas are against U4 "none".

Normal queries of VisA/MTD are limited to 250 per task (fixed before the run): if a task has more,
random.Random(f"p47u4cap|{task}").sample(sorted normal query sha256, 250); all defect queries are kept.

Usage (Windows, the author's 4090 machine):
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u4_influence --stage mvtec --mvtec-root D:\\intergration\\MVTEC --device cuda --workers 6
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u4_influence --stage external --device cuda --workers 6
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u4_influence --stage analyze
    python -m experiments.p47.u4_influence --selftest        # CPU, synthetic data, no checkpoints needed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from experiments.p47 import common as cm  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p47" / f"u4_{s}" for s in ("mvtec", "external", "analysis")}
K_SHOT = 5
ALPHA = 0.05
NORMAL_CAP = 250
QUERY_CONDS = ("none", *cm.P_CONDITIONS, *cm.J_CONDITIONS)
RC_CONDS = ("none", *cm.P_CONDITIONS)


# --------------------------------------------------------------------------- feature extraction (U4 path)

def features(factory, loader, jobs, device, batch_size, with_global=True):
    """As experiments.p10.external_evaluate.extract, with the images decoded and perturbed in the loader pool."""
    import torch
    import torch.nn.functional as F
    from experiments.p6 import p6b_stage2_models as models
    from experiments.p10.extract_dev_features import CLIP_LAYER
    glob, clip, buf = [], [], []

    def flush():
        batch = torch.from_numpy(np.concatenate(buf)).to(device)
        with torch.no_grad():
            pooled, tokens = factory.clip_model.visual(batch, [CLIP_LAYER])
            if with_global:
                dino = factory.dino_extractor.forward_patches(batch)
                glob.append(factory.fusion.fuse(models.l2_rows(pooled), models.l2_rows(dino.mean(dim=1))).float().cpu())
            clip.append(F.normalize(tokens[0][:, 1:, :].float(), dim=-1).half().cpu())
        buf.clear()
    for view in loader.views(jobs):
        buf.append(view)
        if len(buf) == batch_size:
            flush()
    if buf:
        flush()
    return (torch.cat(glob) if with_global else None), torch.cat(clip)


def _max_diff(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


# --------------------------------------------------------------------------- stage 1: MVTec-FS

def run_mvtec(mvtec_root: Path, device, batch_size: int, workers: int) -> int:
    import torch
    from types import SimpleNamespace
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p6.p6b_stage3_common import classification_metrics, load_query_contract
    from experiments.p10 import run_pilot as rp
    from experiments.p10.formal_evaluate import MANIFEST_DIR
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    from experiments.p11 import formal_ab as fa
    from experiments.p14 import final_formal as ff
    from experiments.revision import revision_common as rc
    published = {(r["method"], r["product"], int(r["seed"])): r["balanced_accuracy"]
                 for r in rc.load_json(ff.OUT["mvtec"] / "p14_mvtec_results.json")["raw_metrics"]
                 if r["method"] in (ff.ARR, ff.SS) and int(r["K"]) == K_SHOT}
    loader = cm.PerturbedLoader(workers)
    t0 = time.time()
    with cm.single_shot(OUT["mvtec"]) as staging:
        provenance = rc.git_provenance()
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=True)
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        rows, products = [], {}
        for product in rc.PRODUCTS:
            recs = [q for q in queries if q["product"] == product]
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            if [r["rgb_sha256"] for r in recs] != list(qf["rgb_sha256"]):
                raise cm.P47Error(f"{product}: query order differs from the cached features")
            path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
            n_pool = len(dev["rel_paths"])
            q_true = [r["true_label"] for r in recs]
            order, eps = None, {}
            for seed in rc.SEEDS:
                man = rc.load_json(MANIFEST_DIR / product / f"k{K_SHOT}_s{seed}.json")
                order = order or list(man["class_order"])
                if list(man["class_order"]) != order:
                    raise cm.P47Error(f"{product}: class order differs between manifests")
                s_idx, s_lab = [], []
                for c, cls in enumerate(order):
                    for item in man["classes"][cls]["items"]:
                        s_idx.append(path_index[item["rel_path"]])
                        s_lab.append(c)
                eps[seed] = (torch.as_tensor(s_idx, device=device), s_lab)
            n = len(order)
            qi = torch.arange(n_pool, n_pool + len(recs), device=device)
            zeros = torch.zeros(len(recs), n, device=device)
            name_index = {x: i for i, x in enumerate(nf["names"])}
            good = mvtec_root / product / "train" / "good"
            bank_jobs_base = [(good / x, cm.file_sha256(good / x)) for x in nf["names"]]
            q_jobs_base = [(Path(r["image_path"]), r["rgb_sha256"]) for r in recs]

            def predict(g_q, clip_q, bank_clip, methods):
                g = torch.cat([dev["global_fusion"], g_q]).to(device)
                clip = torch.cat([dev["clip18_patches"], clip_q]).to(device).float()
                out = {}
                for seed in rc.SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in nf["draws"][str(seed)][:20]])
                    bank_imgs = bank_clip[idx].to(device).float()
                    per_m = {"": ff.c1_descriptors(clip, bank_imgs, fa.tau99(bank_imgs))}   # whole tensor, as P14
                    si, s_lab = eps[seed]
                    sc = ff.episode_scores(g, per_m, si, s_lab, qi, n, zeros, zeros)
                    for m in methods:
                        out[(m, seed)] = sc[m].argmax(dim=1).cpu().tolist()
                return out

            with torch.no_grad():
                ref = predict(qf["global_fusion"], qf["clip18_patches"], nf["clip18_patches"], (ff.ARR, ff.SS))
                worst = max(abs(classification_metrics(q_true, [order[i] for i in ref[(m, s)]], order)["balanced_accuracy"]
                                - published[(m, product, s)]) for m in (ff.ARR, ff.SS) for s in rc.SEEDS)
                if worst >= 1e-9:
                    raise cm.P47Error(f"{product}: reference does not reproduce P14 at K = 5 (max |dBA| = {worst:.2e})")
                info = {"class_order": order, "true": q_true, "query_sha256": list(qf["rgb_sha256"]),
                        "reference_vs_p14_max_abs_dBA": worst}
                for cond in QUERY_CONDS:
                    g_q, clip_q = features(factory, loader, [(p, s, cond) for p, s in q_jobs_base], device, batch_size)
                    d = predict(g_q, clip_q, nf["clip18_patches"], (ff.ARR, ff.SS))
                    for (m, s), pred in d.items():
                        rows.append({"product": product, "cond": cond, "scenario": "D", "seed": s, "method": m,
                                     "pred": pred})
                    if cond == "none":
                        info["none_vs_cached"] = {
                            "max_abs_clip": _max_diff(clip_q, qf["clip18_patches"]),
                            "max_abs_global": _max_diff(g_q, qf["global_fusion"]),
                            "prediction_agreement": statistics.fmean(
                                a == b for k in d for a, b in zip(d[k], ref[k]))}
                    if cond in RC_CONDS:
                        _, bank_c = features(factory, loader, [(p, s, cond) for p, s in bank_jobs_base], device,
                                             batch_size, with_global=False)
                        if cond == "none":
                            info["none_vs_cached"]["max_abs_bank_clip"] = _max_diff(bank_c, nf["clip18_patches"])
                        r = predict(g_q, clip_q, bank_c, (ff.ARR,))
                        for (m, s), pred in r.items():
                            rows.append({"product": product, "cond": cond, "scenario": "RC", "seed": s, "method": m,
                                         "pred": pred})
            products[product] = info
            print(f"[p47-u4-mvtec] {product}: {len(recs)} queries, {len(nf['names'])} bank images; "
                  f"none vs cached agreement {info['none_vs_cached']['prediction_agreement']:.4f}; "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        rc.write_jsonl(staging / "u4_mvtec_preds.jsonl", rows)
        rc.write_json(staging / "u4_mvtec_run.json", {
            "schema_version": "p47_u4_mvtec_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u4",
            "provenance": provenance, "K": K_SHOT, "seeds": list(rc.SEEDS),
            "methods": {"ARR": ff.ARR, "SimpleShot": ff.SS}, "conditions": list(QUERY_CONDS),
            "rc_conditions": list(RC_CONDS), "products": products, "minutes": (time.time() - t0) / 60})
    loader.close()
    return 0


# --------------------------------------------------------------------------- stage 2: VisA / MTD

def capped_queries(task, queries, normal):
    """Defect queries all; normal queries capped at NORMAL_CAP (rule fixed before the run). Keeps the sha order."""
    normals = sorted(r["rgb_sha256"] for r in queries if r["class_label"] == normal)
    keep = set(normals) if len(normals) <= NORMAL_CAP else set(random.Random(f"p47u4cap|{task}").sample(normals, NORMAL_CAP))
    return [r for r in queries if r["class_label"] != normal or r["rgb_sha256"] in keep]


def run_external(device, batch_size: int, workers: int) -> int:
    import torch
    from types import SimpleNamespace
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p10 import external_evaluate as ee
    from experiments.p11 import formal_ab as fa
    from experiments.p14 import final_formal as ff
    from experiments.p15 import screening as sc
    from experiments.p15 import two_stage as ts
    from experiments.revision import revision_common as rc
    stat = ts.selected_statistic()
    p14 = {}
    for line in (ff.OUT["external"] / "p14_external_predictions.jsonl").open(encoding="utf-8"):
        u = json.loads(line)
        if u["method"] in (ff.ARR, ff.SS) and int(u["K"]) == K_SHOT:
            p14[(u["method"], u["task"], int(u["seed"]))] = (u["rgb_sha256"], u["pred"])
    p15 = {(u["task"], int(u["seed"])): u for u in rc.load_json(ts.OUT["external"] / "p15_external_results.json")["units"]
           if int(u["K"]) == K_SHOT}
    loader = cm.PerturbedLoader(workers)
    t0 = time.time()
    with cm.single_shot(OUT["external"]) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        rows, tasks = [], {}
        for task in ee.TASKS:
            order = orders[task]
            dataset = task.split("__")[0]
            records = [json.loads(line) for line in (ee.BUNDLE / f"{task}.images.jsonl").open(encoding="utf-8")]
            draws = json.load((ee.BUNDLE / f"{task}.support_draws.json").open(encoding="utf-8"))
            by_sha = {r["rgb_sha256"]: r for r in records}
            queries = sorted((r for r in records if r["partition"] == "query"), key=lambda r: r["rgb_sha256"])
            normal = ee.NORMAL_CLASS[dataset]
            normal_candidates = [r["rgb_sha256"] for r in records
                                 if r["partition"] == "support_candidate" and r["class_label"] == normal]
            query_shas = {r["rgb_sha256"] for r in queries}
            banks, needed, used_normals = {}, set(), set()
            for seed in rc.SEEDS:                                   # as P15 external_task
                banks[seed] = ee.bank_draw(task, seed, normal_candidates, draws[normal][str(seed)]["K5"])
                support_all = {s for cls in order for s in draws[cls][str(seed)]["K5"]}
                if set(banks[seed]) & (support_all | query_shas) or support_all & query_shas:
                    raise cm.P47Error(f"{task}/s{seed}: bank/support/query overlap")
                needed |= support_all | set(banks[seed])
                used_normals |= set(banks[seed]) | set(draws[normal][str(seed)]["K5"])
            shas = sorted(needed) + [r["rgb_sha256"] for r in queries]
            index = {s: i for i, s in enumerate(shas)}
            path = {s: ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in by_sha}
            cal = random.Random(f"p15cal|{task}").sample(sorted(set(normal_candidates) - used_normals - query_shas), ts.N_CAL)
            with torch.no_grad():
                g, clip = ee.extract(factory, [path[s] for s in shas], device, batch_size)        # the P14 / P15 call
                _, clip_cal = ee.extract(factory, [path[s] for s in cal], device, batch_size)
            g = g.to(device)
            sub = capped_queries(task, queries, normal)
            sub_shas = [r["rgb_sha256"] for r in sub]
            sub_true = [r["class_label"] for r in sub]
            bank_union = sorted({s for b in banks.values() for s in b})
            bank_pos = {s: i for i, s in enumerate(bank_union)}
            eps = {}
            for seed in rc.SEEDS:
                s_idx, s_lab = [], []
                for c, cls in enumerate(order):
                    for s in draws[cls][str(seed)][f"K{K_SHOT}"]:
                        s_idx.append(index[s])
                        s_lab.append(c)
                eps[seed] = (s_idx, s_lab)

            def evaluate(g_q, clip_q, bank_clip, cal_clip, methods):
                """bank_clip rows follow bank_union. Returns {(method|'score'|'cal', seed): list}."""
                out = {}
                n = len(order)
                for seed in rc.SEEDS:
                    s_idx, s_lab = eps[seed]
                    bank_imgs = bank_clip[torch.as_tensor([bank_pos[s] for s in banks[seed]])].to(device).float()
                    bank = bank_imgs.reshape(-1, clip.shape[-1])
                    tau = fa.tau99(bank_imgs)
                    rows_clip = torch.cat([clip[torch.as_tensor(s_idx)], clip_q])
                    parts = [ff.c1_descriptors(c.to(device).float(), bank_imgs, tau) for c in rows_clip.split(ee.Q_CHUNK)]
                    per_m = {"": tuple(torch.cat(x) for x in zip(*parts))}
                    gg = torch.cat([g[torch.as_tensor(s_idx, device=device)], g_q.to(device)])
                    si = torch.arange(len(s_idx), device=device)
                    qi = torch.arange(len(s_idx), len(s_idx) + clip_q.shape[0], device=device)
                    zeros = torch.zeros(clip_q.shape[0], n, device=device)
                    scores = ff.episode_scores(gg, per_m, si, s_lab, qi, n, zeros, zeros)
                    for m in methods:
                        out[(m, seed)] = scores[m].argmax(dim=1).cpu().tolist()
                    out[("score", seed)] = ts.scores_of(clip_q, bank, stat, device, ee.Q_CHUNK)
                    out[("cal", seed)] = ts.scores_of(cal_clip, bank, stat, device, ee.Q_CHUNK)
                return out

            info = {"class_order": order, "normal": normal, "true": sub_true, "query_sha256": sub_shas,
                    "queries_total": len(queries), "queries_used": len(sub),
                    "normal_queries_total": sum(r["class_label"] == normal for r in queries),
                    "normal_queries_used": sum(t == normal for t in sub_true), "bank_images": len(bank_union),
                    "calibration_normals": len(cal)}
            with torch.no_grad():
                # reference exactly as P15 external_task computes it (all rows, Q_CHUNK chunks) -> integrity
                qi_all = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
                ref = {}
                for seed in rc.SEEDS:
                    bank_imgs = clip[torch.as_tensor([index[s] for s in banks[seed]])].to(device).float()
                    bank = bank_imgs.reshape(-1, clip.shape[-1])
                    tau = fa.tau99(bank_imgs)
                    parts = [ff.c1_descriptors(ch.to(device).float(), bank_imgs, tau) for ch in clip.split(ee.Q_CHUNK)]
                    per_m = {"": tuple(torch.cat(x) for x in zip(*parts))}
                    s_all = ts.scores_of(clip, bank, stat, device, ee.Q_CHUNK)
                    ref[("score", seed)] = [s_all[i] for i in qi_all.tolist()]
                    ref[("cal", seed)] = ts.scores_of(clip_cal, bank, stat, device, ee.Q_CHUNK)
                    s_idx, s_lab = eps[seed]
                    zeros = torch.zeros(len(queries), len(order), device=device)
                    scores = ff.episode_scores(g, per_m, torch.as_tensor(s_idx, device=device), s_lab, qi_all,
                                               len(order), zeros, zeros)
                    for m in (ff.ARR, ff.SS):
                        ref[(m, seed)] = scores[m].argmax(dim=1).cpu().tolist()
                all_true = [r["class_label"] for r in queries]
                for seed in rc.SEEDS:
                    for m in (ff.ARR, ff.SS):
                        shas_pub, pred_pub = p14[(m, task, seed)]
                        if shas_pub != [r["rgb_sha256"] for r in queries] or pred_pub != [order[i] for i in ref[(m, seed)]]:
                            raise cm.P47Error(f"{task}/s{seed}/{m}: reference does not reproduce P14")
                    s_q, s_cal = ref[("score", seed)], ref[("cal", seed)]
                    t05 = sc.conformal_threshold(s_cal, ALPHA)
                    good = [x for x, t in zip(s_q, all_true) if t == normal]
                    bad = [x for x, t in zip(s_q, all_true) if t != normal]
                    frr = sum(x > t05 for x in good) / len(good)
                    u = p15[(task, seed)]
                    if abs(sc.auroc(good, bad) - u["auroc"]) > 1e-12 or \
                            abs(frr - u["two_level"][ff.ARR][str(ALPHA)]["false_reject_rate"]) > 1e-12:
                        raise cm.P47Error(f"{task}/s{seed}: reference does not reproduce P15 (AUROC / FRR)")
                ref_pos = [index[s] - len(needed) for s in sub_shas]           # reference rows of the subset
                q_jobs = [(path[s], s) for s in sub_shas]
                b_jobs = [(path[s], s) for s in bank_union]
                c_jobs = [(path[s], s) for s in cal]
                for cond in QUERY_CONDS:
                    g_q, clip_q = features(factory, loader, [(p, s, cond) for p, s in q_jobs], device, batch_size)
                    d = evaluate(g_q, clip_q, clip[torch.as_tensor([index[s] for s in bank_union])], clip_cal,
                                 (ff.ARR, ff.SS))
                    if cond == "none":
                        ref_rows = torch.as_tensor([index[s] for s in sub_shas])
                        info["none_vs_reference"] = {
                            "max_abs_clip": _max_diff(clip_q, clip[ref_rows]),
                            "max_abs_global": _max_diff(g_q.to(device), g[ref_rows.to(device)]),
                            "type_agreement": statistics.fmean(
                                a == ref[(m, s)][i] for m in (ff.ARR, ff.SS) for s in rc.SEEDS
                                for a, i in zip(d[(m, s)], ref_pos)),
                            "max_abs_score": max(abs(a - ref[("score", s)][i]) for s in rc.SEEDS
                                                 for a, i in zip(d[("score", s)], ref_pos))}
                    for seed in rc.SEEDS:
                        rows.append({"task": task, "cond": cond, "scenario": "D", "seed": seed,
                                     "pred": {"ARR": d[(ff.ARR, seed)], "SimpleShot": d[(ff.SS, seed)]},
                                     "score": d[("score", seed)], "cal": d[("cal", seed)]})
                    if cond in RC_CONDS:
                        _, bank_c = features(factory, loader, [(p, s, cond) for p, s in b_jobs], device, batch_size,
                                             with_global=False)
                        _, cal_c = features(factory, loader, [(p, s, cond) for p, s in c_jobs], device, batch_size,
                                            with_global=False)
                        r = evaluate(g_q, clip_q, bank_c, cal_c, (ff.ARR,))
                        for seed in rc.SEEDS:
                            rows.append({"task": task, "cond": cond, "scenario": "RC", "seed": seed,
                                         "pred": {"ARR": r[(ff.ARR, seed)]}, "score": r[("score", seed)],
                                         "cal": r[("cal", seed)]})
            tasks[task] = info
            print(f"[p47-u4-ext] {task}: {len(sub)} of {len(queries)} queries; none vs reference agreement "
                  f"{info['none_vs_reference']['type_agreement']:.4f}; {(time.time() - t0) / 60:.1f} min", flush=True)
        rc.write_jsonl(staging / "u4_external_preds.jsonl", rows)
        rc.write_json(staging / "u4_external_run.json", {
            "schema_version": "p47_u4_external_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u4",
            "provenance": provenance, "K": K_SHOT, "alpha": ALPHA, "statistic": stat, "seeds": list(rc.SEEDS),
            "methods": {"ARR": ff.ARR, "SimpleShot": ff.SS}, "conditions": list(QUERY_CONDS),
            "rc_conditions": list(RC_CONDS), "normal_cap": NORMAL_CAP, "tasks": tasks,
            "integrity": "reference reproduces P14 predictions (ARR, SimpleShot) and P15 AUROC / FRR at alpha = 5 %, K = 5",
            "minutes": (time.time() - t0) / 60})
    loader.close()
    return 0


# --------------------------------------------------------------------------- stage 3: analysis

def cluster_boot(values: dict, reps=10000, seed=20260813) -> list:
    """Percentile cluster bootstrap of the mean; values {cluster: [paired observations]} (repository convention)."""
    rng = random.Random(seed)
    keys = list(values)
    est = sorted(statistics.fmean([v for _ in keys for v in values[rng.choice(keys)]]) for _r in range(reps))
    return [est[int(round(0.025 * (reps - 1)))], est[int(round(0.975 * (reps - 1)))]]


def slope(points) -> float | None:
    """Least squares through the origin (the reference point x = 0, delta = 0)."""
    sxx = sum(x * x for x, _ in points)
    return None if sxx == 0 else sum(x * y for x, y in points) / sxx


def screening(score, cal, truth, normal) -> dict:
    t = cm.conformal_threshold(cal, ALPHA)
    good = [x for x, y in zip(score, truth) if y == normal]
    bad = [x for x, y in zip(score, truth) if y != normal]
    return {"frr": sum(x > t for x in good) / len(good), "escape": sum(x <= t for x in bad) / len(bad),
            "auroc": cm.auroc(good, bad), "reject": [x > t for x in score]}


def unit_metrics(units: dict, meta: dict, clusters: list, with_screening: bool) -> dict:
    """units {(cluster, cond, scenario, seed): {'pred': {method: [idx]}, 'score', 'cal'}};
    meta {cluster: {'order', 'true', 'normal'}}. Returns per-unit metrics and deltas against 'none'."""
    out = {}
    for (cl, cond, scen, seed), u in units.items():
        order, true = meta[cl]["order"], meta[cl]["true"]
        ti = np.asarray([order.index(t) for t in true])
        ref = units[(cl, "none", scen, seed)]
        m = {}
        for meth, pred in u["pred"].items():
            p = np.asarray(pred)
            m[f"BA|{meth}"] = cm.balanced_accuracy(ti, p, len(order))
            m[f"flip|{meth}"] = float((p != np.asarray(ref["pred"][meth])).mean())
        if with_screening:
            s = screening(u["score"], u["cal"], true, meta[cl]["normal"])
            r = screening(ref["score"], ref["cal"], true, meta[cl]["normal"])
            m.update({"FRR": s["frr"], "escape": s["escape"], "AUROC": s["auroc"],
                      "flip|pass_reject": float(np.mean(np.asarray(s["reject"]) != np.asarray(r["reject"])))})
        out[(cl, cond, scen, seed)] = m
    return out


def summarize(units: dict, meta: dict, clusters: list, seeds: list, conds: list, scen: str, metrics: list,
              cluster_ci: bool) -> dict:
    """Cluster-macro per seed, mean over seeds; paired delta vs 'none' with cluster bootstrap (or seed SD)."""
    res = {}
    for cond in conds:
        if (clusters[0], cond, scen, seeds[0]) not in units:
            continue
        row = {}
        for k in metrics:
            if k not in units[(clusters[0], cond, scen, seeds[0])]:
                continue
            per_seed = [statistics.fmean(units[(c, cond, scen, s)][k] for c in clusters) for s in seeds]
            entry = {"mean": statistics.fmean(per_seed)}
            if not k.startswith("flip") and cond != "none":
                d = {c: [units[(c, cond, scen, s)][k] - units[(c, "none", scen, s)][k] for s in seeds] for c in clusters}
                d_seed = [statistics.fmean(d[c][i] for c in clusters) for i in range(len(seeds))]
                entry["delta"] = statistics.fmean(d_seed)
                if cluster_ci:
                    entry["delta_ci95"] = cluster_boot(d)
                else:
                    entry["delta_sd_seeds"] = statistics.stdev(d_seed)
            row[k] = entry
        res[cond] = row
    return res


def agreement(units: dict, meta: dict, clusters: list, seeds: list, method: str, decision: bool) -> dict:
    """AIAG attribute agreement over the R jitter replicates (scenario D); cluster macro, mean over seeds."""
    per = {"within_repeatability": [], "fleiss_kappa": [], "agreement_vs_standard": [], "agreement_vs_unperturbed": []}
    for s in seeds:
        acc = {k: [] for k in per}
        for c in clusters:
            true, normal = meta[c]["true"], meta[c].get("normal")
            if decision:
                reps = [screening(units[(c, j, "D", s)]["score"], units[(c, j, "D", s)]["cal"], true, normal)["reject"]
                        for j in cm.J_CONDITIONS]
                ref = screening(units[(c, "none", "D", s)]["score"], units[(c, "none", "D", s)]["cal"], true, normal)["reject"]
                ratings = np.asarray(reps, dtype=int).T
                std = np.asarray([t != normal for t in true], dtype=int)
                r = cm.attribute_agreement(ratings, std, np.asarray(ref, dtype=int), 2)
            else:
                order = meta[c]["order"]
                ratings = np.asarray([units[(c, j, "D", s)]["pred"][method] for j in cm.J_CONDITIONS]).T
                std = np.asarray([order.index(t) for t in true])
                r = cm.attribute_agreement(ratings, std, np.asarray(units[(c, "none", "D", s)]["pred"][method]), len(order))
            for k in acc:
                acc[k].append(r[k])
        for k in per:
            vals = [v for v in acc[k] if not math.isnan(v)]
            per[k].append(statistics.fmean(vals) if vals else float("nan"))
    return {k: {"mean": statistics.fmean(v), "per_seed": v} for k, v in per.items()}


def derived(summary_d: dict, summary_rc: dict | None, metric_keys: list) -> dict:
    """Sensitivity coefficients per factor and drift tolerance (FRR <= 2 alpha without recalibration)."""
    out = {}
    for code in cm.LEVELS:
        conds = [c for c in cm.P_CONDITIONS if c.startswith(code + "=")]
        f = {"name": cm.NAMES[code], "unit": cm.UNITS[code], "levels": {}}
        for key in metric_keys:
            for scen, summ in (("D", summary_d), ("RC", summary_rc)):
                if summ is None or key not in summ.get(conds[0], {}) or "delta" not in summ[conds[0]][key]:
                    continue
                pts = [(cm.level_x(c), summ[c][key]["delta"]) for c in conds]
                f.setdefault("sensitivity", {})[f"{key}|{scen}"] = slope(pts)
        for c in conds:
            f["levels"][c] = {"x": cm.level_x(c),
                              **{f"{k}|D": summary_d[c][k]["mean"] for k in metric_keys if k in summary_d[c]},
                              **({f"{k}|RC": summary_rc[c][k]["mean"] for k in metric_keys if summary_rc and k in summary_rc.get(c, {})})}
        if "FRR" in summary_d.get(conds[0], {}):
            tol = {}
            for side in (-1, 1):
                lv = sorted((c for c in conds if math.copysign(1, cm.level_x(c)) == side), key=lambda c: abs(cm.level_x(c)))
                if not lv:
                    continue
                ok = None
                for c in lv:
                    if summary_d[c]["FRR"]["mean"] <= 2 * ALPHA:
                        ok = c
                    else:
                        break
                tol["decrease" if side < 0 else "increase"] = ok or "none tolerated (smallest level exceeds 2 alpha)"
            f["drift_tolerance_FRR_le_2alpha"] = tol
        out[code] = f
    return out


def analyze_mvtec(rows: list, run: dict) -> dict:
    meta = {p: {"order": v["class_order"], "true": v["true"]} for p, v in run["products"].items()}
    names = {v: k for k, v in run["methods"].items()}
    units = {}
    for r in rows:
        units.setdefault((r["product"], r["cond"], r["scenario"], r["seed"]), {"pred": {}})["pred"][names[r["method"]]] = r["pred"]
    m = unit_metrics(units, meta, list(meta), False)
    clusters, seeds = list(meta), run["seeds"]
    keys = ["BA|ARR", "BA|SimpleShot", "flip|ARR", "flip|SimpleShot"]
    d = summarize(m, meta, clusters, seeds, run["conditions"], "D", keys, True)
    rc_ = summarize(m, meta, clusters, seeds, run["rc_conditions"], "RC", keys, True)
    return {"D": d, "RC": rc_,
            "jitter_agreement": {meth: agreement(units, meta, clusters, seeds, meth, False) for meth in ("ARR", "SimpleShot")},
            "derived": derived(d, rc_, ["BA|ARR"]),
            "integrity": {p: {"reference_vs_p14_max_abs_dBA": v["reference_vs_p14_max_abs_dBA"],
                              **v["none_vs_cached"]} for p, v in run["products"].items()}}


def analyze_external(rows: list, run: dict, visa_tasks: list, mtd_task: str) -> dict:
    meta = {t: {"order": v["class_order"], "true": v["true"], "normal": v["normal"]} for t, v in run["tasks"].items()}
    units = {(r["task"], r["cond"], r["scenario"], r["seed"]): r for r in rows}
    m = unit_metrics(units, meta, list(meta), True)
    seeds = run["seeds"]
    keys = ["BA|ARR", "BA|SimpleShot", "flip|ARR", "flip|SimpleShot", "FRR", "escape", "AUROC", "flip|pass_reject"]
    out = {}
    for name, clusters, ci in (("visa_object_macro", visa_tasks, True), ("mtd", [mtd_task], False)):
        d = summarize(m, meta, clusters, seeds, run["conditions"], "D", keys, ci)
        rc_ = summarize(m, meta, clusters, seeds, run["rc_conditions"], "RC", keys, ci)
        out[name] = {"D": d, "RC": rc_,
                     "jitter_agreement": {"type_ARR": agreement(units, meta, clusters, seeds, "ARR", False),
                                          "type_SimpleShot": agreement(units, meta, clusters, seeds, "SimpleShot", False),
                                          "pass_reject": agreement(units, meta, clusters, seeds, None, True)},
                     "derived": derived(d, rc_, ["BA|ARR", "FRR", "AUROC"])}
    out["integrity"] = {t: v["none_vs_reference"] for t, v in run["tasks"].items()}
    return out


def report(res: dict) -> str:
    def pp(x):
        return f"{100 * x:+.2f}"

    def cell(e, ci_key="delta_ci95"):
        if "delta" not in e:
            return "—"
        if ci_key in e:
            return f"{pp(e['delta'])} [{pp(e[ci_key][0])}, {pp(e[ci_key][1])}]"
        return f"{pp(e['delta'])} ± {100 * e.get('delta_sd_seeds', 0):.2f}"
    lines = ["# P47 U4 - influence quantities (K = 5, 5 seeds; deltas in pp against the unperturbed U4 run)", ""]
    if "mvtec" in res:
        r = res["mvtec"]
        lines += ["## MVTec-FS formal (classification; product-cluster bootstrap CI)", "",
                  "| Condition | dBA ARR (D) | dBA ARR (RC) | dBA SimpleShot | Flip ARR (D) |", "|---|---|---|---|---|"]
        for c, v in r["D"].items():
            rc_ = r["RC"].get(c, {})
            lines.append(f"| {c} | {cell(v['BA|ARR'])} | {cell(rc_['BA|ARR']) if rc_ else '—'} | "
                         f"{cell(v['BA|SimpleShot'])} | {100 * v['flip|ARR']['mean']:.1f} % |")
        a = r["jitter_agreement"]["ARR"]
        lines += ["", f"Re-acquisition jitter (R = {cm.R_JITTER}), ARR type decision: within {100 * a['within_repeatability']['mean']:.1f} %, "
                      f"Fleiss kappa {a['fleiss_kappa']['mean']:.3f}, vs standard {100 * a['agreement_vs_standard']['mean']:.1f} %, "
                      f"vs unperturbed {100 * a['agreement_vs_unperturbed']['mean']:.1f} %.", ""]
    if "external" in res:
        for name, title, ci in (("visa_object_macro", "VisA (object macro; object-cluster bootstrap CI)", True),
                                ("mtd", "MTD (mean ± SD over seeds)", False)):
            r = res["external"][name]
            lines += [f"## {title}", "",
                      "| Condition | dBA ARR (D) | FRR (D) | FRR (RC) | AUROC (D) | Flip pass/reject (D) | Flip type ARR (D) |",
                      "|---|---|---|---|---|---|---|"]
            for c, v in r["D"].items():
                rc_ = r["RC"].get(c, {})
                frr_rc = f"{100 * rc_['FRR']['mean']:.1f} %" if rc_ else "—"
                lines.append(f"| {c} | {cell(v['BA|ARR'], 'delta_ci95' if ci else 'x')} | {100 * v['FRR']['mean']:.1f} % | "
                             f"{frr_rc} | {v['AUROC']['mean']:.3f} | "
                             f"{100 * v['flip|pass_reject']['mean']:.1f} % | {100 * v['flip|ARR']['mean']:.1f} % |")
            lines += [""]
            for kind, a in r["jitter_agreement"].items():
                lines.append(f"- J agreement, {kind}: within {100 * a['within_repeatability']['mean']:.1f} %, "
                             f"kappa {a['fleiss_kappa']['mean']:.3f}, vs standard {100 * a['agreement_vs_standard']['mean']:.1f} %, "
                             f"vs unperturbed {100 * a['agreement_vs_unperturbed']['mean']:.1f} %")
            lines += ["", "| Factor | Drift tolerance (FRR ≤ 2α, no recalibration) | Sensitivity dFRR (D) | dBA ARR (D) |",
                      "|---|---|---|---|"]
            for code, f in r["derived"].items():
                s = f.get("sensitivity", {})
                fr, ba = s.get("FRR|D"), s.get("BA|ARR|D")
                fr_s = "—" if fr is None else f"{100 * fr:+.3f} pp {f['unit']}"
                ba_s = "—" if ba is None else f"{100 * ba:+.3f} pp {f['unit']}"
                tol = f.get("drift_tolerance_FRR_le_2alpha", "—")
                lines.append(f"| {code} {f['name']} | {tol} | {fr_s} | {ba_s} |")
            lines.append("")
    return "\n".join(lines)


def run_analyze() -> int:
    from experiments.p10 import external_evaluate as ee
    from experiments.revision import revision_common as rc
    res = {}
    with cm.single_shot(OUT["analysis"]) as staging:
        if (OUT["mvtec"] / "u4_mvtec_run.json").is_file():
            rows = [json.loads(x) for x in (OUT["mvtec"] / "u4_mvtec_preds.jsonl").open(encoding="utf-8")]
            res["mvtec"] = analyze_mvtec(rows, rc.load_json(OUT["mvtec"] / "u4_mvtec_run.json"))
        if (OUT["external"] / "u4_external_run.json").is_file():
            rows = [json.loads(x) for x in (OUT["external"] / "u4_external_preds.jsonl").open(encoding="utf-8")]
            res["external"] = analyze_external(rows, rc.load_json(OUT["external"] / "u4_external_run.json"),
                                               list(ee.VISA_TASKS), ee.MTD_TASK)
        if not res:
            raise cm.P47Error("no U4 run outputs found")
        res.update({"schema_version": "p47_u4_analysis_v1", "provenance": rc.git_provenance(), "alpha": ALPHA})
        rc.write_json(staging / "u4_analysis.json", json.loads(json.dumps(res, default=str)))
        text = report(res)
        (staging / "u4_analysis.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- self-test

def selftest() -> int:
    from PIL import Image
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (90, 120, 3), dtype=np.uint8))
    base = np.asarray(img).astype(float)
    assert np.array_equal(np.asarray(cm.perturb(img, "none", "s")), np.asarray(img))
    for cond in (*cm.P_CONDITIONS, *cm.J_CONDITIONS):
        a, b = cm.perturb(img, cond, "abc"), cm.perturb(img, cond, "abc")
        assert a.size == img.size and np.array_equal(np.asarray(a), np.asarray(b)), cond
    assert not np.array_equal(np.asarray(cm.perturb(img, "P3=5", "abc")), np.asarray(cm.perturb(img, "P3=5", "abd")))
    g = np.asarray(cm.perturb(img, "P1=0.8", "x")).astype(float)
    assert np.abs(g - np.rint(base * 0.8)).max() <= 1
    n = np.asarray(cm.perturb(Image.fromarray(np.full((200, 200, 3), 128, np.uint8)), "P3=10", "x")).astype(float)
    assert abs(n.std() - 10) < 0.5
    flat = Image.fromarray(np.full((64, 64, 3), 77, np.uint8))
    for cond in ("P4=2", "P5=5", "P6=5"):           # reflect padding keeps a flat image flat
        assert np.abs(np.asarray(cm.perturb(flat, cond, "x")).astype(int) - 77).max() <= 1, cond
    assert abs(cm.level_x("P1=1.2") - 20) < 1e-9 and abs(cm.level_x("P7=70") - 30) < 1e-9
    # Fleiss kappa: perfect agreement 1; the Wikipedia worked example 0.210
    assert cm.fleiss_kappa(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), 3) == 1.0
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    ratings = np.array([[j for j, cnt in enumerate(row) for _ in range(cnt)] for row in table])
    assert abs(cm.fleiss_kappa(ratings, 5) - 0.210) < 1e-3
    assert abs(cm.auroc([0.1, 0.2, 0.3], [0.25, 0.3, 0.9]) - (2 + 2.5 + 3) / 9) < 1e-12
    assert len(capped_queries("t", [{"rgb_sha256": f"{i:03d}", "class_label": "n" if i < 300 else "d"}
                                    for i in range(310)], "n")) == NORMAL_CAP + 10
    # analysis on synthetic external units: a perturbation that shifts scores up raises FRR under D, not under RC
    seeds, conds = [1, 2, 3], ["none", *cm.P_CONDITIONS, *cm.J_CONDITIONS]
    meta_run = {"seeds": seeds, "conditions": conds, "rc_conditions": ["none", *cm.P_CONDITIONS],
                "methods": {"ARR": "arr", "SimpleShot": "ss"}, "tasks": {}}
    rows = []
    for t in ("a", "b", "m"):
        order = ["normal", "x", "y"]
        true = ["normal"] * 60 + ["x"] * 10 + ["y"] * 10
        ti = np.array([order.index(v) for v in true])
        meta_run["tasks"][t] = {"class_order": order, "true": true, "normal": "normal", "none_vs_reference": {}}
        for s in seeds:
            r0 = np.random.default_rng(s)
            base_s = np.where(ti == 0, r0.normal(0, 1, 80), r0.normal(2.5, 1, 80))
            cal = r0.normal(0, 1, 100)
            for cond in conds:
                shift = 0.0 if cond == "none" or cond.startswith("J") else 0.2 * abs(cm.level_x(cond))
                if cond.startswith("J"):
                    shift = 0.05 * r0.normal()
                for scen in ("D", "RC"):
                    if scen == "RC" and cond.startswith("J"):
                        continue
                    noise = np.random.default_rng(abs(hash((t, s, cond, scen))) % 2 ** 32)
                    pred = np.where(noise.random(80) < 0.8 - 0.01 * shift, ti, (ti + 1) % 3)
                    rows.append({"task": t, "cond": cond, "scenario": scen, "seed": s,
                                 "pred": {"ARR": pred.tolist(), "SimpleShot": pred.tolist()},
                                 "score": (base_s + shift).tolist(),
                                 "cal": (cal + (shift if scen == "RC" else 0)).tolist()})
    res = analyze_external(rows, meta_run, ["a", "b"], "m")
    v = res["visa_object_macro"]
    assert v["D"]["P3=10"]["FRR"]["mean"] > v["D"]["none"]["FRR"]["mean"] + 0.2
    assert abs(v["RC"]["P3=10"]["FRR"]["mean"] - v["RC"]["none"]["FRR"]["mean"]) < 0.06
    assert v["derived"]["P3"]["sensitivity"]["FRR|D"] > 0
    assert 0 <= v["jitter_agreement"]["pass_reject"]["within_repeatability"]["mean"] <= 1
    # MVTec-FS branch: classification only
    mrun = {"seeds": seeds, "conditions": conds, "rc_conditions": ["none", *cm.P_CONDITIONS],
            "methods": {"ARR": "arr", "SimpleShot": "ss"}, "products": {}}
    mrows = []
    for p in ("p1", "p2", "p3"):
        order = ["x", "y", "z"]
        true = [order[i % 3] for i in range(30)]
        ti = np.array([i % 3 for i in range(30)])
        mrun["products"][p] = {"class_order": order, "true": true, "reference_vs_p14_max_abs_dBA": 0.0,
                               "none_vs_cached": {"prediction_agreement": 1.0}}
        for s in seeds:
            for cond in conds:
                for scen, meths in (("D", ("arr", "ss")), ("RC", ("arr",))):
                    if scen == "RC" and cond.startswith("J"):
                        continue
                    for meth in meths:
                        drop = 0.0 if cond == "none" else 0.01 * abs(cm.level_x(cond)) if cond.startswith("P") else 0.02
                        noise = np.random.default_rng(abs(hash((p, s, cond, scen, meth))) % 2 ** 32)
                        pred = np.where(noise.random(30) < 0.85 - drop, ti, (ti + 1) % 3)
                        mrows.append({"product": p, "cond": cond, "scenario": scen, "seed": s, "method": meth,
                                      "pred": pred.tolist()})
    mres = analyze_mvtec(mrows, mrun)
    assert mres["D"]["P1=1.2"]["BA|ARR"]["delta"] < 0 and "delta_ci95" in mres["D"]["P1=1.2"]["BA|ARR"]
    assert mres["derived"]["P1"]["sensitivity"]["BA|ARR|D"] is not None
    assert 0 <= mres["jitter_agreement"]["ARR"]["fleiss_kappa"]["mean"] <= 1
    report({"external": res, "mvtec": mres})
    print("u4 selftest passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("mvtec", "external", "analyze"))
    ap.add_argument("--mvtec-root", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4, help="processes for decoding + perturbation (0 = in-process)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "analyze":
        return run_analyze()
    import torch
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if a.stage == "mvtec":
        if not a.mvtec_root:
            raise SystemExit("--mvtec-root is required")
        return run_mvtec(a.mvtec_root, device, a.batch_size, a.workers)
    if a.stage == "external":
        return run_external(device, a.batch_size, a.workers)
    ap.error("--stage or --selftest is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
