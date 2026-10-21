#!/usr/bin/env python3
"""P47 U4-A: geometric influence quantities without padding (amendment to U4).

Protocol: docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, "U4-A". U4 moved the part by translating or rotating the whole
image with reflect padding, so mirrored image content entered at the border. A diagnostic on MVTec AD good parts
(experiments/p47/diag_geometry_padding.py) showed that this synthetic border content, not the displacement of the
part, produced most of the U4 false-reject increase. U4-A repeats the geometric conditions with a field of view that
moves inside the original image, so only real image content is ever seen.

View. Every image (supports, bank, calibration normals, queries) is evaluated through a window of FR x the width and
FR x the height of the decoded image, centred for the reference, then the locked 336 px evaluation view.
FR = min(0.88, the largest window a 5 degree rotation keeps inside every image of the product or task, less 3 px).
Conditions (same per-image seeds and draw order as U4, random.Random(f"p47pert|{sha}|{cond}")):
- P5=1, 2.5, 5: the window moves by v % of its width (x) and v % of its height (y) along the U4 direction theta;
- P6=2, 5: the image is rotated about its centre by +-v degrees (U4 sign) and the centred window is taken;
- J1-J5: as U4 J (gain, noise, rotation), with the window moved by U[-0.5, 0.5] % of its width / height.
The window never reaches pixels that were not in the original image (checked by the self-test).

Scenarios: D for all conditions; RC (bank and calibration normals in the same condition) for none, P5=5 and P6=5.
Data, K, seeds, metrics and the normal-query cap are those of U4. The reference is the windowed view itself, so it
is not compared with the published (full-view) numbers; the analysis reports the windowed reference next to U4's.

    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u4a_geometry --stage mvtec --mvtec-root D:\\intergration\\MVTEC --device cuda --workers 6
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u4a_geometry --stage external --device cuda --workers 6
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u4a_geometry --stage analyze
    python -m experiments.p47.u4a_geometry --selftest     # CPU, synthetic data, no checkpoints needed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p47 import common as cm  # noqa: E402
from experiments.p47 import u4_influence as u4  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p47" / f"u4a_{s}" for s in ("mvtec", "external", "analysis")}
FR_MAX = 0.88
MAX_DEG = 5.0
GEO_LEVELS = {"P5": (1.0, 2.5, 5.0), "P6": (2.0, 5.0)}
GEO_CONDITIONS = tuple(f"{c}={v:g}" for c, levels in GEO_LEVELS.items() for v in levels)
QUERY_CONDS = ("none", *GEO_CONDITIONS, *cm.J_CONDITIONS)
RC_CONDS = ("none", "P5=5", "P6=5")


# --------------------------------------------------------------------------- the moving window

def fr_for(sizes) -> float:
    """Largest window fraction (<= FR_MAX) that a MAX_DEG rotation about the centre keeps inside every image."""
    th = math.radians(MAX_DEG)
    worst = min(1.0 / (math.cos(th) + max(w / h, h / w) * math.sin(th)) - 3.0 / min(w, h) for w, h in sizes)
    return round(min(FR_MAX, worst), 4)        # 3 px margin for the pixel grid


def window_box(w, h, fr, dx=0.0, dy=0.0):
    ww, hh = round(fr * w), round(fr * h)
    x0, y0 = round((w - ww) / 2 + dx), round((h - hh) / 2 + dy)
    if x0 < 0 or y0 < 0 or x0 + ww > w or y0 + hh > h:
        raise cm.P47Error(f"window {x0, y0, ww, hh} leaves the {w} x {h} image")
    return x0, y0, x0 + ww, y0 + hh


def window_params(cond: str, sha: str, w: int, h: int, fr: float) -> dict:
    """Displacement (px), rotation (deg) and photometric jitter of one condition; draws as in common.perturb."""
    p = {"dx": 0.0, "dy": 0.0, "deg": 0.0, "gain": 1.0, "sigma": 0.0, "noise_seed": None}
    if cond == "none":
        return p
    rng = random.Random(f"p47pert|{sha}|{cond}")
    ww, hh = fr * w, fr * h
    if cond.startswith("J"):
        p["gain"] = rng.uniform(0.97, 1.03)
        p["sigma"] = rng.uniform(0.0, 2.0)
        p["dx"] = rng.uniform(-0.5, 0.5) / 100 * ww
        p["dy"] = rng.uniform(-0.5, 0.5) / 100 * hh
        p["deg"] = rng.uniform(-0.5, 0.5)
        p["noise_seed"] = rng.getrandbits(63)
        return p
    code, v = cond.split("=")
    v = float(v)
    if code == "P5":
        theta = rng.uniform(0.0, 2 * math.pi)
        p["dx"], p["dy"] = v / 100 * ww * math.cos(theta), v / 100 * hh * math.sin(theta)
    elif code == "P6":
        p["deg"] = v * rng.choice((-1, 1))
    else:
        raise cm.P47Error(f"U4-A has no condition {cond}")
    return p


def window_view(img: Image.Image, cond: str, sha: str, fr: float) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    p = window_params(cond, sha, w, h, fr)
    box = window_box(w, h, fr, p["dx"], p["dy"])
    if p["deg"] == 0.0 and p["noise_seed"] is None:
        return img.crop(box)
    arr = np.asarray(img).astype(np.float64)
    if p["deg"] != 0.0:
        arr = cm._rotate(arr, p["deg"])          # about the image centre; the window never reaches padded pixels
    arr = arr[box[1]:box[3], box[0]:box[2]]
    if p["noise_seed"] is not None:
        arr = arr * p["gain"]
        arr = arr + np.random.default_rng(p["noise_seed"]).normal(0.0, p["sigma"], arr.shape)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def _load_window_view(job):
    """(path, sha, cond, fr) -> the locked evaluation tensor of the windowed view, as a numpy array."""
    from experiments.p6 import p6b_stage2_models as models
    path, sha, cond, fr = job
    with Image.open(path) as handle:
        img = handle.convert("RGB").copy()
    return models.deterministic_eval_tensor(window_view(img, cond, sha, fr)).numpy()


class WindowLoader(cm.PerturbedLoader):
    def __init__(self, workers: int):
        self.pool = ProcessPoolExecutor(workers, initializer=cm._init_worker) if workers > 0 else None

    def views(self, jobs):
        if self.pool is None:
            return map(_load_window_view, jobs)
        return self.pool.map(_load_window_view, jobs, chunksize=4)


def image_sizes(paths) -> list:
    out = []
    for p in paths:
        with Image.open(p) as handle:
            out.append(handle.size)
    return out


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
    loader = WindowLoader(workers)
    t0 = time.time()
    with cm.single_shot(OUT["mvtec"]) as staging:
        provenance = rc.git_provenance()
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=True)
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        rows, products = [], {}
        for product in rc.PRODUCTS:
            recs = [q for q in queries if q["product"] == product]
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
            dev_paths = [rc.DATASET_ROOT / p for p in dev["rel_paths"]]
            n_pool = len(dev_paths)
            q_true = [r["true_label"] for r in recs]
            order, eps = None, {}
            for seed in rc.SEEDS:
                man = rc.load_json(MANIFEST_DIR / product / f"k{u4.K_SHOT}_s{seed}.json")
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
            bank_paths = [good / x for x in nf["names"]]
            q_paths = [Path(r["image_path"]) for r in recs]
            fr = fr_for(image_sizes(dev_paths + bank_paths + q_paths))
            bank_jobs = [(p, cm.file_sha256(p)) for p in bank_paths]
            q_jobs = [(p, r["rgb_sha256"]) for p, r in zip(q_paths, recs)]
            with torch.no_grad():
                g_dev, clip_dev = u4.features(factory, loader, [(p, "", "none", fr) for p in dev_paths], device,
                                              batch_size)
                _, bank0 = u4.features(factory, loader, [(p, s, "none", fr) for p, s in bank_jobs], device, batch_size,
                                       with_global=False)

                def predict(g_q, clip_q, bank_clip, methods):
                    g = torch.cat([g_dev, g_q]).to(device)
                    clip = torch.cat([clip_dev, clip_q]).to(device).float()
                    out = {}
                    for seed in rc.SEEDS:
                        idx = torch.as_tensor([name_index[x] for x in nf["draws"][str(seed)][:20]])
                        bank_imgs = bank_clip[idx].to(device).float()
                        per_m = {"": ff.c1_descriptors(clip, bank_imgs, fa.tau99(bank_imgs))}
                        si, s_lab = eps[seed]
                        sc = ff.episode_scores(g, per_m, si, s_lab, qi, n, zeros, zeros)
                        for m in methods:
                            out[(m, seed)] = sc[m].argmax(dim=1).cpu().tolist()
                    return out

                info = {"class_order": order, "true": q_true, "query_sha256": [r["rgb_sha256"] for r in recs],
                        "window_fraction": fr}
                for cond in QUERY_CONDS:
                    g_q, clip_q = u4.features(factory, loader, [(p, s, cond, fr) for p, s in q_jobs], device,
                                              batch_size)
                    d = predict(g_q, clip_q, bank0, (ff.ARR, ff.SS))
                    for (m, s), pred in d.items():
                        rows.append({"product": product, "cond": cond, "scenario": "D", "seed": s, "method": m,
                                     "pred": pred})
                    if cond == "none":
                        info["windowed_reference_BA"] = {
                            m: statistics.fmean(classification_metrics(q_true, [order[i] for i in d[(m, s)]], order)
                                                ["balanced_accuracy"] for s in rc.SEEDS) for m in (ff.ARR, ff.SS)}
                    if cond in RC_CONDS:
                        bank_c = bank0 if cond == "none" else u4.features(
                            factory, loader, [(p, s, cond, fr) for p, s in bank_jobs], device, batch_size,
                            with_global=False)[1]
                        r = predict(g_q, clip_q, bank_c, (ff.ARR,))
                        for (m, s), pred in r.items():
                            rows.append({"product": product, "cond": cond, "scenario": "RC", "seed": s, "method": m,
                                         "pred": pred})
            products[product] = info
            print(f"[p47-u4a-mvtec] {product}: {len(recs)} queries, window {fr:.3f}; windowed reference BA "
                  f"{100 * info['windowed_reference_BA'][ff.ARR]:.2f} %; {(time.time() - t0) / 60:.1f} min", flush=True)
        rc.write_jsonl(staging / "u4a_mvtec_preds.jsonl", rows)
        rc.write_json(staging / "u4a_mvtec_run.json", {
            "schema_version": "p47_u4a_mvtec_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u4-a",
            "provenance": provenance, "K": u4.K_SHOT, "seeds": list(rc.SEEDS),
            "methods": {"ARR": ff.ARR, "SimpleShot": ff.SS}, "conditions": list(QUERY_CONDS),
            "rc_conditions": list(RC_CONDS), "products": products, "minutes": (time.time() - t0) / 60})
    loader.close()
    return 0


# --------------------------------------------------------------------------- stage 2: VisA / MTD

def run_external(device, batch_size: int, workers: int) -> int:
    import torch
    from types import SimpleNamespace
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p10 import external_evaluate as ee
    from experiments.p11 import formal_ab as fa
    from experiments.p14 import final_formal as ff
    from experiments.p15 import two_stage as ts
    from experiments.revision import revision_common as rc
    stat = ts.selected_statistic()
    loader = WindowLoader(workers)
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
            for seed in rc.SEEDS:                                   # as P15 / U4
                banks[seed] = ee.bank_draw(task, seed, normal_candidates, draws[normal][str(seed)]["K5"])
                support_all = {s for cls in order for s in draws[cls][str(seed)]["K5"]}
                if set(banks[seed]) & (support_all | query_shas) or support_all & query_shas:
                    raise cm.P47Error(f"{task}/s{seed}: bank/support/query overlap")
                needed |= support_all | set(banks[seed])
                used_normals |= set(banks[seed]) | set(draws[normal][str(seed)]["K5"])
            needed = sorted(needed)
            index = {s: i for i, s in enumerate(needed)}
            path = {s: ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in by_sha}
            cal = random.Random(f"p15cal|{task}").sample(sorted(set(normal_candidates) - used_normals - query_shas), ts.N_CAL)
            sub = u4.capped_queries(task, queries, normal)
            sub_shas = [r["rgb_sha256"] for r in sub]
            sub_true = [r["class_label"] for r in sub]
            bank_union = sorted({s for b in banks.values() for s in b})
            bank_pos = {s: i for i, s in enumerate(bank_union)}
            fr = fr_for(image_sizes([path[s] for s in needed + cal + sub_shas]))
            eps = {}
            for seed in rc.SEEDS:
                s_idx, s_lab = [], []
                for c, cls in enumerate(order):
                    for s in draws[cls][str(seed)][f"K{u4.K_SHOT}"]:
                        s_idx.append(index[s])
                        s_lab.append(c)
                eps[seed] = (s_idx, s_lab)
            with torch.no_grad():
                g, clip = u4.features(factory, loader, [(path[s], s, "none", fr) for s in needed], device, batch_size)
                g = g.to(device)
                _, cal0 = u4.features(factory, loader, [(path[s], s, "none", fr) for s in cal], device, batch_size,
                                      with_global=False)
                bank0 = clip[torch.as_tensor([index[s] for s in bank_union])]

                def evaluate(g_q, clip_q, bank_clip, cal_clip, methods):
                    """As U4: bank_clip rows follow bank_union. Returns {(method|'score'|'cal', seed): list}."""
                    out = {}
                    n = len(order)
                    for seed in rc.SEEDS:
                        s_idx, s_lab = eps[seed]
                        bank_imgs = bank_clip[torch.as_tensor([bank_pos[s] for s in banks[seed]])].to(device).float()
                        bank = bank_imgs.reshape(-1, clip.shape[-1])
                        tau = fa.tau99(bank_imgs)
                        rows_clip = torch.cat([clip[torch.as_tensor(s_idx)], clip_q])
                        parts = [ff.c1_descriptors(c.to(device).float(), bank_imgs, tau)
                                 for c in rows_clip.split(ee.Q_CHUNK)]
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
                        "queries_total": len(queries), "queries_used": len(sub), "bank_images": len(bank_union),
                        "calibration_normals": len(cal), "window_fraction": fr}
                q_jobs = [(path[s], s) for s in sub_shas]
                for cond in QUERY_CONDS:
                    g_q, clip_q = u4.features(factory, loader, [(p, s, cond, fr) for p, s in q_jobs], device,
                                              batch_size)
                    d = evaluate(g_q, clip_q, bank0, cal0, (ff.ARR, ff.SS))
                    for seed in rc.SEEDS:
                        rows.append({"task": task, "cond": cond, "scenario": "D", "seed": seed,
                                     "pred": {"ARR": d[(ff.ARR, seed)], "SimpleShot": d[(ff.SS, seed)]},
                                     "score": d[("score", seed)], "cal": d[("cal", seed)]})
                    if cond in RC_CONDS:
                        if cond == "none":
                            bank_c, cal_c = bank0, cal0
                        else:
                            _, bank_c = u4.features(factory, loader, [(path[s], s, cond, fr) for s in bank_union],
                                                    device, batch_size, with_global=False)
                            _, cal_c = u4.features(factory, loader, [(path[s], s, cond, fr) for s in cal], device,
                                                   batch_size, with_global=False)
                        r = evaluate(g_q, clip_q, bank_c, cal_c, (ff.ARR,))
                        for seed in rc.SEEDS:
                            rows.append({"task": task, "cond": cond, "scenario": "RC", "seed": seed,
                                         "pred": {"ARR": r[(ff.ARR, seed)]}, "score": r[("score", seed)],
                                         "cal": r[("cal", seed)]})
            tasks[task] = info
            print(f"[p47-u4a-ext] {task}: {len(sub)} of {len(queries)} queries, window {fr:.3f}; "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        rc.write_jsonl(staging / "u4a_external_preds.jsonl", rows)
        rc.write_json(staging / "u4a_external_run.json", {
            "schema_version": "p47_u4a_external_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u4-a",
            "provenance": provenance, "K": u4.K_SHOT, "alpha": u4.ALPHA, "statistic": stat, "seeds": list(rc.SEEDS),
            "methods": {"ARR": ff.ARR, "SimpleShot": ff.SS}, "conditions": list(QUERY_CONDS),
            "rc_conditions": list(RC_CONDS), "normal_cap": u4.NORMAL_CAP, "tasks": tasks,
            "minutes": (time.time() - t0) / 60})
    loader.close()
    return 0


# --------------------------------------------------------------------------- stage 3: analysis

def derived_geo(summary_d: dict, summary_rc: dict | None, keys: list) -> dict:
    """Sensitivity (least squares through the reference) and drift tolerance (FRR <= 2 alpha) for P5 and P6."""
    out = {}
    for code in GEO_LEVELS:
        conds = [c for c in GEO_CONDITIONS if c.startswith(code + "=")]
        f = {"name": cm.NAMES[code], "unit": cm.UNITS[code], "levels": {}}
        for key in keys:
            for scen, summ in (("D", summary_d), ("RC", summary_rc)):
                pts = [(cm.level_x(c), summ[c][key]["delta"]) for c in conds
                       if summ and c in summ and key in summ[c] and "delta" in summ[c][key]]
                if pts:
                    f.setdefault("sensitivity", {})[f"{key}|{scen}"] = u4.slope(pts)
        for c in conds:
            f["levels"][c] = {"x": cm.level_x(c), **{f"{k}|D": summary_d[c][k]["mean"] for k in keys if k in summary_d[c]}}
        if "FRR" in summary_d.get(conds[0], {}):
            ok = None
            for c in sorted(conds, key=lambda c: cm.level_x(c)):
                if summary_d[c]["FRR"]["mean"] <= 2 * u4.ALPHA:
                    ok = c
                else:
                    break
            f["drift_tolerance_FRR_le_2alpha"] = ok or "none tolerated (smallest level exceeds 2 alpha)"
        out[code] = f
    return out


def analyze_mvtec(rows: list, run: dict) -> dict:
    meta = {p: {"order": v["class_order"], "true": v["true"]} for p, v in run["products"].items()}
    names = {v: k for k, v in run["methods"].items()}
    units = {}
    for r in rows:
        units.setdefault((r["product"], r["cond"], r["scenario"], r["seed"]), {"pred": {}})["pred"][names[r["method"]]] = r["pred"]
    m = u4.unit_metrics(units, meta, list(meta), False)
    clusters, seeds = list(meta), run["seeds"]
    keys = ["BA|ARR", "BA|SimpleShot", "flip|ARR", "flip|SimpleShot"]
    d = u4.summarize(m, meta, clusters, seeds, run["conditions"], "D", keys, True)
    rc_ = u4.summarize(m, meta, clusters, seeds, run["rc_conditions"], "RC", keys, True)
    per_product = {p: {c: statistics.fmean(m[(p, c, "D", s)]["BA|ARR"] - m[(p, "none", "D", s)]["BA|ARR"]
                                           for s in seeds) for c in GEO_CONDITIONS} for p in clusters}
    return {"D": d, "RC": rc_, "per_product_dBA_ARR_D": per_product,
            "jitter_agreement": {meth: u4.agreement(units, meta, clusters, seeds, meth, False)
                                 for meth in ("ARR", "SimpleShot")},
            "derived": derived_geo(d, rc_, ["BA|ARR"]),
            "windowed_reference": {p: {"window_fraction": v["window_fraction"], **v["windowed_reference_BA"]}
                                   for p, v in run["products"].items()}}


def analyze_external(rows: list, run: dict, visa_tasks: list, mtd_task: str) -> dict:
    meta = {t: {"order": v["class_order"], "true": v["true"], "normal": v["normal"]} for t, v in run["tasks"].items()}
    units = {(r["task"], r["cond"], r["scenario"], r["seed"]): r for r in rows}
    m = u4.unit_metrics(units, meta, list(meta), True)
    seeds = run["seeds"]
    keys = ["BA|ARR", "BA|SimpleShot", "flip|ARR", "flip|SimpleShot", "FRR", "escape", "AUROC", "flip|pass_reject"]
    out = {}
    for name, clusters, ci in (("visa_object_macro", visa_tasks, True), ("mtd", [mtd_task], False)):
        d = u4.summarize(m, meta, clusters, seeds, run["conditions"], "D", keys, ci)
        rc_ = u4.summarize(m, meta, clusters, seeds, run["rc_conditions"], "RC", keys, ci)
        out[name] = {"D": d, "RC": rc_,
                     "per_task_FRR_D": {t: {c: statistics.fmean(m[(t, c, "D", s)]["FRR"] for s in seeds)
                                            for c in ("none", *GEO_CONDITIONS)} for t in clusters},
                     "jitter_agreement": {"type_ARR": u4.agreement(units, meta, clusters, seeds, "ARR", False),
                                          "type_SimpleShot": u4.agreement(units, meta, clusters, seeds, "SimpleShot", False),
                                          "pass_reject": u4.agreement(units, meta, clusters, seeds, None, True)},
                     "derived": derived_geo(d, rc_, ["BA|ARR", "FRR", "AUROC"])}
    out["window_fraction"] = {t: v["window_fraction"] for t, v in run["tasks"].items()}
    return out


def report(res: dict, u4res: dict | None) -> str:
    def pp(x):
        return f"{100 * x:+.2f}"

    def cell(e):
        if "delta" not in e:
            return "—"
        if "delta_ci95" in e:
            return f"{pp(e['delta'])} [{pp(e['delta_ci95'][0])}, {pp(e['delta_ci95'][1])}]"
        return f"{pp(e['delta'])} ± {100 * e.get('delta_sd_seeds', 0):.2f}"
    lines = ["# P47 U4-A - geometry without padding (K = 5, 5 seeds; deltas against the windowed reference)", "",
             "U4 columns: the same condition in U4 (whole image moved, reflect padding), for comparison.", ""]
    if "mvtec" in res:
        r = res["mvtec"]
        u = (u4res or {}).get("mvtec", {})
        lines += ["## MVTec-FS formal (classification; product-cluster bootstrap CI)", "",
                  "| Condition | dBA ARR (D) | dBA ARR (RC) | dBA SimpleShot | Flip ARR | U4: dBA ARR (D) |",
                  "|---|---|---|---|---|---|"]
        for c, v in r["D"].items():
            rc_ = r["RC"].get(c, {})
            old = u.get("D", {}).get(c, {}).get("BA|ARR", {})
            lines.append(f"| {c} | {cell(v['BA|ARR'])} | {cell(rc_['BA|ARR']) if rc_ else '—'} | "
                         f"{cell(v['BA|SimpleShot'])} | {100 * v['flip|ARR']['mean']:.1f} % | {cell(old) if old else '—'} |")
        a = r["jitter_agreement"]["ARR"]
        lines += ["", f"J (R = {cm.R_JITTER}), ARR type decision: within {100 * a['within_repeatability']['mean']:.1f} %, "
                      f"Fleiss kappa {a['fleiss_kappa']['mean']:.3f}, vs standard {100 * a['agreement_vs_standard']['mean']:.1f} %.", ""]
    if "external" in res:
        for name, title in (("visa_object_macro", "VisA (object macro; object-cluster bootstrap CI)"),
                            ("mtd", "MTD (mean ± SD over seeds)")):
            r = res["external"][name]
            u = (u4res or {}).get("external", {}).get(name, {})
            lines += [f"## {title}", "",
                      "| Condition | FRR (D) | FRR (RC) | AUROC (D) | dBA ARR (D) | Flip pass/reject | U4: FRR (D) | U4: dBA ARR (D) |",
                      "|---|---|---|---|---|---|---|---|"]
            for c, v in r["D"].items():
                rc_ = r["RC"].get(c, {})
                old = u.get("D", {}).get(c, {})
                frr_rc = f"{100 * rc_['FRR']['mean']:.1f} %" if rc_ else "—"
                frr_old = f"{100 * old['FRR']['mean']:.1f} %" if old else "—"
                ba_old = cell(old["BA|ARR"]) if old else "—"
                lines.append(f"| {c} | {100 * v['FRR']['mean']:.1f} % | {frr_rc} | {v['AUROC']['mean']:.3f} | "
                             f"{cell(v['BA|ARR'])} | {100 * v['flip|pass_reject']['mean']:.1f} % | {frr_old} | {ba_old} |")
            lines += [""]
            for kind, a in r["jitter_agreement"].items():
                lines.append(f"- J agreement, {kind}: within {100 * a['within_repeatability']['mean']:.1f} %, "
                             f"kappa {a['fleiss_kappa']['mean']:.3f}, vs standard {100 * a['agreement_vs_standard']['mean']:.1f} %")
            for code, f in r["derived"].items():
                lines.append(f"- {code} {f['name']}: drift tolerance (FRR <= 2 alpha) {f.get('drift_tolerance_FRR_le_2alpha', '—')}")
            lines.append("")
    return "\n".join(lines)


def run_analyze() -> int:
    from experiments.p10 import external_evaluate as ee
    from experiments.revision import revision_common as rc
    res = {}
    u4_path = u4.OUT["analysis"] / "u4_analysis.json"
    u4res = rc.load_json(u4_path) if u4_path.is_file() else None
    with cm.single_shot(OUT["analysis"]) as staging:
        if (OUT["mvtec"] / "u4a_mvtec_run.json").is_file():
            rows = [json.loads(x) for x in (OUT["mvtec"] / "u4a_mvtec_preds.jsonl").open(encoding="utf-8")]
            res["mvtec"] = analyze_mvtec(rows, rc.load_json(OUT["mvtec"] / "u4a_mvtec_run.json"))
        if (OUT["external"] / "u4a_external_run.json").is_file():
            rows = [json.loads(x) for x in (OUT["external"] / "u4a_external_preds.jsonl").open(encoding="utf-8")]
            res["external"] = analyze_external(rows, rc.load_json(OUT["external"] / "u4a_external_run.json"),
                                               list(ee.VISA_TASKS), ee.MTD_TASK)
        if not res:
            raise cm.P47Error("no U4-A run outputs found")
        res.update({"schema_version": "p47_u4a_analysis_v1", "provenance": rc.git_provenance(), "alpha": u4.ALPHA})
        rc.write_json(staging / "u4a_analysis.json", json.loads(json.dumps(res, default=str)))
        text = report(res, u4res)
        (staging / "u4a_analysis.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- self-test

def selftest() -> int:
    rng = np.random.default_rng(3)
    # 1. the centred window of the reference, and FR = 1 returns the image itself
    img = Image.fromarray(rng.integers(0, 256, (300, 450, 3), dtype=np.uint8))
    assert np.array_equal(np.asarray(window_view(img, "none", "s", 1.0)), np.asarray(img))
    v = window_view(img, "none", "s", 0.88)
    assert v.size == (round(0.88 * 450), round(0.88 * 300))
    # 2. deterministic, sized like the reference, and inside the image for every condition and aspect up to 3:1
    for size in ((450, 300), (300, 450), (1024, 1024), (600, 200), (200, 600), (1500, 1000)):
        im = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
        fr = fr_for([size])
        ref = window_view(im, "none", "x", fr)
        for cond in QUERY_CONDS:
            for sha in ("a", "b", "c"):
                o1, o2 = window_view(im, cond, sha, fr), window_view(im, cond, sha, fr)
                assert np.array_equal(np.asarray(o1), np.asarray(o2)) and o1.size == ref.size, (size, cond)
    # 3. no padded pixel reaches the window: rotate a sentinel image (constant fill 0) and check the window
    for size in ((450, 300), (300, 450), (1024, 1024), (600, 200), (1500, 1000)):
        w, h = size
        fr = fr_for([size])
        ones = np.ones((h, w))
        for deg in (MAX_DEG, -MAX_DEG):
            from scipy import ndimage
            rot = ndimage.rotate(ones, deg, axes=(1, 0), reshape=False, order=0, mode="constant", cval=0.0)
            x0, y0, x1, y1 = window_box(w, h, fr)
            assert rot[y0:y1, x0:x1].min() == 1.0, (size, deg, fr)
        for v_ in GEO_LEVELS["P5"]:                 # every shift direction stays inside
            for th in np.linspace(0, 2 * math.pi, 17):
                window_box(w, h, fr, v_ / 100 * fr * w * math.cos(th), v_ / 100 * fr * h * math.sin(th))
    # 4. a shift moves real content: shifting a horizontal ramp by +5 % of the window changes the mean by the ramp step
    ramp = Image.fromarray(np.tile(np.linspace(0, 255, 1000).astype(np.uint8), (800, 1)).repeat(3).reshape(800, 1000, 3))
    p = window_params("P5=5", "k", 1000, 800, 0.88)
    shifted = np.asarray(window_view(ramp, "P5=5", "k", 0.88)).astype(float).mean()
    centred = np.asarray(window_view(ramp, "none", "k", 0.88)).astype(float).mean()
    assert abs((shifted - centred) - p["dx"] * 255 / 999) < 1.0
    # 5. the same direction as U4 (first draw of the same generator)
    th = random.Random("p47pert|k|P5=5").uniform(0.0, 2 * math.pi)
    assert abs(p["dx"] - 0.05 * 0.88 * 1000 * math.cos(th)) < 1e-9
    # 6. analysis on synthetic units (the U4 functions are reused; P5 raises the scores, so FRR rises under D)
    seeds, conds = [1, 2, 3], list(QUERY_CONDS)
    run = {"seeds": seeds, "conditions": conds, "rc_conditions": list(RC_CONDS),
           "methods": {"ARR": "arr", "SimpleShot": "ss"}, "tasks": {}}
    rows = []
    for t in ("a", "b", "m"):
        order = ["normal", "x", "y"]
        true = ["normal"] * 60 + ["x"] * 10 + ["y"] * 10
        ti = np.array([order.index(v) for v in true])
        run["tasks"][t] = {"class_order": order, "true": true, "normal": "normal", "window_fraction": 0.88}
        for s in seeds:
            r0 = np.random.default_rng(s)
            base = np.where(ti == 0, r0.normal(0, 1, 80), r0.normal(2.5, 1, 80))
            cal = r0.normal(0, 1, 100)
            for cond in conds:
                shift = 0.3 * cm.level_x(cond) if cond.startswith("P") else 0.0
                for scen in ("D", "RC"):
                    if scen == "RC" and cond not in RC_CONDS:
                        continue
                    pred = np.where(np.random.default_rng(abs(hash((t, s, cond, scen))) % 2 ** 32).random(80) < 0.8, ti,
                                    (ti + 1) % 3)
                    rows.append({"task": t, "cond": cond, "scenario": scen, "seed": s,
                                 "pred": {"ARR": pred.tolist(), "SimpleShot": pred.tolist()},
                                 "score": (base + shift).tolist(), "cal": (cal + (shift if scen == "RC" else 0)).tolist()})
    res = analyze_external(rows, run, ["a", "b"], "m")
    v = res["visa_object_macro"]
    assert v["D"]["P5=5"]["FRR"]["mean"] > v["D"]["none"]["FRR"]["mean"] + 0.2
    assert abs(v["RC"]["P5=5"]["FRR"]["mean"] - v["RC"]["none"]["FRR"]["mean"]) < 0.06
    assert v["derived"]["P5"]["sensitivity"]["FRR|D"] > 0
    report({"external": res}, None)
    print("u4a selftest passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("mvtec", "external", "analyze"))
    ap.add_argument("--mvtec-root", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4, help="processes for decoding + the window view (0 = in-process)")
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
