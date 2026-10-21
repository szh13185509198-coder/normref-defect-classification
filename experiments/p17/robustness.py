#!/usr/bin/env python3
"""P17: report-only robustness analyses (formal + dev) and the patch-level conformal check.

Protocol: docs/P17_ROBUSTNESS_AND_CONFORMAL_CHECK_PROTOCOL.md. No result here changes the main
methods (ARR, CONS); every grid cell is reported, none is selected.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p17.robustness --stage dev --mvtec-root D:\\intergration\\MVTEC --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p17.robustness --stage mvtec --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p17.robustness --stage external --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
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
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p6 import p6b_stage2_models as models  # noqa: E402
from experiments.p6 import run_p6b_stage2_support as support  # noqa: E402
from experiments.p6.p6b_stage3_common import classification_metrics, load_query_contract  # noqa: E402
from experiments.p10 import external_evaluate as ee  # noqa: E402
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.extract_dev_features import CLIP_LAYER  # noqa: E402
from experiments.p10.formal_evaluate import MANIFEST_DIR  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p13 import pilot as p13  # noqa: E402
from experiments.p14 import final_formal as ff  # noqa: E402
from experiments.p12.threshold_controls import P11_DEV, exact_equal  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402
from experiments.revision.c1_paired_bootstrap import contrast  # noqa: E402

OUT = {s: PROJECT_ROOT / "ckpt" / "p17" / s for s in ("dev", "mvtec", "external")}
GUARDED = ("experiments/p10", "experiments/p11", "experiments/p12", "experiments/p13", "experiments/p14",
           "experiments/p15", "experiments/p17", "experiments/revision", "docs/P17_ROBUSTNESS_AND_CONFORMAL_CHECK_PROTOCOL.md")
SS, A, ARR, CONS = ff.SS, ff.A, ff.ARR, ff.CONS
Q_GRID = (0.95, 0.97, 0.98, 0.99, 0.995)
MIN_GRID = (1, 3, 6, 12)
T0_GRID = (0.80, 0.85, 0.90, 0.95)
W_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
D1_Q = (0.95, 0.97, 0.98, 0.99, 0.995, 0.999)
C1_KEYS = sorted({(q, 6) for q in Q_GRID} | {(0.99, m) for m in MIN_GRID})
MAIN = (0.99, 6)
FUSION = ("g_only", "a_only", "r_only", "g_a", "g_r", "a_r", "g_a_r", "arr_devweights")
GROUPS = {
    "threshold quantile (ARR)": [f"arr_q{q}" for q in Q_GRID],
    "fallback size (ARR)": [f"arr_min{m}" for m in MIN_GRID],
    "C0 threshold (CONS)": [f"cons_t{t}" for t in T0_GRID],
    "fusion ablation": list(FUSION),
    "patch backbone": ["a_dino", "arr_dino"],
}


# --------------------------------------------------------------------------- helpers

def region_weights(a: torch.Tensor, tau, min_region: int) -> torch.Tensor:
    """Soft weights max(a - tau, 0) with a top-`min_region` uniform fallback (== ab.region_weights at 6)."""
    w = (a - tau).clamp_min(0)
    small = (w > 0).sum(dim=1) < min_region
    if bool(small.any()):
        fallback = torch.zeros_like(a[small])
        fallback.scatter_(1, a[small].topk(min_region, dim=1).indices, 1.0)
        w[small] = fallback
    return w


def c1_views(patches: torch.Tensor, bank_imgs: torch.Tensor, taus: dict, keys) -> dict:
    """(q, min_region) -> (appearance descriptor, deviation descriptor); P11/P14 path at MAIN."""
    bank = bank_imgs.reshape(-1, bank_imgs.shape[-1])
    a = ra.patch_anomaly(patches, bank)
    out = {}
    for q, m in keys:
        w = ab.region_weights(a, taus[q], soft=True) if m == ab.MIN_REGION else region_weights(a, taus[q], m)
        out[(q, m)] = (ab.weighted_pool(patches, w), p13.region_residual(patches, bank, w))
    return out


def loo_taus(bank_imgs: torch.Tensor, qs) -> dict:
    loo = ab.loo_anomalies(bank_imgs)
    return {q: float(torch.quantile(loo, q)) for q in qs}


def cons_views(clip, si, s_lab, qi, n) -> dict:
    """t0 -> SimpleShot scores of the consensus descriptor (== p13.consensus_scores at 0.90)."""
    a_s, a_q = p13.consensus_anomaly(clip, si, s_lab, qi, n)
    flat = a_s.reshape(-1).double()
    out = {}
    for t0 in T0_GRID:
        t = float(torch.quantile(flat, t0))
        hs = ab.weighted_pool(clip[si], ab.region_weights(a_s, t, soft=True))
        hq = ab.weighted_pool(clip[qi], ab.region_weights(a_q, t, soft=True))
        out[t0] = rp.simpleshot(hs, s_lab, hq, n)
    return out


def episode_scores(g, views, dino_views, cons, si, s_lab, qi, n, dev_w=None) -> dict:
    base = rp.simpleshot(g[si], s_lab, g[qi], n)
    zb = rp._zscore(base)
    out = {SS: base}
    comp = {}
    for key, (ha, hr) in views.items():
        sa = rp.simpleshot(ha[si], s_lab, ha[qi], n)
        sr = rp.simpleshot(hr[si], s_lab, hr[qi], n)
        comp[key] = (sa, sr)
    for q, m in C1_KEYS:
        sa, sr = comp[(q, m)]
        s = zb + rp._zscore(sa) + rp._zscore(sr)
        if m == 6:
            out[f"arr_q{q}"] = s
        if q == 0.99:
            out[f"arr_min{m}"] = s
    sa, sr = comp[MAIN]
    za, zr = rp._zscore(sa), rp._zscore(sr)
    out.update({"g_only": base, "a_only": sa, "r_only": sr, "g_a": zb + za, "g_r": zb + zr, "a_r": za + zr,
                "g_a_r": zb + za + zr, A: zb + za, ARR: zb + za + zr})
    if dev_w is not None:
        out["arr_devweights"] = dev_w[0] * zb + dev_w[1] * za + dev_w[2] * zr
    if dino_views is not None:
        ha, hr = dino_views[MAIN]
        sad = rp.simpleshot(ha[si], s_lab, ha[qi], n)
        srd = rp.simpleshot(hr[si], s_lab, hr[qi], n)
        out["a_dino"] = zb + rp._zscore(sad)
        out["arr_dino"] = zb + rp._zscore(sad) + rp._zscore(srd)
    for t0, sc in cons.items():
        out[f"cons_t{t0}"] = zb + rp._zscore(sc)
    out[CONS] = out["cons_t0.9"]
    return out, (zb, za, zr)


def signflip_p(d: list[float]) -> float:
    obs = abs(statistics.fmean(d))
    hit = sum(abs(statistics.fmean(s * x for s, x in zip(signs, d))) >= obs - 1e-15
              for signs in itertools.product((1, -1), repeat=len(d)))
    return hit / 2 ** len(d)


def select_weights(zparts: list, truths: list, orders: list, cells: list) -> tuple:
    """Dev grid search of (w_S, w_A, w_R); objective = mean over K of product-macro BA."""
    best, best_key = None, None
    for w in itertools.product(W_GRID, repeat=3):
        if w == (0.0, 0.0, 0.0):
            continue
        per = {}
        for (zb, za, zr), truth, order, cell in zip(zparts, truths, orders, cells):
            sc = w[0] * zb + w[1] * za + w[2] * zr
            pred = [order[int(i)] for i in sc.argmax(dim=1).cpu()]
            per.setdefault((cell[1], cell[2]), []).append(classification_metrics(truth, pred, order)["balanced_accuracy"])
        by_k = {}
        for (k, s), v in per.items():
            by_k.setdefault(k, []).append(statistics.fmean(v))
        value = statistics.fmean(statistics.fmean(v) for v in by_k.values())
        key = (round(value, 12), -sum(abs(x - 1.0) for x in w))
        if best_key is None or key > best_key:
            best, best_key = w, key
    return best, best_key[0]


@contextlib.contextmanager
def single_shot(out: Path):
    status = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise rc.P6BFailClosedError(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise rc.P6BFailClosedError(f"{out} exists; this P17 stage is single-shot")
    lock = out.with_name(out.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        raise rc.P6BFailClosedError(f"{lock} exists: another run is active or was interrupted")
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    yield staging
    staging.rename(out)


def dev_weights() -> tuple:
    path = OUT["dev"] / "p17_dev_results.json"
    if not path.is_file():
        raise rc.P6BFailClosedError("run the P17 dev stage first (dev-tuned weights are read from it)")
    return tuple(rc.load_json(path)["dev_tuned_weights"])


# --------------------------------------------------------------------------- dev

def run_dev(mvtec_root: Path, device) -> int:
    with single_shot(OUT["dev"]) as staging:
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        rows, zparts, truths, orders, cells = [], [], [], [], []
        d1 = {q: {"normal_exceed": [], "normal_any": [], "normal_count": [], "defect_mask_recall": [],
                  "defect_bg_exceed": []} for q in D1_Q}
        for product in rc.PRODUCTS:
            data = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            normal = torch.load(ra.NORMAL_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            good_dir = mvtec_root / product / "train" / "good"
            held = sorted(p.name for p in good_dir.glob("*.png") if p.name not in set(normal["names"]))
            pick = random.Random(f"p15dev|{product}").sample(held, min(200, len(held)))
            with torch.no_grad():
                _, held_clip = ee.extract(factory, [good_dir / n for n in pick], device, 4)
            g = data["global_fusion"].to(device)
            clip = data["clip18_patches"].to(device).float()
            dino = data["dino_patches"].to(device).float()
            masks = data["mask_grid"].to(device)
            name_index = {x: i for i, x in enumerate(normal["names"])}
            with torch.no_grad():
                for seed in rp.DEV_SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in normal["draws"][str(seed)][:ra.M_PRIMARY]])
                    bank_imgs = normal["clip18_patches"][idx].to(device).float()
                    dino_bank = normal["dino_patches"][idx].to(device).float()
                    taus = loo_taus(bank_imgs, sorted(set(Q_GRID) | set(D1_Q)))
                    bank = bank_imgs.reshape(-1, clip.shape[-1])
                    a_held = torch.cat([ra.patch_anomaly(c.to(device).float(), bank) for c in held_clip.split(16)])
                    a_def = ra.patch_anomaly(clip, bank)
                    for q in D1_Q:
                        sel = a_held > taus[q]
                        d1[q]["normal_exceed"].append(float(sel.float().mean()))
                        d1[q]["normal_any"].append(float(sel.any(dim=1).float().mean()))
                        d1[q]["normal_count"].append(float(sel.sum(dim=1).float().mean()))
                        m = masks > 0
                        sd = a_def > taus[q]
                        d1[q]["defect_mask_recall"].append(float((sd & m).sum() / m.sum().clamp_min(1)))
                        d1[q]["defect_bg_exceed"].append(float((sd & ~m).sum() / (~m).sum().clamp_min(1)))
                    views = c1_views(clip, bank_imgs, taus, C1_KEYS)
                    dviews = c1_views(dino, dino_bank, loo_taus(dino_bank, (0.99,)), [MAIN])
                    for k_shot, order, s_idx, s_lab, q_idx, q_lab in rp.episodes(data["labels"], seed, product):
                        n = len(order)
                        si = torch.as_tensor(s_idx, device=device)
                        qi = torch.as_tensor(q_idx, device=device)
                        cons = cons_views(clip, si, s_lab, qi, n)
                        sc, zp = episode_scores(g, views, dviews, cons, si, s_lab, qi, n)
                        truth = [order[c] for c in q_lab]
                        rows += fa.score_rows(sc, order, truth, {"product": product, "K": k_shot, "seed": seed})
                        zparts.append(zp)
                        truths.append(truth)
                        orders.append(order)
                        cells.append((product, k_shot, seed))
            print(f"[p17-dev] {product} done", flush=True)

        integrity = {"A_ARR_CONS_SS_vs_p13_dev": exact_equal(rows, rc.load_json(p13.OUT["dev"] / "p13_dev_results.json")["rows"],
                                                             (SS, A, ARR, CONS), ("product", "K", "seed"))}
        if integrity["A_ARR_CONS_SS_vs_p13_dev"] >= 1e-12:
            raise rc.P6BFailClosedError(f"dev integrity failed: {integrity}")
        weights, value = select_weights(zparts, truths, orders, cells)
        summary = rp.summarize(rows)
        for v in summary["methods"].values():
            v.pop("gate_pass", None)
        cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rows}

        def vs(m, ref):
            per = {p: 100 * statistics.fmean(cell[(m, p, k, s)] - cell[(ref, p, k, s)] for k in rp.DEV_K for s in rp.DEV_SEEDS)
                   for p in rc.PRODUCTS}
            return {"mean_delta_pp": statistics.fmean(per.values()), "positive_products": sum(v > 0 for v in per.values())}
        grids = {grp: {m: vs(m, CONS if m.startswith("cons") else ARR) for m in ms if m != "arr_devweights"}
                 for grp, ms in GROUPS.items()}
        d1_summary = {str(q): {k: statistics.fmean(v) for k, v in d.items()} for q, d in d1.items()}
        result = {"schema_version": "p17_dev_v1", "protocol": "docs/P17_ROBUSTNESS_AND_CONFORMAL_CHECK_PROTOCOL.md#d",
                  "provenance": rc.git_provenance(), "integrity": integrity, "dev_tuned_weights": list(weights),
                  "dev_tuned_objective": value, "conformal_check": d1_summary, "grids_vs_reference": grids,
                  "vs_simpleshot": summary, "rows": rows}
        lines = ["# P17 dev — conformal calibration check and robustness grids (dev only)", "",
                 f"Integrity vs P13 dev: max|Δ| = {integrity['A_ARR_CONS_SS_vs_p13_dev']:.1e}. "
                 f"Dev-tuned weights (w_S, w_A, w_R) = {weights} (objective {100 * value:.2f}).", "",
                 "## D1 Patch-level calibration (held-out MVTec AD normals; defect patches of the dev pool, masks for evaluation only)", "",
                 "| q | target normal-patch rate | realized normal-patch rate | normal images with ≥1 selected patch | selected patches per normal image | defect-mask recall | defect-image background rate |",
                 "|---|---|---|---|---|---|---|"]
        for q, v in d1_summary.items():
            lines.append(f"| {q} | {100 * (1 - float(q)):.2f}% | {100 * v['normal_exceed']:.2f}% | {100 * v['normal_any']:.1f}% | "
                         f"{v['normal_count']:.1f} | {100 * v['defect_mask_recall']:.1f}% | {100 * v['defect_bg_exceed']:.2f}% |")
        lines += ["", "## D2 Dev grids (Δ BA vs ARR, or vs CONS for C0 rows; mean over K = 1/3; +products/13)", ""]
        for grp, g2 in grids.items():
            lines.append(f"- **{grp}:** " + ", ".join(f"{m} {v['mean_delta_pp']:+.2f} ({v['positive_products']})" for m, v in g2.items()))
        text = "\n".join(lines)
        rc.write_json(staging / "p17_dev_results.json", result)
        (staging / "p17_dev_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- MVTec-FS formal

def run_mvtec(device) -> int:
    dev_w = dev_weights()
    with single_shot(OUT["mvtec"]) as staging:
        provenance = rc.git_provenance()
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
        truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
        rows = []
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
            n_pool = len(dev["rel_paths"])
            g = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
            clip = torch.cat([dev["clip18_patches"], qf["clip18_patches"]]).to(device).float()
            dino = torch.cat([dev["dino_patches"], qf["dino_patches"]]).to(device).float()
            name_index = {x: i for i, x in enumerate(nf["names"])}
            q_true = [truth[(product, sha)] for sha in qf["rgb_sha256"]]
            qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
            with torch.no_grad():
                for seed in rc.SEEDS:
                    idx = torch.as_tensor([name_index[x] for x in nf["draws"][str(seed)][:ra.M_PRIMARY]])
                    bank_imgs = nf["clip18_patches"][idx].to(device).float()
                    dino_bank = nf["dino_patches"][idx].to(device).float()
                    views = c1_views(clip, bank_imgs, loo_taus(bank_imgs, Q_GRID), C1_KEYS)
                    dviews = c1_views(dino, dino_bank, loo_taus(dino_bank, (0.99,)), [MAIN])
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
                        cons = cons_views(clip, si, s_lab, qi, n)
                        sc, _ = episode_scores(g, views, dviews, cons, si, s_lab, qi, n, dev_w)
                        rows += fa.score_rows(sc, order, q_true, {"product": product, "K": k_shot, "seed": seed})
            print(f"[p17-mvtec] {product} done", flush=True)
        integrity = exact_equal(rows, rc.load_json(ff.OUT["mvtec"] / "p14_mvtec_results.json")["raw_metrics"],
                                (SS, A, ARR, CONS), ("product", "K", "seed"))
        if integrity >= 1e-9:
            raise rc.P6BFailClosedError(f"integrity vs P14 formal failed: {integrity}")
        grid = rc.metric_grid_from_rows(rows)
        report = {}
        for grp, ms in GROUPS.items():
            report[grp] = {}
            for m in ms:
                ref = CONS if m.startswith("cons") else ARR
                c = contrast(grid, ref, "balanced_accuracy", target=m)
                lv = rc.product_macro_summary(grid, m)
                report[grp][m] = {"reference": ref, "levels": [100 * lv[str(k)]["balanced_accuracy"]["mean"] for k in rc.K_VALUES],
                                  "delta_pp": 100 * c["Dbar"], "ci95_pp": [100 * x for x in c["Dbar_ci95"]],
                                  "positive": c["Dbar_positive_products"],
                                  "signflip_p": signflip_p(list(c["Dbar_per_product_mean_difference"].values()))}
        write_report(staging, "mvtec", provenance, integrity, dev_w, report, rows, 13)
    return 0


def write_report(staging, stage, provenance, integrity, dev_w, report, rows, n_clusters, mtd=None):
    result = {"schema_version": f"p17_{stage}_v1", "protocol": "docs/P17_ROBUSTNESS_AND_CONFORMAL_CHECK_PROTOCOL.md#f",
              "label": "report-only robustness analyses; no selection", "provenance": provenance,
              "integrity_max_abs_diff_vs_p14": integrity, "dev_tuned_weights": list(dev_w), "report": report,
              "mtd_descriptive": mtd, "raw_metrics": rows}
    unit = "products" if stage == "mvtec" else "VisA objects"
    lines = [f"# P17 {stage} — robustness analyses (report-only, single run)", "",
             f"Commit `{provenance['head']}`. Integrity vs P14 (SimpleShot, A, ARR, CONS): max|Δ| = {integrity:.1e}. "
             f"Dev-tuned weights = {tuple(dev_w)}.", "",
             f"| Group | Row | BA K=1 / 3 / 5 | Δ vs reference (pooled, 95% CI) | +{unit}/{n_clusters} | sign-flip p |",
             "|---|---|---|---|---|---|"]
    for grp, rs in report.items():
        for m, v in rs.items():
            mt = ""
            lines.append(f"| {grp} | {m} (vs {v['reference']}) | " + " / ".join(f"{x:.2f}" for x in v["levels"]) +
                         f" | {v['delta_pp']:+.2f} [{v['ci95_pp'][0]:+.2f}, {v['ci95_pp'][1]:+.2f}] | {v['positive']} | {v['signflip_p']:.4f} |{mt}")
    if mtd:
        lines += ["", "MTD (descriptive, Δ BA vs reference, seed mean over K=1/3/5):"]
        lines += [f"- {m}: " + " / ".join(f"{x:+.2f}" for x in v) for m, v in mtd.items()]
    text = "\n".join(lines)
    rc.write_json(staging / f"p17_{stage}_results.json", result)
    (staging / f"p17_{stage}_table.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)


# --------------------------------------------------------------------------- VisA / MTD

def extract_with_dino(factory, paths, device, batch_size):
    """ee.extract (identical global and CLIP outputs) plus L2-normalized DINOv2 patches."""
    glob, clip, dino_p = [], [], []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB").copy())
        batch = torch.cat([models.deterministic_eval_tensor(im) for im in images]).to(device)
        with torch.no_grad():
            pooled, tokens = factory.clip_model.visual(batch, [CLIP_LAYER])
            dino = factory.dino_extractor.forward_patches(batch)
            glob.append(factory.fusion.fuse(models.l2_rows(pooled), models.l2_rows(dino.mean(dim=1))).float().cpu())
            clip.append(F.normalize(tokens[0][:, 1:, :].float(), dim=-1).half().cpu())
            dino_p.append(F.normalize(dino.float(), dim=-1).half().cpu())
    return torch.cat(glob), torch.cat(clip), torch.cat(dino_p)


def external_task(task, order, factory, device, batch_size, dev_w):
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
    g, clip, dino = extract_with_dino(factory, [ee.ROOTS[dataset] / by_sha[s]["image_rel"] for s in shas], device, batch_size)
    qi = torch.as_tensor([index[r["rgb_sha256"]] for r in queries], device=device)
    q_true = [r["class_label"] for r in queries]
    q_patches = clip[qi.cpu()]
    g = g.to(device)
    rows = []
    for seed in rc.SEEDS:
        bidx = torch.as_tensor([index[s] for s in banks[seed]])
        bank_imgs = clip[bidx].to(device).float()
        dino_bank = dino[bidx].to(device).float()
        taus = loo_taus(bank_imgs, Q_GRID)
        dtaus = loo_taus(dino_bank, (0.99,))
        parts = [c1_views(ch.to(device).float(), bank_imgs, taus, C1_KEYS) for ch in clip.split(ee.Q_CHUNK)]
        views = {k: tuple(torch.cat([p[k][i] for p in parts]) for i in range(2)) for k in C1_KEYS}
        dparts = [c1_views(ch.to(device).float(), dino_bank, dtaus, [MAIN]) for ch in dino.split(ee.Q_CHUNK)]
        dviews = {MAIN: tuple(torch.cat([p[MAIN][i] for p in dparts]) for i in range(2))}
        for k_shot in rc.K_VALUES:
            s_idx, s_lab = [], []
            for c, cls in enumerate(order):
                for s in draws[cls][str(seed)][f"K{k_shot}"]:
                    s_idx.append(index[s])
                    s_lab.append(c)
            n = len(order)
            si = torch.as_tensor(s_idx, device=device)
            sup = clip[si.cpu()].to(device).float()
            s_range = torch.arange(len(s_idx), device=device)
            chunks = {t0: [] for t0 in T0_GRID}
            for ch in q_patches.split(ee.Q_CHUNK):  # the P14 external consensus path
                both = torch.cat([sup, ch.to(device).float()])
                cv = cons_views(both, s_range, s_lab, torch.arange(len(s_idx), both.shape[0], device=device), n)
                for t0 in T0_GRID:
                    chunks[t0].append(cv[t0])
            cons = {t0: torch.cat(v) for t0, v in chunks.items()}
            sc, _ = episode_scores(g, views, dviews, cons, si, s_lab, qi, n, dev_w)
            rows += fa.score_rows(sc, order, q_true, {"task": task, "K": k_shot, "seed": seed})
    return rows


def run_external(device, batch_size) -> int:
    dev_w = dev_weights()
    with single_shot(OUT["external"]) as staging:
        provenance = rc.git_provenance()
        orders, _ = ee.frozen_artifacts()
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        rows = []
        for task in ee.TASKS:
            with torch.no_grad():
                rows += external_task(task, orders[task], factory, device, batch_size, dev_w)
            print(f"[p17-ext] {task} done", flush=True)
        integrity = exact_equal(rows, rc.load_json(ff.OUT["external"] / "p14_external_results.json")["raw_metrics"],
                                (SS, A, ARR, CONS), ("task", "K", "seed"))
        if integrity >= 1e-9:
            raise rc.P6BFailClosedError(f"integrity vs P14 external failed: {integrity}")
        grid = ee.grid_of(rows)
        report, mtd = {}, {}
        for grp, ms in GROUPS.items():
            report[grp] = {}
            for m in ms:
                ref = CONS if m.startswith("cons") else ARR
                c = ee.visa_contrast(grid, m, ref, "balanced_accuracy")
                objs = c["per_K"]["1"]["per_object"].keys()
                per_obj = [statistics.fmean(c["per_K"][str(k)]["per_object"][o] for k in rc.K_VALUES) for o in objs]
                lv = ee.visa_macro(grid, m)
                report[grp][m] = {"reference": ref, "levels": [100 * lv[str(k)]["mean"] for k in rc.K_VALUES],
                                  "delta_pp": 100 * c["Dbar"], "ci95_pp": [100 * x for x in c["Dbar_ci95"]],
                                  "positive": c["Dbar_positive_objects"], "signflip_p": signflip_p(per_obj)}
                mc = ee.mtd_contrast(grid, m, ref, "balanced_accuracy")
                mtd[m] = [100 * mc[str(k)]["mean"] for k in rc.K_VALUES]
        write_report(staging, "external", provenance, integrity, dev_w, report, rows, 11, mtd)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("dev", "mvtec", "external"))
    parser.add_argument("--mvtec-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if args.stage == "dev":
        if not args.mvtec_root:
            raise SystemExit("--mvtec-root is required for the dev stage")
        return run_dev(Path(args.mvtec_root), device)
    return run_mvtec(device) if args.stage == "mvtec" else run_external(device, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
