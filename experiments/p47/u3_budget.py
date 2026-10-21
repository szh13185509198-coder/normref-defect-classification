#!/usr/bin/env python3
"""P47 U3: uncertainty budget of the reported balanced accuracy (MVTec-FS formal).

Protocol: docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md, "U3". Three stages, each write-once:

1. extract  - CLIP layer-18 patch features of the MVTec AD train/good images needed by the bank-size sweep
              (union over M in {5, 10, 20, 50} and 10 draws random.Random(f"p47bank|{product}|{M}|{j}")), the same
              extraction call as P10/P14 (ee.extract, batch 4). -> ckpt/p47/u3_features/
2. run      - ARR and SimpleShot predictions for
              (a) the full cross S x B (5 formal support manifests x the 5 formal bank draws, M = 20, K in {1, 3, 5});
                  the diagonal cells (S = B seed) are the published P14 cells and must reproduce P14 exactly;
              (b) the sweep, M in {5, 10, 20, 50}: 5 support manifests x 10 bank draws from train/good.
              -> ckpt/p47/u3/ (u3_cells.jsonl, u3_run.json)
3. analyze  - two-way random-effects ANOVA per product x K, stratified query bootstrap (1000), law of propagation for
              the macro mean, Welch-Satterthwaite, Type B implementation component. -> ckpt/p47/u3_analysis/
              Needs no GPU and no features; it can be re-derived from u3_cells.jsonl anywhere.

Usage (Windows, the author's 4090 machine):
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u3_budget --stage extract --mvtec-root D:\\intergration\\MVTEC --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u3_budget --stage run --device cuda
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p47.u3_budget --stage analyze --impl-halfwidth-pp <a> --impl-source "<files>"
    python -m experiments.p47.u3_budget --selftest          # CPU, synthetic data, no checkpoints needed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from experiments.p47 import common as cm  # noqa: E402

OUT_FEAT = PROJECT_ROOT / "ckpt" / "p47" / "u3_features"
OUT_RUN = PROJECT_ROOT / "ckpt" / "p47" / "u3"
OUT_ANA = PROJECT_ROOT / "ckpt" / "p47" / "u3_analysis"
M_SWEEP = (5, 10, 20, 50)
J_SWEEP = 10
M_CROSS = 20
Q_REPS = 1000
K_BUDGET = 5


def sweep_draws(names: list[str], product: str) -> dict:
    """{(M, j): names} from the sorted train/good names (protocol U3)."""
    if len(names) < max(M_SWEEP):
        raise cm.P47Error(f"{product}: only {len(names)} train/good images")
    return {(m, j): random.Random(f"p47bank|{product}|{m}|{j}").sample(names, m)
            for m in M_SWEEP for j in range(J_SWEEP)}


# --------------------------------------------------------------------------- stage 1

def run_extract(mvtec_root: Path, device, batch_size: int) -> int:
    import torch
    from types import SimpleNamespace
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p10 import external_evaluate as ee
    from experiments.revision import revision_common as rc
    with cm.single_shot(OUT_FEAT) as staging:
        factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
        info = {}
        for product in rc.PRODUCTS:
            good = mvtec_root / product / "train" / "good"
            names = sorted(p.name for p in good.glob("*.png"))
            draws = sweep_draws(names, product)
            needed = sorted({n for d in draws.values() for n in d})
            with torch.no_grad():
                _, clip = ee.extract(factory, [good / n for n in needed], device, batch_size)
            torch.save({"schema_version": "p47_u3_traingood_features_v1", "product": product,
                        "source": f"{product}/train/good", "names": needed,
                        "draws": {f"{m}|{j}": d for (m, j), d in draws.items()},
                        "file_sha256": {n: cm.file_sha256(good / n) for n in needed},
                        "clip18_patches": clip, "provenance": rc.git_provenance()}, staging / f"{product}.pt")
            info[product] = {"train_good": len(names), "extracted": len(needed)}
            print(f"[p47-u3-extract] {product}: {len(needed)} of {len(names)} train/good images", flush=True)
        rc.write_json(staging / "u3_features_info.json", {"provenance": rc.git_provenance(), "products": info})
    return 0


# --------------------------------------------------------------------------- stage 2

def run_scoring(device) -> int:
    import torch
    from experiments.p6.p6b_stage3_common import classification_metrics, load_query_contract
    from experiments.p10 import run_pilot as rp
    from experiments.p10.formal_evaluate import MANIFEST_DIR
    from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES
    from experiments.p11 import formal_ab as fa
    from experiments.p14 import final_formal as ff
    from experiments.revision import revision_common as rc
    if not (OUT_FEAT / "u3_features_info.json").is_file():
        raise cm.P47Error("run --stage extract first")
    published = {(r["method"], r["product"], int(r["K"]), int(r["seed"])): r["balanced_accuracy"]
                 for r in rc.load_json(ff.OUT["mvtec"] / "p14_mvtec_results.json")["raw_metrics"]
                 if r["method"] in (ff.ARR, ff.SS)}
    with cm.single_shot(OUT_RUN) as staging:
        provenance = rc.git_provenance()
        queries, _ = load_query_contract(rc.RECORDS_PATH, rc.DATASET_ROOT, rc.STAGE1_BUNDLE, verify_rgb=False)
        truth = {(q["product"], q["rgb_sha256"]): q["true_label"] for q in queries}
        cells, products, worst = [], {}, 0.0
        for product in rc.PRODUCTS:
            dev = torch.load(rp.FEATURE_DIR / f"{product}.pt", map_location="cpu", weights_only=False)
            qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
            nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
            sw = torch.load(OUT_FEAT / f"{product}.pt", map_location="cpu", weights_only=False)
            path_index = {p: i for i, p in enumerate(dev["rel_paths"])}
            n_pool = len(dev["rel_paths"])
            g = torch.cat([dev["global_fusion"], qf["global_fusion"]]).to(device)
            clip = torch.cat([dev["clip18_patches"], qf["clip18_patches"]]).to(device).float()
            q_true = [truth[(product, sha)] for sha in qf["rgb_sha256"]]
            qi = torch.arange(n_pool, n_pool + len(q_true), device=device)
            episodes, order0 = {}, None
            for seed in rc.SEEDS:
                for k in rc.K_VALUES:
                    man = rc.load_json(MANIFEST_DIR / product / f"k{k}_s{seed}.json")
                    order = list(man["class_order"])
                    if order0 is None:
                        order0 = order
                    if order != order0:
                        raise cm.P47Error(f"{product}: class order differs between manifests")
                    s_idx, s_lab = [], []
                    for c, cls in enumerate(order):
                        for item in man["classes"][cls]["items"]:
                            s_idx.append(path_index[item["rel_path"]])
                            s_lab.append(c)
                    episodes[(seed, k)] = (torch.as_tensor(s_idx, device=device), s_lab)
            n = len(order0)
            zeros = torch.zeros(len(q_true), n, device=device)   # cross-class / consensus are not needed here
            products[product] = {"class_order": order0, "true": q_true, "query_sha256": list(qf["rgb_sha256"])}

            def score_bank(bank_imgs, design, m, b):
                per_m = {"": ff.c1_descriptors(clip, bank_imgs, fa.tau99(bank_imgs))}   # whole tensor, as P14
                out = {}
                for (seed, k), (si, s_lab) in episodes.items():
                    sc = ff.episode_scores(g, per_m, si, s_lab, qi, n, zeros, zeros)
                    for method in (ff.ARR, ff.SS):
                        pred = sc[method].argmax(dim=1).cpu().tolist()
                        out[(method, seed, k)] = pred
                        if method == ff.ARR or (design == "cross" and b == rc.SEEDS[0]):
                            cells.append({"product": product, "design": design, "M": m, "B": b, "S": seed, "K": k,
                                          "method": method, "pred": pred})
                return out

            with torch.no_grad():
                normal_clip = nf["clip18_patches"].to(device).float()
                name_index = {x: i for i, x in enumerate(nf["names"])}
                for b in rc.SEEDS:
                    draw = nf["draws"][str(b)][:M_CROSS]
                    bank_imgs = normal_clip[torch.as_tensor([name_index[x] for x in draw], device=device)]
                    out = score_bank(bank_imgs, "cross", M_CROSS, b)
                    for k in rc.K_VALUES:       # integrity: the diagonal cell is the published P14 cell
                        for method in (ff.ARR, ff.SS):
                            pred = [order0[i] for i in out[(method, b, k)]]
                            ba = classification_metrics(q_true, pred, order0)["balanced_accuracy"]
                            worst = max(worst, abs(ba - published[(method, product, k, b)]))
                del normal_clip
                sw_clip = sw["clip18_patches"].to(device).float()
                sw_index = {x: i for i, x in enumerate(sw["names"])}
                for m in M_SWEEP:
                    for j in range(J_SWEEP):
                        draw = sw["draws"][f"{m}|{j}"]
                        bank_imgs = sw_clip[torch.as_tensor([sw_index[x] for x in draw], device=device)]
                        score_bank(bank_imgs, "sweep", m, j)
                del sw_clip
            if worst >= 1e-9:
                raise cm.P47Error(f"{product}: diagonal cells do not reproduce P14 (max |dBA| = {worst:.2e})")
            print(f"[p47-u3] {product}: {len(q_true)} queries, {n} classes; diagonal reproduces P14", flush=True)
        rc.write_jsonl(staging / "u3_cells.jsonl", cells)
        rc.write_json(staging / "u3_run.json", {
            "schema_version": "p47_u3_run_v1", "protocol": "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md#u3",
            "provenance": provenance, "methods": {"ARR": ff.ARR, "SimpleShot": ff.SS},
            "integrity_diagonal_vs_p14_max_abs_dBA": worst, "products": products,
            "design": {"cross": {"S": list(rc.SEEDS), "B": list(rc.SEEDS), "M": M_CROSS, "K": list(rc.K_VALUES)},
                       "sweep": {"S": list(rc.SEEDS), "M": list(M_SWEEP), "draws": J_SWEEP,
                                 "rule": "random.Random(f'p47bank|{product}|{M}|{j}').sample(sorted train/good, M)"}}})
    print(f"[p47-u3] done; {len(cells)} cells; diagonal max |dBA| vs P14 = {worst:.1e}")
    return 0


# --------------------------------------------------------------------------- stage 3

def cluster_ci(values: dict, reps=10000, seed=20260813) -> list:
    """Percentile product-cluster bootstrap of the mean (the repository's percentile_bootstrap convention)."""
    rng = random.Random(seed)
    keys = list(values)
    est = []
    for _ in range(reps):
        vals = [v for _k in keys for v in values[rng.choice(keys)]]
        est.append(statistics.fmean(vals))
    est.sort()
    return [est[int(round(0.025 * (reps - 1)))], est[int(round(0.975 * (reps - 1)))]]


def analyze(cells: list[dict], products: dict, seeds, k_values, arr_key: str, ss_key: str,
            impl_halfwidth_pp=None, impl_source=None) -> dict:
    """All U3 numbers from the cell predictions. BA in fractions; tables print pp."""
    by = {}
    for c in cells:
        by[(c["product"], c["design"], c["M"], c["B"], c["S"], c["K"], c["method"])] = np.asarray(c["pred"])
    prods = list(products)
    meta = {p: (np.asarray([products[p]["class_order"].index(t) for t in products[p]["true"]]),
                len(products[p]["class_order"])) for p in prods}
    seeds = list(seeds)
    out = {"per_K": {}}
    for k in k_values:
        per_p, ss_p = {}, {}
        for p in prods:
            t, n = meta[p]
            preds = np.stack([by[(p, "cross", M_CROSS, b, s, k, arr_key)] for s in seeds for b in seeds])
            y = np.array([cm.balanced_accuracy(t, x, n) for x in preds]).reshape(len(seeds), len(seeds))
            av = cm.two_way_random(y)
            boot = cm.stratified_bootstrap_ba(t, preds, n, Q_REPS, f"p47q|{p}|{k}")      # [reps, S*B]
            diag = [i * len(seeds) + i for i in range(len(seeds))]
            av["var_Q_single"] = float(boot.var(axis=0, ddof=1).mean())
            av["var_Q_bench"] = float(boot[:, diag].mean(axis=1).var(ddof=1))
            av["nu_Q"] = float(len(t) - 1)
            av["diag_mean"] = float(np.mean(np.diag(y)))
            av["diag"] = np.diag(y).tolist()
            av["var_dep"] = av["var_S"] + av["var_B"] + av["var_SB"] + av["var_Q_single"]
            av["nu_dep"] = cm.welch_satterthwaite([(av["var_S"], av["nu_S"]), (av["var_B"], av["nu_B"]),
                                                   (av["var_SB"], av["nu_SB"]), (av["var_Q_single"], av["nu_Q"])])
            per_p[p] = av
            sp = np.stack([by[(p, "cross", M_CROSS, seeds[0], s, k, ss_key)] for s in seeds])
            ys = np.array([cm.balanced_accuracy(t, x, n) for x in sp])
            bs = cm.stratified_bootstrap_ba(t, sp, n, Q_REPS, f"p47q|{p}|{k}")
            ss_p[p] = {"mean": float(ys.mean()), "var_S": float(ys.var(ddof=1)), "nu_S": len(seeds) - 1.0,
                       "var_Q_single": float(bs.var(axis=0, ddof=1).mean()), "var_Q_bench": float(bs.mean(axis=1).var(ddof=1)),
                       "nu_Q": float(len(t) - 1)}
        P, ns = len(prods), len(seeds)
        comps = {}
        for name, key, nu in (("support draw S", "var_S", "nu_S"), ("bank draw B", "var_B", "nu_B"),
                              ("interaction S x B", "var_SB", "nu_SB")):
            parts = [(per_p[p][key] / ns / P ** 2, per_p[p][nu]) for p in prods]
            comps[name] = {"u2": sum(v for v, _ in parts), "nu": cm.welch_satterthwaite(parts)}
        parts = [(per_p[p]["var_Q_bench"] / P ** 2, per_p[p]["nu_Q"]) for p in prods]
        comps["query sampling Q"] = {"u2": sum(v for v, _ in parts), "nu": cm.welch_satterthwaite(parts)}
        if impl_halfwidth_pp is not None:
            comps["implementation I (Type B, rectangular)"] = {"u2": (impl_halfwidth_pp / 100) ** 2 / 3, "nu": math.inf,
                                                               "half_width_pp": impl_halfwidth_pp, "source": impl_source}
        uc2 = sum(c["u2"] for c in comps.values())
        nu_eff = cm.welch_satterthwaite([(c["u2"], c["nu"]) for c in comps.values()])
        kf = cm.t975(nu_eff)
        for c in comps.values():
            c["u_pp"] = 100 * math.sqrt(c["u2"])
            c["share"] = c["u2"] / uc2 if uc2 > 0 else 0.0
        macro = statistics.fmean(per_p[p]["diag_mean"] for p in prods)
        ci = cluster_ci({p: per_p[p]["diag"] for p in prods})
        ss_parts = [(ss_p[p]["var_S"] / ns / P ** 2, ss_p[p]["nu_S"]) for p in prods] + \
                   [(ss_p[p]["var_Q_bench"] / P ** 2, ss_p[p]["nu_Q"]) for p in prods]
        ss_uc = math.sqrt(sum(v for v, _ in ss_parts))
        ss_nu = cm.welch_satterthwaite(ss_parts)
        pooled_dep = math.sqrt(statistics.fmean(per_p[p]["var_dep"] for p in prods))
        out["per_K"][str(k)] = {
            "benchmark_mean": {"value": macro, "components": comps, "u_c": math.sqrt(uc2), "nu_eff": nu_eff,
                               "k": kf, "U95": kf * math.sqrt(uc2),
                               "existing_product_cluster_bootstrap_ci95": ci,
                               "note": "the cluster-bootstrap CI also contains between-product heterogeneity"},
            "per_deployment": {"pooled_sd": pooled_dep,
                               "per_product": {p: {"mean": per_p[p]["mean"], "sd": math.sqrt(per_p[p]["var_dep"]),
                                                   "nu": per_p[p]["nu_dep"],
                                                   "pi95": [per_p[p]["mean"] - cm.t975(per_p[p]["nu_dep"]) * math.sqrt(per_p[p]["var_dep"]),
                                                            per_p[p]["mean"] + cm.t975(per_p[p]["nu_dep"]) * math.sqrt(per_p[p]["var_dep"])]}
                                               for p in prods}},
            "anova_per_product": per_p,
            "simpleshot": {"benchmark_mean": statistics.fmean(ss_p[p]["mean"] for p in prods), "u_c": ss_uc,
                           "nu_eff": ss_nu, "U95": cm.t975(ss_nu) * ss_uc, "per_product": ss_p},
        }
        if any(c["design"] == "sweep" for c in cells):
            curve = {}
            for m in M_SWEEP:
                rows = {}
                for p in prods:
                    t, n = meta[p]
                    y = np.array([[cm.balanced_accuracy(t, by[(p, "sweep", m, j, s, k, arr_key)], n)
                                   for j in range(J_SWEEP)] for s in seeds])
                    rows[p] = cm.two_way_random(y)
                curve[str(m)] = {"macro_mean_BA": statistics.fmean(r["mean"] for r in rows.values()),
                                 "pooled_sigma_B": math.sqrt(statistics.fmean(r["var_B"] for r in rows.values())),
                                 "pooled_sigma_S": math.sqrt(statistics.fmean(r["var_S"] for r in rows.values())),
                                 "pooled_sigma_SB": math.sqrt(statistics.fmean(r["var_SB"] for r in rows.values())),
                                 "per_product": {p: {x: r[x] for x in ("mean", "var_S", "var_B", "var_SB")}
                                                 for p, r in rows.items()}}
            out["per_K"][str(k)]["bank_size_sweep"] = curve
    return out


def table(res: dict) -> str:
    def f(x):
        return "inf" if not math.isfinite(x) else f"{x:.3g}"
    lines = ["# P47 U3 - uncertainty budget of the benchmark balanced accuracy (ARR, MVTec-FS formal)", ""]
    for k, r in res["per_K"].items():
        b = r["benchmark_mean"]
        lines += [f"## K = {k}", "",
                  f"Benchmark mean BA {100 * b['value']:.2f} %; u_c = {100 * b['u_c']:.2f} pp, nu_eff = {f(b['nu_eff'])}, "
                  f"k = {b['k']:.2f}, U95 = {100 * b['U95']:.2f} pp. Existing product-cluster bootstrap CI: "
                  f"[{100 * b['existing_product_cluster_bootstrap_ci95'][0]:.2f}, "
                  f"{100 * b['existing_product_cluster_bootstrap_ci95'][1]:.2f}] % (also contains between-product spread).", "",
                  "| Component | u (pp) | nu | Share of u_c^2 |", "|---|---|---|---|"]
        for name, c in b["components"].items():
            lines.append(f"| {name} | {c['u_pp']:.3f} | {f(c['nu'])} | {100 * c['share']:.1f} % |")
        d = r["per_deployment"]
        lines += ["", f"Single deployment (one product, one support set, one bank): pooled SD {100 * d['pooled_sd']:.2f} pp.", "",
                  "| Product | Mean BA | SD (pp) | 95 % prediction interval (clipped to [0, 100]) |", "|---|---|---|---|"]
        for p, v in d["per_product"].items():
            lo, hi = max(0.0, 100 * v["pi95"][0]), min(100.0, 100 * v["pi95"][1])
            lines.append(f"| {p} | {100 * v['mean']:.2f} | {100 * v['sd']:.2f} | [{lo:.1f}, {hi:.1f}] |")
        s = r["simpleshot"]
        lines += ["", f"SimpleShot (S and Q only): mean {100 * s['benchmark_mean']:.2f} %, u_c {100 * s['u_c']:.2f} pp, "
                      f"U95 {100 * s['U95']:.2f} pp."]
        if "bank_size_sweep" in r:
            lines += ["", "| M | Macro BA | sigma_B (pp) | sigma_S (pp) | sigma_SB (pp) |", "|---|---|---|---|---|"]
            for m, v in r["bank_size_sweep"].items():
                lines.append(f"| {m} | {100 * v['macro_mean_BA']:.2f} | {100 * v['pooled_sigma_B']:.2f} | "
                             f"{100 * v['pooled_sigma_S']:.2f} | {100 * v['pooled_sigma_SB']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def run_analyze(impl_halfwidth_pp, impl_source) -> int:
    from experiments.revision import revision_common as rc
    run = rc.load_json(OUT_RUN / "u3_run.json")
    cells = [json.loads(line) for line in (OUT_RUN / "u3_cells.jsonl").open(encoding="utf-8")]
    with cm.single_shot(OUT_ANA) as staging:
        res = analyze(cells, run["products"], run["design"]["cross"]["S"], run["design"]["cross"]["K"],
                      run["methods"]["ARR"], run["methods"]["SimpleShot"], impl_halfwidth_pp, impl_source)
        res.update({"schema_version": "p47_u3_analysis_v1", "provenance": rc.git_provenance(),
                    "run_provenance": run["provenance"], "query_bootstrap_reps": Q_REPS,
                    "implementation_component": "included" if impl_halfwidth_pp is not None else
                    "NOT included (no --impl-halfwidth-pp given)"})
        rc.write_json(staging / "u3_budget.json", json.loads(json.dumps(res, default=float).replace("Infinity", "null")))
        text = table(res)
        (staging / "u3_budget.md").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


# --------------------------------------------------------------------------- self-test

def selftest() -> int:
    # ANOVA recovers known components on a large synthetic design
    rng = np.random.default_rng(0)
    a, b, reps = 40, 40, 30
    est = []
    for _ in range(reps):
        y = 0.7 + rng.normal(0, 0.03, (a, 1)) + rng.normal(0, 0.02, (1, b)) + rng.normal(0, 0.01, (a, b))
        r = cm.two_way_random(y)
        est.append((r["var_S"], r["var_B"], r["var_SB"]))
    m = np.mean(est, axis=0)
    assert abs(m[0] / 0.03 ** 2 - 1) < 0.15 and abs(m[1] / 0.02 ** 2 - 1) < 0.15 and abs(m[2] / 0.01 ** 2 - 1) < 0.05, m
    # negative estimates are set to zero
    r = cm.two_way_random(np.array([[0.5, 0.6], [0.6, 0.5]]))
    assert r["var_S"] == 0 and r["var_B"] == 0 and r["var_SB"] > 0
    # Welch-Satterthwaite: equal parts double the dof
    assert abs(cm.welch_satterthwaite([(1, 10), (1, 10)]) - 20) < 1e-9
    # BA matches the repository definition, bootstrap mean matches the point value
    t = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    p = np.array([0, 1, 0, 1, 1, 2, 0, 2, 2])
    assert abs(cm.balanced_accuracy(t, p, 3) - (2 / 3 + 1 + 3 / 4) / 3) < 1e-12
    assert cm.balanced_accuracy(t, p, 4) == cm.balanced_accuracy(t, p, 3) * 3 / 4   # empty class counts 0
    bs = cm.stratified_bootstrap_ba(t, p[None, :], 3, 4000, "x")
    assert abs(bs.mean() - cm.balanced_accuracy(t, p, 3)) < 0.01
    # full analysis on a synthetic cell set
    seeds, ks, prods = (1, 2, 3, 4, 5), (1, 5), {}
    cells = []
    for pi in range(3):
        order = ["a", "b", "c"]
        true = [order[i % 3] for i in range(30)]
        prods[f"p{pi}"] = {"class_order": order, "true": true}
        ti = np.array([order.index(x) for x in true])
        for k in ks:
            for s in seeds:
                for bb in seeds:
                    pr = np.where(rng.random(30) < 0.75, ti, (ti + 1) % 3)
                    cells.append({"product": f"p{pi}", "design": "cross", "M": M_CROSS, "B": bb, "S": s, "K": k,
                                  "method": "arr", "pred": pr.tolist()})
                cells.append({"product": f"p{pi}", "design": "cross", "M": M_CROSS, "B": seeds[0], "S": s, "K": k,
                              "method": "ss", "pred": np.where(rng.random(30) < 0.6, ti, (ti + 2) % 3).tolist()})
                for mm in M_SWEEP:
                    for j in range(J_SWEEP):
                        pr = np.where(rng.random(30) < 0.6 + mm / 500, ti, (ti + 1) % 3)
                        cells.append({"product": f"p{pi}", "design": "sweep", "M": mm, "B": j, "S": s, "K": k,
                                      "method": "arr", "pred": pr.tolist()})
    res = analyze(cells, prods, seeds, ks, "arr", "ss", impl_halfwidth_pp=0.35, impl_source="test")
    b = res["per_K"]["5"]["benchmark_mean"]
    assert abs(sum(c["share"] for c in b["components"].values()) - 1) < 1e-9
    assert b["U95"] > b["u_c"] > 0 and 0.6 < b["value"] < 0.9
    assert res["per_K"]["5"]["bank_size_sweep"]["50"]["macro_mean_BA"] > res["per_K"]["5"]["bank_size_sweep"]["5"]["macro_mean_BA"]
    table(res)
    print("u3 selftest passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("extract", "run", "analyze"))
    ap.add_argument("--mvtec-root", type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--impl-halfwidth-pp", type=float, default=None,
                    help="Type B implementation half-width in pp: the largest |CI bound| of the P32/P38/P41 device "
                         "recomputations of the pooled delta BA")
    ap.add_argument("--impl-source", default=None, help="the files the half-width was read from")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "analyze":
        return run_analyze(a.impl_halfwidth_pp, a.impl_source)
    import torch
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if a.stage == "extract":
        if not a.mvtec_root:
            raise SystemExit("--mvtec-root is required")
        return run_extract(a.mvtec_root, device, a.batch_size)
    if a.stage == "run":
        return run_scoring(device)
    ap.error("--stage or --selftest is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
