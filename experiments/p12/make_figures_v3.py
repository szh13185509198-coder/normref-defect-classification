#!/usr/bin/env python3
"""Figures v3 with the adaptive-region method A (P12 protocol, section F).

Reads only frozen result files (P10, P11, P12) plus cached features; writes PNG/PDF/CSV to
``docs/paper/figures_v3/``. Run after the P12 T-mvtec, R1' and R2' stages.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p12.make_figures_v3 --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402  (deliberately before torch)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.p10 import make_figures as mf  # noqa: E402  (sets the Agg backend)
from experiments.p10 import run_pilot as rp  # noqa: E402
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p12 import cable_ab, normal_aware_ab  # noqa: E402
from experiments.p12.threshold_controls import OUT as T_OUT, P11_EXTERNAL, P11_MVTEC  # noqa: E402
from experiments.p12.thresholds import tau_mebin, tau_otsu  # noqa: E402
from experiments.p18 import baselines as bl  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

plt = mf.plt
OUT = PROJECT_ROOT / "docs" / "paper" / "figures_v3"
K = mf.K
A, NR, SS = fa.A, fa.NR, fa.SS
LABELS = {**mf.LABELS, A: "Adaptive normal region (ours, C1)", NR: "Fixed-fraction normal ref. (C1)"}
STYLE = {"oracle_maskpool_clip18": mf.STYLE["oracle_maskpool_clip18"],
         A: dict(color="0.0", ls="-", lw=2.6, marker="o", ms=7.5, mfc="0.0"),
         NR: dict(color="0.35", ls="-", lw=1.8, marker="o", ms=6, mfc="white"),
         "residual_mean_clip18": mf.STYLE["residual_mean_clip18"],
         "fused_xclass_clip18": mf.STYLE["fused_xclass_clip18"],
         SS: mf.STYLE[SS], "ours": mf.STYLE["ours"], "mvrec_rgb": mf.STYLE["mvrec_rgb"]}


def formal_cells() -> dict:
    cell = {}
    for r in rc.load_json(mf.FORMAL / "formal_results.json")["raw_metrics"] + rc.load_json(P11_MVTEC)["raw_metrics"]:
        cell[(r["method"], r["product"], r["K"], r["seed"])] = r["balanced_accuracy"]
    return cell


def fig_line():
    summ = rc.load_json(mf.FORMAL / "formal_results.json")["product_macro_summary"]
    summ.update(rc.load_json(P11_MVTEC)["product_macro_summary"])
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    rows = []
    for m, st in STYLE.items():
        mean = [100 * summ[m][str(k)]["balanced_accuracy"]["mean"] for k in K]
        sd = [100 * summ[m][str(k)]["balanced_accuracy"]["std_ddof1"] for k in K]
        ax.plot(K, mean, label=LABELS[m], color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"],
                ms=st["ms"], mfc=st["mfc"], mec=st["color"])
        rows += [[m, k, round(v, 4), round(s, 4)] for k, v, s in zip(K, mean, sd)]
    ax.set_xticks(K)
    ax.set_xlabel("Shots per class, K")
    ax.set_ylabel("Product-macro balanced accuracy (%)")
    ax.set_ylim(40, 95)
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False, fontsize=6.9, loc="lower right", ncol=2)
    ax.set_title("MVTec-FS-RGB (13 products, 5 seeds)", fontsize=9.5)
    mf.save(fig, "fig_mvtecfs_line", rows, ["method", "K", "ba_mean_pct", "ba_sd_pct"])


def fig_per_product():
    cell = formal_cells()

    def gain(m, p):
        return 100 * statistics.fmean(cell[(m, p, k, s)] - cell[(SS, p, k, s)] for k in K for s in rc.SEEDS)
    products = sorted(rc.PRODUCTS, key=lambda p: gain(A, p))
    y = np.arange(len(products))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    series = ((A, "0.0", 0.27), (NR, "0.45", 0.0), ("fused_xclass_clip18", "0.75", -0.27))
    for m, color, off in series:
        ax.barh(y + off, [gain(m, p) for p in products], height=0.26, color=color, label=LABELS[m])
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks(y, products)
    ax.set_xlabel("Δ balanced accuracy vs SimpleShot (pp), mean over K and seeds")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="x", color="0.9")
    mf.save(fig, "fig_mvtecfs_per_product", [[p, *(round(gain(m, p), 4) for m, _, _ in series)] for p in products],
            ["product", "adaptive_minus_simpleshot_pp", "fixed_normref_minus_simpleshot_pp", "cross_class_minus_simpleshot_pp"])


def fig_visa_forest():
    p11 = rc.load_json(P11_EXTERNAL)
    p10 = rc.load_json(mf.EXTERNAL)["hypotheses"]
    groups = [("Adaptive − fixed normal ref. (E4)", p11["hypotheses"]["E4_primary_A"]["visa"]["balanced_accuracy"]),
              ("Adaptive − SimpleShot", p11["descriptive_contrasts"][f"{A}_minus_{SS}"]["visa_BA"]),
              ("Fixed normal ref. − SimpleShot (E1)", p10["E1_primary_C1"]["visa"]["balanced_accuracy"]),
              ("Cross-class − SimpleShot (E3)", p10["E3_strict_C0"]["visa"]["balanced_accuracy"])]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    rows, ypos, ticks = [], 0, []
    for label, c in groups:
        entries = [(f"K={k}", c["per_K"][str(k)]["D"], c["per_K"][str(k)]["ci95"]) for k in K] + [("mean", c["Dbar"], c["Dbar_ci95"])]
        for sub, d, ci in entries:
            color = "0.0" if sub == "mean" else "0.45"
            ax.errorbar(100 * d, ypos, xerr=[[100 * (d - ci[0])], [100 * (ci[1] - d)]], fmt="o" if sub == "mean" else "s",
                        color=color, ms=6 if sub == "mean" else 4.5, capsize=2.5, lw=1.2)
            ticks.append((ypos, f"{label}  ·  {sub}"))
            rows.append([label, sub, round(100 * d, 4), round(100 * ci[0], 4), round(100 * ci[1], 4)])
            ypos -= 1
        ypos -= 0.6
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7)
    ax.set_xlabel("Paired Δ balanced accuracy (pp), 95% object-cluster CI (11 VisA objects)")
    ax.grid(axis="x", color="0.9")
    mf.save(fig, "fig_visa_forest", rows, ["contrast", "K", "delta_pp", "ci_low_pp", "ci_high_pp"])


def fig_industrial():
    summ = rc.load_json(normal_aware_ab.OUT / "normal_aware_ab_results.json")["summary"]
    methods = [SS, "fused_xclass_clip18", NR, A, "ours", "mvrec_rgb_adaptation"]
    short = {SS: "SimpleShot\n(C0)", "fused_xclass_clip18": "Cross-class\n(C0)", NR: "Fixed normal\nref. (C1)",
             A: "Adaptive\nregion (C1)", "ours": "Adaptation\n(C0)", "mvrec_rgb_adaptation": "MVREC-RGB\nadapt."}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    rows = []
    for ax, (ds, key) in zip(axes, (("VisA (object macro)", "visa_per_1000_at_95pct_good"), ("MTD", "mtd_per_1000_at_95pct_good"))):
        x = np.arange(len(methods))
        for j, k in enumerate(K):
            vals = [summ[m][str(k)][key]["false_rejects_per_1000"] for m in methods]
            esc = [summ[m][str(k)][key]["escapes_per_1000"] for m in methods]
            ax.bar(x + (j - 1) * 0.26, vals, width=0.26, color=str(0.2 + 0.25 * j), label=f"K={k}")
            rows += [[ds, m, k, round(v, 2), round(e, 2)] for m, v, e in zip(methods, vals, esc)]
        ax.set_xticks(x, [short[m] for m in methods], fontsize=6.6)
        ax.set_title(ds, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("False rejects per 1,000 parts\n(95% good, argmax decision)")
        ax.grid(axis="y", color="0.9")
    axes[0].legend(frameon=False, fontsize=7)
    mf.save(fig, "fig_industrial_false_rejects", rows, ["dataset", "method", "K", "false_rejects_per_1000", "escapes_per_1000"])


# --------------------------------------------------------------------------- maps

def upsample(values: torch.Tensor) -> np.ndarray:
    side = int(round(values.numel() ** 0.5))
    v = values.float().reshape(1, 1, side, side)
    return F.interpolate(v, size=(336, 336), mode="bilinear", align_corners=False)[0, 0].numpy()


def region_mask(sel: torch.Tensor) -> np.ndarray:
    side = int(round(sel.numel() ** 0.5))
    return np.kron(sel.reshape(side, side).cpu().numpy().astype(float), np.ones((14, 14)))


def query_context(product: str):
    """Formal query features, the seed-111 M=20 bank and its tau, query records."""
    qf = torch.load(mf.FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
    nf = torch.load(mf.FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
    name_index = {n: i for i, n in enumerate(nf["names"])}
    bank_imgs = nf["clip18_patches"][[name_index[n] for n in nf["draws"]["111"][:ra.M_PRIMARY]]].float()
    return qf, bank_imgs, fa.tau99(bank_imgs)


def query_records() -> dict:
    out = {}
    with rc.RECORDS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("source_pools") == ["query"]:
                out[(r["product"], r["rgb_sha256"])] = r
    return out


def k5s111_predictions(methods) -> dict:
    preds = {}
    with (T_OUT["mvtec"] / "threshold_mvtec_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            u = json.loads(line)
            if u["K"] == 5 and u["seed"] == 111 and u["method"] in methods:
                for sha, t, p in zip(u["rgb_sha256"], u["true"], u["pred"]):
                    preds[(u["method"], u["product"], sha)] = (p, t)
    return preds


def image_of(rec):
    from PIL import Image
    with Image.open(rc.DATASET_ROOT / rec["canonical_relative_path"]) as handle:
        return handle.convert("RGB").copy()


def fig_qualitative():
    cell = formal_cells()
    gains = {p: statistics.fmean(cell[(A, p, k, s)] - cell[(SS, p, k, s)] for k in K for s in rc.SEEDS) for p in rc.PRODUCTS}
    ranked = sorted(rc.PRODUCTS, key=lambda p: -gains[p])
    products = ranked[:3] + ranked[-3:]
    preds = k5s111_predictions((A, SS))
    records = query_records()
    fig, axes = plt.subplots(len(products), 4, figsize=(7.2, 1.95 * len(products)))
    rows = []
    for row, product in enumerate(products):
        qf, bank_imgs, tau = query_context(product)
        picks = {}
        for i in sorted(range(len(qf["rgb_sha256"])), key=lambda j: qf["rgb_sha256"][j]):
            sha = qf["rgb_sha256"][i]
            pa, true = preds[(A, product, sha)]
            ps, _ = preds[(SS, product, sha)]
            if "success" not in picks and pa == true and ps != true:
                picks["success"] = (i, pa, ps, true)
            if "failure" not in picks and pa != true:
                picks["failure"] = (i, pa, ps, true)
        for col, kind in enumerate(("success", "failure")):
            ax_img, ax_map = axes[row, 2 * col], axes[row, 2 * col + 1]
            if kind not in picks:
                for ax in (ax_img, ax_map):
                    ax.axis("off")
                ax_img.set_title(f"{product}: no {kind} case", fontsize=6.4, loc="left")
                continue
            i, pa, ps, true = picks[kind]
            rec = records[(product, qf["rgb_sha256"][i])]
            a = ra.patch_anomaly(qf["clip18_patches"][i:i + 1].float(), bank_imgs.reshape(-1, bank_imgs.shape[-1]))
            w = ab.region_weights(a, tau, soft=True)[0]
            mf.draw_panel(ax_img, ax_map, image_of(rec), upsample(a[0]), mf.polygons_of(rec),
                          f"{product} ({kind}) · true: {true}\nOurs: {pa} · SimpleShot: {ps}")
            ax_map.contour(region_mask(w > 0), levels=[0.5], colors="cyan", linewidths=0.8)
            rows.append([product, kind, qf["rgb_sha256"][i], true, pa, ps, int((w > 0).sum()), round(100 * gains[product], 4)])
    fig.suptitle("Anomaly maps with the adaptive region (cyan) — K=5, seed 111; green: GT outline, visualization only", fontsize=8)
    fig.tight_layout()
    mf.save(fig, "fig_qualitative_adaptive_region", rows,
            ["product", "case", "rgb_sha256", "true", "ours_pred", "simpleshot_pred", "region_patches", "product_gain_pp"])


def fig_region_comparison():
    cell = formal_cells()
    gains = {p: statistics.fmean(cell[(A, p, k, s)] - cell[(NR, p, k, s)] for k in K for s in rc.SEEDS) for p in rc.PRODUCTS}
    products = sorted(rc.PRODUCTS, key=lambda p: -gains[p])[:3]
    preds = k5s111_predictions((A, NR))
    records = query_records()
    titles = ("Image + GT outline", "Fixed top 10%", "Otsu (per image)", "MEBin-style (per image)", "Ours: normal-calibrated")
    fig, axes = plt.subplots(len(products), 5, figsize=(9.0, 1.95 * len(products)))
    rows = []
    for row, product in enumerate(products):
        qf, bank_imgs, tau = query_context(product)
        pick = next((i for i in sorted(range(len(qf["rgb_sha256"])), key=lambda j: qf["rgb_sha256"][j])
                     if preds[(A, product, qf["rgb_sha256"][i])][0] == preds[(A, product, qf["rgb_sha256"][i])][1]
                     and preds[(NR, product, qf["rgb_sha256"][i])][0] != preds[(NR, product, qf["rgb_sha256"][i])][1]), None)
        if pick is None:
            for ax in axes[row]:
                ax.axis("off")
            axes[row, 0].set_title(f"{product}: no case", fontsize=6.4, loc="left")
            continue
        sha = qf["rgb_sha256"][pick]
        rec = records[(product, sha)]
        image = image_of(rec)
        view = np.asarray(image.resize((336, 336)))
        a = ra.patch_anomaly(qf["clip18_patches"][pick:pick + 1].float(), bank_imgs.reshape(-1, bank_imgs.shape[-1]))
        fixed = torch.zeros_like(a).scatter_(1, a.topk(math.ceil(rp.RHO * a.shape[1]), dim=1).indices, 1.0)[0] > 0
        regions = [fixed, (a > tau_otsu(a))[0], (a > tau_mebin(a))[0]]
        w = ab.region_weights(a, tau, soft=True)[0]
        # evidence purity lambda (S12.1 definition, lambda_any) of the weights each panel shows; mask = evaluation only
        mask = qf["mask_grid_oracle_only"][pick:pick + 1].float()
        lam = [float(bl.purity(r[None].float(), mask)[0]) for r in regions] + [float(bl.purity(w[None], mask)[0])]
        mf.draw_panel(axes[row, 0], axes[row, 1], image, upsample(a[0]), mf.polygons_of(rec),
                      f"{product} · true: {preds[(A, product, sha)][1]}\nOurs: {preds[(A, product, sha)][0]} · fixed: {preds[(NR, product, sha)][0]}")
        for col in range(1, 5):
            ax = axes[row, col]
            ax.clear()
            ax.imshow(view)
            ax.axis("off")
            if col < 4:
                ax.imshow(region_mask(regions[col - 1]), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
                n_sel = int(regions[col - 1].sum())
            else:
                ax.imshow(upsample(w), cmap="inferno", alpha=0.6)
                n_sel = int((w > 0).sum())
            ax.set_title(f"{titles[col]}\n{n_sel} patches, λ = {lam[col - 1]:.2f}", fontsize=6.2, loc="left")
        rows.append([product, sha, *(int(r.sum()) for r in regions), int((w > 0).sum()), *(round(x, 4) for x in lam)])
    fig.tight_layout()
    mf.save(fig, "fig_region_comparison", rows,
            ["product", "rgb_sha256", "fixed_top10_patches", "otsu_patches", "mebin_patches", "ours_patches",
             "fixed_top10_lambda", "otsu_lambda", "mebin_lambda", "ours_lambda"])


def fig_cable(mvtec_root: Path, device):
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p10.external_evaluate import extract
    res = rc.load_json(cable_ab.OUT / "cable_ab_results.json")
    persistent = [p for p in res["persistent_errors_all_seeds"][A] if p.startswith("combined/")][:2]
    if not persistent:
        print("[fig] cable: no persistent combined errors for A; figure skipped")
        return
    bank_names = rc.load_json(mf.CABLE)["normal_bank_draws"]["111"]
    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    normal_dir = mvtec_root / "cable" / "train" / "good"
    with torch.no_grad():
        _, clip = extract(factory, [mf.CABLE_IMAGES / p for p in persistent] + [normal_dir / n for n in bank_names], device, 4)
    bank_imgs = clip[len(persistent):].float()
    tau = fa.tau99(bank_imgs)
    preds = {}
    with (cable_ab.OUT / "cable_ab_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["method"] == A and r["seed"] == 111:
                preds[r["image_path"]] = r["pred"]
    fig, axes = plt.subplots(1, 2 * len(persistent), figsize=(3.6 * len(persistent), 2.0))
    rows = []
    for j, path in enumerate(persistent):
        from PIL import Image
        with Image.open(mf.CABLE_IMAGES / path) as handle:
            image = handle.convert("RGB").copy()
        a = ra.patch_anomaly(clip[j:j + 1].float(), bank_imgs.reshape(-1, bank_imgs.shape[-1]))
        w = ab.region_weights(a, tau, soft=True)[0]
        mf.draw_panel(axes[2 * j], axes[2 * j + 1], image, upsample(a[0]), [],
                      f"Cable {path} · true: combined\nOurs (seed 111): {preds[path]}")
        axes[2 * j + 1].contour(region_mask(w > 0), levels=[0.5], colors="cyan", linewidths=0.8)
        rows.append([path, "combined", preds[path], int((w > 0).sum())])
    fig.tight_layout()
    mf.save(fig, "fig_cable_combined_failures", rows, ["image_path", "true", "ours_pred_seed111", "region_patches"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    mf.OUT = OUT  # redirect the shared save() helper to figures_v3
    mf.setup()
    with torch.no_grad():
        fig_line()
        fig_per_product()
        fig_visa_forest()
        fig_industrial()
        fig_qualitative()
        fig_region_comparison()
        fig_cable(Path(args.mvtec_root), torch.device(args.device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
