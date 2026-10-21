#!/usr/bin/env python3
"""Figures v4: final methods ARR (C1) and CONS (C0), ablation chain, threshold controls, two-stage.

Reads only frozen result files (P10–P15) plus cached features; no evaluation. Writes PNG / PDF /
CSV to ``docs/paper/figures_v4/`` (docs/P16_FIGURES_V4_PROTOCOL.md).

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p16.make_figures_v4 --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402  (deliberately before torch)
import torch  # noqa: E402

from experiments.p10 import make_figures as mf  # noqa: E402  (sets the Agg backend)
from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p11 import formal_ab as fa  # noqa: E402
from experiments.p11 import run_pilot_ab as ab  # noqa: E402
from experiments.p12 import make_figures_v3 as v3  # noqa: E402
from experiments.p12.threshold_controls import OUT as T_OUT  # noqa: E402
from experiments.p14 import final_formal as ff  # noqa: E402
from experiments.p15 import two_stage as ts  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

plt = mf.plt
OUT = PROJECT_ROOT / "docs" / "paper" / "figures_v4"
K = mf.K
SS, XC, CONS, NR, A, ARR = ff.SS, ff.XC, ff.CONS, ff.NR, ff.A, ff.ARR
P14_MVTEC = ff.OUT["mvtec"] / "p14_mvtec_results.json"
P14_EXT = ff.OUT["external"] / "p14_external_results.json"
P14_CABLE = ff.OUT["cable"] / "p14_cable_results.json"
LABELS = {"oracle_maskpool_clip18": "Mask oracle (upper bound)", ARR: "Ours: dual-branch normal ref. (C1)",
          A: "Adaptive region only (C1)", NR: "Fixed-fraction normal ref. (C1)", CONS: "Ours: support consensus (C0)",
          XC: "Cross-class contrast (C0)", SS: "SimpleShot (C0)", "ours": "Support-side adaptation (C0)",
          "mvrec_rgb": "MVREC-RGB adaptation"}
STYLE = {"oracle_maskpool_clip18": mf.STYLE["oracle_maskpool_clip18"],
         ARR: dict(color="0.0", ls="-", lw=2.6, marker="o", ms=7.5, mfc="0.0"),
         A: dict(color="0.3", ls="--", lw=1.5, marker="o", ms=5.5, mfc="white"),
         NR: dict(color="0.55", ls="-", lw=1.4, marker="o", ms=5, mfc="white"),
         CONS: dict(color="0.15", ls="-", lw=2.0, marker="s", ms=6.5, mfc="0.15"),
         XC: dict(color="0.5", ls=":", lw=1.4, marker="s", ms=5, mfc="white"),
         SS: mf.STYLE[SS], "ours": mf.STYLE["ours"], "mvrec_rgb": mf.STYLE["mvrec_rgb"]}


def mvtec_summary() -> dict:
    s = rc.load_json(mf.FORMAL / "formal_results.json")["product_macro_summary"]
    s.update(rc.load_json(P14_MVTEC)["product_macro_summary"])
    return s


def ci_text(c) -> str:
    return f"{100 * c['Dbar']:+.2f} [{100 * c['Dbar_ci95'][0]:+.2f}, {100 * c['Dbar_ci95'][1]:+.2f}]"


def fig_line():
    summ = mvtec_summary()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
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
    ax.legend(frameon=False, fontsize=6.6, loc="lower right", ncol=2)
    ax.set_title("MVTec-FS-RGB (13 products, 5 seeds)", fontsize=9.5)
    mf.save(fig, "fig_mvtecfs_line", rows, ["method", "K", "ba_mean_pct", "ba_sd_pct"])


def fig_per_product():
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"]
            for r in rc.load_json(P14_MVTEC)["raw_metrics"]}

    def gain(m, p):
        return 100 * statistics.fmean(cell[(m, p, k, s)] - cell[(SS, p, k, s)] for k in K for s in rc.SEEDS)
    products = sorted(rc.PRODUCTS, key=lambda p: gain(ARR, p))
    y = np.arange(len(products))
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    series = ((ARR, "0.0", 0.27), (NR, "0.5", 0.0), (CONS, "0.78", -0.27))
    for m, color, off in series:
        ax.barh(y + off, [gain(m, p) for p in products], height=0.26, color=color, label=LABELS[m])
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks(y, products)
    ax.set_xlabel("Δ balanced accuracy vs SimpleShot (pp), mean over K and seeds")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax.grid(axis="x", color="0.9")
    mf.save(fig, "fig_mvtecfs_per_product", [[p, *(round(gain(m, p), 4) for m, _, _ in series)] for p in products],
            ["product", "dual_branch_minus_simpleshot_pp", "fixed_normref_minus_simpleshot_pp", "consensus_minus_simpleshot_pp"])


def fig_ablation_chain():
    """Step-wise paired Δ (MVTec-FS formal, BA pooled over K, 95 % product-cluster CI)."""
    h10 = rc.load_json(mf.FORMAL / "formal_results.json")["hypotheses"]
    t12 = rc.load_json(T_OUT["mvtec"] / "threshold_mvtec_results.json")["contrasts"]
    h14 = rc.load_json(P14_MVTEC)["hypotheses"]
    steps = [("C1", "SimpleShot → fixed 10% hard pooling (P10)", h10["H1_primary_C1"]["contrast"]["balanced_accuracy"]),
             ("C1", "→ soft weights within fixed 10% (P12)", t12[f"fused_ctrl_soft_top10_minus_{NR}"]["balanced_accuracy"]),
             ("C1", "→ normal-calibrated threshold (P12)", t12[f"{A}_minus_fused_ctrl_soft_top10"]["balanced_accuracy"]),
             ("C1", "→ + deviation branch (P14)", h14["H6_ARR_minus_A"]["results"]["balanced_accuracy"]),
             ("C0", "SimpleShot → cross-class contrast (P10)", h10["H3_strict_C0"]["contrast"]["balanced_accuracy"]),
             ("C0", "→ support-consensus background (P14)", h14["H7_CONS_minus_xclass"]["results"]["balanced_accuracy"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.set_axisbelow(True)
    rows = []
    for i, (cond, label, c) in enumerate(steps):
        y = -i - (0.6 if cond == "C0" else 0)
        d, lo, hi = (100 * c["Dbar"], 100 * c["Dbar_ci95"][0], 100 * c["Dbar_ci95"][1])
        ax.barh(y, d, height=0.6, color="0.2" if cond == "C1" else "0.65")
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="none", color="0.0", capsize=2.5, lw=1)
        ax.text(max(hi, d) + 0.3, y, f"{d:+.2f}", va="center", fontsize=7)
        rows.append([cond, label, round(d, 4), round(lo, 4), round(hi, 4)])
    ax.set_yticks([-i - (0.6 if s[0] == "C0" else 0) for i, s in enumerate(steps)], [f"[{s[0]}] {s[1]}" for s in steps], fontsize=7)
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_xlabel("Paired Δ balanced accuracy (pp), pooled over K, 95% product-cluster CI")
    ax.grid(axis="x", color="0.9")
    mf.save(fig, "fig_ablation_chain", rows, ["condition", "step", "delta_pp", "ci_low_pp", "ci_high_pp"])


def fig_threshold_controls():
    tm = rc.load_json(T_OUT["mvtec"] / "threshold_mvtec_results.json")["contrasts"]
    te = rc.load_json(T_OUT["external"] / "threshold_external_results.json")["contrasts"]
    names = {"fused_ctrl_soft_otsu": "Otsu (per image)", "fused_ctrl_soft_mebin": "MEBin-style (per image)",
             "fused_ctrl_soft_top10": "Fixed top 10% (soft)"}
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    rows, ticks = [], []
    for i, (m, label) in enumerate(names.items()):
        for j, (ds, c) in enumerate((("MVTec-FS", tm[f"{A}_minus_{m}"]["balanced_accuracy"]),
                                     ("VisA", te[f"{A}_minus_{m}"]["visa"]["balanced_accuracy"]))):
            y = -(2.6 * i + j)
            d, lo, hi = 100 * c["Dbar"], 100 * c["Dbar_ci95"][0], 100 * c["Dbar_ci95"][1]
            ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o" if j == 0 else "s", color="0.0" if j == 0 else "0.5",
                        capsize=2.5, ms=5)
            ticks.append((y, f"{label} · {ds}"))
            rows.append([label, ds, round(d, 4), round(lo, 4), round(hi, 4)])
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7)
    ax.set_xlabel("Normal-calibrated threshold − image-only threshold: Δ BA (pp), 95% CI")
    ax.grid(axis="x", color="0.9")
    mf.save(fig, "fig_threshold_controls", rows, ["control", "dataset", "delta_pp", "ci_low_pp", "ci_high_pp"])


def fig_visa_forest():
    p14 = rc.load_json(P14_EXT)
    p10 = rc.load_json(mf.EXTERNAL)["hypotheses"]
    groups = [("Dual-branch − adaptive region (E6)", p14["hypotheses"]["E6_ARR_minus_A"]["visa"]["balanced_accuracy"]),
              ("Dual-branch − SimpleShot", p14["descriptive_contrasts"][f"{ARR}_minus_{SS}"]["visa_BA"]),
              ("Consensus − cross-class (E7)", p14["hypotheses"]["E7_CONS_minus_xclass"]["visa"]["balanced_accuracy"]),
              ("Consensus − SimpleShot", p14["descriptive_contrasts"][f"{CONS}_minus_{SS}"]["visa_BA"]),
              ("Fixed normal ref. − SimpleShot (E1)", p10["E1_primary_C1"]["visa"]["balanced_accuracy"])]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    rows, ypos, ticks = [], 0, []
    for label, c in groups:
        entries = [(f"K={k}", c["per_K"][str(k)]["D"], c["per_K"][str(k)]["ci95"]) for k in K] + [("mean", c["Dbar"], c["Dbar_ci95"])]
        for sub, d, ci in entries:
            ax.errorbar(100 * d, ypos, xerr=[[100 * (d - ci[0])], [100 * (ci[1] - d)]], fmt="o" if sub == "mean" else "s",
                        color="0.0" if sub == "mean" else "0.45", ms=6 if sub == "mean" else 4.5, capsize=2.5, lw=1.2)
            ticks.append((ypos, f"{label}  ·  {sub}"))
            rows.append([label, sub, round(100 * d, 4), round(100 * ci[0], 4), round(100 * ci[1], 4)])
            ypos -= 1
        ypos -= 0.6
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=6.8)
    ax.set_xlabel("Paired Δ balanced accuracy (pp), 95% object-cluster CI (11 VisA objects)")
    ax.grid(axis="x", color="0.9")
    mf.save(fig, "fig_visa_forest", rows, ["contrast", "K", "delta_pp", "ci_low_pp", "ci_high_pp"])


def fig_industrial():
    na = rc.load_json(P14_EXT)["normal_aware"]
    old = rc.load_json(mf.NORMAL_AWARE)["summary"]
    methods = [SS, XC, CONS, NR, ARR, "ours", "mvrec_rgb_adaptation"]
    src = {m: (na[m] if m in na else old[m]) for m in methods}
    short = {SS: "SimpleShot\n(C0)", XC: "Cross-class\n(C0)", CONS: "Consensus\n(C0, ours)", NR: "Fixed normal\nref. (C1)",
             ARR: "Dual-branch\n(C1, ours)", "ours": "Adaptation\n(C0)", "mvrec_rgb_adaptation": "MVREC-RGB\nadapt."}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    rows = []
    for ax, (ds, key) in zip(axes, (("VisA (object macro)", "visa_per_1000_at_95pct_good"), ("MTD", "mtd_per_1000_at_95pct_good"))):
        x = np.arange(len(methods))
        for j, k in enumerate(K):
            vals = [src[m][str(k)][key]["false_rejects_per_1000"] for m in methods]
            esc = [src[m][str(k)][key]["escapes_per_1000"] for m in methods]
            ax.bar(x + (j - 1) * 0.26, vals, width=0.26, color=str(0.2 + 0.25 * j), label=f"K={k}")
            rows += [[ds, m, k, round(v, 2), round(e, 2)] for m, v, e in zip(methods, vals, esc)]
        ax.set_xticks(x, [short[m] for m in methods], fontsize=6.3)
        ax.set_title(ds, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("False rejects per 1,000 parts\n(95% good, argmax decision)")
        ax.grid(axis="y", color="0.9")
    axes[0].legend(frameon=False, fontsize=7)
    mf.save(fig, "fig_industrial_false_rejects", rows, ["dataset", "method", "K", "false_rejects_per_1000", "escapes_per_1000"])


def fig_two_stage():
    """Operating points: false rejects vs escapes per 1,000 (95 % good); review items annotated."""
    ext = rc.load_json(ts.OUT["external"] / "p15_external_results.json")
    cab = rc.load_json(ts.OUT["cable_test"] / "p15_cable_test_results.json")
    argmax_ext = rc.load_json(P14_EXT)["normal_aware"][ARR]
    panels = [("VisA (K=5)", ext["visa_object_macro"]["5"], argmax_ext["5"]["visa_per_1000_at_95pct_good"]),
              ("MTD (K=5)", ext["mtd"]["5"], argmax_ext["5"]["mtd_per_1000_at_95pct_good"])]
    c = cab["classification_argmax"][ARR]
    panels.append(("Cable held-out split (K=8)", cab["two_stage"]["8"],
                   {"false_rejects_per_1000": 950 * c["false_reject_rate"]["mean"], "escapes_per_1000": 50 * c["escape_rate"]["mean"]}))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    rows = []
    for ax, (title, v, am) in zip(axes, panels):
        pts = [(f"α={a}", v[f"{ARR}|a={a}|false_rejects_per_1000"]["mean"], v[f"{ARR}|a={a}|escapes_per_1000"]["mean"], 0.0, "o")
               for a in ("0.01", "0.02", "0.05")]
        pts += [(f"3-way ({a})", v[f"3way|pass={a}|false_rejects_per_1000"]["mean"], v[f"3way|pass={a}|escapes_per_1000"]["mean"],
                 v[f"3way|pass={a}|review_items_per_1000"]["mean"], "s") for a in ("0.1", "0.3")]
        pts.append(("argmax (no screening)", am["false_rejects_per_1000"], am["escapes_per_1000"], 0.0, "X"))
        for name, fr, es, rev, mk in pts:
            ax.scatter(fr, es, marker=mk, s=42, color="0.0" if mk != "X" else "0.55", zorder=3)
            ax.annotate(name + (f"\nreview {rev:.0f}" if rev else ""), (fr, es), fontsize=6, xytext=(4, 3), textcoords="offset points")
            rows.append([title, name, round(fr, 3), round(es, 3), round(rev, 3)])
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("False rejects per 1,000")
        ax.grid(color="0.92")
    axes[0].set_ylabel("Escapes per 1,000")
    fig.tight_layout()
    mf.save(fig, "fig_two_stage_operating_points", rows, ["dataset", "operating_point", "false_rejects_per_1000",
                                                          "escapes_per_1000", "review_items_per_1000"])


def fig_qualitative():
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in rc.load_json(P14_MVTEC)["raw_metrics"]}
    gains = {p: statistics.fmean(cell[(ARR, p, k, s)] - cell[(SS, p, k, s)] for k in K for s in rc.SEEDS) for p in rc.PRODUCTS}
    ranked = sorted(rc.PRODUCTS, key=lambda p: -gains[p])
    products = ranked[:3] + ranked[-3:]
    preds = {}
    with (ff.OUT["mvtec"] / "p14_mvtec_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            u = json.loads(line)
            if u["K"] == 5 and u["seed"] == 111 and u["method"] in (ARR, SS):
                for sha, t, p in zip(u["rgb_sha256"], u["true"], u["pred"]):
                    preds[(u["method"], u["product"], sha)] = (p, t)
    records = v3.query_records()
    fig, axes = plt.subplots(len(products), 4, figsize=(7.2, 1.95 * len(products)))
    rows = []
    for row, product in enumerate(products):
        qf, bank_imgs, tau = v3.query_context(product)
        picks = {}
        for i in sorted(range(len(qf["rgb_sha256"])), key=lambda j: qf["rgb_sha256"][j]):
            sha = qf["rgb_sha256"][i]
            pa, true = preds[(ARR, product, sha)]
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
            mf.draw_panel(ax_img, ax_map, v3.image_of(rec), v3.upsample(a[0]), mf.polygons_of(rec),
                          f"{product} ({kind}) · true: {true}\nOurs: {pa} · SimpleShot: {ps}")
            ax_map.contour(v3.region_mask(w > 0), levels=[0.5], colors="cyan", linewidths=0.8)
            rows.append([product, kind, qf["rgb_sha256"][i], true, pa, ps, int((w > 0).sum()), round(100 * gains[product], 4)])
    fig.suptitle("Anomaly maps with the normal-calibrated region (cyan) — K=5, seed 111; green: GT outline, visualization only",
                 fontsize=7.8)
    fig.tight_layout()
    mf.save(fig, "fig_qualitative", rows,
            ["product", "case", "rgb_sha256", "true", "ours_pred", "simpleshot_pred", "region_patches", "product_gain_pp"])


def fig_cable(mvtec_root: Path, device):
    """ARR's persistent errors on the Cable OOF pool (combined class), maps from the seed-111 bank."""
    from PIL import Image
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p10.external_evaluate import extract
    res = rc.load_json(P14_CABLE)
    persistent = [p for p in res["persistent_errors_all_seeds"][ARR] if p.startswith("combined/")][:2]
    if not persistent:
        print("[fig] cable: no persistent combined errors for ARR; figure skipped")
        return
    bank_names = rc.load_json(mf.CABLE)["normal_bank_draws"]["111"]
    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    normal_dir = mvtec_root / "cable" / "train" / "good"
    _, clip = extract(factory, [mf.CABLE_IMAGES / p for p in persistent] + [normal_dir / n for n in bank_names], device, 4)
    bank_imgs = clip[len(persistent):].float()
    tau = fa.tau99(bank_imgs)
    preds = {}
    with (ff.OUT["cable"] / "p14_cable_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["method"] == ARR and r["seed"] == 111:
                preds[r["image_path"]] = r["pred"]
    fig, axes = plt.subplots(1, 2 * len(persistent), figsize=(3.6 * len(persistent), 2.0))
    rows = []
    for j, path in enumerate(persistent):
        with Image.open(mf.CABLE_IMAGES / path) as handle:
            image = handle.convert("RGB").copy()
        a = ra.patch_anomaly(clip[j:j + 1].float(), bank_imgs.reshape(-1, bank_imgs.shape[-1]))
        w = ab.region_weights(a, tau, soft=True)[0]
        mf.draw_panel(axes[2 * j], axes[2 * j + 1], image, v3.upsample(a[0]), [],
                      f"Cable {path} · true: combined\nOurs (seed 111): {preds[path]}")
        axes[2 * j + 1].contour(v3.region_mask(w > 0), levels=[0.5], colors="cyan", linewidths=0.8)
        rows.append([path, "combined", preds[path], int((w > 0).sum())])
    fig.tight_layout()
    mf.save(fig, "fig_cable_combined_failures", rows, ["image_path", "true", "ours_pred_seed111", "region_patches"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    mf.OUT = OUT  # redirect the shared save() helper
    mf.setup()
    with torch.no_grad():
        fig_line()
        fig_per_product()
        fig_ablation_chain()
        fig_threshold_controls()
        fig_visa_forest()
        fig_industrial()
        fig_two_stage()
        fig_qualitative()
        v3.fig_region_comparison()
        fig_cable(Path(args.mvtec_root), torch.device(args.device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
