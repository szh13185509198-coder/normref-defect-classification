#!/usr/bin/env python3
"""Figures v5 (P17 robustness, P18 backbone swap / evidence purity, A8 cost). Reads result files only.

Grayscale with marker / line-style encoding, matching figures v4. Each figure is saved as PNG + PDF
with a CSV of the plotted numbers.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p20.make_figures_v5
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT = PROJECT_ROOT / "ckpt"
OUT = PROJECT_ROOT / "docs" / "paper" / "figures_v5"
ARR, SS, A, NR, XC, CONS = ("fused_adareg_regres_soft_q99", "simpleshot_cl2n", "fused_adareg_soft_q99",
                            "fused_normref_clip18", "fused_xclass_clip18", "fused_cons_soft_supq90")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
INK, GRID = "0.0", "0.9"


def load(rel):
    return json.loads((CKPT / rel).read_text(encoding="utf-8"))


def save(fig, name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(header)
        w.writerows(rows)
    print("saved", name)


# ------------------------------------------------------------------ backbone swap (P18 + P14/P11/P10 main)

MAIN = {  # main configuration, pooled over K, 95 % cluster-bootstrap CI (ledger P10/P11/P14)
    "MVTec-FS": {"arr-ss": (14.36, 9.50, 19.60), "a-nr": (6.04, 3.68, 8.64), "arr-a": (2.11, 0.47, 4.13), "cons-xc": (5.08, 2.33, 8.13)},
    "VisA": {"arr-ss": (18.16, 10.77, 24.21), "a-nr": (9.64, 3.68, 14.53), "arr-a": (2.34, 0.24, 4.24), "cons-xc": (3.81, 1.15, 6.47)},
}
ROWS = [("arr-ss", "Dual-branch − SimpleShot"), ("a-nr", "Normal-calibrated − fixed fraction"),
        ("arr-a", "Deviation branch (dual − single)"), ("cons-xc", "Consensus − cross-class (C0)")]
SERIES = [("main", "CLIP-L/14@336 + DINOv2-S (main)", "o", "0.0"), ("dinov2_b", "DINOv2-B/14 only", "s", "0.35"),
          ("clip_b16", "CLIP-B/16 only", "^", "0.6")]


def alt_value(res, b, key):
    t, r = {"arr-ss": (f"arr_{b}", f"ss_{b}"), "a-nr": (f"a_{b}", f"nr_{b}"), "arr-a": (f"arr_{b}", f"a_{b}"),
            "cons-xc": (f"cons_{b}", f"xc_{b}")}[key]
    v = res["contrasts"][f"{t} − {r}"]
    return v["delta_pp"], v["ci95_pp"][0], v["ci95_pp"][1]


def fig_backbone_swap():
    res = {"MVTec-FS": load("p18/mvtec/p18_mvtec_results.json"), "VisA": load("p18/external/p18_external_results.json")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    rows_csv = []
    for ax, ds in zip(axes, ("MVTec-FS", "VisA")):
        for i, (key, _) in enumerate(ROWS):
            for j, (sid, label, marker, color) in enumerate(SERIES):
                d, lo, hi = MAIN[ds][key] if sid == "main" else alt_value(res[ds], sid, key)
                y = -i + (1 - j) * 0.22
                ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt=marker, color=color, mfc=color if j < 2 else "white",
                            ms=6, capsize=2, lw=1.1, label=label if i == 0 else None)
                rows_csv.append([ds, key, sid, f"{d:.2f}", f"{lo:.2f}", f"{hi:.2f}"])
        ax.axvline(0, color="0.2", lw=0.8)
        ax.grid(axis="x", color=GRID)
        ax.set_title(ds, fontsize=9)
    axes[0].set_yticks([-i for i in range(len(ROWS))], [r[1] for r in ROWS])
    fig.supxlabel("Δ balanced accuracy (pp, pooled over K; 95% cluster-bootstrap CI)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.07))
    fig.tight_layout()
    save(fig, "fig_backbone_swap", ["dataset", "contrast", "features", "delta_pp", "ci_lo", "ci_hi"], rows_csv)


# ------------------------------------------------------------------ evidence purity (P18 dev)

PURITY = [("global", "Global mean", "-", "x", "0.6"), ("fixed_hard", "Fixed top-10%, uniform", ":", "o", "0.45"),
          ("ours_hard", "Normal-calibrated, uniform", "--", "s", "0.3"), ("mebin_soft", "MEBin-style, soft", "-.", "D", "0.45"),
          ("ours_soft", "Normal-calibrated, soft (ours)", "-", "o", "0.0")]
BIN_LABEL = {"0.0-0.02": "<2%", "0.02-0.05": "2–5%", "0.05-0.15": "5–15%", "0.15-1.01": "≥15%"}


def fig_evidence_purity():
    d = load("p18/dev/p18_dev_results.json")
    bins = list(d["purity"]["by_rho_bin"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.35, 1]})
    rows_csv = []
    x = list(range(len(bins)))
    for key, label, ls, marker, color in PURITY:
        ys = [d["purity"]["by_rho_bin"][b][key] for b in bins]
        lw = 2.2 if key == "ours_soft" else 1.2
        ax.plot(x, ys, ls=ls, marker=marker, color=color, lw=lw, ms=5, mfc=color if key == "ours_soft" else "white", label=label)
        rows_csv += [["purity", key, b, f"{v:.4f}"] for b, v in zip(bins, ys)]
    ax.set_xticks(x, [f"{BIN_LABEL[b]}\n(n={d['purity']['by_rho_bin'][b]['n']})" for b in bins])
    ax.set_xlabel("Defect share of patches ρ (mask, evaluation only)")
    ax.set_ylabel("Evidence purity λ")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2)
    ax.set_title("(a) Evidence purity vs defect size (dev)", fontsize=8.5)
    ex = d["background_exceedance_by_distance"]
    labels = {"1-1": "1", "2-2": "2", "3-3": "3", "4-5": "4–5", "6-1000000": "≥6"}
    vals = [100 * v for v in ex.values()]
    bx.bar(range(len(vals)), vals, color="0.55", width=0.62)
    for i, v in enumerate(vals):
        bx.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=6.8)
    bx.axhline(1.05, color=INK, ls="--", lw=1, label="Normal images (P17): 1.05%")
    bx.legend(frameon=False, fontsize=6.8, loc="upper right")
    bx.set_xticks(range(len(vals)), [labels[k] for k in ex])
    bx.set_xlabel("Distance to the defect mask (patch cells)")
    bx.set_ylabel("Background exceedance (%)")
    bx.grid(axis="y", color=GRID)
    bx.set_title("(b) Background exceedance on defect images", fontsize=8.5)
    rows_csv += [["exceedance", k, "", f"{v:.4f}"] for k, v in ex.items()]
    fig.tight_layout()
    save(fig, "fig_evidence_purity", ["panel", "series", "bin", "value"], rows_csv)


# ------------------------------------------------------------------ robustness (P17)

GROUPS = [("threshold quantile (ARR)", "q"), ("fallback size (ARR)", "fallback"), ("C0 threshold (CONS)", "C0 τ0")]


def fig_robustness():
    res = {"MVTec-FS": load("p17/mvtec/p17_mvtec_results.json")["report"],
           "VisA": load("p17/external/p17_external_results.json")["report"]}
    items = []
    for g, short in GROUPS:
        for m, v in sorted(res["MVTec-FS"][g].items(), key=lambda kv: float(kv[0].replace("arr_q", "").replace("arr_min", "").replace("cons_t", ""))):
            if abs(v["delta_pp"]) < 1e-12 and v["ci95_pp"] == [0.0, 0.0]:
                continue  # the locked setting itself
            val = m.replace("arr_q", "").replace("arr_min", "").replace("cons_t", "")
            items.append((g, m, f"{short} = {val}"))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    rows_csv = []
    for i, (g, m, label) in enumerate(items):
        for j, (ds, marker, color) in enumerate((("MVTec-FS", "o", "0.0"), ("VisA", "s", "0.55"))):
            v = res[ds][g][m]
            d, lo, hi = v["delta_pp"], *v["ci95_pp"]
            ax.errorbar(d, -i + (0.5 - j) * 0.3, xerr=[[d - lo], [hi - d]], fmt=marker, color=color, ms=5,
                        mfc=color if j == 0 else "white", capsize=2, lw=1, label=ds if i == 0 else None)
            rows_csv.append([ds, g, m, f"{d:.2f}", f"{lo:.2f}", f"{hi:.2f}"])
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks([-i for i in range(len(items))], [it[2] for it in items], fontsize=7)
    ax.set_xlabel("Δ BA vs the locked setting (pp; 95% CI)")
    ax.grid(axis="x", color=GRID)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("Locked: q = 0.99, fallback = 6, C0 τ0 = 0.90", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_robustness", ["dataset", "group", "row", "delta_pp", "ci_lo", "ci_hi"], rows_csv)


# ------------------------------------------------------------------ cost vs accuracy (A8 + P18)

def fig_cost_accuracy():
    cost = load("p18_resource/resource_results.json")["results"]
    lv = {"MVTec-FS": load("p18/mvtec/p18_mvtec_results.json")["levels"], "VisA": load("p18/external/p18_external_results.json")["levels"]}
    cfg = [("main", ARR, "CLIP-L + DINOv2-S\n(main)"), ("dinov2_b", "arr_dinov2_b", "DINOv2-B only"), ("clip_b16", "arr_clip_b16", "CLIP-B/16 only")]
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    rows_csv = []
    for ds, marker, color in (("MVTec-FS", "o", "0.0"), ("VisA", "s", "0.55")):
        xs, ys = [], []
        for key, method, label in cfg:
            ms = cost[key]["online"]["dual_branch_ARR"]["end_to_end_from_pil"]["median_ms"]
            ba = statistics.fmean(lv[ds][method])
            xs.append(ms)
            ys.append(ba)
            rows_csv.append([ds, key, f"{ms:.1f}", f"{ba:.2f}", cost[key]["online"]["dual_branch_ARR"]["peak_allocated_bytes"]])
            if ds == "MVTec-FS":
                ax.annotate(label, (ms, ba), textcoords="offset points", xytext=(6, 4), fontsize=6.6)
        ax.plot(xs, ys, marker=marker, color=color, ls="none", ms=7, mfc=color if ds == "MVTec-FS" else "white", label=ds)
    ax.set_xlabel("End-to-end latency per image (ms, RTX 4050 laptop, FP32, batch 1)")
    ax.set_ylabel("Dual-branch BA (%, mean over K = 1/3/5)")
    ax.set_xlim(0, 150)
    ax.grid(color=GRID)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_cost_accuracy", ["dataset", "features", "latency_ms", "ba_mean_over_K", "peak_gpu_bytes"], rows_csv)


def fig_cost_accuracy_p22():
    """Main, DINOv2-B only and DINOv2-S only, latency from ONE session (P22 profile), BA from the P22 formal
    results (which carry the main and DINOv2-B rows, the latter equal to P18 bit for bit)."""
    prof = load("p22/profile/p22_profile_results.json")["arr_end_to_end_median_ms"]
    mv = load("p22/formal/mvtec/p22_formal_mvtec_results.json")["levels"]
    ex = load("p22/formal/external/p22_formal_external_results.json")["levels"]
    cfg = [("main", ARR, "CLIP-L + DINOv2-S (main)\n326M params"),
           ("dinov2_b_r336_fp32", "arr_dinov2_b", "DINOv2-B only\n87M"),
           ("dinov2_s_r336_fp32", "arr_dinov2_s_r336_fp32", "DINOv2-S only\n22M")]
    ba = {"MVTec-FS": lambda m: statistics.fmean(mv[m][k]["mean"] for k in ("1", "3", "5")),
          "VisA": lambda m: statistics.fmean(ex[m]["visa"]),
          "MTD": lambda m: statistics.fmean(ex[m]["mtd"])}
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    rows_csv = []
    for ds, marker, color in (("MVTec-FS", "o", "0.0"), ("VisA", "s", "0.45"), ("MTD", "^", "0.7")):
        xs, ys = [], []
        for key, method, label in cfg:
            xs.append(prof[key])
            ys.append(ba[ds](method))
            rows_csv.append([ds, key, f"{prof[key]:.1f}", f"{ys[-1]:.2f}"])
            if ds == "MVTec-FS":
                dx, dy, ha = {"main": (0, 9, "center"), "dinov2_b_r336_fp32": (0, 9, "center"),
                              "dinov2_s_r336_fp32": (7, -22, "left")}[key]
                ax.annotate(label, (prof[key], ys[-1]), textcoords="offset points", xytext=(dx, dy),
                            fontsize=6.4, ha=ha)
        ax.plot(xs, ys, marker=marker, color=color, ls=":", lw=0.8, ms=6,
                mfc=color if ds == "MVTec-FS" else "white", label=ds)
    ax.set_xlabel("End-to-end latency per image (ms)\nRTX 4050 laptop, FP32, batch 1, measured in one session",
                  fontsize=7.4)
    ax.set_ylabel("Dual-branch BA (%, mean over K = 1/3/5)")
    ax.set_xlim(0, 140)
    ax.set_ylim(48, 86)
    ax.grid(color=GRID)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout()
    save(fig, "fig_cost_accuracy_p22", ["dataset", "features", "latency_ms", "ba_mean_over_K"], rows_csv)


def fig_cost_accuracy_p26():
    """Accuracy vs deployment cost on ONE RTX 4090 run (P26 r2, mean of the forward and reverse passes).
    (a) model forward per image; (b) time to add a new task (log). BA = MVTec-FS formal, mean over K."""
    cost = load("p26_cloud/rtx4090_r3/cost_results.json")["summary"]   # P26 r3 (P30): r2 + two CLIP-L/14-only blocks
    p25 = load("p25/mvtec/p25_results.json")["levels_pct"]
    p21 = load("p21/mvtec/p21_results.json")["levels_pct"]
    p22 = load("p22/formal/mvtec/p22_formal_mvtec_results.json")["levels"]
    p29 = load("p29/mvtec/p29_results.json")["levels_pct"]
    lv = lambda d, m: statistics.fmean(d[m][k]["balanced_accuracy"]["mean"] for k in ("1", "3", "5"))
    ba = {"ours_main": lv(p25, ARR), "ours_clipb16_336": lv(p25, "arr_clip_b16"),
          "ours_clipb16_224": lv(p25, "arr_clip_b16_r224"),
          "ours_b336_fp32": statistics.fmean(p22["arr_dinov2_b"][k]["mean"] for k in ("1", "3", "5")),
          "ours_s336_fp32": statistics.fmean(p22["arr_dinov2_s_r336_fp32"][k]["mean"] for k in ("1", "3", "5")),
          "restricted_adaptation": lv(p21, "ours"),
          "cliplora_b16_224": lv(p25, "cliplora_b16"), "cliplora_b16_336": lv(p25, "cliplora_b16_r336"),
          "cliplora_l14_336": lv(p29, "cliplora_l14_336"),
          # P17 `a_r` (CLIP-L/14@336 only); the FP16 point keeps the FP32 accuracy (P30: FP16 - FP32 = -0.07 pp)
          "ours_l14_336_fp32": lv(p29, "a_r"), "ours_l14_336_fp16": lv(p29, "a_r")}
    mean = lambda b, key: statistics.fmean(cost[b][p][key] for p in ("1", "2"))
    style = {"ours": dict(marker="o", color="0.0", mfc="0.0"),
             "restricted": dict(marker="s", color="0.45", mfc="0.45"),
             "lora": dict(marker="^", color="0.25", mfc="white")}
    group = lambda b: "ours" if b.startswith("ours") else ("restricted" if b.startswith("restricted") else "lora")
    label = {"ours_main": "Ours, main", "ours_b336_fp32": "Ours, DINOv2-B", "ours_s336_fp32": "Ours, DINOv2-S",
             "ours_clipb16_224": "Ours, B/16@224", "ours_clipb16_336": "Ours, B/16@336",
             "restricted_adaptation": "Trainable adapter", "cliplora_b16_224": "CLIP-LoRA, B/16 (224 / 336)",
             "cliplora_l14_336": "CLIP-LoRA, L/14@336", "ours_l14_336_fp32": "Ours, L/14 only (FP32)",
             "ours_l14_336_fp16": "L/14 only (FP16)"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    rows_csv = []
    offsets_a = {"ours_main": (-4, 7, "right"), "ours_b336_fp32": (-6, 3, "right"), "ours_s336_fp32": (7, -3, "left"),
                 "ours_clipb16_224": (6, -3, "left"), "ours_clipb16_336": (6, -3, "left"),
                 "restricted_adaptation": (0, -12, "center"), "cliplora_b16_224": (6, 4, "left"),
                 "cliplora_l14_336": (6, 2, "left"), "ours_l14_336_fp32": (6, -10, "left"),
                 "ours_l14_336_fp16": (0, 8, "center")}
    for b in ba:
        fwd, prep = mean(b, "model_forward_ms"), mean(b, "prep_or_train_s")
        comp = mean(b, "train_compute_only_s") if "train_compute_only_s" in cost[b]["1"] else None
        rows_csv.append([b, f"{ba[b]:.2f}", f"{fwd:.2f}", f"{prep:.1f}", "" if comp is None else f"{comp:.1f}"])
        st = style[group(b)]
        ax1.plot(fwd, ba[b], ls="none", ms=7, mec=st["color"], **st)
        ax2.plot(prep, ba[b], ls="none", ms=7, mec=st["color"], **st)
        if comp is not None:           # official loop (marker) and compute-only (small bar end)
            ax2.plot([comp, prep], [ba[b], ba[b]], color="0.6", lw=1.0, zorder=0)
            ax2.plot(comp, ba[b], marker="|", color="0.25", ms=8, ls="none")
        if b in label:
            dx, dy, ha = offsets_a[b]
            ax1.annotate(label[b], (fwd, ba[b]), textcoords="offset points", xytext=(dx, dy), fontsize=6.4, ha=ha)
    ax2.annotate("Ours (7 configurations)", (2.2, ba["ours_s336_fp32"]), textcoords="offset points", xytext=(8, 0),
                 fontsize=6.4, ha="left", va="center")
    ax2.annotate("Trainable adapter\n(80 steps)", (mean("restricted_adaptation", "prep_or_train_s"),
                 ba["restricted_adaptation"]), textcoords="offset points", xytext=(0, -20), fontsize=6.4, ha="center")
    ax2.annotate("CLIP-LoRA, 2,500 steps\n| GPU training only   △ incl. data loading", (mean("cliplora_b16_224", "prep_or_train_s"),
                 ba["cliplora_b16_224"]), textcoords="offset points", xytext=(-6, 9), fontsize=6.4, ha="right")
    ax2.annotate("L/14@336", (mean("cliplora_l14_336", "prep_or_train_s"), ba["cliplora_l14_336"]),
                 textcoords="offset points", xytext=(-6, 6), fontsize=6.4, ha="right")
    ax1.set_xlabel("Model forward per image (ms)", fontsize=7.4)
    ax2.set_xlabel("Time to add a new task (s, log scale)", fontsize=7.4)
    ax1.set_ylabel("MVTec-FS BA (%, mean over K = 1/3/5)")
    ax1.set_xlim(0, 27)
    ax2.set_xscale("log")
    ax2.set_xlim(1, 6000)
    ax1.set_ylim(62, 84)
    for ax, t in ((ax1, "(a) Inference"), (ax2, "(b) New task")):
        ax.grid(color=GRID)
        ax.set_title(t, fontsize=8, loc="left")
    fig.text(0.5, 0.005, "RTX 4090, batch 1, one run; mean of forward and reverse passes", ha="center", fontsize=6.4,
             color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig_cost_accuracy_p26", ["block", "mvtec_ba_mean_over_K", "model_forward_ms", "new_task_s",
                                        "lora_compute_only_s"], rows_csv)


def fig_ablation_steps():
    """Fig. 5 (v5): step decomposition SimpleShot -> dual branch (Table 8), three datasets, paired Δ with 95% CI.

    MVTec-FS: product-cluster bootstrap (c1_paired_bootstrap.contrast); VisA: object-cluster bootstrap
    (external_evaluate.visa_contrast); Real-IAD: P19's own contrast (object-cluster bootstrap), registered steps only."""
    from experiments.p10 import external_evaluate as ee
    from experiments.p19 import run as p19
    from experiments.revision.c1_paired_bootstrap import contrast
    NR, CEN, T10 = "fused_normref_clip18", "fused_normcenter_mean_clip18", "fused_ctrl_soft_top10"
    A_, SS_, ARR_ = "fused_adareg_soft_q99", "simpleshot_cl2n", "fused_adareg_regres_soft_q99"
    mv_rows = (load("p10_formal/results/formal_results.json")["raw_metrics"]
               + load("p12/threshold_mvtec/threshold_mvtec_results.json")["raw_metrics"]
               + load("p14_formal/mvtec/p14_mvtec_results.json")["raw_metrics"])
    mv = {}
    for r in mv_rows:   # later files repeat SS / NR / A bit for bit (integrity-checked); keep the first
        mv.setdefault(r["method"], {}).setdefault((r["product"], int(r["K"]), int(r["seed"])), r)
    vx_rows = (load("p10_external/results/external_results.json")["raw_metrics"]
               + load("p12/threshold_external/threshold_external_results.json")["raw_metrics"]
               + load("p14_formal/external/p14_external_results.json")["raw_metrics"])
    vx = {}
    for r in vx_rows:
        vx.setdefault(r["method"], {}).setdefault((r["task"], int(r["K"]), int(r["seed"])), r)
    ri = load("p19/blind/p19_results.json")["raw_metrics"]

    def m(t, r):
        c = contrast(mv, r, "balanced_accuracy", target=t)
        return 100 * c["Dbar"], 100 * c["Dbar_ci95"][0], 100 * c["Dbar_ci95"][1]

    def v(t, r):
        c = ee.visa_contrast(vx, t, r, "balanced_accuracy")
        return 100 * c["Dbar"], 100 * c["Dbar_ci95"][0], 100 * c["Dbar_ci95"][1]

    def x(t, r):
        c = p19.contrast(ri, "defect", t, r)
        d = c.get("delta_pp", c.get("Dbar_pp"))
        lo, hi = c.get("ci95", c.get("ci95_pp"))
        return d, lo, hi

    steps = [("① Normal-image centering", (CEN, SS_), False, False),
             ("② Select by normal reference (fixed 10%)", (NR, CEN), False, False),
             ("③ Soft weights by excess", (T10, NR), False, False),
             ("④ Calibrated threshold (vs fixed 10%)", (A_, T10), False, False),
             ("⑤ Deviation branch", (ARR_, A_), True, False),
             ("①+②  Fixed-fraction normal ref. − SimpleShot", (NR, SS_), True, True),
             ("③+④  Calibrated soft region − fixed fraction", (A_, NR), True, True),
             ("①–⑤  Dual branch − SimpleShot", (ARR_, SS_), True, True)]
    series = (("MVTec-FS (13 products)", m, "o", "0.0", "0.0", -0.22),
              ("VisA (11 objects)", v, "s", "0.45", "0.45", 0.0),
              ("Real-IAD blind (30 objects)", x, "^", "0.2", "white", 0.22))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    rows_csv = []
    ys = []
    for i, (label, (t, r), on_realiad, subtotal) in enumerate(steps):
        y = -i - (0.7 if subtotal else 0.0)
        ys.append(y)
        if subtotal:
            ax.axhspan(y - 0.45, y + 0.45, color="0.95", zorder=0)
        for name, fn, marker, color, mfc, off in series:
            if fn is x and not on_realiad:
                continue
            d, lo, hi = fn(t, r)
            ax.plot([lo, hi], [y + off, y + off], color=color, lw=1.2, zorder=2)
            ax.plot(d, y + off, marker=marker, color=color, mfc=mfc, mec=color if mfc != "white" else "0.2",
                    ms=6.5, ls="none", zorder=3, label=name if i == 4 else None)
            rows_csv.append([label, name, f"{d:.2f}", f"{lo:.2f}", f"{hi:.2f}"])
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_yticks(ys, [s[0] for s in steps], fontsize=7.2)
    ax.set_xlabel("Paired Δ balanced accuracy (pp), pooled over K = 1/3/5, 95% cluster-bootstrap CI", fontsize=7.4)
    ax.grid(axis="x", color=GRID)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_ablation_steps", ["step", "dataset", "delta_pp", "ci_low_pp", "ci_high_pp"], rows_csv)


def main() -> int:
    fig_backbone_swap()
    fig_evidence_purity()
    fig_robustness()
    fig_cost_accuracy()
    fig_cost_accuracy_p22()
    fig_cost_accuracy_p26()
    return 0


if __name__ == "__main__":
    sys.exit(main())
