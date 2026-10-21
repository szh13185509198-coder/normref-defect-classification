#!/usr/bin/env python3
"""Figures for the Measurement Science and Technology version of the paper.

Reads only the result-file copies in paper_figures/data/ckpt (hashes in paper_figures/data/README.md) and the plotted
data of the published figures (paper_figures/legacy_main, paper_figures/legacy_supplementary). Writes, for every figure, a vector PDF, a
300 dpi PNG and the plotted numbers as CSV to paper_figures/mst/out/.

House style (MST author guidelines): final width 8.5 cm or 15 cm, text 8 pt (7 pt minimum), greyscale-safe (grey
levels plus line style and marker), no acronyms in tick labels or legends, one name per method.

Usage:
    python paper_figures/mst/make_mst_figures.py            # all figures
    python paper_figures/mst/make_mst_figures.py fig3 fig7  # selected figures
"""

from __future__ import annotations

import collections
import csv
import json
import math
import random
import statistics as st
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent
DATA = FIGS / "data" / "ckpt"
OLD_MAIN = FIGS / "legacy_main"
OLD_SUPP = FIGS / "legacy_supplementary"
OUT = HERE / "out"

CM = 1 / 2.54
W1, W2 = 8.5 * CM, 15.0 * CM  # single- and double-column widths (inches)
INK, G1, G2, G3 = "0.0", "0.35", "0.6", "0.8"  # grey levels, darkest first

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    # no text below 8.5 pt at the printed size (figures are saved at their final width); check_pdf.py enforces >= 8 pt
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2, "lines.markersize": 4.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
    "mathtext.default": "regular",
})

# One name per method, used in every figure (no internal codes).
ARR, SS, NR, A, CONS, XC = ("fused_adareg_regres_soft_q99", "simpleshot_cl2n", "fused_normref_clip18",
                            "fused_adareg_soft_q99", "fused_cons_soft_supq90", "fused_xclass_clip18")
NAME = {ARR: "Proposed method", SS: "SimpleShot (whole image)", NR: "Fixed-ratio normal reference",
        A: "Calibrated region, appearance only", CONS: "Support consensus (no conforming images)",
        XC: "Cross-class contrast", "ours": "Trainable support adapter",  # legacy key: NOT the proposed method
        "mvrec_rgb": "MVREC (RGB)", "oracle": "Ground-truth mask pooling (evaluation only)"}
DATASET = {"mvtec": "MVTec-FS (13 products)", "visa": "VisA (11 objects)", "realiad": "Real-IAD, blind (30 objects)",
           "mtd": "Magnetic tile (MTD)", "cable": "Cable, held-out split"}

PRODUCT = {"metal_nut": "metal nut", "pipe_fryum": "pipe fryum", "chewinggum": "chewing gum",
           "macaroni1": "macaroni 1", "macaroni2": "macaroni 2", "pcb1": "PCB 1", "pcb2": "PCB 2", "pcb3": "PCB 3",
           "pcb4": "PCB 4"}

NOTES: list[str] = []  # provenance notes collected while drawing, written to out/NOTES.txt


# --------------------------------------------------------------------------- helpers

def load(rel: str):
    return json.loads((DATA / rel).read_text(encoding="utf-8"))


def old_csv(folder: Path, name: str) -> list[dict]:
    with (folder / name / f"{name}.csv").open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


def save(fig, name: str, header: list[str], rows: list[list]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(header)
        w.writerows(rows)
    print(f"[mst-fig] {name}")


def panel_label(ax, s: str, x: float = -0.02, y: float = 1.02) -> None:
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="right", va="bottom")


def hgrid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color="0.9", lw=0.5)
    ax.set_axisbelow(True)


def raw_grid(rows: list[dict], unit: str) -> dict:
    """{method: {(unit, K, seed): balanced accuracy}}"""
    g = collections.defaultdict(dict)
    for r in rows:
        g[r["method"]][(r[unit], int(r["K"]), int(r["seed"]))] = r["balanced_accuracy"]
    return g


def macro_by_seed(g: dict, method: str, units: list[str], k: int) -> list[float]:
    seeds = sorted({key[2] for key in g[method]})
    return [100 * st.fmean(g[method][(u, k, s)] for u in units) for s in seeds]


def cluster_bootstrap_mean(values: list[float], n_boot: int = 10000, seed: int = 47) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(st.fmean(rng.choice(values) for _ in range(n)) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def clopper_pearson(k: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    k = int(round(k))
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else stats.beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a, k + 1, n - k)
    return lo, hi


# --------------------------------------------------------------------------- figure 2: where accuracy is lost

def fig2_headroom():
    lv21 = load("p21/mvtec/p21_results.json")["levels_pct"]
    lv29 = load("p29/mvtec/p29_results.json")["levels_pct"]
    lv25 = load("p25/mvtec/p25_results.json")["levels_pct"]
    p14 = load("p14_formal/mvtec/p14_mvtec_results.json")["product_macro_summary"]
    p11 = load("p11_formal/mvtec/p11_formal_mvtec_results.json")["product_macro_summary"]
    k5 = lambda lv, m: lv[m]["5"]["balanced_accuracy"]["mean"]  # noqa: E731
    pm = lambda s, m: 100 * s[m]["5"]["balanced_accuracy"]["mean"]  # noqa: E731
    # Three values come from files not in paper_figures/data (ckpt/revision_c/c2_frozen_feature_baselines.json and
    # ckpt/p24/mvtec/p24_results.json); until they are added, the published figure's printed values are used.
    missing = {"logreg": ("revision_c/c2_frozen_feature_baselines.json", 72.3),
               "ss_b16_224": ("p24/mvtec/p24_results.json", 61.1),
               "oracle_b16_224": ("p24/mvtec/p24_results.json", 85.7)}
    got = {}
    for key, (rel, printed) in missing.items():
        got[key] = printed
        if not (DATA / rel).is_file():
            NOTES.append(f"fig2: {key} = {printed} transcribed from the published figure; add ckpt/{rel} to read it")
    if (DATA / "revision_c/c2_frozen_feature_baselines.json").is_file():
        rows = [r for r in load("revision_c/c2_frozen_feature_baselines.json")["raw_metrics"] if r["method"] == "logreg_rfs"]
        got["logreg"] = 100 * st.fmean(st.fmean(r["balanced_accuracy"] for r in rows if r["K"] == 5 and r["seed"] == s)
                                       for s in sorted({r["seed"] for r in rows}))
    if (DATA / "p24/mvtec/p24_results.json").is_file():
        g = raw_grid(load("p24/mvtec/p24_results.json")["raw_metrics"], "product")
        units = sorted({u for u, _, _ in g["ss_clip_b16_r224"]})
        got["ss_b16_224"] = st.fmean(macro_by_seed(g, "ss_clip_b16_r224", units, 5))
        got["oracle_b16_224"] = st.fmean(macro_by_seed(g, "oracle_clip_b16_r224", units, 5))

    groups = [
        ("Main features (CLIP ViT-L/14 at 336 px with DINOv2-S)", [
            ("Classifier changed, same frozen features", [
                ("SimpleShot (whole image)", k5(lv21, SS), G3),
                ("Logistic regression", got["logreg"], G3),
                ("Trainable support adapter", k5(lv21, "ours"), G3)]),
            ("Encoder fine-tuned (uses class names)", [
                ("CLIP-LoRA, official code", k5(lv29, "cliplora_l14_336"), G2)]),
            ("Better local evidence, no training", [
                ("Cross-class contrast", k5(lv21, XC), G1),
                ("Support consensus (no conforming images)", k5(lv21, CONS), G1),
                ("Fixed-ratio normal reference", pm(p14, NR), G1),
                ("Proposed method", k5(lv21, ARR), INK)]),
            ("Reference, not a method", [
                ("Ground-truth mask pooling", pm(p11, "oracle_maskpool_clip18"), "hatch")])]),
        ("CLIP ViT-B/16 at 224 px, the backbone of official CLIP-LoRA", [
            (None, [("SimpleShot (whole image)", got["ss_b16_224"], G3),
                    ("CLIP-LoRA, official code", k5(lv21, "cliplora_b16"), G2),
                    ("Proposed method", k5(lv25, "arr_clip_b16_r224"), INK),
                    ("Ground-truth mask pooling", got["oracle_b16_224"], "hatch")])])]
    fig, ax = plt.subplots(figsize=(W2, 10.2 * CM))
    y, ticks, labels, out = 0.0, [], [], []
    trans = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for gi, (title, sub) in enumerate(groups):
        if gi:
            y -= 0.5
            ax.plot([0.01, 0.99], [y + 0.6] * 2, color=G2, lw=0.5, transform=trans, clip_on=False)
        ax.text(0.01, y, title, fontweight="bold", va="center", transform=trans)
        y -= 1.05
        top, ss = y + 0.5, None
        for sname, bars in sub:
            if sname:
                ax.text(0.03, y, sname, style="italic", va="center", transform=trans, color="0.25")
                y -= 0.95
            for label, v, shade in bars:
                ss = v if label.startswith("SimpleShot") else ss
                hatch = shade == "hatch"
                ax.barh(y, v, height=0.68, color="white" if hatch else shade, edgecolor=INK if hatch else "none",
                        hatch="////" if hatch else None, lw=0.5)
                ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=8.5,
                        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
                ticks.append(y)
                labels.append(label)
                out.append([title, sname or "", label, round(v, 2)])
                y -= 0.95
            y -= 0.2
        ax.plot([ss, ss], [top, y + 0.7], color=INK, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.5, 0.7)
    ax.set_xlim(55, 95)
    ax.set_xlabel("Balanced accuracy on MVTec-FS (%), 5 shots per defect type")
    hgrid(ax, "x")
    fig.subplots_adjust(left=0.36, right=0.98, top=0.99, bottom=0.09)
    save(fig, "fig2_headroom", ["section", "group", "method", "balanced_accuracy_pct_K5"], out)


# --------------------------------------------------------------------------- figure 3: gains, curves and per object

def fig3_gains():
    mv = raw_grid(load("p14_formal/mvtec/p14_mvtec_results.json")["raw_metrics"]
                  + load("p10_formal/results/formal_results.json")["raw_metrics"], "product")
    ex = raw_grid(load("p14_formal/external/p14_external_results.json")["raw_metrics"]
                  + load("p10_external/results/external_results.json")["raw_metrics"], "task")
    lv21 = load("p21/mvtec/p21_results.json")["levels_pct"]
    p11 = load("p11_formal/mvtec/p11_formal_mvtec_results.json")["product_macro_summary"]["oracle_maskpool_clip18"]
    products = sorted({u for u, _, _ in mv[ARR]})
    visa = sorted(t for t in {u for u, _, _ in ex[ARR]} if t.startswith("visa__"))
    ks = [1, 3, 5]

    def curve(g, m, units):
        per = [macro_by_seed(g, m, units, k) for k in ks]
        return [st.fmean(p) for p in per], [st.stdev(p) for p in per]

    series = {
        "mvtec": [("oracle", ([100 * p11[str(k)]["balanced_accuracy"]["mean"] for k in ks],
                              [100 * p11[str(k)]["balanced_accuracy"]["std_ddof1"] for k in ks])),
                  (ARR, curve(mv, ARR, products)), (NR, curve(mv, NR, products)),
                  ("ours", ([lv21["ours"][str(k)]["balanced_accuracy"]["mean"] for k in ks],
                            [lv21["ours"][str(k)]["balanced_accuracy"]["sd"] for k in ks])),
                  (SS, curve(mv, SS, products))],
        "visa": [(ARR, curve(ex, ARR, visa)), (NR, curve(ex, NR, visa)), (SS, curve(ex, SS, visa))]}
    style = {"oracle": dict(color=G2, ls=(0, (4, 2)), marker="*", mfc="white", ms=6),
             ARR: dict(color=INK, ls="-", marker="o", mfc=INK, lw=1.6),
             NR: dict(color=G1, ls="-", marker="o", mfc="white"),
             "ours": dict(color=G2, ls=(0, (1, 1.2)), marker="v", mfc=G2),
             SS: dict(color=G1, ls=(0, (5, 1.5, 1, 1.5)), marker="^", mfc="white")}
    fig = plt.figure(figsize=(W2, 8.2 * CM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.45, left=0.075, right=0.99, top=0.91, bottom=0.31)
    out = []
    for i, key in enumerate(["mvtec", "visa"]):
        ax = fig.add_subplot(gs[0, i])
        for m, (mean, sd) in series[key]:
            ax.errorbar(ks, mean, yerr=sd, capsize=2, elinewidth=0.6, capthick=0.6, label=NAME[m], **style[m])
            out += [[DATASET[key], NAME[m], k, round(a, 2), round(b, 2)] for k, a, b in zip(ks, mean, sd)]
        ax.set_xticks(ks)
        ax.set_xlabel("Shots per defect type")
        ax.set_ylim(40, 95)
        hgrid(ax)
        ax.set_title(DATASET[key], pad=3)
        panel_label(ax, "(a)" if i == 0 else "(b)", x=-0.12)
        if i == 0:
            ax.set_ylabel("Balanced accuracy (%)")
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, columnspacing=1.0, handlelength=2.6,
               fontsize=8.5)

    # (c) one point per product / object: gain of the proposed method over SimpleShot, mean over shots and seeds
    def per_unit(g, units):
        return {u: 100 * st.fmean(g[ARR][key] - g[SS][key] for key in g[ARR] if key[0] == u) for u in units}

    hyp = load("p19/blind/p19_results.json")["hypotheses"]["B1"]
    cols = [("MVTec-FS\n(13)", per_unit(mv, products)),
            ("VisA\n(11)", per_unit(ex, visa)),
            ("Real-IAD\nblind (30)", dict(hyp["per_object_pp"]))]
    mtd = per_unit(ex, ["mtd__mtd6way"])["mtd__mtd6way"]
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(3)
    for j, (lab, d) in enumerate(cols):
        v = np.array(list(d.values()))
        ax.scatter(j + rng.uniform(-0.17, 0.17, len(v)), v, s=11, facecolor=G3, edgecolor=G1, lw=0.4, zorder=2)
        lo, hi = cluster_bootstrap_mean(list(v))
        ax.errorbar(j + 0.33, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="D", color=INK, ms=4,
                    capsize=2, elinewidth=0.8, zorder=3)
        ax.text(j + 0.08, 49, f"{int((v > 0).sum())}/{len(v)}\n> 0", ha="center", va="top", fontsize=8.5)
        out += [[lab.replace("\n", " "), "per unit", u, round(x, 2), ""] for u, x in d.items()]
        out.append([lab.replace("\n", " "), "mean [95% CI]", "", round(v.mean(), 2), f"{lo:.2f}..{hi:.2f}"])
    ax.scatter([3], [mtd], s=14, marker="s", facecolor=G3, edgecolor=G1, lw=0.4)
    out.append(["MTD", "single dataset", "mtd", round(mtd, 2), ""])
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(4), [c[0] for c in cols] + ["MTD\n(1)"], fontsize=8.5)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-15, 50)
    ax.set_ylabel("Gain over SimpleShot\n(percentage points)", labelpad=1)
    hgrid(ax)
    ax.set_title("Gain per product or object", pad=3)
    panel_label(ax, "(c)", x=-0.2)
    NOTES.append("fig3(c): grey dots = products/objects (mean over shots and seeds); black diamond = mean with 95% "
                 "cluster-bootstrap interval over products/objects (10 000 resamples, seed 47); for Real-IAD the "
                 f"published interval is {hyp['ci95_pp'][0]:.2f}..{hyp['ci95_pp'][1]:.2f}")
    save(fig, "fig3_gains", ["dataset", "series", "K_or_unit", "value", "sd_or_ci"], out)


# --------------------------------------------------------------------------- figure 4 and supplementary forests

def forest(rows, names, datasets, markers, xlabel, name, width=W2, height=None, groups=None, xlim=None,
           legend_loc="lower right"):
    """rows: list of (row_label, dataset, delta, lo, hi). One line per row label, datasets offset vertically."""
    height = height or (1.6 + 0.62 * len(names)) * CM * 1.0 + 1.6 * CM
    fig, ax = plt.subplots(figsize=(width, height))
    off = np.linspace(0.22, -0.22, len(datasets)) if len(datasets) > 1 else [0]
    pos = {n: -i for i, n in enumerate(names)}
    for (lab, ds, d, lo, hi) in rows:
        j = datasets.index(ds)
        m = markers[ds]
        ax.errorbar(d, pos[lab] + off[j], xerr=[[d - lo], [hi - d]], fmt=m["marker"], color=m["color"],
                    mfc=m.get("mfc", m["color"]), ms=4.5, capsize=1.8, elinewidth=0.8, capthick=0.6, label=ds)
    if groups:
        for lo_i, hi_i in groups:
            ax.axhspan(pos[names[hi_i]] - 0.45, pos[names[lo_i]] + 0.45, color="0.94", lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(list(pos.values()), names)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-len(names) + 0.4, 0.6)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    hgrid(ax, "x")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend([uniq[d] for d in datasets if d in uniq], [d for d in datasets if d in uniq], loc=legend_loc,
              handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    save(fig, name, ["row", "dataset", "delta_pp", "ci_low_pp", "ci_high_pp"],
         [[a, b, c, d, e] for a, b, c, d, e in rows])


MARK3 = {DATASET["mvtec"]: dict(marker="o", color=INK),
         DATASET["visa"]: dict(marker="s", color=G1),
         DATASET["realiad"]: dict(marker="^", color=G1, mfc="white")}


def fig4_steps():
    step = {"① Normal-image centering": "(1) Centre features on normal images",
            "② Select by normal reference (fixed 10%)": "(2) Select patches by normal reference (fixed 10%)",
            "③ Soft weights by excess": "(3) Weight patches by excess over the threshold",
            "④ Calibrated threshold (vs fixed 10%)": "(4) Calibrated threshold instead of fixed 10%",
            "⑤ Deviation branch": "(5) Add the deviation descriptor",
            "①+②  Fixed-fraction normal ref. − SimpleShot": "(1)+(2) vs SimpleShot",
            "③+④  Calibrated soft region − fixed fraction": "(3)+(4) vs fixed 10% region",
            "①–⑤  Dual branch − SimpleShot": "(1)–(5): proposed method vs SimpleShot"}
    ds = {"MVTec-FS (13 products)": DATASET["mvtec"], "VisA (11 objects)": DATASET["visa"],
          "Real-IAD blind (30 objects)": DATASET["realiad"]}
    rows = [(step[r["step"]], ds[r["dataset"]], float(r["delta_pp"]), float(r["ci_low_pp"]), float(r["ci_high_pp"]))
            for r in old_csv(OLD_MAIN, "fig06_stepwise_ablation")]
    names = list(step.values())
    forest(rows, names, list(ds.values()), MARK3,
           "Change in balanced accuracy (percentage points), 95% interval", "fig4_steps",
           height=9.0 * CM, groups=[(5, 7)], xlim=(-2, 26), legend_loc="upper right")


def figS_visa_forest():
    lab = {"Dual-branch − adaptive region (E6)": "Proposed vs appearance-only region",
           "Dual-branch − SimpleShot": "Proposed vs SimpleShot",
           "Consensus − cross-class (E7)": "Support consensus vs cross-class contrast",
           "Consensus − SimpleShot": "Support consensus vs SimpleShot",
           "Fixed normal ref. − SimpleShot (E1)": "Fixed 10% normal-referenced vs SimpleShot"}
    kname = {"K=1": "1 shot", "K=3": "3 shots", "K=5": "5 shots", "mean": "mean"}
    rows = [(lab[r["contrast"]], kname[r["K"]], float(r["delta_pp"]), float(r["ci_low_pp"]), float(r["ci_high_pp"]))
            for r in old_csv(OLD_MAIN, "fig07_visa_forest")]
    marks = {"1 shot": dict(marker="o", color=G2, mfc="white"), "3 shots": dict(marker="s", color=G2, mfc="white"),
             "5 shots": dict(marker="^", color=G2, mfc="white"), "mean": dict(marker="D", color=INK)}
    forest(rows, list(lab.values()), list(marks), marks,
           "Change in balanced accuracy on VisA (percentage points)\n95% interval over the 11 objects", "figS_visa_forest",
           height=8.0 * CM)


def figS_threshold_source():
    lab = {"Otsu (per image)": "Otsu threshold per image", "MEBin-style (per image)": "MEBin threshold per image",
           "Fixed top 10% (soft)": "Fixed top 10% per image"}
    rows = [(lab[r["control"]], DATASET["mvtec"] if r["dataset"] == "MVTec-FS" else DATASET["visa"],
             float(r["delta_pp"]), float(r["ci_low_pp"]), float(r["ci_high_pp"]))
            for r in old_csv(OLD_SUPP, "figS11-2_threshold_source")]
    forest(rows, list(lab.values()), [DATASET["mvtec"], DATASET["visa"]], MARK3,
           "Normal-calibrated threshold minus image-only threshold:\nchange in balanced accuracy (percentage points)",
           "figS_threshold_source", width=W2, height=5.6 * CM, xlim=(-1, 10.5))


def figS_robustness():
    lab = {"arr_q0.95": "Threshold quantile 0.95", "arr_q0.97": "Threshold quantile 0.97",
           "arr_q0.98": "Threshold quantile 0.98", "arr_q0.995": "Threshold quantile 0.995",
           "arr_min1": "Minimum region 1 patch", "arr_min3": "Minimum region 3 patches",
           "arr_min12": "Minimum region 12 patches", "cons_t0.8": "Consensus threshold 0.80",
           "cons_t0.85": "Consensus threshold 0.85", "cons_t0.95": "Consensus threshold 0.95"}
    rows = [(lab[r["row"]], DATASET["mvtec"] if r["dataset"] == "MVTec-FS" else DATASET["visa"],
             float(r["delta_pp"]), float(r["ci_lo"]), float(r["ci_hi"])) for r in old_csv(OLD_SUPP, "figS11-1_robustness")]
    forest(rows, list(lab.values()), [DATASET["mvtec"], DATASET["visa"]], MARK3,
           "Change in balanced accuracy vs the fixed setting (percentage points)\n"
           "fixed: quantile 0.99, minimum region 6 patches, consensus threshold 0.90",
           "figS_robustness", width=W2, height=9.0 * CM, legend_loc="lower left")


def figS_backbone():
    lab = {"arr-ss": "Proposed vs SimpleShot", "a-nr": "Calibrated vs fixed 10% region",
           "arr-a": "Deviation descriptor added", "cons-xc": "Support consensus vs cross-class"}
    feat = {"main": "CLIP ViT-L/14 + DINOv2-S (main)", "dinov2_b": "DINOv2-B/14 only", "clip_b16": "CLIP ViT-B/16 only"}
    marks = {feat["main"]: dict(marker="o", color=INK), feat["dinov2_b"]: dict(marker="s", color=G1),
             feat["clip_b16"]: dict(marker="^", color=G1, mfc="white")}
    allrows = old_csv(OLD_SUPP, "figS12-2_backbone_swap")
    for dsn, tag in (("MVTec-FS", "mvtec"), ("VisA", "visa")):
        rows = [(lab[r["contrast"]], feat[r["features"]], float(r["delta_pp"]), float(r["ci_lo"]), float(r["ci_hi"]))
                for r in allrows if r["dataset"] == dsn]
        forest(rows, list(lab.values()), list(feat.values()), marks,
               f"{DATASET[tag]}:\nchange in balanced accuracy (percentage points)", f"figS_backbone_{tag}",
               width=W2, height=6.2 * CM, xlim=(-2, 28))


# --------------------------------------------------------------------------- figure 6: false-reject rate of one calibration

def fig6_frr_theory():
    """Split-conformal screening with N calibration normals: k = ceil((N + 1)(1 - alpha)); the false-reject rate realised
    by one calibration set is Beta(N - k + 1, k) (Vovk 2013). (a) its density for alpha = 5 %; (b) its 5-95 % range."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 6.0 * CM), gridspec_kw=dict(wspace=0.32))
    out = []
    x = np.linspace(0, 0.16, 801)
    for n, ls, c in ((50, (0, (1, 1.2)), G1), (100, (0, (4, 1.5)), G1), (500, "-", INK)):
        k = math.ceil((n + 1) * 0.95)
        d = stats.beta(n - k + 1, k)
        a.plot(100 * x, d.pdf(x) / 100, color=c, ls=ls, label=f"N = {n}")
        lo, hi = d.ppf([0.05, 0.95])
        out.append(["a", 0.05, n, k, round(100 * d.mean(), 3), round(100 * lo, 3), round(100 * hi, 3),
                    round(1 - d.cdf(0.05), 3)])
    a.axvline(5, color=G2, lw=0.6)
    a.annotate("α = 5%", xy=(5, 0.3), xytext=(7.5, 0.3), fontsize=8.5, va="center",
               arrowprops=dict(arrowstyle="-", lw=0.5, color=G1))
    a.set_xlabel("False-reject rate of one calibration (%)")
    a.set_ylabel("Probability density (per %)")
    a.set_xlim(0, 16)
    a.set_ylim(bottom=0)
    a.legend(loc="upper right")
    panel_label(a, "(a)", x=-0.13)
    hgrid(a)

    styles = {0.01: ("o", INK, "-"), 0.02: ("s", G1, (0, (4, 1.5))), 0.05: ("^", G1, "-"), 0.10: ("D", G2, (0, (1, 1.2)))}
    for alpha, (mk, c, ls) in styles.items():
        lo, hi, nn = [], [], []
        step = round(1 / alpha)  # N + 1 a multiple of 1/alpha: (N + 1)(1 - alpha) is an integer, coverage exactly 1 - alpha
        for n in range(step - 1, 2001, step):
            k = round((n + 1) * (1 - alpha))
            if k > n:
                continue
            d = stats.beta(n - k + 1, k)
            l5, h5 = d.ppf([0.05, 0.95])
            nn.append(n), lo.append(100 * l5), hi.append(100 * h5)
            if n + 1 in (20, 50, 100, 200, 300, 500, 1000):
                out.append(["b", alpha, n, k, round(100 * d.mean(), 3), round(100 * l5, 3), round(100 * h5, 3),
                            round(1 - d.cdf(alpha), 3)])
        b.fill_between(nn, lo, hi, color=c, alpha=0.12, lw=0)
        b.plot(nn, hi, color=c, ls=ls, lw=1.0)
        b.plot(nn, lo, color=c, ls=ls, lw=1.0)
        b.plot(nn, [100 * alpha] * len(nn), color=c, ls=ls, lw=0.5)
        b.plot([nn[-1]], [100 * alpha], marker=mk, color=c, ms=4, clip_on=False)
        b.text(nn[-1] * 1.15, 100 * alpha, f"α = {100 * alpha:g}%", va="center", fontsize=8.5)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(8, 2000)
    b.set_ylim(0.03, 40)
    b.set_yticks([0.03, 0.1, 0.3, 1, 3, 10, 30], ["0.03", "0.1", "0.3", "1", "3", "10", "30"])
    b.set_xticks([10, 30, 100, 300, 1000], ["10", "30", "100", "300", "1000"])
    b.set_xlabel("Number of calibration normals, N")
    b.set_ylabel("False-reject rate (%)")
    hgrid(b)
    panel_label(b, "(b)", x=-0.13)
    fig.subplots_adjust(left=0.09, right=0.9, top=0.93, bottom=0.17)
    NOTES.append("fig6: theory only (Beta(N-k+1,k), k = (N+1)(1-alpha)); N plotted where (N+1)alpha is an integer, so "
                 "the mean false-reject rate (N-k+1)/(N+1) equals alpha exactly; the P47 U1 "
                 "resampling results are to be overlaid")
    save(fig, "fig6_frr_theory", ["panel", "alpha", "N", "k", "mean_frr_pct", "q05_pct", "q95_pct", "P_frr_gt_alpha"],
         out)


# --------------------------------------------------------------------------- figure 7: operating points with intervals

def eval_sizes(units: list[dict]) -> dict:
    """Evaluation normals / defects per task, recovered from the denominators of the reported rates (checked
    against the query counts)."""
    rates = collections.defaultdict(lambda: ([], []))
    for u in units:
        for a in u["two_level"][ARR].values():
            rates[u["task"]][0].append(a["false_reject_rate"])
            rates[u["task"]][1].append(a["escape_rate"])
    sizes = {}
    for t, (fr, es) in rates.items():
        n0 = math.lcm(*[Fraction(v).limit_denominator(2000).denominator for v in fr])
        n1 = math.lcm(*[Fraction(v).limit_denominator(2000).denominator for v in es])
        sizes[t] = (n0, n1)
    return sizes


def fig7_operating_points():
    p15 = load("p15/external/p15_external_results.json")
    cab = load("p15/cable_test/p15_cable_test_results.json")
    p14 = load("p14_formal/external/p14_external_results.json")["normal_aware"][ARR]["5"]
    sizes = eval_sizes(p15["units"])
    for t, (n0, n1) in sizes.items():
        q = p15["task_info"][t]["queries"]
        if n0 + n1 != q:
            raise SystemExit(f"{t}: recovered {n0} + {n1} != {q} queries")
    comp = cab["test_composition"]
    cab_n = (comp["good"], sum(v for c, v in comp.items() if c != "good"))
    good = 0.95

    def unit_points(u):
        pts = {}
        for a, v in u["two_level"][ARR].items():
            pts[f"a={a}"] = (v["false_reject_rate"], v["escape_rate"], 0.0)
        for p, v in u["three_way"].items():
            pts[f"3way={p}"] = (v["auto_false_reject_rate"], v["escape_rate"], v["per_1000"]["review_items"])
        return pts

    def summarise(per_seed_by_unit: dict, sizes_by_unit: dict | None, cluster: bool):
        """per_seed_by_unit: {unit: [points per seed]} -> {point: (fr, es, review, fr_lo, fr_hi, es_lo, es_hi)}"""
        res = {}
        keys = next(iter(per_seed_by_unit.values()))[0].keys()
        for key in keys:
            unit_mean = {u: [st.fmean(p[key][i] for p in pts) for i in range(3)] for u, pts in per_seed_by_unit.items()}
            fr = st.fmean(v[0] for v in unit_mean.values())
            es = st.fmean(v[1] for v in unit_mean.values())
            rv = st.fmean(v[2] for v in unit_mean.values())
            if cluster:
                fr_lo, fr_hi = cluster_bootstrap_mean([v[0] for v in unit_mean.values()])
                es_lo, es_hi = cluster_bootstrap_mean([v[1] for v in unit_mean.values()])
            else:
                (u, (n0, n1)), = sizes_by_unit.items()
                fr_lo, fr_hi = clopper_pearson(fr * n0, n0)
                es_lo, es_hi = clopper_pearson(es * n1, n1)
            res[key] = (fr, es, rv, fr_lo, fr_hi, es_lo, es_hi)
        return res

    k5 = [u for u in p15["units"] if u["K"] == 5]
    visa = collections.defaultdict(list)
    mtd = collections.defaultdict(list)
    for u in k5:
        (visa if u["task"].startswith("visa__") else mtd)[u["task"]].append(unit_points(u))
    cable = {"cable": [unit_points(v) for v in cab["units"].values()]}
    res_visa, res_mtd = summarise(visa, None, True), summarise(mtd, {"mtd": sizes["mtd__mtd6way"]}, False)
    u2b_path = HERE.parents[1] / "ckpt" / "p47" / "u2b" / "u2b_risks.json"
    if u2b_path.is_file():
        # P47 U2-B: 95 % intervals combining the calibration draw (order-statistic law, one draw per task shared by the
        # bank seeds) and the test sample, at 95 % conforming parts; the points must be the same P15 values.
        u2b = json.loads(u2b_path.read_text(encoding="utf-8"))["groups"]
        rule = {"a=0.01": "two|0.01", "a=0.02": "two|0.02", "a=0.05": "two|0.05", "3way=0.1": "three|0.1",
                "3way=0.3": "three|0.3"}
        for res, grp in ((res_visa, "visa_object_macro"), (res_mtd, "mtd")):
            for key, (fr, es, rv, *_) in list(res.items()):
                e = u2b[grp]["risks"][rule[key]][str(good)]
                if abs(e["R_P"]["value"] - good * fr) > 1e-12 or abs(e["R_C"]["value"] - (1 - good) * es) > 1e-12:
                    raise SystemExit(f"fig7: U2-B point differs from P15 ({grp}, {key})")
                (pl, ph), (cl, ch) = e["R_P"]["ci95_combined"], e["R_C"]["ci95_combined"]
                res[key] = (fr, es, rv, pl / good, ph / good, cl / (1 - good), ch / (1 - good))
    panels = [("VisA (11 objects), 5 shots", res_visa,
               (p14["visa_per_1000_at_95pct_good"]["false_rejects_per_1000"],
                p14["visa_per_1000_at_95pct_good"]["escapes_per_1000"])),
              ("Magnetic tile, 5 shots", res_mtd,
               (p14["mtd_per_1000_at_95pct_good"]["false_rejects_per_1000"],
                p14["mtd_per_1000_at_95pct_good"]["escapes_per_1000"])),
              ("Cable, held-out, 8 shots", summarise(cable, {"cable": cab_n}, False), None)]
    argmax_cable = cab["classification_argmax"][ARR]
    panels[2] = (panels[2][0], panels[2][1], (1000 * good * argmax_cable["false_reject_rate"]["mean"],
                                              1000 * (1 - good) * argmax_cable["escape_rate"]["mean"]))
    # Each operating point has its own marker (no text inside the axes); error bars are light and drawn underneath.
    marker = {"a=0.01": dict(marker="o", mfc=INK), "a=0.02": dict(marker="^", mfc=INK), "a=0.05": dict(marker="D", mfc=INK, ms=4.2),
              "3way=0.1": dict(marker="s", mfc="white"), "3way=0.3": dict(marker="p", mfc="white", ms=5.6)}
    names = {"a=0.01": "Screening, α = 1%", "a=0.02": "Screening, α = 2%", "a=0.05": "Screening, α = 5%",
             "3way=0.1": "With review band, 0.1", "3way=0.3": "With review band, 0.3"}
    fig, axes = plt.subplots(1, 3, figsize=(W2, 7.6 * CM), gridspec_kw=dict(wspace=0.38))
    out = []
    for ax, (title, res, argmax) in zip(axes, panels):
        alphas = [k for k in res if k.startswith("a=")]
        xs = [1000 * good * res[k][0] for k in alphas]
        ys = [1000 * (1 - good) * res[k][1] for k in alphas]
        ax.plot(xs, ys, color=INK, lw=0.8, zorder=2)
        for k in res:
            fr, es, rv, fl, fh, el, eh = res[k]
            x, y = 1000 * good * fr, 1000 * (1 - good) * es
            xe = [[x - 1000 * good * fl], [1000 * good * fh - x]]
            ye = [[y - 1000 * (1 - good) * el], [1000 * (1 - good) * eh - y]]
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=G2, elinewidth=0.6, capsize=1.5, capthick=0.5,
                        zorder=1)
            m = {"ms": 4.8, "mec": INK, "mew": 0.8, **marker[k]}
            ax.plot(x, y, ls="", zorder=3, **m)
            out.append([title, k, round(x, 2), round(1000 * good * fl, 2), round(1000 * good * fh, 2), round(y, 2),
                        round(1000 * (1 - good) * el, 2), round(1000 * (1 - good) * eh, 2), round(rv, 1)])
        xmax = max(1000 * good * res[k][4] for k in res) * 1.12 + 3
        if argmax:
            ax_x, ax_y = argmax
            out.append([title, "no screening (type classifier only)", round(ax_x, 2), "", "", round(ax_y, 2), "", "", 0])
            if ax_x <= xmax * 2.2:
                xmax = max(xmax, ax_x * 1.12)
                ax.plot(ax_x, ax_y, "X", color=G2, mec=INK, mew=0.4, ms=6, clip_on=False, zorder=4)
            else:
                ax.text(0.97, 0.05, f"No screening\n(off scale):\n{ax_x:.0f} false rejects,\n{ax_y:.1f} escapes",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="0.25")
        ax.set_xlim(-0.04 * xmax, xmax)
        ax.set_ylim(bottom=-0.03 * ax.get_ylim()[1])
        ax.set_title(title, pad=3)
        ax.set_xlabel("False rejects per 1000 parts")
        hgrid(ax)
        hgrid(ax, "x")
    axes[0].set_ylabel("Escapes per 1000 parts")
    for ax, s in zip(axes, "abc"):
        panel_label(ax, f"({s})", x=-0.1, y=1.09)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", color=INK, **{"ms": 4.8, "mec": INK, "mew": 0.8, **marker[k]}) for k in names]
    handles.append(Line2D([], [], marker="X", ls="", color=G2, mec=INK, mew=0.4, ms=6))
    fig.legend(handles, list(names.values()) + ["No screening"], loc="lower center", ncol=3, fontsize=8.5,
               columnspacing=1.2, handletextpad=0.3, handlelength=1.2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.31)
    NOTES.append("fig7: rates per 1000 parts with 95% conforming parts; intervals: VisA and MTD = P47 U2-B (calibration draw by the "
                 "order-statistic law, one draw per task shared by the bank seeds, combined with the test sample: VisA "
                 "object-cluster bootstrap, MTD stratified query bootstrap); Cable = Clopper-Pearson on the evaluation "
                 "set (50 good / 28 defective), mean over support draws" if u2b_path.is_file() else
                 "fig7: rates per 1000 parts with 95% conforming parts; intervals: VisA = object-cluster bootstrap (11 objects, "
                 "seeds averaged); MTD and Cable = Clopper-Pearson on the evaluation set (MTD 476 good / 194 defective, "
                 "Cable 50 / 28), mean over support draws. Provisional until P47 U2")
    save(fig, "fig7_operating_points", ["panel", "point", "false_rejects_per_1000", "fr_lo", "fr_hi",
                                        "escapes_per_1000", "es_lo", "es_hi", "review_per_1000"], out)


# --------------------------------------------------------------------------- figure 9: accuracy and deployment cost

def fig9_cost():
    acc = {r["block"]: r for r in old_csv(OLD_MAIN, "fig10_accuracy_and_cost")}
    cost = load("p26_cloud/rtx4090_r3/cost_results.json")["summary"]
    label = {"ours_main": "Proposed, main features", "ours_l14_336_fp32": "Proposed, CLIP-L/14 only",
             "ours_l14_336_fp16": "Proposed, CLIP-L/14 only, half precision", "ours_b336_fp32": "Proposed, DINOv2-B",
             "ours_s336_fp32": "Proposed, DINOv2-S", "ours_clipb16_336": "Proposed, CLIP-B/16 at 336 px",
             "ours_clipb16_224": "Proposed, CLIP-B/16 at 224 px", "restricted_adaptation": "Trainable adapter",
             "cliplora_b16_224": "CLIP-LoRA, B/16 at 224 px", "cliplora_b16_336": "CLIP-LoRA, B/16 at 336 px",
             "cliplora_l14_336": "CLIP-LoRA, L/14 at 336 px"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 7.0 * CM), sharey=True, gridspec_kw=dict(wspace=0.08))
    out = []
    for key, lab in label.items():
        passes = cost[key]
        fw = [passes[p]["model_forward_ms"] for p in sorted(passes)]
        tr = [passes[p]["prep_or_train_s"] for p in sorted(passes)]
        y = float(acc[key]["mvtec_ba_mean_over_K"])
        ours = key.startswith("ours")
        mk = dict(fmt="o" if ours else ("^" if key.startswith("clip") else "s"), color=INK if ours else G1,
                  mfc=INK if key == "ours_main" else ("white" if not ours else G2), ms=4.5, elinewidth=0.8, capsize=0)
        a.errorbar(st.fmean(fw), y, xerr=[[st.fmean(fw) - min(fw)], [max(fw) - st.fmean(fw)]], **mk)
        b.errorbar(st.fmean(tr), y, xerr=[[st.fmean(tr) - min(tr)], [max(tr) - st.fmean(tr)]], **mk)
        out.append([lab, round(y, 2), *[round(v, 3) for v in fw], *[round(v, 2) for v in tr]])
        place = {"ours_main": (-5, 0, "right", "center"), "ours_clipb16_224": (5, -1, "left", "top"),
                 "ours_s336_fp32": (5, -1, "left", "top"), "cliplora_l14_336": (0, 5, "center", "bottom"),
                 "cliplora_b16_224": (6, 0, "left", "center"), "restricted_adaptation": (0, -6, "center", "top")}
        if key in place:
            dx, dy, ha, va = place[key]
            short = lab.replace(" at 224 px", "").replace(" at 336 px", "").replace("Proposed, ", "")
            a.annotate(short, (st.fmean(fw), y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, ha=ha,
                       va=va)
    b.set_xscale("log")
    b.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    b.set_xticks([], minor=True)
    a.set_xlabel("Model forward time per image (ms)")
    b.set_xlabel("Time to add a new inspection task (s)")
    a.set_ylabel("Balanced accuracy on MVTec-FS (%)\nmean over 1, 3 and 5 shots")
    a.set_xlim(0, 25)
    a.set_ylim(64, 82)
    for ax, s in ((a, "(a)"), (b, "(b)")):
        hgrid(ax)
        hgrid(ax, "x")
        panel_label(ax, s, x=-0.04 if s == "(b)" else -0.15)
    from matplotlib.lines import Line2D
    b.legend([Line2D([], [], marker="o", color=INK, ls=""), Line2D([], [], marker="o", color=INK, mfc=G2, ls=""),
              Line2D([], [], marker="s", color=G1, mfc="white", ls=""),
              Line2D([], [], marker="^", color=G1, mfc="white", ls="")],
             ["Proposed, main features", "Proposed, other feature sets", "Trainable adapter",
              "CLIP-LoRA, official code"], loc="center", bbox_to_anchor=(0.55, 0.5), fontsize=8.5)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.16)
    NOTES.append("fig9: RTX 4090, batch 1; horizontal bars = range of the two passes (forward and reverse order). "
                 "Jetson panel to be added from P47 U6-D")
    save(fig, "fig9_cost", ["configuration", "ba_mean_over_K_pct", "forward_ms_pass1", "forward_ms_pass2",
                            "new_task_s_pass1", "new_task_s_pass2"], out)


# --------------------------------------------------------------------------- supplementary bars

def figS_false_rejects():
    order = [("simpleshot_cl2n", "SimpleShot (whole image)"), ("fused_xclass_clip18", "Cross-class contrast"),
             ("fused_cons_soft_supq90", "Support consensus"), ("fused_normref_clip18", "Fixed 10% region"),
             ("fused_adareg_regres_soft_q99", "Proposed method"), ("ours", "Trainable support adapter"),
             ("mvrec_rgb_adaptation", "MVREC (RGB)")]
    rows = old_csv(OLD_SUPP, "figS10-1_false_rejects")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 6.6 * CM), sharey=True, gridspec_kw=dict(wspace=0.08))
    out = []
    shade = {"1": G3, "3": G2, "5": G1}
    for ax, (dsn, title) in zip(axes, (("VisA (object macro)", "VisA (11 objects)"), ("MTD", "Magnetic tile (MTD)"))):
        for i, (m, lab) in enumerate(order):
            for j, k in enumerate("135"):
                v = next(float(r["false_rejects_per_1000"]) for r in rows
                         if r["dataset"] == dsn and r["method"] == m and r["K"] == k)
                ax.barh(-i + (1 - j) * 0.26, v, height=0.24, color=shade[k], edgecolor="white", lw=0.4,
                        label=f"{k} shot{'s' if k != '1' else ''}" if i == 0 else None)
                out.append([title, lab, k, v])
        ax.set_title(title, pad=3)
        ax.set_xlabel("False rejects per 1000 parts\n(type classifier only, 95% conforming parts)")
        hgrid(ax, "x")
    axes[0].set_yticks([-i for i in range(len(order))], [o[1] for o in order])
    axes[0].tick_params(axis="y", length=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.0))
    fig.subplots_adjust(left=0.28, right=0.98, top=0.84, bottom=0.2)
    save(fig, "figS_false_rejects_without_screening", ["dataset", "method", "K", "false_rejects_per_1000"], out)


def figS_per_product():
    rows = old_csv(OLD_SUPP, "figS14-1_per_product")
    rows.sort(key=lambda r: float(r["dual_branch_minus_simpleshot_pp"]))
    fig, ax = plt.subplots(figsize=(W1, 9.0 * CM))
    cols = [("dual_branch_minus_simpleshot_pp", "Proposed method", INK),
            ("fixed_normref_minus_simpleshot_pp", "Fixed-ratio normal reference", G2),
            ("consensus_minus_simpleshot_pp", "Support consensus (no conforming images)", G3)]
    out = []
    for i, r in enumerate(rows):
        for j, (c, lab, col) in enumerate(cols):
            ax.barh(i + (1 - j) * 0.27, float(r[c]), height=0.25, color=col, edgecolor="white", lw=0.3,
                    label=lab if i == 0 else None)
        out.append([r["product"], *[float(r[c]) for c, _, _ in cols]])
    ax.set_yticks(range(len(rows)), [PRODUCT.get(r["product"], r["product"]) for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("Gain over SimpleShot (percentage points)\nmean over shots and seeds")
    hgrid(ax, "x")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(pad=0.3)
    save(fig, "figS_per_product", ["product", "proposed_pp", "fixed10_pp", "consensus_pp"], out)


def figS_evidence_purity():
    rows = old_csv(OLD_SUPP, "figS12-1_evidence_purity")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 8.4 * CM), gridspec_kw=dict(wspace=0.32, width_ratios=[1.25, 1]))
    series = [("global", "Whole-image mean", dict(color=G2, ls="-", marker="x")),
              ("fixed_hard", "Fixed top 10%, uniform", dict(color=G1, ls=(0, (1, 1.2)), marker="o", mfc="white")),
              ("ours_hard", "Normal-calibrated, uniform", dict(color=G1, ls=(0, (4, 1.5)), marker="s", mfc="white")),
              ("mebin_soft", "MEBin (per image), weighted", dict(color=G2, ls=(0, (5, 1.5, 1, 1.5)), marker="D", mfc="white")),
              ("ours_soft", "Normal-calibrated, weighted (proposed)", dict(color=INK, ls="-", marker="o", lw=1.6))]
    bins = ["0.0-0.02", "0.02-0.05", "0.05-0.15", "0.15-1.01"]
    ticks = ["< 2%", "2–5%", "5–15%", "≥ 15%"]
    out = []
    for key, lab, sty in series:
        v = [float(next(r["value"] for r in rows if r["panel"] == "purity" and r["series"] == key and r["bin"] == bn))
             for bn in bins]
        a.plot(range(4), v, label=lab, **sty)
        out += [["purity", lab, t, x] for t, x in zip(ticks, v)]
    a.set_xticks(range(4), ticks)
    a.set_xlabel("Defect share of the image's patches")
    a.set_ylabel("Weight on defect patches")
    a.set_ylim(0, 1)
    fig.legend(*a.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, fontsize=8.5,
               columnspacing=1.2)
    hgrid(a)
    ex = [r for r in rows if r["panel"] == "exceedance"]
    lab = ["1", "2", "3", "4–5", "≥ 6"]
    vals = [100 * float(r["value"]) for r in ex]
    b.bar(range(5), vals, color=G2, width=0.62, edgecolor="none")
    for i, v in enumerate(vals):
        b.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=8.5)
    b.axhline(1.05, color=INK, lw=0.7, ls=(0, (4, 2)), label="Normal images: 1.05%")
    b.legend(loc="upper right", fontsize=8.5)
    b.set_xticks(range(5), lab)
    b.set_xlabel("Distance to the defect (patches)")
    b.set_ylabel("Background patches\nabove the threshold (%)")
    hgrid(b)
    out += [["exceedance", "distance", l_, v] for l_, v in zip(lab, vals)]
    panel_label(a, "(a)", x=-0.12)
    panel_label(b, "(b)", x=-0.2)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.36)
    NOTES.append("figS_evidence_purity: development split; the defect share uses ground-truth masks (evaluation only)")
    save(fig, "figS_evidence_purity", ["panel", "series", "bin", "value"], out)


# --------------------------------------------------------------------------- main

FIGURES = {"fig2": fig2_headroom, "fig3": fig3_gains, "fig4": fig4_steps, "fig6": fig6_frr_theory,
           "fig7": fig7_operating_points, "fig9": fig9_cost, "figS_visa": figS_visa_forest,
           "figS_threshold": figS_threshold_source, "figS_robustness": figS_robustness, "figS_backbone": figS_backbone,
           "figS_false_rejects": figS_false_rejects, "figS_per_product": figS_per_product,
           "figS_purity": figS_evidence_purity}


def main(argv=None) -> int:
    names = (argv or sys.argv[1:]) or list(FIGURES)
    for n in names:
        FIGURES[n]()
    if NOTES:
        (OUT / "NOTES.txt").write_text("\n".join(NOTES) + "\n", encoding="utf-8")
        print("\n".join(f"[note] {n}" for n in NOTES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
