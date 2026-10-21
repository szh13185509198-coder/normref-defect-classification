#!/usr/bin/env python3
"""Headroom figure (bottleneck, MVTec-FS K = 5) and Fig. 2 (method overview) for paper v5.

Fig. 1 (information boundary, query access) is the draw.io figure
docs/paper/figures/editable/FIGURE1_INFORMATION_BOUNDARY_V4.drawio.

Images: MVTec AD / MVTec-FS files only (cable for Fig. 1, metal_nut for Fig. 2); the soft-weight panel of Fig. 2 is
cropped from the already published region-comparison figure (same query image, same method). Numbers in Fig. 1(c)
are read from the result files (MVTec-FS formal, K = 5). Writes PNG / PDF to docs/paper/figures_v5/.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p20.make_figures_method
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p21 import score as p21  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "paper" / "figures_v5"
MVTEC = Path(r"D:\intergration\MVTEC")
FS = PROJECT_ROOT / "data" / "mvtec-fs" / "image"
REGION_FIG = PROJECT_ROOT / "docs" / "paper" / "figures_v4" / "fig_region_comparison.png"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def thumb(path, size=220):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    return np.asarray(im)


def box(ax, x, y, w, h, text="", fc="white", ec="0.2", lw=0.8, ls="-", fs=7, weight="normal", va="center", r=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ty = y + h / 2 if va == "center" else y + h - 1.2
        ax.text(x + w / 2, ty, text, ha="center", va="center" if va == "center" else "top", fontsize=fs,
                fontweight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, ls="-", color="0.15", lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0))


def image_at(ax, img, x, y, w, h, border="0.3"):
    sub = ax.inset_axes([x, y, w, h], transform=ax.transData)
    sub.imshow(img)
    sub.set_xticks([])
    sub.set_yticks([])
    for s in sub.spines.values():
        s.set_color(border)
        s.set_linewidth(0.6)
    return sub


def ba_k5() -> tuple[dict, dict]:
    """K = 5 BA (%) on MVTec-FS-RGB: the main frozen features, and the CLIP-B/16 @ 224 backbone of official CLIP-LoRA."""
    lv = rc.load_json(PROJECT_ROOT / "ckpt" / "p21" / "mvtec" / "p21_results.json")["levels_pct"]
    grid = rc.metric_grid_from_rows(rc.load_json(PROJECT_ROOT / "ckpt" / "p14_formal" / "mvtec" / "p14_mvtec_results.json")["raw_metrics"])
    grid.update(rc.metric_grid_from_rows([r for r in rc.load_json(PROJECT_ROOT / "ckpt" / "p11_formal" / "mvtec" / "p11_formal_mvtec_results.json")["raw_metrics"]
                                          if r["method"] == "oracle_maskpool_clip18"]))
    grid.update(rc.metric_grid_from_rows([r for r in rc.load_json(rc.C2_RESULTS)["raw_metrics"] if r["method"] == "logreg_rfs"]))
    grid.update(rc.metric_grid_from_rows(rc.load_json(p21.P24)["raw_metrics"]))
    k5 = lambda m: p21.summary(grid, m)["5"]["balanced_accuracy"]["mean"]  # noqa: E731
    lv29 = rc.load_json(PROJECT_ROOT / "ckpt" / "p29" / "mvtec" / "p29_results.json")["levels_pct"]
    main = {"SimpleShot": lv["simpleshot_cl2n"]["5"]["balanced_accuracy"]["mean"],
            "CLIP-LoRA (official code; fine-tunes CLIP-L/14@336)": lv29["cliplora_l14_336"]["5"]["balanced_accuracy"]["mean"],
            "Logistic regression": k5("logreg_rfs"),
            "Trainable support-side adaptation": lv["ours"]["5"]["balanced_accuracy"]["mean"],
            "Cross-class contrast (C0)": lv["fused_xclass_clip18"]["5"]["balanced_accuracy"]["mean"],
            "Consensus background (C0)": lv["fused_cons_soft_supq90"]["5"]["balanced_accuracy"]["mean"],
            "Fixed-fraction normal reference (C1)": k5("fused_normref_clip18"),
            "Ours: dual-branch, normal-referenced (C1)": lv["fused_adareg_regres_soft_q99"]["5"]["balanced_accuracy"]["mean"],
            "Mask oracle (evaluation only)": k5("oracle_maskpool_clip18")}
    b16 = {"SimpleShot": k5("ss_clip_b16_r224"),
           "CLIP-LoRA (official code; fine-tunes the encoders)": lv["cliplora_b16"]["5"]["balanced_accuracy"]["mean"],
           "Ours: dual-branch, zero training (C1)": k5("arr_clip_b16_r224"),
           "Mask oracle (evaluation only)": k5("oracle_clip_b16_r224")}
    return main, b16


def fig_headroom():
    """Where the headroom is (MVTec-FS-RGB formal, K = 5). Top: on the main CLIP-L/14@336 backbone, classifier changes on
    the same frozen features, official CLIP-LoRA fine-tuned on that CLIP (P29), better local evidence and the mask oracle.
    Bottom: official CLIP-LoRA against methods on its default backbone (CLIP-B/16, 224 px)."""
    main, b16 = ba_k5()
    x0 = 56
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sections = [("Main backbone CLIP-L/14@336 (frozen-feature methods add DINOv2-S)", main,
                 [("Classifier changes on the same frozen features", ["SimpleShot", "Logistic regression",
                                                                      "Trainable support-side adaptation"], "0.8"),
                  ("Fine-tuning the encoder (CLIP only, uses class names)", ["CLIP-LoRA (official code; fine-tunes CLIP-L/14@336)"], None),
                  ("Better local evidence", ["Cross-class contrast (C0)", "Consensus background (C0)",
                                             "Fixed-fraction normal reference (C1)",
                                             "Ours: dual-branch, normal-referenced (C1)"], "0.45"),
                  ("Upper bound", ["Mask oracle (evaluation only)"], "white")]),
                ("Same backbone as official CLIP-LoRA (CLIP-B/16, 224 px)", b16,
                 [(None, list(b16), None)])]
    shade = {"SimpleShot": "0.8", "Mask oracle (evaluation only)": "white"}
    from matplotlib.transforms import blended_transform_factory
    left = blended_transform_factory(fig.transFigure, ax.transData)
    y, ticks, labels = 0, [], []
    for si, (title, v, groups) in enumerate(sections):
        if si:
            y -= 0.35
            ax.plot([0.015, 0.985], [y + 0.55] * 2, color="0.55", lw=0.8, transform=left, clip_on=False)
        ax.text(0.015, y, title, fontsize=7.6, fontweight="bold", va="center", transform=left)
        y -= 1
        top = y + 0.5
        for gname, names, color in groups:
            if gname:
                ax.text(0.035, y, gname, fontsize=7, style="italic", va="center", transform=left)
                y -= 1
            for n in names:
                c = color or shade.get(n, "0.62")
                c = "0.0" if n.startswith("Ours") else c
                ax.barh(y, v[n], color=c, edgecolor="0.2", lw=0.6, hatch="////" if c == "white" else None, height=0.72)
                ax.text(v[n] + 0.25, y, f"{v[n]:.1f}", va="center", fontsize=6.8)
                ticks.append(y)
                labels.append(n)
                y -= 1
            y -= 0.3
        ax.plot([v["SimpleShot"]] * 2, [top, y + 0.8], color="0.4", lw=0.7, ls=":")
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.5, 0.9)
    ax.set_xlim(x0, 94)
    ax.set_xlabel("MVTec-FS-RGB balanced accuracy (%), K = 5", fontsize=7.5)
    ax.grid(axis="x", color="0.9")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_headroom_v5.{ext}", dpi=300)
    plt.close(fig)
    return main, b16


def fig2():
    fig = plt.figure(figsize=(7.2, 6.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 94)
    ax.axis("off")
    nut = FS / "metal_nut"
    # inputs -------------------------------------------------------------------------------------------------
    box(ax, 1, 69, 17, 24, fc="0.95", ec="0.4")
    ax.text(9.5, 91.6, "Support set", ha="center", fontsize=7, fontweight="bold")
    for i, c in enumerate(("bent", "color", "flip", "scratch")):
        f = sorted((nut / "training" / c).glob("*.png"))[0]
        image_at(ax, thumb(f), 2.0 + (i % 2) * 8, 80.3 - (i // 2) * 10.4, 7.0, 7.4)
        ax.text(5.5 + (i % 2) * 8, 79.9 - (i // 2) * 10.4, c, ha="center", va="top", fontsize=5.4)
    box(ax, 1, 44, 17, 22, fc="0.95", ec="0.4")
    ax.text(9.5, 64.6, "Query image", ha="center", fontsize=7, fontweight="bold")
    image_at(ax, thumb(nut / "testing" / "bent" / "013.png"), 3.5, 45.5, 12, 16.5, border="0.0")
    box(ax, 1, 13, 17, 28, fc="white", ec="0.0", lw=1.3)
    ax.text(9.5, 39.6, "5–20 normal images", ha="center", fontsize=7, fontweight="bold")
    ax.text(9.5, 37.3, "(unlabeled, \"good\" only)", ha="center", fontsize=5.8)
    for i in range(4):
        image_at(ax, thumb(MVTEC / "metal_nut" / "train" / "good" / f"00{i}.png"), 2.2 + (i % 2) * 7.8,
                 25.5 - (i // 2) * 10.0, 7.0, 8.6)
    # backbone and global branch ---------------------------------------------------------------------------------
    box(ax, 22, 55, 17, 38, "Frozen backbones\n\nCLIP ViT-L/14@336\nlayer-18 patch tokens t\n+ global embedding\n\n"
        "DINOv2 ViT-S/14\npatch mean\n\n(no training)", fc="0.88", ec="0.3", fs=6.5)
    arrow(ax, 18, 81, 22, 81)
    arrow(ax, 18, 55, 22, 64)
    box(ax, 43, 81, 22, 12, "Global fused feature g(x)\n→ SimpleShot score S", fc="white", ec="0.3", fs=6.8)
    arrow(ax, 39, 85, 43, 87)
    box(ax, 43, 56, 21, 21, "C0 (no normal images):\nsupport-only local contrast\n\ncross-class contrast, or\n"
        "consensus background\nb(t) → a₀ = 1 − b,\nτ₀ = Q₀.₉₀ → soft weights", fc="white", ec="0.5", ls="--", fs=5.8)
    arrow(ax, 39, 66, 43, 66, ls="--", color="0.45")
    # C1: the normal bank sets the evidence ------------------------------------------------------------------
    box(ax, 21.5, 13, 58.5, 39.5, fc="0.96", ec="0.0", lw=1.2)
    ax.text(55, 51.0, "C1 (main): the normal bank sets the evidence", ha="center", fontsize=7.5, fontweight="bold",
            va="center")
    box(ax, 23, 37, 16, 10, "Normal patch bank B\n(M × 576 patches)", fc="white", ec="0.0", fs=6.5)
    arrow(ax, 18, 27, 23, 40)
    arrow(ax, 26, 55, 26, 47)
    ax.text(25.4, 53.8, "patches t", fontsize=5.6, ha="right", va="center")
    roles = [("① Score", "a(t) = 1 − max_b cos(t, b)"), ("② Gate", "τ = Q₀.₉₉ of leave-one-out\nnormal-patch scores"),
             ("③ Weight", "w(t) = max(a(t) − τ, 0)\n(< 6 patches: top-6 fallback)")]
    for i, (r, f) in enumerate(roles):
        y = 42.5 - i * 9.4
        box(ax, 41, y - 3.8, 20, 8.2, fc="white", ec="0.3")
        ax.text(42, y + 3.4, r, fontsize=6.8, fontweight="bold", va="top")
        ax.text(51, y - 0.7, f, ha="center", va="center", fontsize=5.8)
    arrow(ax, 39, 42, 41, 42)
    # example of the weights on the query
    reg = Image.open(REGION_FIG).convert("RGB")
    W, H = reg.size
    crop = reg.crop((int(W * 0.8075), int(H * 0.7065), int(W * 0.9885), int(H * 0.9835)))
    image_at(ax, np.asarray(crop), 63.5, 16.5, 14.5, 14.5, border="0.0")
    ax.text(70.75, 14.8, "soft weights on the query", ha="center", va="center", fontsize=5.4)
    arrow(ax, 61, 23.8, 63.5, 23.8)
    # descriptors (same weights)
    box(ax, 62.5, 42, 16.5, 8, "Appearance\nh_A = Σ w t / Σ w", fc="white", ec="0.0", fs=6.2)
    box(ax, 62.5, 32.5, 16.5, 8.5, "Deviation\nh_R = Σ w (t − nn_B(t)) / Σ w", fc="white", ec="0.0", fs=5.6)
    arrow(ax, 70.75, 31, 70.75, 32.5)
    ax.text(72.0, 31.6, "same w", fontsize=5.3, va="center")
    # fusion
    box(ax, 84, 35, 15, 30, "z-score fusion\n\nargmax_c\nz(S) + z(SS[h_A])\n+ z(SS[h_R])\n\n→ defect type ŷ",
        fc="0.88", ec="0.0", lw=1.2, fs=6.4)
    arrow(ax, 65, 87, 91.5, 65)
    arrow(ax, 64, 66.5, 84, 60, ls="--", color="0.45")
    arrow(ax, 79, 46, 84, 51)
    arrow(ax, 79, 36.7, 84, 41)
    # application strip: two-stage inspection ----------------------------------------------------------------
    box(ax, 21.5, 1, 77.5, 9.5, fc="white", ec="0.5")
    ax.text(28, 5.75, "Application:\ntwo-stage\ninspection", ha="center", va="center", fontsize=6.3, fontweight="bold")
    steps = [(34.5, "image-level score\n(same normal bank)"), (51, "split-conformal\nthreshold (set α)"),
             (67.5, "pass / manual review\n/ reject"), (84, "rejected → defect\ntype (as above)")]
    for i, (x, t) in enumerate(steps):
        box(ax, x, 2.5, 14, 6.5, t, fc="0.96", ec="0.6", fs=5.6)
        if i:
            arrow(ax, x - 2.5, 5.75, x, 5.75)
    arrow(ax, 31, 37, 38, 9)
    arrow(ax, 91.5, 35, 91.5, 9)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_method_v5.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # fig2() (the matplotlib method overview) is superseded by the draw.io Figure 3 (FIGURE3_CORE_METHOD_V4.drawio)
    print(fig_headroom())
    print("saved fig_headroom_v5")
