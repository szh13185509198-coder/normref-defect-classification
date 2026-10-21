#!/usr/bin/env python3
"""Paper figures from frozen P10 result files (docs/P10_FIGURES_AND_SENSITIVITY_PROTOCOL.md §2).

Every figure also writes the plotted numbers to a CSV next to it. Output directory:
``docs/paper/figures_v2/``.

Usage:
    D:\\Anaconda\\envs\\cole\\python.exe -m experiments.p10.make_figures --mvtec-root D:\\intergration\\MVTEC --device cuda
"""

from __future__ import annotations

import argparse
import csv
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
import torch.nn.functional as F  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.p10 import run_pilot_a as ra  # noqa: E402
from experiments.p10.formal_evaluate import RESULTS_DIR as FORMAL  # noqa: E402
from experiments.p10.formal_extract import OUT_DIR as FORMAL_FEATURES  # noqa: E402
from experiments.revision import revision_common as rc  # noqa: E402

OUT = PROJECT_ROOT / "docs" / "paper" / "figures_v2"
SUPP = PROJECT_ROOT / "ckpt" / "p10_formal" / "supplement_s" / "supplement_s_results.json"
EXTERNAL = PROJECT_ROOT / "ckpt" / "p10_external" / "results" / "external_results.json"
NORMAL_AWARE = PROJECT_ROOT / "ckpt" / "p10_external" / "normal_aware" / "normal_aware_results.json"
CABLE = PROJECT_ROOT / "ckpt" / "p10_cable" / "results" / "cable_results.json"
CABLE_IMAGES = PROJECT_ROOT / "data" / "mvtecad-cable"
K = (1, 3, 5)

LABELS = {
    "oracle_maskpool_clip18": "Mask oracle (upper bound)",
    "fused_normref_clip18": "Normal reference (C1)",
    "residual_mean_clip18": "Normal residual (C1)",
    "fused_xclass_clip18": "Cross-class contrast (C0)",
    "simpleshot_cl2n": "SimpleShot (C0)",
    "ours": "Support-side adaptation (C0)",
    "mvrec_rgb": "MVREC-RGB adaptation",
    "mvrec_rgb_adaptation": "MVREC-RGB adaptation",
}
STYLE = {
    "oracle_maskpool_clip18": dict(color="0.55", ls=(0, (4, 2)), lw=1.6, marker="*", ms=9, mfc="white"),
    "fused_normref_clip18": dict(color="0.0", ls="-", lw=2.4, marker="o", ms=7, mfc="0.0"),
    "residual_mean_clip18": dict(color="0.25", ls=(0, (1, 1.5)), lw=1.8, marker="D", ms=5.5, mfc="white"),
    "fused_xclass_clip18": dict(color="0.3", ls="-", lw=2.0, marker="s", ms=6.5, mfc="0.3"),
    "simpleshot_cl2n": dict(color="0.5", ls="-.", lw=1.5, marker="^", ms=7, mfc="white"),
    "ours": dict(color="0.62", ls="--", lw=1.5, marker="v", ms=6.5, mfc="0.62"),
    "mvrec_rgb": dict(color="0.75", ls=":", lw=1.5, marker="x", ms=6.5, mfc="0.75"),
}


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name, rows, header):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[fig] {name}")


def fig_line():
    formal = rc.load_json(FORMAL / "formal_results.json")["product_macro_summary"]
    formal["residual_mean_clip18"] = rc.load_json(SUPP)["product_macro_summary"]["residual_mean_clip18"]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    rows = []
    for m, st in STYLE.items():
        mean = [100 * formal[m][str(k)]["balanced_accuracy"]["mean"] for k in K]
        sd = [100 * formal[m][str(k)]["balanced_accuracy"]["std_ddof1"] for k in K]
        ax.plot(K, mean, label=LABELS[m], color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"],
                ms=st["ms"], mfc=st["mfc"], mec=st["color"])
        rows += [[m, k, round(v, 4), round(s, 4)] for k, v, s in zip(K, mean, sd)]
    ax.set_xticks(K)
    ax.set_xlabel("Shots per class, K")
    ax.set_ylabel("Product-macro balanced accuracy (%)")
    ax.set_ylim(40, 95)
    ax.grid(axis="y", color="0.9")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right", ncol=2)
    ax.set_title("MVTec-FS-RGB (13 products, 5 seeds)", fontsize=9.5)
    save(fig, "fig_mvtecfs_line", rows, ["method", "K", "ba_mean_pct", "ba_sd_pct"])


def fig_per_product():
    raw = rc.load_json(FORMAL / "formal_results.json")["raw_metrics"]
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in raw}
    def gain(m, p):
        return 100 * statistics.fmean(cell[(m, p, k, s)] - cell[("simpleshot_cl2n", p, k, s)] for k in K for s in rc.SEEDS)
    products = sorted(rc.PRODUCTS, key=lambda p: gain("fused_normref_clip18", p))
    y = np.arange(len(products))
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    c1 = [gain("fused_normref_clip18", p) for p in products]
    c0 = [gain("fused_xclass_clip18", p) for p in products]
    ax.barh(y + 0.19, c1, height=0.38, color="0.15", label=LABELS["fused_normref_clip18"])
    ax.barh(y - 0.19, c0, height=0.38, color="0.65", label=LABELS["fused_xclass_clip18"])
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks(y, products)
    ax.set_xlabel("Δ balanced accuracy vs SimpleShot (pp), mean over K and seeds")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.grid(axis="x", color="0.9")
    save(fig, "fig_mvtecfs_per_product", [[p, round(a, 4), round(b, 4)] for p, a, b in zip(products, c1, c0)],
         ["product", "normal_reference_minus_simpleshot_pp", "cross_class_minus_simpleshot_pp"])


def fig_visa_forest():
    hyp = rc.load_json(EXTERNAL)["hypotheses"]
    names = {"E1_primary_C1": "Normal reference − SimpleShot", "E3_strict_C0": "Cross-class − SimpleShot",
             "E2_attribution_C1": "Normal reference − normal centering"}
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    rows, ypos, ticks = [], 0, []
    for key, label in names.items():
        c = hyp[key]["visa"]["balanced_accuracy"]
        entries = [(f"K={k}", c["per_K"][str(k)]["D"], c["per_K"][str(k)]["ci95"]) for k in K] + \
                  [("mean", c["Dbar"], c["Dbar_ci95"])]
        for sub, d, ci in entries:
            color = "0.0" if sub == "mean" else "0.45"
            ax.errorbar(100 * d, ypos, xerr=[[100 * (d - ci[0])], [100 * (ci[1] - d)]], fmt="o" if sub == "mean" else "s",
                        color=color, ms=6 if sub == "mean" else 4.5, capsize=2.5, lw=1.2)
            ticks.append((ypos, f"{label}  ·  {sub}"))
            rows.append([key, sub, round(100 * d, 4), round(100 * ci[0], 4), round(100 * ci[1], 4)])
            ypos -= 1
        ypos -= 0.6
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7.2)
    ax.set_xlabel("Paired Δ balanced accuracy (pp), 95% object-cluster CI (11 VisA objects)")
    ax.grid(axis="x", color="0.9")
    save(fig, "fig_visa_forest", rows, ["hypothesis", "K", "delta_pp", "ci_low_pp", "ci_high_pp"])


def fig_industrial():
    summ = rc.load_json(NORMAL_AWARE)["summary"]
    methods = ["simpleshot_cl2n", "fused_xclass_clip18", "fused_normref_clip18", "ours", "mvrec_rgb_adaptation"]
    short = {"simpleshot_cl2n": "SimpleShot\n(C0)", "fused_xclass_clip18": "Cross-class\n(C0)",
             "fused_normref_clip18": "Normal ref.\n(C1)", "ours": "Adaptation\n(C0)",
             "mvrec_rgb_adaptation": "MVREC-RGB\nadapt."}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=False)
    rows = []
    for ax, (ds, key) in zip(axes, (("VisA (object macro)", "visa_per_1000_at_95pct_good"),
                                     ("MTD", "mtd_per_1000_at_95pct_good"))):
        x = np.arange(len(methods))
        for j, k in enumerate(K):
            vals = [summ[m][str(k)][key]["false_rejects_per_1000"] for m in methods]
            esc = [summ[m][str(k)][key]["escapes_per_1000"] for m in methods]
            ax.bar(x + (j - 1) * 0.26, vals, width=0.26, color=str(0.2 + 0.25 * j), label=f"K={k}")
            rows += [[ds, m, k, round(v, 2), round(e, 2)] for m, v, e in zip(methods, vals, esc)]
        ax.set_xticks(x, [short[m] for m in methods], fontsize=6.8)
        ax.set_title(ds, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("False rejects per 1,000 parts\n(95% good, argmax decision)")
        ax.grid(axis="y", color="0.9")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "fig_industrial_false_rejects", rows, ["dataset", "method", "K", "false_rejects_per_1000", "escapes_per_1000"])


def anomaly_map(patches: torch.Tensor, bank: torch.Tensor) -> np.ndarray:
    a = 1.0 - (patches.float() @ bank.float().t()).max(dim=1).values
    side = int(round(a.numel() ** 0.5))
    a = a.reshape(1, 1, side, side)
    return F.interpolate(a, size=(336, 336), mode="bilinear", align_corners=False)[0, 0].numpy()


def draw_panel(ax_img, ax_map, image: Image.Image, amap: np.ndarray, polygons, title: str):
    w, h = image.size
    view = np.asarray(image.convert("RGB").resize((336, 336), Image.BICUBIC))
    for ax in (ax_img, ax_map):
        ax.imshow(view)
        ax.axis("off")
        for poly in polygons:
            xs = [p[0] * 336 / w for p in poly] + [poly[0][0] * 336 / w]
            ys = [p[1] * 336 / h for p in poly] + [poly[0][1] * 336 / h]
            ax.plot(xs, ys, color="lime", lw=0.9)
    ax_map.imshow(amap, cmap="inferno", alpha=0.55)
    ax_img.set_title(title, fontsize=6.4, loc="left")


def polygons_of(record) -> list:
    polys = []
    for rel in record.get("annotation_paths") or []:
        path = rc.DATASET_ROOT / rel
        if path.is_file():
            for shape in json.loads(path.read_text(encoding="utf-8")).get("shapes", []):
                pts = [tuple(map(float, p)) for p in shape.get("points", [])]
                if shape.get("shape_type") == "rectangle" and len(pts) == 2:
                    (x0, y0), (x1, y1) = pts
                    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                if len(pts) >= 3:
                    polys.append(pts)
    return polys


def fig_qualitative():
    raw = rc.load_json(FORMAL / "formal_results.json")["raw_metrics"]
    cell = {(r["method"], r["product"], r["K"], r["seed"]): r["balanced_accuracy"] for r in raw}
    gains = {p: statistics.fmean(cell[("fused_normref_clip18", p, k, s)] - cell[("simpleshot_cl2n", p, k, s)]
                                 for k in K for s in rc.SEEDS) for p in rc.PRODUCTS}
    ranked = sorted(rc.PRODUCTS, key=lambda p: -gains[p])
    products = ranked[:3] + ranked[-3:]
    preds = {}
    with (FORMAL / "formal_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["K"] == 5 and r["seed"] == 111 and r["method"] in ("fused_normref_clip18", "simpleshot_cl2n"):
                preds[(r["method"], r["product"], r["rgb_sha256"])] = (r["predicted_label"], r["true_label"])
    records = {}
    with rc.RECORDS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("source_pools") == ["query"]:
                records[(r["product"], r["rgb_sha256"])] = r
    fig, axes = plt.subplots(len(products), 4, figsize=(7.2, 1.95 * len(products)))
    rows = []
    for row, product in enumerate(products):
        qf = torch.load(FORMAL_FEATURES / f"queries_{product}.pt", map_location="cpu", weights_only=False)
        nf = torch.load(FORMAL_FEATURES / f"normals_{product}.pt", map_location="cpu", weights_only=False)
        name_index = {n: i for i, n in enumerate(nf["names"])}
        bank = nf["clip18_patches"][[name_index[n] for n in nf["draws"]["111"][:ra.M_PRIMARY]]].reshape(-1, 1024)
        order = sorted(range(len(qf["rgb_sha256"])), key=lambda i: qf["rgb_sha256"][i])
        picks = {}
        for i in order:
            sha = qf["rgb_sha256"][i]
            nr, true = preds[("fused_normref_clip18", product, sha)]
            ss, _ = preds[("simpleshot_cl2n", product, sha)]
            if "success" not in picks and nr == true and ss != true:
                picks["success"] = (i, nr, ss, true)
            if "failure" not in picks and nr != true:
                picks["failure"] = (i, nr, ss, true)
        for col, kind in enumerate(("success", "failure")):
            ax_img, ax_map = axes[row, 2 * col], axes[row, 2 * col + 1]
            if kind not in picks:
                for ax in (ax_img, ax_map):
                    ax.axis("off")
                ax_img.set_title(f"{product}: no {kind} case", fontsize=6.4, loc="left")
                continue
            i, nr, ss, true = picks[kind]
            sha = qf["rgb_sha256"][i]
            rec = records[(product, sha)]
            with Image.open(rc.DATASET_ROOT / rec["canonical_relative_path"]) as handle:
                image = handle.convert("RGB").copy()
            amap = anomaly_map(qf["clip18_patches"][i], bank)
            draw_panel(ax_img, ax_map, image, amap, polygons_of(rec),
                       f"{product} ({kind}) · true: {true}\nNR: {nr} · SimpleShot: {ss}")
            rows.append([product, kind, sha, true, nr, ss, round(gains[product] * 100, 4)])
    fig.suptitle("Normal-reference anomaly maps (K=5, seed 111; green: GT outline, visualization only)", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_qualitative_anomaly_maps", rows,
         ["product", "case", "rgb_sha256", "true", "normal_reference_pred", "simpleshot_pred", "product_gain_pp"])


def fig_cable(mvtec_root: Path, device):
    from experiments.p6 import run_p6b_stage2_support as support
    from experiments.p10.external_evaluate import extract
    res = rc.load_json(CABLE)
    persistent = [p for p in res["persistent_errors_all_seeds"]["fused_normref_clip18"] if p.startswith("combined/")][:2]
    bank_names = res["normal_bank_draws"]["111"]
    factory = support._build_lazy_clip_and_dino(SimpleNamespace(project_root=PROJECT_ROOT, device=device))
    normal_dir = mvtec_root / "cable" / "train" / "good"
    with torch.no_grad():
        _, clip = extract(factory, [CABLE_IMAGES / p for p in persistent] + [normal_dir / n for n in bank_names],
                          device, 4)
    bank = clip[len(persistent):].reshape(-1, clip.shape[-1])
    preds = {}
    with (CABLE.parent / "cable_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["method"] == "fused_normref_clip18" and r["seed"] == 111:
                preds[r["image_path"]] = r["pred"]
    fig, axes = plt.subplots(1, 2 * len(persistent), figsize=(3.6 * len(persistent), 2.0))
    rows = []
    for j, path in enumerate(persistent):
        with Image.open(CABLE_IMAGES / path) as handle:
            image = handle.convert("RGB").copy()
        draw_panel(axes[2 * j], axes[2 * j + 1], image, anomaly_map(clip[j], bank), [],
                   f"Cable {path} · true: combined\nNR (seed 111): {preds[path]}")
        rows.append([path, "combined", preds[path]])
    fig.tight_layout()
    save(fig, "fig_cable_combined_failures", rows, ["image_path", "true", "normal_reference_pred_seed111"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    setup()
    fig_line()
    fig_per_product()
    fig_visa_forest()
    fig_industrial()
    fig_qualitative()
    fig_cable(Path(args.mvtec_root), torch.device(args.device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
