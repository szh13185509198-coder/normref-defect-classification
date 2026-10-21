#!/usr/bin/env python3
"""Colour data charts as editable draw.io files, rendered to PDF / PNG.

Input: the plotted numbers that paper_figures/mst/make_mst_figures.py writes to paper_figures/mst/out/*.csv (run that
first); figure 6 is recomputed from the Beta(N - k + 1, k) theory. Output: paper_figures/mst/schematics/charts/.

    python paper_figures/mst/make_mst_figures.py
    python paper_figures/mst/schematics/make_color_charts.py

Colour roles (validated palette, at most four hues per chart; everything else grey):
    orange = proposed method; aqua = normal-referenced comparison; blue = comparisons without normal images;
    violet = trained methods (support adapter, CLIP-LoRA); grey = SimpleShot; black dashed outline = ground-truth
    mask pooling (evaluation only). Datasets: MVTec-FS blue, VisA violet, Real-IAD aqua, MTD grey.
Every series also has its own marker or a direct label, so colour is never the only cue.
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drawchart import Plot, bar, legend, line, marker  # noqa: E402
from drawio import AQUA, BLUE, F, F_TITLE, INKC, ORANGE, VIOLET, Diagram  # noqa: E402

CSV_DIR = HERE.parent / "out"
W = 565                     # content width: with the 1 px border the rendered page is 567 px = 15 cm
OUT = HERE / "charts"
GREY_M, GREY_L = "#7a7a7a", "#b8b8b8"
B250, B400, B600 = "#86b6ef", "#3987e5", "#184f95"   # ordinal blue ramp (1, 3, 5 shots)

METHOD = {  # label -> (colour, marker, filled)
    "Proposed method": (ORANGE, "o", True),
    "SimpleShot (whole image)": (GREY_M, "^", False),
    "Fixed-ratio normal reference": (AQUA, "s", False),
    "Trainable support adapter": (VIOLET, "v", True),
    "Ground-truth mask pooling (evaluation only)": (INKC, "h", False),
}
DATASET = {"MVTec-FS (13 products)": (BLUE, "o", True), "VisA (11 objects)": (VIOLET, "s", True),
           "Real-IAD, blind (30 objects)": (AQUA, "^", False)}


def rows(name):
    with (CSV_DIR / f"{name}.csv").open(encoding="utf-8") as h:
        return list(csv.DictReader(h))


# --------------------------------------------------------------------------- figure 2: headroom (bars)

def fig2():
    r = rows("fig2_headroom")
    colour = {"SimpleShot (whole image)": GREY_L, "Logistic regression": GREY_L, "Trainable support adapter": VIOLET,
              "CLIP-LoRA, official code": VIOLET, "Cross-class contrast": BLUE,
              "Support consensus (no conforming images)": BLUE, "Fixed-ratio normal reference": AQUA,
              "Proposed method": ORANGE, "Ground-truth mask pooling": "#ffffff"}
    # pass 1: vertical layout (px)
    items, cur, ROW, prev_sec, prev_grp = [], 4.0, 16.5, None, None
    for row in r:
        if row["section"] != prev_sec:
            if prev_sec is not None:
                cur += 6
                items.append(("rule", cur))
                cur += 6
            items.append(("section", cur, row["section"]))
            cur += 18
            prev_sec, prev_grp = row["section"], None
        if row["group"] and row["group"] != prev_grp:
            items.append(("group", cur, row["group"]))
            cur += 15
        prev_grp = row["group"]
        items.append(("bar", cur + ROW / 2, row["method"], float(row["balanced_accuracy_pct_K5"])))
        cur += ROW
    bottom = cur + 4
    d = Diagram("fig2_headroom", W, int(bottom + 36))
    p = Plot(d, 236, 4, 300, bottom - 4, (55, 95), (0, 1))
    ticks = [55, 60, 65, 70, 75, 80, 85, 90, 95]
    for v in ticks:
        line(d, [(p.X(v), 4), (p.X(v), bottom)], "#e6e6e6", 0.6)
        line(d, [(p.X(v), bottom), (p.X(v), bottom + 3)], "#333333", 0.8)
        d.text(p.X(v) - 20, bottom + 4, 40, 14, f"{v}", font=F, align="center")
    line(d, [(p.x, bottom), (p.x + p.w, bottom)], "#333333", 0.8)
    d.text(p.x - 40, bottom + 20, p.w + 80, 14, "Balanced accuracy on MVTec-FS (%), 5 shots per defect type", font=F,
           align="center")
    # pass 2: draw; SimpleShot reference line per section (value labels are moved past it, never drawn across it)
    ref = {}
    sec = None
    for it in items:
        if it[0] == "section":
            sec = it[2]
        elif it[0] == "bar" and it[2].startswith("SimpleShot"):
            ref[sec] = it[3]
    sec_bars = []
    for it in items:
        if it[0] == "rule":
            line(d, [(4, it[1]), (W - 4, it[1])], GREY_L, 0.6)
        elif it[0] == "section":
            d.text(4, it[1], 540, 15, it[2], font=F, bold=True)
            sec = it[2]
            sec_bars.append([])
        elif it[0] == "group":
            d.text(12, it[1], 300, 14, it[2], font=F, italic=True, color="#444444")
        else:
            _, yc, m, v = it
            c = colour[m]
            bar(d, p.X(55), yc - 6, p.X(v) - p.X(55), 12, c, edge=INKC if c == "#ffffff" else "none",
                dash=c == "#ffffff")
            lx, rx = p.X(v) + 3, p.X(ref[sec])
            if lx - 4 < rx < lx + 28:
                lx = rx + 4
            d.text(lx, yc - 7, 30, 14, f"{v:.1f}", font=F)
            d.text(4, yc - 7, 226, 14, m, font=F, align="right", bold=m == "Proposed method")
            sec_bars[-1].append((yc, m, v))
    for bars in sec_bars:
        ss = next(v for _, m, v in bars if m.startswith("SimpleShot"))
        line(d, [(p.X(ss), bars[0][0] - 9), (p.X(ss), bars[-1][0] + 9)], INKC, 0.7, "2 2")
    return d


# --------------------------------------------------------------------------- figure 3: curves and per-object gains

def fig3():
    r = rows("fig3_gains")
    d = Diagram("fig3_gains", W, 470)
    for i, ds in enumerate(("MVTec-FS (13 products)", "VisA (11 objects)")):
        p = Plot(d, 50 + i * 290, 22, 205, 150, (0.6, 5.4), (40, 95))
        p.yticks([40, 50, 60, 70, 80, 90], width=24)
        p.xticks([1, 3, 5], grid=False)
        p.axes()
        p.title(ds)
        p.panel("(a)" if i == 0 else "(b)", dx=-46)
        p.xlabel("Shots per defect type")
        p.ylabel("Balanced accuracy (%)", dx=38)
        series = {}
        for row in r:
            if row["dataset"] == ds and row["series"] in METHOD:
                series.setdefault(row["series"], []).append((int(row["K_or_unit"]), float(row["value"]),
                                                             float(row["sd_or_ci"])))
        for m in ["Ground-truth mask pooling (evaluation only)", "SimpleShot (whole image)",
                  "Trainable support adapter", "Fixed-ratio normal reference", "Proposed method"]:
            if m not in series:
                continue
            c, mk, filled = METHOD[m]
            pts = sorted(series[m])
            dash = "4 2" if m.startswith("Ground") else ("1 2" if m.startswith("Trainable") else
                                                          ("5 2 1 2" if m.startswith("SimpleShot") else None))
            p.polyline([k for k, _, _ in pts], [v for _, v, _ in pts], c, 2.0 if m == "Proposed method" else 1.2,
                       dash)
            for k, v, sd in pts:
                p.erry(k, v - sd, v + sd, c, 0.7, 2)
                p.marker(k, v, mk, c, filled, 7 if m == "Proposed method" else 6)
    legend(d, 12, 218, [("Proposed method", "o", ORANGE, True, None, True),
                        ("Fixed-ratio normal reference", "s", AQUA, False, None, True),
                        ("SimpleShot (whole image)", "^", GREY_M, False, "5 2 1 2", True),
                        ("Trainable support adapter (MVTec-FS only)", "v", VIOLET, True, "1 2", True),
                        ("Ground-truth mask pooling (evaluation only)", "h", INKC, False, "4 2", True)],
           col_w=276, cols=2, row_h=16)

    # (c) per-unit gains: one row per dataset, one open mark per product or object, mean with 95 % interval below
    cols = [("MVTec-FS (13)", BLUE, "o", "MVTec-FS, 13 products"), ("VisA (11)", VIOLET, "s", "VisA, 11 objects"),
            ("Real-IAD blind (30)", AQUA, "^", "Real-IAD (blind), 30 objects")]
    p = Plot(d, 196, 296, 360, 4 * 32, (-15, 40), (3.5, -0.5))
    p.xticks([-10, 0, 10, 20, 30, 40])
    p.axes(left=False)
    p.vline(0, INKC, 0.8)
    p.panel("(c)", dx=-190, dy=-22)
    d.text(p.x, p.y - 22, p.w, 15, "Gain of the proposed method per product or object", font=F_TITLE, align="center")
    p.xlabel("Gain over SimpleShot (percentage points)")
    rng = random.Random(3)
    for j, (key, c, mk, lab) in enumerate(cols):
        vals = [float(x["value"]) for x in r if x["dataset"] == key and x["series"] == "per unit"]
        for v in vals:
            marker(d, p.X(v), p.Y(j) - 3 + rng.uniform(-3.5, 3.5), mk, c, False, 5.5)
        mean = next(x for x in r if x["dataset"] == key and x["series"].startswith("mean"))
        lo, hi = (float(t) for t in mean["sd_or_ci"].split(".."))
        p.errx(j, lo, hi, INKC, 1.0, 2.5, dy=6)
        marker(d, p.X(float(mean["value"])), p.Y(j) + 6, "D", c, True, 7.5, edge=INKC)
        d.text(4, p.Y(j) - 15, 186, 14, lab, font=F, align="right")
        d.text(4, p.Y(j) + 1, 186, 14, f"{sum(v > 0 for v in vals)} of {len(vals)} above 0", font=F, align="right",
               color="#444444")
    mtd = next(x for x in r if x["series"] == "single dataset")
    marker(d, p.X(float(mtd["value"])), p.Y(3), "s", GREY_M, True, 6.5)
    d.text(4, p.Y(3) - 7, 186, 14, "MTD, one task", font=F, align="right")
    return d


# --------------------------------------------------------------------------- forests (figure 4 and supplementary)

def forest(name, rows_, names, series, colours, xlabel, xlim, xticks, legend_at=None, width=W, label_w=210,
           groups=None, height=None, rh=30):
    """One row per contrast (label wraps onto two lines), one marker + 95 % interval per series."""
    n = len(names)
    xl_lines = 2 if len(xlabel) * 6.0 > width - label_w - 26 else 1
    height = height or 10 + n * rh + 22 + 15 * xl_lines + (legend_at[4] if legend_at and len(legend_at) > 4 else 0)
    d = Diagram(name, width, height)
    x0 = label_w + 14
    p = Plot(d, x0, 8, width - x0 - 8, n * rh, xlim, (n - 0.5, -0.5))
    if groups:
        for a, b in groups:
            bar(d, p.x, p.Y(a - 0.5), p.w, p.Y(b + 0.5) - p.Y(a - 0.5), "#f3f3f3")
    p.xticks(xticks)
    p.axes(left=False)
    p.vline(0, INKC, 0.8)
    step = min(6.0, (rh - 10) / max(1, len(series) - 1)) if len(series) > 1 else 0
    off = [step * ((len(series) - 1) / 2 - j) for j in range(len(series))]
    for i, nm in enumerate(names):
        d.text(4, p.Y(i) - rh / 2 + 1, label_w, rh - 2, nm, font=F, align="right")
    for (lab, s_, dlt, lo, hi) in rows_:
        i, j = names.index(lab), series.index(s_)
        c, mk, filled = colours[s_]
        p.errx(i, lo, hi, c, 0.9, 2.2, dy=-off[j])
        marker(d, p.X(dlt), p.Y(i) - off[j], mk, c, filled, 6.5)
    d.text(p.x - 10, p.y + p.h + 19, p.w + 20, 15 * xl_lines, xlabel, font=F, align="center", valign="top")
    if legend_at:
        legend(d, legend_at[0], legend_at[1], [(s_, colours[s_][1], colours[s_][0], colours[s_][2], None, False)
                                               for s_ in series], col_w=legend_at[2], cols=legend_at[3])
    return d


def fr(name):
    return [(x[list(x)[0]], x[list(x)[1]], float(x["delta_pp"]), float(x["ci_low_pp"]), float(x["ci_high_pp"]))
            for x in rows(name)]


def fig4():
    rr = fr("fig4_steps")
    names = list(dict.fromkeys(x[0] for x in rr))
    series = list(DATASET)
    return forest("fig4_steps", rr, names, series, DATASET,
                  "Change in balanced accuracy (percentage points), 95% interval", (-2, 26),
                  [0, 5, 10, 15, 20, 25], legend_at=(390, 12, 175, 1), groups=[(5, 7)], label_w=196)


def figS_visa():
    rr = fr("figS_visa_forest")
    names = list(dict.fromkeys(x[0] for x in rr))
    cols = {"1 shot": (B250, "o", True), "3 shots": (B400, "s", True), "5 shots": (B600, "^", True),
            "mean": (ORANGE, "D", True)}
    return forest("figS_visa_forest", rr, names, list(cols), cols,
                  "Change in balanced accuracy on VisA (percentage points), 95% interval over 11 objects",
                  (-2, 27), [0, 5, 10, 15, 20, 25], legend_at=(488, 100, 76, 1), label_w=180)


def figS_threshold():
    rr = fr("figS_threshold_source")
    names = list(dict.fromkeys(x[0] for x in rr))
    cols = {k: v for k, v in DATASET.items() if k != "Real-IAD, blind (30 objects)"}
    return forest("figS_threshold_source", rr, names, list(cols), cols,
                  "Normal-calibrated minus image-only threshold: change in balanced accuracy (percentage points)",
                  (-1, 10.5), [0, 2, 4, 6, 8, 10], legend_at=(200, 168, 175, 2, 20), label_w=180)


def figS_robustness():
    rr = [(x["row"], x["dataset"], float(x["delta_pp"]), float(x["ci_low_pp"]), float(x["ci_high_pp"]))
          for x in rows("figS_robustness")]
    names = list(dict.fromkeys(x[0] for x in rr))
    cols = {k: v for k, v in DATASET.items() if k != "Real-IAD, blind (30 objects)"}
    return forest("figS_robustness", rr, names, list(cols), cols,
                  "Change vs the fixed setting (percentage points); fixed: quantile 0.99, minimum region 6, "
                  "consensus 0.90", (-6.5, 2.5), [-6, -5, -4, -3, -2, -1, 0, 1, 2], legend_at=(184, 8, 160, 1),
                  label_w=160, rh=26)


def figS_backbone(tag):
    rr = [(x["row"], x["dataset"], float(x["delta_pp"]), float(x["ci_low_pp"]), float(x["ci_high_pp"]))
          for x in rows(f"figS_backbone_{tag}")]
    names = list(dict.fromkeys(x[0] for x in rr))
    feats = list(dict.fromkeys(x[1] for x in rr))
    cols = {feats[0]: (ORANGE, "o", True), feats[1]: (AQUA, "s", True), feats[2]: (VIOLET, "^", False)}
    title = "MVTec-FS (13 products)" if tag == "mvtec" else "VisA (11 objects)"
    return forest(f"figS_backbone_{tag}", rr, names, feats, cols,
                  f"{title}: change in balanced accuracy (percentage points)", (-2, 28), [0, 5, 10, 15, 20, 25],
                  legend_at=(356, 72, 206, 1), label_w=178)


# --------------------------------------------------------------------------- figure 6: theory

def fig6():
    d = Diagram("fig6_frr_theory", W, 222)
    a = Plot(d, 56, 18, 214, 158, (0, 16), (0, 0.45))
    a.xticks([0, 5, 10, 15], grid=False)
    a.yticks([0, 0.1, 0.2, 0.3, 0.4], ["0", "0.1", "0.2", "0.3", "0.4"], width=24)
    a.axes()
    a.panel("(a)", dx=-54)
    a.xlabel("False-reject rate of one calibration (%)")
    a.ylabel("Probability density (per %)", dx=44)
    # measured (P47 U1-D): realised false-reject rate of one calibration, deployed engine, N = 100, alpha = 5 %
    import json
    u1d = json.loads((DEVICE / "p47" / "u1d" / "p47_u1d_results.json").read_text(encoding="utf-8"))
    draws = json.loads((DEVICE / "p47" / "u1d" / "p47_u1d_draws_marginal.json").read_text(encoding="utf-8"))
    pooled = np.array([v for k, vals in draws.items() if k.endswith("|100|marginal|0.05") for v in vals])
    width = 0.5
    edges = np.arange(0, 16 + width, width)
    hist, _ = np.histogram(100 * pooled, bins=edges)
    dens = hist / (pooled.size * width)
    for lo_e, hv in zip(edges[:-1], dens):
        if hv > 0:
            bar(d, a.X(lo_e) + 0.3, a.Y(hv), a.X(lo_e + width) - a.X(lo_e) - 0.6, a.Y(0) - a.Y(hv), "#cde2fb")
    x = np.linspace(0.0005, 0.16, 120)
    for n, c, w, lab_at in ((50, B250, 1.6, (0.4, 0.245)), (100, B400, 1.6, (7.4, 0.13)), (500, B600, 2.0, (6.0, 0.36))):
        k = math.ceil((n + 1) * 0.95)
        y = stats.beta(n - k + 1, k).pdf(x) / 100
        a.polyline(100 * x, y, c, w)
        a.label(lab_at[0], lab_at[1], f"N = {n}", dx=3, color=c, bold=True, w=50)
    a.vline(5, ORANGE, 1.0, "3 2")
    a.label(5, 0.43, "α = 5%", dx=4, color=ORANGE, bold=True)

    b = Plot(d, 346, 18, 204, 158, (15, 2000), (0, 16), xlog=True)
    b.xticks([20, 50, 100, 200, 500, 2000])
    b.yticks([0, 4, 8, 12, 16], width=20)
    b.axes()
    b.panel("(b)", dx=-58)
    b.xlabel("Number of calibration normals, N")
    b.ylabel("False-reject rate (%)", dx=36)
    # alpha = 5 % only: theory 5-95 % band (stepped fill between thinned N points) and the measured quantiles
    alpha, step = 0.05, 20
    ns, lo, hi = [], [], []
    for n in range(step - 1, 2001, step):
        k = round((n + 1) * (1 - alpha))
        dd = stats.beta(n - k + 1, k)
        l5, h5 = dd.ppf([0.05, 0.95])
        ns.append(n), lo.append(100 * l5), hi.append(100 * h5)
    # fill: thin columns every 2 px, band edges interpolated in log N between the exact theory points
    lg = np.log10(ns)
    px = b.X(ns[0])
    while px < b.X(ns[-1]):
        n_mid = 10 ** np.interp(math.log10(ns[0]) + (px + 1 - b.X(ns[0])) / b.w * (math.log10(2000) - math.log10(15)),
                                lg, lg)
        h_, l_ = np.interp(math.log10(n_mid), lg, hi), np.interp(math.log10(n_mid), lg, lo)
        bar(d, px, b.Y(h_), 2.05, b.Y(l_) - b.Y(h_), "#fde3d6")
        px += 2
    b.polyline(ns, hi, ORANGE, 1.4)
    b.polyline(ns, lo, ORANGE, 1.4)
    b.hline(5, ORANGE, 0.8, "4 2")
    b.label(16, 5, "α = 5%", dx=2, dy=-9, color=ORANGE, bold=True, w=50)

    for n_cal in (20, 50, 100):
        cell = u1d["summary"][f"{n_cal}|marginal|0.05"]
        for qk in ("q05", "q95"):
            b.marker(n_cal, 100 * cell["empirical"][qk], "o", "#8a2b1a", True, 6.5, edge="#ffffff")
    legend(d, b.X(42), 22, [("theory, 5–95% range", "s", "#fde3d6", True, None, False),
                              ("measured, 5% and 95%", "o", "#8a2b1a", True, None, False)], col_w=160)
    return d


# --------------------------------------------------------------------------- figure 7: operating points

def fig7():
    r = rows("fig7_operating_points")
    d = Diagram("fig7_operating_points", W, 262)
    panels = list(dict.fromkeys(x["panel"] for x in r))
    style_ = {"a=0.01": ("o", ORANGE, True), "a=0.02": ("^", ORANGE, True), "a=0.05": ("D", ORANGE, True),
              "3way=0.1": ("s", VIOLET, False), "3way=0.3": ("h", VIOLET, False)}
    for i, pn in enumerate(panels):
        pts = [x for x in r if x["panel"] == pn]
        main = [x for x in pts if x["point"] in style_]
        arg = [x for x in pts if x["point"].startswith("no screening")]
        xmax = max(float(x["fr_hi"]) for x in main) * 1.12 + 3
        if arg and float(arg[0]["false_rejects_per_1000"]) <= 2.2 * xmax:
            xmax = max(xmax, float(arg[0]["false_rejects_per_1000"]) * 1.12)
        ymax = max(float(x["es_hi"]) for x in main) * 1.08
        step_x = [5, 10, 20, 25, 50][np.searchsorted([25, 50, 100, 150, 300], xmax)]
        step_y = [5, 10][int(ymax > 30)]
        p = Plot(d, 44 + i * 186, 22, 142, 150, (-0.04 * xmax, xmax), (-0.03 * ymax, ymax))
        p.xticks(list(range(0, int(xmax) + 1, step_x)))
        p.yticks(list(range(0, int(ymax) + 1, step_y)), width=24)
        p.axes()
        p.title(pn.replace("Magnetic tile (MTD)", "Magnetic tile").replace("held-out split", "held-out"))
        p.panel(f"({'abc'[i]})", dx=-42)
        p.xlabel("False rejects per 1000 parts")
        if i == 0:
            p.ylabel("Escapes per 1000 parts", dx=32)
        al = [x for x in main if x["point"].startswith("a=")]
        p.polyline([float(x["false_rejects_per_1000"]) for x in al], [float(x["escapes_per_1000"]) for x in al],
                   ORANGE, 1.2)
        for x in main:
            mk, c, filled = style_[x["point"]]
            fx, ey = float(x["false_rejects_per_1000"]), float(x["escapes_per_1000"])
            p.errx(ey, float(x["fr_lo"]), float(x["fr_hi"]), "#c8c8c8", 0.7, 1.8)
            p.erry(fx, float(x["es_lo"]), float(x["es_hi"]), "#c8c8c8", 0.7, 1.8)
            p.marker(fx, ey, mk, c, filled, 7)
        if arg:
            ax_, ay_ = float(arg[0]["false_rejects_per_1000"]), float(arg[0]["escapes_per_1000"])
            if ax_ <= xmax:
                p.marker(ax_, ay_, "x", GREY_M, True, 7)
            else:
                d.text(p.x + p.w - 122, p.y + p.h - 50, 120, 46,
                       f"No screening<br>(off scale): {ax_:.0f}<br>false rejects, {ay_:.1f} escapes",
                       font=F, align="right", color="#444444")
    legend(d, 30, 222, [("Screening, α = 1%", "o", ORANGE, True, None, False),
                        ("Screening, α = 2%", "^", ORANGE, True, None, False),
                        ("Screening, α = 5%", "D", ORANGE, True, None, False),
                        ("Review band, 10%", "s", VIOLET, False, None, False),
                        ("Review band, 30%", "h", VIOLET, False, None, False),
                        ("No screening", "x", GREY_M, True, None, False)], col_w=176, cols=3, row_h=17)
    return d


# --------------------------------------------------------------------------- figure 9: cost

DEVICE = HERE.parents[2] / "ckpt"


def device_sessions():
    """P47 U6-D on the Jetson Orin Nano: 10 fresh-process sessions of the deployed engine, energy windows."""
    import json
    summ = json.loads((DEVICE / "p47" / "p47_jetson" / "p47_u6d_summary.json").read_text(encoding="utf-8"))
    return summ["D1_sessions"], summ["D2_energy"]


def fig9():
    r = rows("fig9_cost")
    d = Diagram("fig9_cost", W, 262)
    a = Plot(d, 62, 18, 160, 160, (0, 25), (64, 82))
    b = Plot(d, 236, 18, 140, 160, (1, 5000), (64, 82), xlog=True)
    a.xticks([0, 5, 10, 15, 20, 25])
    a.yticks([64, 67, 70, 73, 76, 79, 82], width=20)
    b.xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    b.yticks([64, 67, 70, 73, 76, 79, 82], labels=[""] * 7, width=4)
    a.axes(), b.axes()
    a.panel("(a)", dx=-60)
    b.panel("(b)", dx=-12)
    a.xlabel("Forward time per image (ms)")
    b.xlabel("Time to add a task (s)")
    a.ylabel("Balanced accuracy (%)<br>MVTec-FS, mean of K = 1, 3, 5", dx=46)
    a.title("RTX 4090")
    b.title("RTX 4090")
    lab = {"Proposed, CLIP-B/16 at 224 px": (-8, 13, 90, "left"), "CLIP-LoRA, L/14 at 336 px": (-60, -12, 110, "left"),
           "CLIP-LoRA, B/16 at 224 px": (7, 0, 100, "left"), "Trainable adapter": (8, 13, 110, "right")}
    for x in r:
        name = x["configuration"]
        if name.startswith("Proposed"):
            c, mk, filled = ORANGE, "o", name == "Proposed, main features"
        elif name.startswith("Trainable"):
            c, mk, filled = VIOLET, "s", False
        else:
            c, mk, filled = BLUE, "^", False
        y = float(x["ba_mean_over_K_pct"])
        fw = (float(x["forward_ms_pass1"]), float(x["forward_ms_pass2"]))
        tr = (float(x["new_task_s_pass1"]), float(x["new_task_s_pass2"]))
        a.errx(y, min(fw), max(fw), c, 0.8, 0)
        b.errx(y, min(tr), max(tr), c, 0.8, 0)
        a.marker(st.fmean(fw), y, mk, c, filled, 7, edge=c)
        b.marker(st.fmean(tr), y, mk, c, filled, 7, edge=c)
        if name in lab:
            dx, dy, w, al = lab[name]
            short = name.replace("Proposed, ", "").replace(" at 224 px", "").replace(" at 336 px", "").replace(",", "")
            a.label(st.fmean(fw), y, short, dx=dx, dy=dy, w=w, align=al)
    legend(d, 20, 226, [("Proposed, main features", "o", ORANGE, True, None, False),
                        ("Proposed, other features", "o", ORANGE, False, None, False),
                        ("Trainable support adapter", "s", VIOLET, False, None, False),
                        ("CLIP-LoRA, official code", "^", BLUE, False, None, False)], col_w=190, cols=2, row_h=17)

    # (c) Jetson Orin Nano, deployed single engine: 10 fresh-process sessions (P47 U6-D)
    d1, d2 = device_sessions()
    c_ = Plot(d, 458, 18, 102, 160, (0.3, 10.7), (66.0, 68.6))
    c_.yticks([66.0, 66.5, 67.0, 67.5, 68.0, 68.5], ["66.0", "66.5", "67.0", "67.5", "68.0", "68.5"], width=26)
    c_.xticks([1, 5, 10], grid=False)
    c_.axes()
    c_.panel("(c)", dx=-62)
    c_.title("Jetson Orin Nano")
    c_.xlabel("Process session")
    c_.ylabel("Latency per part (ms)", dx=44)
    g, u = d1["grand_mean_ms"], d1["u_single_measurement_ms"]
    bar(d, c_.x, c_.Y(g + 2 * u), c_.w, c_.Y(g - 2 * u) - c_.Y(g + 2 * u), "#fdeee7")
    c_.hline(g, ORANGE, 1.0, "4 2")
    for i, sess in enumerate(d1["sessions"], 1):
        c_.erry(i, sess["mean_ms"] - sess["sd_ms"], sess["mean_ms"] + sess["sd_ms"], ORANGE, 0.8, 2)
        c_.marker(i, sess["mean_ms"], "o", ORANGE, True, 5.5)
    return d


def figS_device_scaling():
    """P47 U6-D (D3): deployed engine against the number of defect types, the classification head against the bank
    size M (PyTorch on the device), and 1 / 4 / 9 tiles per part."""
    import json
    sc = json.loads((DEVICE / "p47" / "p47_jetson" / "p47_u6_scaling.json").read_text(encoding="utf-8"))
    d = Diagram("figS_device_scaling", W, 192)
    a = Plot(d, 54, 16, 128, 140, (1, 17), (60, 72))
    cl = sorted((int(k), v) for k, v in sc["classes_e2e"].items())
    a.xticks([2, 4, 8, 16], grid=False)
    a.yticks([60, 63, 66, 69, 72], width=24)
    a.axes()
    a.panel("(a)", dx=-52)
    a.xlabel("Defect types")
    a.ylabel("End-to-end latency (ms)", dx=38)
    a.polyline([k for k, _ in cl], [v["p50_ms"] for _, v in cl], ORANGE, 1.4)
    for k, v in cl:
        a.erry(k, v["p50_ms"] - v["sd_ms"], v["p50_ms"] + v["sd_ms"], ORANGE, 0.8, 2)
        a.marker(k, v["p50_ms"], "o", ORANGE, True, 6)
    b = Plot(d, 244, 16, 130, 140, (3.5, 140), (0.8, 80), xlog=True, ylog=True)
    hm = sorted((int(k), v) for k, v in sc["head_vs_M"].items())
    b.xticks([5, 10, 20, 50, 100])
    b.yticks([1, 3, 10, 30], width=20)
    b.axes()
    b.panel("(b)", dx=-50)
    b.xlabel("Bank size, M")
    b.ylabel("Head time (ms)", dx=32)
    for prec, c, mk in (("fp32", BLUE, "s"), ("fp16", ORANGE, "o")):
        b.polyline([m for m, _ in hm], [v[prec]["p50_ms"] for _, v in hm], c, 1.4)
        for m, v in hm:
            b.marker(m, v[prec]["p50_ms"], mk, c, True, 6)
    legend(d, 250, 18, [("FP16", "o", ORANGE, True, None, True), ("FP32", "s", BLUE, True, None, True)], col_w=64)
    c_ = Plot(d, 450, 16, 110, 140, (0.5, 3.5), (0, 600))
    tl = sorted((int(k), v) for k, v in sc["tiles_per_part"].items())
    c_.xticks([1, 2, 3], [f"{n}" for n, _ in tl], grid=False)
    c_.yticks([0, 200, 400, 600], width=24)
    c_.axes()
    c_.panel("(c)", dx=-56)
    c_.xlabel("Tiles per part")
    c_.ylabel("Latency per part (ms)", dx=40)
    for i, (n, v) in enumerate(tl, 1):
        c_.vbar(i, v["p50_ms"], 24, ORANGE if n == 1 else "#f4a582")
        c_.label(i, v["p50_ms"], f"{v['p50_ms']:.0f}", dx=0, dy=-9, w=40, align="center")
    return d


# --------------------------------------------------------------------------- supplementary bars and lines

def figS_false_rejects():
    r = rows("figS_false_rejects_without_screening")
    methods = list(dict.fromkeys(x["method"] for x in r))
    d = Diagram("figS_false_rejects", W, 262)
    shade = {"1": B250, "3": B400, "5": B600}
    for i, ds in enumerate(dict.fromkeys(x["dataset"] for x in r)):
        xmax = 520 if i == 0 else 850
        p = Plot(d, 154 + i * 214, 24, 190, 182, (0, xmax), (len(methods) - 0.5, -0.5))
        p.xticks(list(range(0, xmax + 1, 100 if i == 0 else 200)))
        p.axes(left=False)
        p.title(ds)
        p.xlabel("False rejects per 1000 parts")
        for j, m in enumerate(methods):
            if i == 0:
                d.text(2, p.Y(j) - 7, 146, 14, m, font=F, align="right", bold=m == "Proposed method")
            for k, kk in enumerate("135"):
                v = next(float(x["false_rejects_per_1000"]) for x in r
                         if x["dataset"] == ds and x["method"] == m and x["K"] == kk)
                p.hbar(j + (k - 1) * 0.27, v, 6, ORANGE if (m == "Proposed method" and False) else shade[kk])
    legend(d, 154, 244, [("1 shot", "s", B250, True, None, False), ("3 shots", "s", B400, True, None, False),
                         ("5 shots", "s", B600, True, None, False)], col_w=80, cols=3)
    return d


def figS_per_product():
    r = rows("figS_per_product")
    d = Diagram("figS_per_product", 319, 340)
    p = Plot(d, 70, 8, 240, 13 * 22, (-2, 37), (12.5, -0.5))
    p.xticks([0, 10, 20, 30])
    p.axes(left=False)
    p.vline(0, INKC, 0.8)
    cols = [("proposed_pp", "Proposed method", ORANGE), ("fixed10_pp", "Fixed 10% region", AQUA),
            ("consensus_pp", "Support consensus", BLUE)]
    for j, x in enumerate(reversed(r)):
        d.text(2, p.Y(j) - 7, 62, 14, x["product"].replace("_", " "), font=F, align="right")
        for k, (key, _, c) in enumerate(cols):
            p.hbar(j + (k - 1) * 0.27, float(x[key]), 5.2, c, base=0)
    p.xlabel("Gain over SimpleShot (percentage points)", dy=16)
    legend(d, 186, 206, [(lab, "s", c, True, None, False) for _, lab, c in cols], col_w=132)
    return d


def figS_purity():
    r = rows("figS_evidence_purity")
    d = Diagram("figS_evidence_purity", W, 270)
    a = Plot(d, 56, 18, 236, 158, (-0.3, 3.3), (0, 1))
    a.xticks([0, 1, 2, 3], ["< 2%", "2–5%", "5–15%", "≥ 15%"], grid=False)
    a.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1.0"], width=24)
    a.axes()
    a.panel("(a)", dx=-54)
    a.xlabel("Defect share of the image's patches")
    a.ylabel("Weight on defect patches", dx=44)
    sty = {"Whole-image mean": (GREY_M, "x", True, "2 2"), "Fixed top 10%, uniform": (BLUE, "o", False, "4 2"),
           "Normal-calibrated, uniform": (AQUA, "s", False, "4 2"),
           "MEBin (per image), weighted": (VIOLET, "D", False, "5 2 1 2"),
           "Normal-calibrated, weighted (proposed)": (ORANGE, "o", True, None)}
    for s_, (c, mk, filled, dash) in sty.items():
        v = [float(x["value"]) for x in r if x["panel"] == "purity" and x["series"] == s_]
        a.polyline(range(4), v, c, 2.0 if dash is None else 1.2, dash)
        for i, y in enumerate(v):
            a.marker(i, y, mk, c, filled, 6)
    legend(d, 20, 220, [(k, v[1], v[0], v[2], v[3], True) for k, v in sty.items()], col_w=270, cols=2, row_h=16)
    b = Plot(d, 366, 18, 194, 158, (-0.6, 4.6), (0, 25))
    ex = [x for x in r if x["panel"] == "exceedance"]
    b.xticks(range(5), [x["bin"] for x in ex], grid=False)
    b.yticks([0, 5, 10, 15, 20, 25], width=20)
    b.axes()
    b.panel("(b)", dx=-58)
    b.xlabel("Distance to the defect (patches)")
    b.ylabel("Background patches above τ (%)", dx=36)
    for i, x in enumerate(ex):
        v = float(x["value"])
        b.vbar(i, v, 22, AQUA)
        b.label(i, v, f"{v:.1f}", dx=0, dy=-9, w=34, align="center")
    b.hline(1.05, ORANGE, 1.2, "4 2")
    b.label(4.6, 19, "dashed line: normal<br>images, 1.05%", dx=0, dy=0, w=118, align="right", color=ORANGE,
            bold=True)
    return d


# --------------------------------------------------------------------------- figure 8: influence quantities (P47 U4)

U4_PHOTO_ROWS = [("none", "unperturbed"),
                 ("P1=0.8", "gain × 0.8"), ("P1=0.9", "gain × 0.9"), ("P1=1.1", "gain × 1.1"), ("P1=1.2", "gain × 1.2"),
                 ("P2=0.8", "gamma 0.8"), ("P2=1.25", "gamma 1.25"),
                 ("P3=2", "noise 2 DN"), ("P3=5", "noise 5 DN"), ("P3=10", "noise 10 DN"),
                 ("P4=0.5", "blur 0.5 px"), ("P4=1", "blur 1 px"), ("P4=2", "blur 2 px"),
                 ("P7=90", "JPEG 90"), ("P7=70", "JPEG 70"), ("P7=50", "JPEG 50")]
U4_GEO_ROWS = [("P5=1", "shift 1 %"), ("P5=2.5", "shift 2.5 %"), ("P5=5", "shift 5 %"),
               ("P6=2", "rotation 2°"), ("P6=5", "rotation 5°")]
U4_GROUP_START = {"P1=0.8", "P2=0.8", "P3=2", "P4=0.5", "P7=90", "P5=1", "A:none", "P6=2", "A:P6=2"}


def fig8():
    """P47 U4 (ckpt/p47/u4_analysis/u4_analysis.json): realized false-reject rate at alpha = 5 % on VisA (object macro)
    and MTD, without and after recalibration, and the change in balanced accuracy on MVTec-FS (classification, with
    the 95 % product-cluster bootstrap interval of the unrecalibrated change). Shift and rotation come from U4-A
    (field of view moved inside the image, ckpt/p47/u4a_analysis/) when it exists; otherwise from U4 (whole image
    moved, reflect padding), marked with an asterisk."""
    import json
    res = json.loads((DEVICE / "p47" / "u4_analysis" / "u4_analysis.json").read_text(encoding="utf-8"))
    a_path = DEVICE / "p47" / "u4a_analysis" / "u4a_analysis.json"
    res_a = json.loads(a_path.read_text(encoding="utf-8")) if a_path.is_file() else None
    rows = list(U4_PHOTO_ROWS)
    if res_a:
        rows += [("A:none", "window view")] + [("A:" + c, lab) for c, lab in U4_GEO_ROWS]
    else:
        rows += [(c, lab + " *") for c, lab in U4_GEO_ROWS]
    sources = {"visa": (res["external"]["visa_object_macro"], res_a and res_a["external"]["visa_object_macro"]),
               "mtd": (res["external"]["mtd"], res_a and res_a["external"]["mtd"]),
               "mvtec": (res["mvtec"], res_a and res_a["mvtec"])}
    rh, gap, top = 13.0, 5.0, 34
    ys, y = {}, top
    for c, _ in rows:
        if c in U4_GROUP_START:
            y += gap
        ys[c] = y + rh / 2
        y += rh
    bottom = y + 2
    d = Diagram("fig8_influence", W, int(bottom + (58 if res_a else 76)))
    D_C, R_C = ORANGE, AQUA
    for c, lab in rows:
        d.text(2, ys[c] - 7, 104, 14, lab, align="right", bold=c in ("none", "A:none"))
    if res_a:  # without the padding artifact every value fits much tighter axes
        frr_lim, frr_t, ba_lim, ba_t = (0, 12), [0, 5, 10], (-4, 2), [-4, -2, 0, 2]
    else:
        frr_lim, frr_t, ba_lim, ba_t = (0, 25), [0, 5, 10, 15, 20, 25], (-11, 3), [-10, -5, 0]
    panels = [("visa", "FRR", frr_lim, frr_t, "(a)", "VisA: false rejects (%)", 116, 126),
              ("mtd", "FRR", frr_lim if res_a else (0, 16), frr_t if res_a else [0, 5, 10, 15], "(b)",
               "MTD: false rejects (%)", 264, 126),
              ("mvtec", "BA|ARR", ba_lim, ba_t, "(c)", "MVTec-FS: Δ BA (pp)", 418, 140)]
    for src, key, xlim, xt, pl, title, x0, w in panels:
        p = Plot(d, x0, top, w, bottom - top, xlim, (bottom, top))
        for v in xt:
            line(d, [(p.X(v), top), (p.X(v), bottom)], "#e6e6e6", 0.6)
        line(d, [(p.x, bottom), (p.x + p.w, bottom)], "#333333", 0.8)
        for v in xt:
            line(d, [(p.X(v), bottom), (p.X(v), bottom + 3)], "#333333", 0.8)
            d.text(p.X(v) - 15, bottom + 4, 30, 14, f"{v:g}".replace("-", "−"), align="center")
        d.text(p.x - 10, top - 30, p.w + 20, 15, f"<b>{pl}</b> {title}", align="center")
        if key == "FRR":
            line(d, [(p.X(5), top - 4), (p.X(5), bottom)], INKC, 0.9, "4 2")
            line(d, [(p.X(10), top - 4), (p.X(10), bottom)], GREY_M, 0.9, "1 2")
            d.text(p.X(5) - 14, top - 15, 28, 13, "α", align="center", italic=True)
            d.text(p.X(10) - 14, top - 15, 28, 13, "2α", align="center", italic=True, color="#444444")
        else:
            line(d, [(p.X(0), top - 4), (p.X(0), bottom)], INKC, 0.9)
        for c, _ in rows:
            yv = ys[c]
            res_ = sources[src][1] if c.startswith("A:") else sources[src][0]
            cond = c[2:] if c.startswith("A:") else c
            rc_ = res_["RC"].get(cond)
            if key == "FRR":
                vd = 100 * res_["D"][cond]["FRR"]["mean"]
                vr = None if rc_ is None else 100 * rc_["FRR"]["mean"]
            elif cond == "none":
                vd, vr = 0.0, (None if rc_ is None else 0.0)
            else:
                e = res_["D"][cond]["BA|ARR"]
                vd = 100 * e["delta"]
                vr = None if rc_ is None else 100 * rc_["BA|ARR"]["delta"]
                lo, hi = (100 * t for t in e["delta_ci95"])
                line(d, [(p.X(max(lo, xlim[0])), yv), (p.X(min(hi, xlim[1])), yv)], "#f3b79c", 1.4)
            if vr is not None and cond != "none" and abs(p.X(vd) - p.X(vr)) > 7:
                line(d, [(p.X(vd), yv), (p.X(vr), yv)], "#9a9a9a", 0.8)
            if vr is not None:
                marker(d, p.X(vr), yv, "s", R_C, True, 6)
            marker(d, p.X(vd), yv, "o", D_C, False, 7)
    legend(d, 116, bottom + 24, [("without recalibration", "o", D_C, False, None, False),
                                 ("after recalibration (bank and limit rebuilt from conforming parts)", "s", R_C, True, None,
                                  False)], col_w=440, cols=1, row_h=16)
    if not res_a:  # with U4-A the caption explains the window-view rows
        d.text(116, bottom + 58, 446, 15, "* whole image moved, reflect padding (U4); U4-A will replace these rows",
               color="#444444")
    return d


# --------------------------------------------------------------------------- figure 6: evidence regions

FIG6_DATA = Path(os.environ.get("FIG6_DATA", HERE.parents[1] / "data" / "ckpt" / "fig6_evidence"))


def _hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


def _evidence_png(view: np.ndarray, weight: np.ndarray | None, tint: str | None, polygons: list) -> bytes:
    """336-px view; with weights (24 x 24, >= 0), patches are shown at a brightness proportional to their weight
    (relative to the largest) and tinted; the ground-truth outline is a white dashed line with a dark halo."""
    import io
    from PIL import Image, ImageDraw
    v = view.astype(float) / 255
    if weight is not None:
        m = np.kron(weight / max(float(weight.max()), 1e-12), np.ones((14, 14)))[..., None]
        c = _hex_rgb(tint)
        v = v * (0.22 + 0.78 * m) * (1 - 0.5 * m) + 0.5 * m * c
    im = Image.fromarray((255 * v.clip(0, 1)).astype(np.uint8)).resize((672, 672), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    for poly in polygons:
        pts = [(2 * x, 2 * y) for x, y in poly] + [(2 * poly[0][0], 2 * poly[0][1])]
        dr.line(pts, fill=(30, 30, 30), width=7, joint="curve")
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):                 # white dashes, 10 px on / 7 px off
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 17))
            for j in range(n):
                t0, t1 = j / n, min(1.0, (j + 10 / 17) / n)
                dr.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0), (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)],
                        fill=(255, 255, 255), width=3)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def fig6_evidence():
    """Evidence regions on three MVTec-FS queries (data: paper_figures/mst/export_fig6_evidence.py, run locally):
    the query with the true defect outlined, the fixed 10 % of patches with the largest a(t), and the
    normal-referenced weights w(t) of the proposed method."""
    import json
    meta = json.loads((FIG6_DATA / "fig6_evidence.json").read_text(encoding="utf-8"))
    arr = np.load(FIG6_DATA / "fig6_evidence.npz")
    side, gap, lab, top, foot = 128, 10, 150, 20, 15
    x0 = W - 3 * side - 2 * gap - 2
    row_h = side + foot + 9
    d = Diagram("fig6_evidence", W, int(top + 3 * row_h - 6))
    for j, (pl, t) in enumerate((("(a)", "Query"), ("(b)", "Fixed 10 % of patches"), ("(c)", "Proposed, w(t)"))):
        d.text(x0 + j * (side + gap) - 6, 1, side + 12, 16, f"<b>{pl}</b> {t}", font=F_TITLE, align="center")

    def name(s):
        return s.replace("_", " ")
    for q, m in enumerate(meta["queries"]):
        y = top + q * row_h
        d.text(2, y + 2, lab - 8, side - 4,
               f"<b>{name(m['product'])}</b><br>true type: {name(m['true'])}<br>proposed: {name(m['pred_ours'])} ✓"
               f"<br>fixed 10 %: {name(m['pred_fixed'])} ✗", align="left", valign="top")
        view = arr[f"view_{q}"]
        polys = m["gt_polygons_336"]
        panels = [(None, None, None), (arr[f"fixed_{q}"].astype(float), AQUA, "fixed"),
                  (arr[f"w_{q}"].astype(float), ORANGE, "ours")]
        for j, (wgt, tint, key) in enumerate(panels):
            xi = x0 + j * (side + gap)
            d.image_png(xi, y, side, side, _evidence_png(view, wgt, tint, polys), border="#9a9a9a")
            if key:
                n = m["patches"]["fixed" if key == "fixed" else "ours_nonzero"]
                d.text(xi - 4, y + side + 1, side + 8, foot,
                       f"{n} patches, λ = {m['purity_lambda'][key]:.2f}", align="center", color="#444444")
    return d


# --------------------------------------------------------------------------- supplementary: U3 budget (P47 U3)

def figS_u3():
    """P47 U3 (ckpt/p47/u3_analysis/u3_budget.json): (a) shares of the variance of the benchmark mean BA; (b) bank-draw
    standard deviation against the bank size M; (c) one deployment of each product at K = 5: mean BA and the 95 %
    prediction interval with and without the test-sample component."""
    import json
    sys.path.insert(0, str(HERE.parents[2]))
    from experiments.p47.common import t975, welch_satterthwaite
    res = json.loads((DEVICE / "p47" / "u3_analysis" / "u3_budget.json").read_text(encoding="utf-8"))
    d = Diagram("figS_u3_budget", W, 494)
    comp = [("support draw S", "support draw", BLUE), ("bank draw B", "bank draw", AQUA),
            ("interaction S x B", "support × bank", VIOLET), ("query sampling Q", "test sample", GREY_L),
            ("implementation I (Type B, rectangular)", "implementation (Type B)", "#555555")]
    # (a) shares
    a = Plot(d, 64, 24, 230, 66, (0, 100), (2.5, -0.5))
    a.xticks([0, 25, 50, 75, 100])
    a.axes(left=False)
    a.panel("(a)", dx=-62, dy=-22)
    d.text(a.x - 10, a.y - 22, a.w + 20, 15, "Shares of u<sub>c</sub>² of the benchmark mean", align="center")
    a.xlabel("Share (%)")
    for j, k in enumerate(("1", "3", "5")):
        b = res["per_K"][k]["benchmark_mean"]
        x = 0.0
        for key, _, c in comp:
            s = 100 * b["components"][key]["share"]
            bar(d, a.X(x), a.Y(j) - 8, a.X(x + s) - a.X(x), 16, c)
            x += s
        d.text(2, a.Y(j) - 7, 56, 14, f"K = {k}", align="right")
        d.text(a.X(100) + 4, a.Y(j) - 7, 70, 14, f"U<sub>95</sub> {100 * b['U95']:.1f}", color="#444444")
    legend(d, 6, 144, [(lab, "s", c, True, None, False) for _, lab, c in comp], col_w=190, cols=3, row_h=16)
    # (b) bank-draw SD against M
    bp = Plot(d, 420, 24, 136, 76, (4, 60), (0, 6), xlog=True)
    bp.xticks([5, 10, 20, 50])
    bp.yticks([0, 2, 4, 6], width=16)
    bp.axes()
    bp.panel("(b)", dx=-52, dy=-22)
    d.text(bp.x - 30, bp.y - 22, bp.w + 40, 15, "Bank draw, one product", align="center")
    bp.xlabel("Bank size, M")
    bp.ylabel("σ<sub>B</sub> (pp)", dx=26)
    for k, c, mk in (("1", B250, "o"), ("3", B400, "s"), ("5", B600, "^")):
        sw = res["per_K"][k]["bank_size_sweep"]
        ms = sorted(int(m) for m in sw)
        bp.polyline(ms, [100 * sw[str(m)]["pooled_sigma_B"] for m in ms], c, 1.4)
        for m in ms:
            bp.marker(m, 100 * sw[str(m)]["pooled_sigma_B"], mk, c, True, 6)
    legend(d, 470, 26, [("K = 1", "o", B250, True, None, True), ("K = 3", "s", B400, True, None, True),
                        ("K = 5", "^", B600, True, None, True)], col_w=60, cols=1, row_h=13)
    # (c) one deployment per product, K = 5
    ap = res["per_K"]["5"]["anova_per_product"]
    prods = sorted(ap, key=lambda p: ap[p]["mean"])
    cp = Plot(d, 96, 210, 456, len(prods) * 16, (40, 100), (len(prods) - 0.5, -0.5))
    cp.xticks([40, 50, 60, 70, 80, 90, 100])
    cp.axes(left=False)
    cp.panel("(c)", dx=-94, dy=-22)
    d.text(cp.x, cp.y - 22, cp.w, 15, "One deployment at K = 5: mean and 95 % prediction interval", align="center")
    cp.xlabel("Balanced accuracy on MVTec-FS (%)")
    inf = float("inf")
    for i, p in enumerate(prods):
        v = ap[p]
        nu = lambda x: inf if x is None else x  # noqa: E731
        parts = [(v["var_S"], nu(v["nu_S"])), (v["var_B"], nu(v["nu_B"])), (v["var_SB"], nu(v["nu_SB"]))]
        vd = sum(x for x, _ in parts)
        va = vd + v["var_Q_single"]
        td = t975(welch_satterthwaite(parts))
        ta = t975(welch_satterthwaite(parts + [(v["var_Q_single"], v["nu_Q"])]))
        m = 100 * v["mean"]
        lo_a, hi_a = max(40, m - 100 * ta * va ** 0.5), min(100, m + 100 * ta * va ** 0.5)
        lo_d, hi_d = max(40, m - 100 * td * vd ** 0.5), min(100, m + 100 * td * vd ** 0.5)
        line(d, [(cp.X(lo_a), cp.Y(i)), (cp.X(hi_a), cp.Y(i))], "#b8b8b8", 1.0)
        bar(d, cp.X(lo_d), cp.Y(i) - 3.5, cp.X(hi_d) - cp.X(lo_d), 7, "#9cc3ef")
        marker(d, cp.X(m), cp.Y(i), "o", ORANGE, True, 6.5)
        d.text(2, cp.Y(i) - 7, 88, 14, p.replace("_", " "), align="right")
    legend(d, 96, 210 + len(prods) * 16 + 40,
           [("mean over 25 support × bank draws", "o", ORANGE, True, None, False),
            ("support and bank draw", "s", "#9cc3ef", True, None, False),
            ("plus test sample (about 40 parts)", None, "#b8b8b8", True, None, True)], col_w=400, cols=1, row_h=16)
    return d


CHARTS = {"fig6_evidence": fig6_evidence, "fig2_headroom": fig2, "fig3_gains": fig3, "fig4_steps": fig4, "fig6_frr_theory": fig6, "fig8_influence": fig8,
          "figS_u3_budget": figS_u3,
          "fig7_operating_points": fig7, "fig9_cost": fig9, "figS_visa_forest": figS_visa,
          "figS_threshold_source": figS_threshold, "figS_robustness": figS_robustness,
          "figS_backbone_mvtec": lambda: figS_backbone("mvtec"), "figS_backbone_visa": lambda: figS_backbone("visa"),
          "figS_false_rejects": figS_false_rejects, "figS_per_product": figS_per_product,
          "figS_evidence_purity": figS_purity, "figS_device_scaling": figS_device_scaling}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    names = [a for a in argv if not a.startswith("--")] or [
        n for n in CHARTS if n != "fig6_evidence" or (FIG6_DATA / "fig6_evidence.json").is_file()]
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for n in names:
        d = CHARTS[n]()
        p = OUT / f"{n}.drawio"
        d.save(p)
        files.append(str(p))
    if "--no-render" not in argv:
        viewer = os.environ.get("DRAWIO_VIEWER", str(HERE / "viewer-static.min.js"))
        subprocess.run(["node", str(HERE / "render.js"), viewer, str(OUT), *files], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
