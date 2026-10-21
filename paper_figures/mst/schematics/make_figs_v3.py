#!/usr/bin/env python3
"""Manuscript figures 8 and 10 as editable draw.io files (final versions, 2026-10-14).

    python paper_figures/mst/schematics/make_figs_v3.py              # both
    python paper_figures/mst/schematics/make_figs_v3.py fig10        # one

fig8   Figure 8 (operating points of the two-stage cascade): legend under the panels, review load under each panel
       title, one accent colour for screening, grey review bands and no-screening mark, light interval bars drawn
       under the markers. Numbers: ../out/fig7_operating_points.csv (make_mst_figures.py).
fig10  Figure 10 (accuracy and deployment cost): (a) time to add a defect class on a log axis labelled 1 ... 10,000 s with
       1 min / 1 h guide lines, the no-training region named in the plot; (b) Jetson latency per session with the
       +-2u band and the grand mean named in the plot. Numbers: ../out/fig9_cost.csv and the P47 U6-D summary.

Outputs: paper_figures/mst/drawio/charts/fig7_operating_points.* and fig9_cost.* (PNG / PDF exported with the
draw.io desktop app, as in make_figs_v2.py). Figures 1 and 6 are drawn by make_figs_v2.py.
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drawchart import Plot, bar  # noqa: E402
from drawio import ORANGE, Diagram  # noqa: E402
import make_figs_v2 as v2  # noqa: E402

CSV_DIR = HERE.parent / "out"
OUT = HERE.parent / "drawio" / "charts"
W = 565
T, TS = 12, 11                      # 9 pt and 8.25 pt at the printed 15 cm width
GREY_D, GREY_M, GREY_L = "#5c5c5c", "#8a8a8a", "#c4c4c4"
ACCENT_TXT = "#b4461c"              # darker shade of the accent for text (contrast)
BAND = "#fbe9e0"


class MathDiagram(Diagram):
    """Diagram with draw.io math typesetting switched on (as exported for the submission)."""
    def xml(self) -> str:
        return super().xml().replace('math="0"', 'math="1"')


# --------------------------------------------------------------------------- Figure 8: operating points

def fig8() -> Diagram:
    """Figure 8 (operating points). Layout of the earlier version (legend under the panels, review load under each title, off-scale note), with
    the v3 colours: one accent for screening, greys for the review bands and no screening, lighter interval bars."""
    from drawchart import legend
    rows = list(csv.DictReader((CSV_DIR / "fig7_operating_points.csv").open(encoding="utf-8")))
    d = MathDiagram("fig7_operating_points", W, 266)
    panels = list(dict.fromkeys(x["panel"] for x in rows))
    style_ = {"a=0.01": ("o", ORANGE, True), "a=0.02": ("^", ORANGE, True), "a=0.05": ("D", ORANGE, True),
              "3way=0.1": ("s", GREY_D, False), "3way=0.3": ("h", GREY_D, False)}
    for i, pn in enumerate(panels):
        pts = [x for x in rows if x["panel"] == pn]
        main = [x for x in pts if x["point"] in style_]
        arg = [x for x in pts if x["point"].startswith("no screening")]
        xmax = max(float(x["fr_hi"]) for x in main) * 1.12 + 3
        if arg and float(arg[0]["false_rejects_per_1000"]) <= 2.2 * xmax:
            xmax = max(xmax, float(arg[0]["false_rejects_per_1000"]) * 1.12)
        ymax = max(float(x["es_hi"]) for x in main) * 1.08
        step_x = [5, 10, 20, 25, 50][int(sum(xmax > t for t in (25, 50, 100, 150)))]
        step_y = [5, 10][int(ymax > 30)]
        p = Plot(d, 44 + i * 186, 36, 142, 146, (-0.04 * xmax, xmax), (-0.03 * ymax, ymax))
        p.xticks(list(range(0, int(xmax) + 1, step_x)), font=T)
        p.yticks(list(range(0, int(ymax) + 1, step_y)), width=24, font=T)
        p.axes()
        p.title(pn.replace("Magnetic tile", "MTD").replace(" (11 objects)", ""), font=T, dy=-34)
        rv = {x["point"]: float(x["review_per_1000"]) for x in main}
        d.text(p.x - 20, p.y - 17, p.w + 40, 14,
               f"review: {v2.half_up(rv['3way=0.1'])} / {v2.half_up(rv['3way=0.3'])} per 1,000", font=TS,
               align="center", color=GREY_D)
        p.panel(f"({'abc'[i]})", dx=-42, dy=-34)
        p.xlabel("False rejects per 1,000", font=T)
        if i == 0:
            p.ylabel("Escapes per 1,000 parts", dx=32, font=T)
        for x in main:                       # intervals first and light
            fx, ey = float(x["false_rejects_per_1000"]), float(x["escapes_per_1000"])
            p.errx(ey, float(x["fr_lo"]), float(x["fr_hi"]), GREY_L, 0.8, 1.6)
            p.erry(fx, float(x["es_lo"]), float(x["es_hi"]), GREY_L, 0.8, 1.6)
        al = [x for x in main if x["point"].startswith("a=")]
        p.polyline([float(x["false_rejects_per_1000"]) for x in al], [float(x["escapes_per_1000"]) for x in al],
                   ORANGE, 1.2)
        for x in main:
            mk, c, filled = style_[x["point"]]
            p.marker(float(x["false_rejects_per_1000"]), float(x["escapes_per_1000"]), mk, c, filled, 8)
        if arg:
            ax_, ay_ = float(arg[0]["false_rejects_per_1000"]), float(arg[0]["escapes_per_1000"])
            if ax_ <= xmax:
                p.marker(ax_, ay_, "x", GREY_M, True, 8)
            else:
                d.text(p.x + p.w - 122, p.y + p.h - 64, 120, 60,
                       f"No screening<br>(off scale): {ax_:.0f}<br>false rejects,<br>{ay_:.1f} escapes",
                       font=TS, align="right", color=GREY_D)
    legend(d, 30, 228, [("Screening, α = 1%", "o", ORANGE, True, None, False),
                        ("Screening, α = 2%", "^", ORANGE, True, None, False),
                        ("Screening, α = 5%", "D", ORANGE, True, None, False),
                        ("Review band, 10%", "s", GREY_D, False, None, False),
                        ("Review band, 30%", "h", GREY_D, False, None, False),
                        ("No screening", "x", GREY_M, True, None, False)], col_w=176, cols=3, row_h=18, font=T)
    return d


# --------------------------------------------------------------------------- Figure 10: accuracy and cost

def fig10() -> Diagram:
    import make_color_charts as mcc
    r = {x["configuration"]: x for x in csv.DictReader((CSV_DIR / "fig9_cost.csv").open(encoding="utf-8"))}
    d = MathDiagram("fig9_cost", W, 214)
    a = Plot(d, 62, 26, 258, 142, (1, 1e4 * 1.6), (62, 84), xlog=True)
    # no-training region first, named inside the plot
    bar(d, a.X(1), a.y, a.X(10) - a.X(1), a.h, BAND)
    d.text(a.X(1) + 3, a.y + a.h - 18, a.X(10) - a.X(1) - 4, 14, "no training", font=TS, color=ACCENT_TXT)
    a.xticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"], font=TS)
    for v, txt in ((60, "1 min"), (3600, "1 h")):
        a.vline(v, GREY_L, 0.8, "2 2")
        d.text(a.X(v) + 3, a.y + 2, 40, 13, txt, font=TS, align="left", color=GREY_M)
    a.yticks([62, 66, 70, 74, 78, 82], width=22, font=TS)
    a.axes()
    d.text(a.x - 54, a.y - 22, a.w + 54, 15, "<b>(a)</b> RTX 4090", font=T)
    a.xlabel("Time to add a defect class (s)", font=T, dy=19)
    a.ylabel("Balanced accuracy (%)", dx=38, font=T)
    show = [("Proposed, main features", "Proposed", ORANGE, "o", True, 8, "left", True),
            ("Proposed, CLIP-B/16 at 224 px", "Proposed, B/16", ORANGE, "o", False, 8, "left", False),
            ("Trainable adapter", "trainable adapter", GREY_D, "s", False, 0, "above", False),
            ("CLIP-LoRA, B/16 at 224 px", "CLIP-LoRA, B/16", GREY_D, "^", False, 0, "below", False),
            ("CLIP-LoRA, L/14 at 336 px", "CLIP-LoRA, L/14", GREY_D, "^", False, -8, "right", False)]
    for key, lab, c, mk, filled, dx, al, bold in show:
        x = r[key]
        y = float(x["ba_mean_over_K_pct"])
        t = st.fmean((float(x["new_task_s_pass1"]), float(x["new_task_s_pass2"])))
        a.marker(t, y, mk, c, filled, 9 if bold else 7.5)
        if al in ("below", "above"):
            a.label(t, y, lab, dx=0, dy=13 if al == "below" else -13, w=110, align="center", font=TS, color=GREY_D)
            continue
        a.label(t, y, lab, dx=dx, w=110, align=al, font=TS if not bold else T,
                color=ACCENT_TXT if c == ORANGE else GREY_D, bold=bold)
    # (b) Jetson: ten sessions
    d1, _ = mcc.device_sessions()
    g, u = d1["grand_mean_ms"], d1["u_single_measurement_ms"]
    b = Plot(d, 412, 26, 146, 142, (0.3, 10.7), (66.0, 68.6))
    bar(d, b.x, b.Y(g + 2 * u), b.w, b.Y(g - 2 * u) - b.Y(g + 2 * u), BAND)
    b.yticks([66.0, 66.5, 67.0, 67.5, 68.0, 68.5], ["66.0", "66.5", "67.0", "67.5", "68.0", "68.5"], width=26,
             font=TS)
    b.xticks([1, 5, 10], grid=False, font=TS)
    b.axes()
    d.text(b.x - 54, b.y - 22, b.w + 54, 15, "<b>(b)</b> Jetson Orin Nano Super", font=T)
    b.xlabel("Process session", font=T, dy=19)
    b.ylabel("Latency per part (ms)", dx=40, font=T)
    b.hline(g, ORANGE, 1.0, "4 2")
    for i, sess in enumerate(d1["sessions"], 1):
        b.erry(i, sess["mean_ms"] - sess["sd_ms"], sess["mean_ms"] + sess["sd_ms"], GREY_M, 0.8, 1.6)
        b.marker(i, sess["mean_ms"], "o", ORANGE, True, 6)
    d.text(b.x + 2, b.Y(g + 2 * u) - 15, b.w - 4, 13, "band: ±2<i>u</i>, one measurement", font=TS,
           color=ACCENT_TXT)
    d.text(b.x + 2, b.Y(g - 2 * u) + 3, b.w - 4, 13, f"dashed: mean {g:.2f} ms", font=TS, color=ACCENT_TXT)
    return d


FIGS = {"fig8": (fig8, OUT / "fig7_operating_points.drawio"),
        "fig10": (fig10, OUT / "fig9_cost.drawio")}


def main(argv):
    for name in argv or list(FIGS):
        fn, path = FIGS[name]
        fn().save(path)
        v2.export(path)
        print("wrote", path, "and PNG/PDF")


if __name__ == "__main__":
    main(sys.argv[1:])
