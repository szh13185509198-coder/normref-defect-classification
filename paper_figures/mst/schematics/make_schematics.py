#!/usr/bin/env python3
"""Editable colour schematics for the MST paper, written as draw.io files and rendered to PDF / PNG.

    python paper_figures/mst/schematics/make_schematics.py            # write .drawio, then render
    python paper_figures/mst/schematics/make_schematics.py --no-render

Colour roles (validated palette, see drawio.py): blue = set-up from labelled defect images, aqua = the normal
reference (conforming parts), orange = per-part inspection, violet = decisions and reported results, grey = uncertainty
tags. Every box carries its own text, so colour is never the only cue. Edit the .drawio files in draw.io; re-render
with render.js (or export from draw.io: File > Export as > PDF, "crop", border 0).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drawio import AQUA, BLUE, F, F_TITLE, GREY, INKC, ORANGE, TINT, VIOLET, Diagram  # noqa: E402

ASSETS = HERE / "assets" / "bottle"   # make_assets.py: official MVTec AD images and maps (ASSETS.json)
MANIFEST = __import__("json").loads((ASSETS / "ASSETS.json").read_text(encoding="utf-8"))
DEFECTS = [("support_broken_large", "broken (large)"), ("support_broken_small", "broken (small)"),
           ("support_contamination", "contamination")]
PRED = "broken (small)"
OUT = HERE / "out"
W = 563                 # panel frames carry a stroke: with it and the 1 px border the page is 567 px = 15 cm


def badge(d: Diagram, x, y, text):
    """Small grey uncertainty tag (U1-U6) sitting on a box corner."""
    d.box(x, y, 24, 15, text, color=GREY, fill="#ffffff", font=F, bold=True, dashed=True, spacing=0)


# --------------------------------------------------------------------------- figure 1: the measurement chain

def arrow(d, x0, y0, x1, y1, color=INKC):
    d.line([(x0, y0), (x1, y1)], color=color, end="block")


def caption(d, x, y, w, text, font=F, h=28):
    d.text(x, y, w, h, text, font=font, align="center", valign="top")


def measurement_chain() -> Diagram:
    import mini
    d = Diagram("measurement_chain", W, 418)
    normals = [ASSETS / f"normal_{j}.png" for j in range(3)]
    T_A = "t<sub>α</sub>"

    # ---------------- (a) set-up
    d.box(0, 0, W, 216, "", color=BLUE, fill="#f6f9fd", rounded=False, stroke_width=0.8)
    d.text(6, 4, 400, 16, "(a) Set-up, once per product: nothing is trained", font=F_TITLE, bold=True)
    # conforming parts -> bank -> tau
    for j in range(3):
        d.image(10 + 7 * j, 28 + 4 * j, 40, 40, normals[j])
    caption(d, 0, 80, 80, "M = 20<br>conforming parts")
    badge(d, 62, 22, "U3")
    arrow(d, 72, 54, 94, 54)
    d.image_png(96, 30, 74, 49, mini.to_png(mini.patch_mosaic(normals)))
    d.symbol(162, 22, "B", AQUA)
    caption(d, 84, 82, 98, "bank B of<br>M × 576 patches")
    arrow(d, 176, 54, 196, 54)
    d.image_png(198, 28, 72, 46, mini.score_histogram(AQUA, seed=1))
    d.symbol(262, 22, "τ", AQUA)
    caption(d, 186, 82, 96, "patch threshold τ<br>(0.99 quantile)")
    # calibration parts -> t_alpha
    for j in range(3):
        d.image(312 + 7 * j, 28 + 4 * j, 40, 40, normals[(j + 1) % 3])
    caption(d, 300, 80, 82, "N = 100<br>conforming parts")
    badge(d, 364, 22, "U1")
    arrow(d, 374, 54, 396, 54)
    d.image_png(398, 28, 72, 46, mini.score_histogram(AQUA, tail=VIOLET, seed=3))
    d.symbol(462, 22, T_A, VIOLET)
    caption(d, 380, 82, 128, "acceptance limit " + T_A + "<br>(false-reject rate α)")
    # defect examples -> prototypes
    for j, (f, lab) in enumerate(DEFECTS):
        d.image(12 + 60 * j, 124, 40, 40, ASSETS / f"{f}.png")
        caption(d, 60 * j, 166, 64, lab.replace(" (", "<br>("))
    caption(d, 0, 196, 190, "K labelled images per defect type", h=15)
    badge(d, 172, 118, "U3")
    arrow(d, 198, 144, 220, 144)
    d.image_png(222, 120, 78, 52, mini.prototype_scatter(query=False, n=3))
    d.symbol(294, 114, "p", BLUE)
    caption(d, 206, 176, 110, "class prototypes p<sub>c</sub>", h=15)
    d.text(330, 118, 230, 92, "Frozen image features only; a new defect type adds one prototype (about 2 s). "
                              "Coloured symbols mark each set-up quantity and the inspection step that uses it; "
                              "grey tags U1–U6 mark the uncertainty components of section 5.5.",
           color="#333333", valign="top")

    # ---------------- (b) inspection
    y0 = 224
    d.box(0, y0, W, 194, "", color=ORANGE, fill="#fef8f4", rounded=False, stroke_width=0.8)
    d.text(6, y0 + 4, 400, 16, "(b) Inspection of each part, independently", font=F_TITLE, bold=True)
    badge(d, W - 30, y0 + 5, "U6")
    yi = y0 + 36
    d.image(10, yi, 66, 66, ASSETS / "query.png")
    badge(d, 58, yi - 9, "U4")
    caption(d, 0, yi + 70, 86, "image of<br>the part")
    arrow(d, 78, yi + 33, 98, yi + 33)
    d.image(100, yi, 66, 66, ASSETS / "query_anomaly_map.png")
    d.symbol(158, yi - 9, "B", AQUA)
    caption(d, 84, yi + 70, 100, "patch scores a(t);<br>image score s(x)")
    arrow(d, 168, yi + 33, 188, yi + 33)
    d.rhombus(190, yi + 3, 88, 60, "s(x) &gt; " + T_A + " ?", color=VIOLET)
    d.symbol(270, yi - 8, T_A, VIOLET)
    badge(d, 190, yi - 9, "U2")
    # no -> accept
    d.text(238, yi + 67, 20, 14, "no")
    arrow(d, 234, yi + 63, 234, yi + 92)
    d.image(209, yi + 94, 50, 50, normals[0])
    d.box(247, yi + 130, 18, 18, "✓", color=AQUA, fill=AQUA, text_color="#ffffff", bold=True, shape="ellipse",
          rounded=False, spacing=0)
    d.text(270, yi + 102, 80, 32, "<b>accept</b><br>conforming")
    # yes -> evidence -> prototype -> reject
    d.text(281, yi + 16, 26, 14, "yes")
    arrow(d, 278, yi + 33, 298, yi + 33)
    d.image(300, yi, 66, 66, ASSETS / "query_weight_map.png")
    d.symbol(358, yi - 9, "τ", AQUA)
    caption(d, 286, yi + 70, 94, "patches above τ,<br>weighted")
    arrow(d, 368, yi + 33, 388, yi + 33)
    d.image_png(390, yi + 6, 82, 55, mini.prototype_scatter(query=True, n=3))
    d.symbol(464, yi - 9, "p", BLUE)
    caption(d, 382, yi + 70, 100, "nearest prototype,<br>three scores")
    arrow(d, 474, yi + 33, 492, yi + 33)
    d.image(494, yi + 4, 58, 58, ASSETS / "query.png")
    d.box(538, yi + 48, 18, 18, "✕", color=ORANGE, fill=ORANGE, text_color="#ffffff", bold=True, shape="ellipse",
          rounded=False, spacing=0)
    d.text(484, yi + 70, 80, 32, "<b>reject:</b><br>" + PRED, align="center")
    return d


# --------------------------------------------------------------------------- uncertainty sources

def uncertainty_sources() -> Diagram:
    import mini
    normals = [ASSETS / f"normal_{j}.png" for j in range(3)]
    supports = [ASSETS / f"{f}.png" for f, _ in DEFECTS]
    cards_top = [
        ("Calibration set", AQUA, "U1", mini.beta_curves(), "Which N conforming parts set t<sub>α</sub>; the size N",
         "Type A: 1000 resampled calibration sets vs Beta(N − k + 1, k); curves: N = 50, 100, 500"),
        ("Normal reference", AQUA, "U3", mini.to_png(mini.grid(normals, n_cols=3, size=120)),
         "Which M conforming parts form the bank B; the size M",
         "Type A: 5 bank draws × 5 support draws; M = 5, 10, 20, 50"),
        ("Support set", BLUE, "U3", mini.to_png(mini.grid(supports, n_cols=3, size=120)),
         "Which K examples represent each defect type; K", "Type A: 5 support draws (crossed with the bank)")]
    cards_bottom = [
        ("Test sample", GREY, "U3", mini.visa_gain_dots(), "Finite number of test parts and objects",
         "Type A: bootstrap over test parts and objects (small plot: gain per VisA object)"),
        ("Influence quantities", ORANGE, "U4", mini.to_png(mini.perturbations(ASSETS / "query.png")),
         "Lighting, gamma, noise, blur, JPEG, shift, rotation",
         "Sensitivity with and without recalibration; agreement over repeats"),
        ("Computing platform", GREY, "U6", mini.chip_icon(),
         "Precision, engine, process session, clock and temperature",
         "Type A: 10 sessions, 5 energy windows. Type B (accuracy only): PyTorch–TensorRT difference")]
    w, gap, x0, hc = 182, 8, 2, 146
    sw = 3 * w + 2 * gap
    d = Diagram("uncertainty_sources", W, 2 * hc + 34 + 2 * 22 + 4)
    spine_y = 2 + hc + 22
    spine = d.box(x0, spine_y, sw, 34, "Reported results: false-reject rate, escapes per 1000 parts, "
                  "balanced accuracy, latency and energy", color=VIOLET, bold=True)

    def card(x, y, title, color, tag, png, what, how):
        head = d.box(x, y, w, 20, title, color=color, fill=color, text_color="#ffffff", bold=True, rounded=False)
        body = d.box(x, y + 20, w, hc - 20, "", color=color, fill=TINT[color], rounded=False)
        d.image_png(x + 4, y + 26, 64, 50, png)
        d.text(x + 72, y + 24, w - 76, 58, what, valign="middle")
        d.text(x + 5, y + 86, w - 10, hc - 88, f"<i>{how}</i>", color="#333333", valign="top")
        badge(d, x + w - 27, y + 2, tag)
        return head, body

    for j, c in enumerate(cards_top):
        x = x0 + j * (w + gap)
        _, b = card(x, 2, *c)
        d.edge(b, spine, color=c[1], entry=(((x + w / 2) - x0) / sw, 0))
    for j, c in enumerate(cards_bottom):
        x = x0 + j * (w + gap)
        h, _ = card(x, spine_y + 34 + 22, *c)
        d.edge(h, spine, color=c[1], entry=(((x + w / 2) - x0) / sw, 1))
    return d


# --------------------------------------------------------------------------- method detail (one query, real maps)

def method_detail() -> Diagram:
    """Assets from make_assets.py (MVTec AD bottle, test/broken_small/014); the number of patches above tau is read
    from its manifest."""
    d = Diagram("method_detail", W, 322)
    d.box(0, 0, W, 98, "", color=BLUE, fill="#f6f9fd", rounded=False, stroke_width=0.8)
    d.text(6, 3, 300, 16, "(a) Fixed before any query", font=F_TITLE, bold=True)
    d.box(0, 102, W, 220, "", color=ORANGE, fill="#fef8f4", rounded=False, stroke_width=0.8)
    d.text(6, 105, 200, 16, "(b) For each query", font=F_TITLE, bold=True)

    # (a) normal images -> bank -> threshold; support images -> prototypes
    for j in range(3):
        d.image(10 + 7 * j, 24 + 4 * j, 38, 38, ASSETS / f"normal_{j}.png")
    d.text(0, 74, 76, 14, "M conforming parts", align="center")
    bank = d.box(84, 32, 100, 40, "Patch bank B<br>M × 576 patches", color=AQUA)
    tau = d.box(196, 28, 90, 48, "Threshold τ:<br>0.99 quantile,<br>leave-one-out", color=AQUA)
    d.line([(64, 52), (84, 52)], color=INKC, end="block")
    d.edge(bank, tau, color=AQUA)
    for j, (f, lab) in enumerate(DEFECTS):
        d.image(298 + 60 * j, 22, 36, 36, ASSETS / f"{f}.png")
        d.text(278 + 60 * j, 60, 76, 30, lab.replace(" (", "<br>("), align="center", valign="top")
    pc = d.box(484, 32, 78, 40, "Class<br>prototypes p<sub>c</sub>", color=BLUE)
    d.line([(472, 52), (484, 52)], color=INKC, end="block")

    # (b) query
    d.image(8, 170, 62, 62, ASSETS / "query.png")
    d.text(0, 236, 78, 30, "Query: bottle<br>(whole image)", align="center", valign="top")
    d.box(82, 124, 262, 182, "", color=AQUA, fill="#ffffff", stroke_width=1.2)
    d.text(88, 286, 250, 15, "Normal-referenced evidence selection", bold=True, color=AQUA, align="center")
    s1 = d.image(94, 140, 62, 62, ASSETS / "query_anomaly_map.png")
    s2 = d.box(170, 146, 76, 50, f"{MANIFEST['patches_above_tau']} of {MANIFEST['patches']}<br>patches<br>exceed τ",
               color=AQUA)
    s3 = d.image(262, 140, 62, 62, ASSETS / "query_weight_map.png")
    d.text(86, 208, 78, 100, "(1) Score:<br>a(t), distance to the nearest patch in B", align="center", valign="top")
    d.text(166, 208, 84, 100, "(2) Gate:<br>normal variation ends at τ", align="center", valign="top")
    d.text(254, 208, 84, 100, "(3) Weight:<br>w = max(a − τ, 0)", align="center", valign="top")
    d.line([(70, 196), (94, 176)], color=INKC, end="block")
    d.edge(s1, s2)
    d.edge(s2, s3)
    d.edge(bank, s1, color=AQUA, dashed=True, exit=(0.2, 1), entry=(0.5, 0))
    d.edge(tau, s2, color=AQUA, dashed=True)

    ha = d.box(356, 126, 98, 46, "Appearance h<sub>A</sub>:<br>weighted patch mean", color=ORANGE)
    hr = d.box(356, 184, 98, 46, "Deviation h<sub>R</sub>:<br>weighted t − nn<sub>B</sub>(t)", color=ORANGE)
    g = d.box(356, 236, 98, 36, "Global feature g", color=ORANGE)
    d.edge(s3, ha, exit=(1, 0.5))
    d.edge(s3, hr, exit=(1, 0.5))
    d.line([(39, 268), (39, 314), (348, 314), (348, 254), (356, 254)], color=INKC, end="block")
    clf = d.box(466, 126, 96, 140, "Distance to each p<sub>c</sub>; z-scores of the three summed", color=BLUE)
    for b, ey in ((ha, 23 / 140), (hr, 81 / 140), (g, 128 / 140)):
        d.edge(b, clf, exit=(1, 0.5), entry=(0, ey))
    d.edge(pc, clf, color=BLUE, dashed=True)
    out = d.box(466, 276, 96, 40, "ŷ = broken (small)", color=VIOLET, bold=True)
    d.edge(clf, out)
    return d

# --------------------------------------------------------------------------- main

FIGS = {"schem_measurement_chain": measurement_chain, "schem_method_detail": method_detail,
        "schem_uncertainty_sources": uncertainty_sources}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for name, fn in FIGS.items():
        p = OUT / f"{name}.drawio"
        fn().save(p)
        files.append(str(p))
        print(f"[schem] wrote {p.name}")
    if "--no-render" not in argv:
        viewer = os.environ.get("DRAWIO_VIEWER", str(HERE / "viewer-static.min.js"))
        if not Path(viewer).is_file():
            print(f"[schem] viewer not found ({viewer}); download https://viewer.diagrams.net/js/viewer-static.min.js "
                  "or set DRAWIO_VIEWER")
            return 1
        subprocess.run(["node", str(HERE / "render.js"), viewer, str(OUT), *files], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
