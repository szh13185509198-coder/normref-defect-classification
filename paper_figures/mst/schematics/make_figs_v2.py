#!/usr/bin/env python3
"""Manuscript figures 1 and 6 as editable draw.io files (figures 8 and 10: make_figs_v3.py).

    python paper_figures/mst/schematics/make_figs_v2.py fig1 fig6   # writes .drawio, exports PNG and PDF

Print rules (IOP: text 8-12 pt at the final size, 15 cm full width; colour never the only cue): the page is drawn at
15 cm (565 px content + 1 px border = 567 px at 96 px/in); body text 12 px = 9 pt, nothing below 11 px = 8.25 pt.

fig1   Figure 1 (measurement chain): edit of the hand-drawn drawio/schematics/schem_measurement_chain_base.drawio.
fig6   Figure 6 (patch evidence): tiles cropped from drawio/schematics/fig5_patch_selection_old_matplotlib.png, the
       export of experiments/p12/make_figures_v3.fig_region_comparison; numbers from fig5_patch_selection.csv.

Outputs go to paper_figures/mst/drawio/. The PNG / PDF export calls the draw.io desktop app (path in DRAWIO); the
.drawio files are written without it.
"""

from __future__ import annotations

import base64
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drawio import AQUA, BLUE, ORANGE, VIOLET, Diagram  # noqa: E402

NEW = HERE.parent / "drawio"                 # editable sources and exports of the final figures
DRAWIO = Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\draw.io\draw.io.exe"))
T, TS = 12, 11                  # 9 pt and 8.25 pt at the printed size
GREY_M = "#7a7a7a"


def half_up(x: float) -> int:
    from decimal import Decimal, ROUND_HALF_UP
    return int(Decimal(str(x)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def export(drawio: Path) -> None:
    for fmt, extra in (("png", ["-s", "3.125"]), ("pdf", [])):
        out = drawio.with_suffix("." + fmt)
        subprocess.run([str(DRAWIO), "-x", "-f", fmt, *extra, "--crop", "-o", str(out), str(drawio)],
                       check=True, capture_output=True, timeout=180)


MIN_REGION, MEBIN_STEPS = 6, 64


def image_jpeg(d: Diagram, x, y, side, jpg: bytes, border="#8a8a8a"):
    data = base64.b64encode(jpg).decode()
    st = (f"shape=image;imageAspect=0;aspect=fixed;imageBorder={border};strokeWidth=0.6;"
          f"image=data:image/jpeg,{data};")
    d.cells.append(f'<mxCell id="{d._id(None)}" value="" style="{st}" vertex="1" parent="1">'
                   f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{side:.2f}" height="{side:.2f}" as="geometry"/></mxCell>')


# --------------------------------------------------------------------------- Figure 1 (edit of the hand-edited file)

def fig1_xml() -> str:
    """Figure 1 is hand-edited in draw.io, so it is changed in place (idempotent: always starts from the copy saved
    before this round): the prose block in panel (a) is removed, the defect-example row is centred, and panel (c)
    shows the image sets (tutor comment A17), replacing the separate data-partition sketch."""
    import re
    src = (NEW / "schematics" / "schem_measurement_chain_base.drawio").read_text(encoding="utf-8")
    blocks = re.compile(r'<mxCell id="(c\d+)"[^>]*?(?:/>|>.*?</mxCell>)', re.S)
    def drop(s, cid):
        m = next(m for m in blocks.finditer(s) if m.group(1) == cid)
        return s[:m.start()] + s[m.end():]
    def shift(s, cid, dx):
        m = next(m for m in blocks.finditer(s) if m.group(1) == cid)
        blk = re.sub(r'(<mx(?:Geometry|Point)[^>]*?x=")([-\d.]+)"', lambda g: f'{g.group(1)}{float(g.group(2)) + dx:g}"',
                     m.group(0))
        if 'x="' not in re.search(r'<mxGeometry[^>]*', blk).group(0) and "edge=" not in blk[:200]:
            blk = blk.replace("<mxGeometry ", f'<mxGeometry x="{dx}" ', 1)
        return s[:m.start()] + blk + s[m.end():]
    src = drop(src, "c37").replace("K labelled images", "K labeled images")
    for cid in ("c25", "c26", "c27", "c28", "c29", "c30", "c31", "c33", "c34", "c35", "c36"):
        src = shift(src, cid, 118)
    # panel (c): image sets
    y0, h = 426, 92
    d = Diagram("tmp", 563, y0 + h)
    d.box(0, y0, 563, h, "", color=GREY_M, fill="#f7f7f7", rounded=False, stroke_width=0.8)
    d.text(6, y0 + 4, 551, 16, "<b>(c) Image sets.</b> Bank, calibration, support and query images are disjoint",
           font=T)
    sets = [("Development", "pseudo-tasks built<br>from support images", GREY_M),
            ("Reference bank", "M = 20 conforming parts<br>→ B, τ", AQUA),
            ("Calibration set", "N = 100 conforming parts<br>→ T<sub>α</sub>", VIOLET),
            ("Support set", "K images per<br>defect type → p<sub>c</sub>", BLUE),
            ("Test queries", "never used<br>in set-up", ORANGE)]
    gap, bw = 6.0, (563 - 12 - 4 * 6.0) / 5
    for j, (head, body, c) in enumerate(sets):
        x = 6 + j * (bw + gap)
        d.box(x, y0 + 26, bw, 60, f"<b>{head}</b><br>{body}", color=c, font=TS, valign="middle")
    cells = "".join(d.cells).replace('id="c', 'id="pc').replace('parent="c', 'parent="pc')
    src = src.replace("</root>", cells + "</root>", 1)
    src = re.sub(r'pageHeight="\d+"', f'pageHeight="{y0 + h}"', src, count=1)
    return src


# --------------------------------------------------------------------------- Figure 6, classic look

def fig6_classic() -> Diagram:
    """The earlier look kept (red regions above each per-image threshold; normal-calibrated weights as a heat map),
    rebuilt at 15 cm so that every label is 11 px = 8.25 pt. Tiles are cropped from the original export
    (experiments/p12/make_figures_v3.fig_region_comparison); numbers from fig5_patch_selection.csv."""
    src = NEW / "schematics" / "fig5_patch_selection_old_matplotlib.png"
    im = Image.open(src).convert("RGB")
    COLS = [(30, 517), (562, 1049), (1095, 1582), (1627, 2114), (2159, 2646)]
    ROWS = [(101, 588), (667, 1154), (1234, 1720)]
    meta = list(csv.DictReader((NEW / "schematics" / "fig5_patch_selection.csv").open(encoding="utf-8")))
    data = {r["product"]: r for r in meta}
    rows_ = [("screw", "screw — true: scratch head; ours: scratch head; fixed 10%: scratch neck"),
             ("capsule", "capsule — true: crack; ours: crack; fixed 10%: scratch"),
             ("metal_nut", "metal nut — true: bent; ours: bent; fixed 10%: scratch")]
    heads = ["<b>Query</b><br>mask in green", "<b>Fixed top 10%</b><br>(per image)", "<b>Otsu</b><br>(per image)",
             "<b>MEBin</b><br>(per image)", "<b>Ours</b><br>(normal-calibrated)"]
    keys = [None, "fixed_top10", "otsu", "mebin", "ours"]
    W_PAGE, TW = 565, 108.0
    GAP = (W_PAGE - 2 - 5 * TW) / 4
    d = Diagram("fig5_patch_selection", W_PAGE, 10)
    for ci, h in enumerate(heads):
        x = 1 + ci * (TW + GAP)
        d.text(x - GAP / 2, 0, TW + GAP, 29, h, font=TS, align="center", valign="top")
    y = 31
    for prod, lab in rows_:
        d.text(1, y, W_PAGE - 2, 14, lab, font=TS)
        ty = y + 15
        for ci in range(5):
            x = 1 + ci * (TW + GAP)
            tile = im.crop((COLS[ci][0], rows_ and ROWS[[r[0] for r in rows_].index(prod)][0], COLS[ci][1] + 1,
                            ROWS[[r[0] for r in rows_].index(prod)][1] + 1)).resize((420, 420), Image.LANCZOS)
            buf = io.BytesIO(); tile.save(buf, "JPEG", quality=93, subsampling=0)
            image_jpeg(d, x, ty, TW, buf.getvalue(), border="#7a7a7a")
            if keys[ci]:
                n = data[prod][keys[ci] + "_patches"]; lam = float(data[prod][keys[ci] + "_lambda"])
                d.text(x - GAP / 2, ty + TW + 1, TW + GAP, 14, f"{n} patches, λ = {lam:.2f}", font=TS, align="center")
        y = ty + TW + 1 + 14 + 8
    d.height = int(y - 6)
    return d


FIGS = {"fig6": (fig6_classic, NEW / "schematics" / "fig5_patch_selection.drawio")}


def main(argv):
    if "fig1" in argv:
        path = NEW / "schematics" / "schem_measurement_chain.drawio"
        path.write_text(fig1_xml(), encoding="utf-8")
        export(path)
        print("wrote", path, "and PNG/PDF")
        argv = [a for a in argv if a != "fig1"]
        if not argv:
            return
    for name in argv or list(FIGS):
        fn, path = FIGS[name]
        d = fn()
        d.save(path)
        export(path)
        print("wrote", path, "and PNG/PDF")


if __name__ == "__main__":
    main(sys.argv[1:])
