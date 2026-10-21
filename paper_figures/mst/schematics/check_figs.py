#!/usr/bin/env python3
"""Print-size checks for the draw.io figures (charts/ and out/), run on the rendered output.

For every figure:
1. width: the rendered page must be at most 15 cm (567 px at 96 px/in, + 2 px border), so it is placed at 100 %;
   a wider page would be scaled down and is reported as a failure;
2. font size: every text run, as measured by the browser, is at least 8 pt at the printed size
   (font px x scale x 72/96); the .drawio source is checked too (fontSize >= MIN_PX on every labelled cell), so the
   rule still holds after editing in draw.io at the same size;
3. overlap: no two text runs overlap (rendered boxes, 0.5 px tolerance);
4. bounds: no text run leaves the page;
5. fonts: every font in the PDF is embedded (pdffonts), and the PDF carries real text (pdftotext).

    TEXT_METRICS_DIR=<dir> node render.js ...      # writes <dir>/<name>.text.json while rendering
    python check_figs.py <dir>                     # checks every figure that has metrics
Exit status 1 if any check fails.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDERS = (HERE / "charts", HERE / "out")
MAX_W = 567            # 15 cm at 96 px/in: the page (content plus the 1 px border) must not be wider
MIN_PT = 8.0
MIN_PX = MIN_PT * 96 / 72  # 10.67 px at 100 %


def source_fonts(drawio: Path) -> list[tuple[float, str]]:
    """(fontSize, label) of every cell that shows text."""
    s = drawio.read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r'<mxCell id="[^"]*" value="([^"]*)" style="([^"]*)"', s):
        val, sty = m.groups()
        if not re.sub(r"&lt;[^&]*&gt;|<[^>]*>", "", val).strip():
            continue
        f = re.search(r"fontSize=([\d.]+)", sty)
        out.append((float(f.group(1)) if f else 11.0, val))   # draw.io default is 11
    return out


def overlaps(runs):
    """Pairs of text runs whose rendered line boxes intersect (each wrapped line is its own box)."""
    bad = []
    for i in range(len(runs)):
        a = runs[i]
        for b in runs[i + 1:]:
            worst = 0.0
            for ax, ay, aw, ah in a.get("lines", [[a["x"], a["y"], a["w"], a["h"]]]):
                for bx, by, bw, bh in b.get("lines", [[b["x"], b["y"], b["w"], b["h"]]]):
                    ix = min(ax + aw, bx + bw) - max(ax, bx) - 0.5
                    iy = min(ay + ah, by + bh) - max(ay, by) - 0.5
                    if ix > 0 and iy > 0:
                        worst = max(worst, ix * iy)
            if worst > 1.0:
                bad.append((a["text"][:30], b["text"][:30], round(worst, 1)))
    return bad


def check(name: str, metrics: dict, drawio: Path, pdf: Path) -> list[str]:
    errs = []
    w, h = metrics["page"]["w"], metrics["page"]["h"]
    scale = min(1.0, MAX_W / w)
    if w > MAX_W + 0.5:
        errs.append(f"page {w:.0f} px = {w / 96 * 2.54:.1f} cm wide (> 15 cm)")
    runs = metrics["runs"]
    small = [(r["text"][:30], round(r["font_px"] * scale * 0.75, 2)) for r in runs if r["font_px"] * scale * 0.75 < MIN_PT - 1e-6]
    if small:
        errs.append(f"{len(small)} text runs below {MIN_PT} pt, e.g. {small[:4]}")
    src = [(f, v[:30]) for f, v in source_fonts(drawio) if f < MIN_PX - 1e-6]
    if src:
        errs.append(f"{len(src)} cells with fontSize < {MIN_PX:.2f} px in the .drawio, e.g. {src[:4]}")
    ov = overlaps(runs)
    if ov:
        errs.append(f"{len(ov)} overlapping text pairs, e.g. {ov[:4]}")
    out = [r["text"][:30] for r in runs if r["x"] < -0.5 or r["y"] < -0.5 or r["x"] + r["w"] > w + 0.5 or r["y"] + r["h"] > h + 0.5]
    if out:
        errs.append(f"{len(out)} text runs outside the page, e.g. {out[:4]}")
    if shutil.which("pdffonts") and pdf.is_file():
        rows = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout.splitlines()[2:]
        not_emb = [r.split()[0] for r in rows if r.split()[-5] != "yes"]
        if not rows:
            errs.append("PDF has no fonts (text converted to outlines?)")
        if not_emb:
            errs.append(f"fonts not embedded: {not_emb}")
        txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
        if len(txt.strip()) < 3:
            errs.append("PDF has no extractable text")
    fonts = sorted({r["family"].split(",")[0].strip('"') for r in runs})
    sizes = sorted({round(r["font_px"] * scale * 0.75, 2) for r in runs})
    print(f"{'FAIL' if errs else 'ok  '} {name:30s} {w / 96 * 2.54:4.1f} x {h / 96 * 2.54:4.1f} cm  "
          f"text {min(sizes) if sizes else 0:.2f}-{max(sizes) if sizes else 0:.2f} pt  font {fonts}  runs {len(runs)}")
    for e in errs:
        print(f"      - {e}")
    return errs


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    mdir = Path(argv[0])
    failed = 0
    for f in sorted(mdir.glob("*.text.json")):
        name = f.name[:-len(".text.json")]
        folder = next((d for d in FOLDERS if (d / f"{name}.drawio").is_file()), None)
        if folder is None:
            continue
        failed += bool(check(name, json.loads(f.read_text()), folder / f"{name}.drawio", folder / f"{name}.pdf"))
    print(f"{failed} figure(s) failed")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
