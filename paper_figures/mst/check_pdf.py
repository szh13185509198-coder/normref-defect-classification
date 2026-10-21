#!/usr/bin/env python3
"""Print-size checks for the matplotlib figures (paper_figures/mst/out/*.pdf), read from the PDF itself.

1. width at most 15 cm (8.5 cm figures are single-column); the figures are saved at their final size, so the PDF is
   placed at 100 %;
2. every glyph run is at least 8 pt;
3. no two text spans overlap (0.3 pt tolerance), and no text is cut by the page edge;
4. every font is embedded (pdffonts) and the text is real text.

    python paper_figures/mst/check_pdf.py [pdf ...]      # default: every PDF in out/
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
MIN_PT, MAX_W_CM = 8.0, 15.0


def check(pdf: Path) -> list[str]:
    errs = []
    doc = pymupdf.open(pdf)
    page = doc[0]
    w_cm, h_cm = page.rect.width / 72 * 2.54, page.rect.height / 72 * 2.54
    if w_cm > MAX_W_CM + 0.01:
        errs.append(f"{w_cm:.2f} cm wide (> 15 cm)")
    spans = [sp for b in page.get_text("dict")["blocks"] for ln in b.get("lines", []) for sp in ln["spans"]
             if sp["text"].strip()]
    small = [(sp["text"][:25], round(sp["size"], 2)) for sp in spans if sp["size"] < MIN_PT - 1e-6]
    if small:
        errs.append(f"{len(small)} spans below {MIN_PT} pt, e.g. {small[:4]}")
    pr = page.rect
    out = [sp["text"][:25] for sp in spans if not pymupdf.Rect(sp["bbox"]) in pr + (-0.5, -0.5, 0.5, 0.5)]
    if out:
        errs.append(f"{len(out)} text spans cut by the page edge, e.g. {out[:4]}")
    bad = []
    for i in range(len(spans)):
        a = pymupdf.Rect(spans[i]["bbox"])
        for sp in spans[i + 1:]:
            r = a & pymupdf.Rect(sp["bbox"])
            if r.width > 0.3 and r.height > 0.3 and r.width * r.height > 1.0:
                bad.append((spans[i]["text"][:20], sp["text"][:20]))
    if bad:
        errs.append(f"{len(bad)} overlapping text pairs, e.g. {bad[:4]}")
    rows = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout.splitlines()[2:]
    if not rows:
        errs.append("no fonts in the PDF")
    if [r for r in rows if r.split()[-5] != "yes"]:
        errs.append("fonts not embedded")
    fonts = sorted({sp["font"] for sp in spans})
    sizes = [sp["size"] for sp in spans]
    print(f"{'FAIL' if errs else 'ok  '} {pdf.stem:38s} {w_cm:4.1f} x {h_cm:4.1f} cm  text "
          f"{min(sizes):.2f}-{max(sizes):.2f} pt  {fonts}")
    for e in errs:
        print(f"      - {e}")
    return errs


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    pdfs = [Path(a) for a in argv] or sorted((HERE / "out").glob("*.pdf"))
    failed = sum(bool(check(p)) for p in pdfs)
    print(f"{failed} figure(s) failed")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
