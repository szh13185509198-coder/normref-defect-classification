"""Minimal writer for editable draw.io diagrams (uncompressed XML, opens directly in draw.io / diagrams.net).

Coordinates are draw.io px at 100 % zoom; the page is drawn at its final printed size (567 px = 15 cm, 96 px per
inch), so fontSize 11 px is 8.25 pt in print. No text is smaller than F (8.25 pt); check_figs.py enforces >= 8 pt on
the rendered output.
"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

FONT = "Arial"          # rendered here with Liberation Sans (metric-compatible); Arial on Windows / macOS
F, F_TITLE, F_PANEL = 11, 11.5, 12   # px at the printed size: 8.25, 8.6 and 9 pt

# Validated categorical palette (dataviz reference palette, light mode): stroke/header colour and a light tint.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
TINT = {BLUE: "#e9f1fb", ORANGE: "#fdeee7", AQUA: "#e6f6ef", VIOLET: "#ecebf6", "#7a7a7a": "#f2f2f2"}
GREY, INKC = "#7a7a7a", "#1a1a1a"


SUB = f'<sub style="font-size:{F}px">'   # subscripts at the minimum text size too (the browser default is 83 %)


def label(text: str) -> str:
    return text.replace("<sub>", SUB)


def style(**kw) -> str:
    return ";".join(f"{k}={v}" for k, v in kw.items()) + ";"


class Diagram:
    def __init__(self, name: str, width: int, height: int):
        self.name, self.width, self.height = name, width, height
        self.cells: list[str] = []
        self.n = 0

    def _id(self, given: str | None) -> str:
        self.n += 1
        return given or f"c{self.n}"

    def box(self, x, y, w, h, text="", color=GREY, fill=None, id=None, font=F, bold=False, rounded=True,
            dashed=False, align="center", valign="middle", stroke_width=1, shape=None, text_color=INKC,
            spacing=3, opacity=None, parent="1") -> str:
        i = self._id(id)
        st = dict(rounded=1 if rounded else 0, whiteSpace="wrap", html=1, fontFamily=FONT, fontSize=font,
                  fontColor=text_color, strokeColor=color, fillColor=fill if fill is not None else TINT.get(color, "#ffffff"),
                  strokeWidth=stroke_width, align=align, verticalAlign=valign, spacing=spacing, arcSize=8)
        if bold:
            st["fontStyle"] = 1
        if dashed:
            st.update(dashed=1, dashPattern="3 2")
        if shape:
            st["shape"] = shape
        if opacity is not None:
            st["opacity"] = opacity
        self.cells.append(f'<mxCell id="{i}" value="{escape(label(text))}" style="{style(**st)}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def text(self, x, y, w, h, text, font=F, bold=False, align="left", valign="middle", color=INKC, italic=False,
             id=None) -> str:
        i = self._id(id)
        if x < 0 or x + w > self.width:          # keep every text box on the page (the page size is the bounding box)
            x0, x1 = max(x, 0), min(x + w, self.width)
            if align == "center":                # stay centred on the same point
                c = x + w / 2
                half = min(c - x0, x1 - c)
                x0, x1 = c - half, c + half
            x, w = x0, x1 - x0
        fs = (1 if bold else 0) + (2 if italic else 0)
        st = dict(text=None, html=1, whiteSpace="wrap", fontFamily=FONT, fontSize=font, fontColor=color, align=align,
                  verticalAlign=valign, strokeColor="none", fillColor="none", spacing=0, fontStyle=fs)
        s = ";".join(k if v is None else f"{k}={v}" for k, v in st.items()) + ";"
        self.cells.append(f'<mxCell id="{i}" value="{escape(label(text))}" style="{s}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def rhombus(self, x, y, w, h, text, color, font=F, id=None) -> str:
        return self.box(x, y, w, h, text, color=color, font=font, id=id, shape="rhombus", rounded=False, spacing=0)

    def image(self, x, y, w, h, path: Path, max_px=160, border=GREY, id=None) -> str:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        data = base64.b64encode(buf.getvalue()).decode()
        i = self._id(id)
        st = (f"shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;"
              f"imageBorder={border};strokeWidth=0.5;image=data:image/jpeg,{data};")
        self.cells.append(f'<mxCell id="{i}" value="" style="{st}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def image_png(self, x, y, w, h, png: bytes, border=None, id=None) -> str:
        """Embed PNG bytes (keeps transparency); border=None draws no frame."""
        from PIL import Image
        iw, ih = Image.open(io.BytesIO(png)).size  # fit inside w x h, keep the aspect ratio, centred
        s = min(w / iw, h / ih)
        x, y, w, h = x + (w - iw * s) / 2, y + (h - ih * s) / 2, iw * s, ih * s
        data = base64.b64encode(png).decode()
        i = self._id(id)
        b = f"imageBorder={border};strokeWidth=0.5;" if border else ""
        st = f"shape=image;imageAspect=0;aspect=fixed;{b}image=data:image/png,{data};"
        self.cells.append(f'<mxCell id="{i}" value="" style="{st}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def symbol(self, x, y, text, color, size=18, font=F) -> str:
        """Filled circle with a white symbol (B, tau, t_alpha, p_c): marks a set-up quantity and where it is used."""
        return self.box(x, y, size, size, text, color=color, fill=color, text_color="#ffffff", bold=True, font=font,
                        shape="ellipse", rounded=False, spacing=0)

    def edge(self, src, tgt, label="", color=INKC, dashed=False, points=(), exit=None, entry=None, width=1, font=F,
             end="block", start="none", id=None, ortho=True) -> str:
        i = self._id(id)
        st = dict(html=1, endArrow=end, endFill=1, startArrow=start, endSize=5, strokeColor=color, strokeWidth=width,
                  fontFamily=FONT, fontSize=font, fontColor=INKC, labelBackgroundColor="#ffffff", rounded=0)
        if ortho:
            st["edgeStyle"] = "orthogonalEdgeStyle"
        if dashed:
            st.update(dashed=1, dashPattern="4 3")
        for k, v in (("exit", exit), ("entry", entry)):
            if v:
                st[f"{k}X"], st[f"{k}Y"] = v
                st[f"{k}Dx"] = st[f"{k}Dy"] = 0
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
        geo = (f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry>' if pts else
               '<mxGeometry relative="1" as="geometry"/>')
        self.cells.append(f'<mxCell id="{i}" value="{escape(label)}" style="{style(**st)}" edge="1" parent="1" '
                          f'source="{src}" target="{tgt}">{geo}</mxCell>')
        return i

    def line(self, pts, color=GREY, dashed=False, width=1, end="none", id=None) -> str:
        """Free polyline from pts[0] to pts[-1] (no source/target)."""
        i = self._id(id)
        st = dict(html=1, endArrow=end, endFill=1, endSize=5, strokeColor=color, strokeWidth=width, rounded=0)
        if dashed:
            st.update(dashed=1, dashPattern="4 3")
        (x0, y0), (x1, y1), mid = pts[0], pts[-1], pts[1:-1]
        arr = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in mid)
        self.cells.append(f'<mxCell id="{i}" value="" style="{style(**st)}" edge="1" parent="1">'
                          f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x0}" y="{y0}" as="sourcePoint"/>'
                          f'<mxPoint x="{x1}" y="{y1}" as="targetPoint"/>'
                          + (f'<Array as="points">{arr}</Array>' if arr else "") + '</mxGeometry></mxCell>')
        return i

    def xml(self) -> str:
        body = "".join(self.cells)
        return (f'<mxfile host="normref-mst" type="device"><diagram id="{self.name}" name="{self.name}">'
                f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="5" guides="1" tooltips="1" connect="1" arrows="1" '
                f'fold="1" page="1" pageScale="1" pageWidth="{self.width}" pageHeight="{self.height}" math="0" '
                f'shadow="0" background="#ffffff"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root>'
                f'</mxGraphModel></diagram></mxfile>')

    def save(self, path: Path) -> None:
        path.write_text(self.xml(), encoding="utf-8")
