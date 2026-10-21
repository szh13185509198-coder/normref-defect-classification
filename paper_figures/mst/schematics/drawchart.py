"""Data charts as native draw.io objects (axes, ticks, bars, markers, error bars, polylines), so every element stays
editable in draw.io. Coordinates are computed here from the plotted numbers; nothing is traced by hand.

A Plot maps data to page px inside the rectangle (x, y, w, h) of a Diagram. Linear or log axes.
"""

from __future__ import annotations

import math

from drawio import F, F_PANEL, F_TITLE, FONT, INKC, Diagram, label, style
from html import escape

GRID = "#e6e6e6"
AXIS = "#333333"
MUTED = "#555555"


class Plot:
    def __init__(self, d: Diagram, x, y, w, h, xlim, ylim, xlog=False, ylog=False):
        self.d, self.x, self.y, self.w, self.h = d, x, y, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    # ---- mapping
    def _t(self, v, lim, log):
        a, b = lim
        if log:
            v, a, b = math.log10(v), math.log10(a), math.log10(b)
        return (v - a) / (b - a)

    def X(self, v):
        return self.x + self._t(v, self.xlim, self.xlog) * self.w

    def Y(self, v):
        return self.y + self.h - self._t(v, self.ylim, self.ylog) * self.h

    # ---- axes
    def axes(self, left=True, bottom=True):
        if bottom:
            self.d.line([(self.x, self.y + self.h), (self.x + self.w, self.y + self.h)], color=AXIS, width=0.8)
        if left:
            self.d.line([(self.x, self.y), (self.x, self.y + self.h)], color=AXIS, width=0.8)

    def xticks(self, vals, labels=None, grid=True, font=F, tick=3):
        for v, lab in zip(vals, labels or [f"{v:g}".replace("-", "−") for v in vals]):
            px = self.X(v)
            if grid:
                self.d.line([(px, self.y), (px, self.y + self.h)], color=GRID, width=0.6)
            self.d.line([(px, self.y + self.h), (px, self.y + self.h + tick)], color=AXIS, width=0.8)
            self.d.text(px - 30, self.y + self.h + tick + 1, 60, 14, lab, font=font, align="center", valign="top")

    def yticks(self, vals, labels=None, grid=True, font=F, tick=3, width=40):
        for v, lab in zip(vals, labels or [f"{v:g}".replace("-", "−") for v in vals]):
            py = self.Y(v)
            if grid:
                self.d.line([(self.x, py), (self.x + self.w, py)], color=GRID, width=0.6)
            self.d.line([(self.x - tick, py), (self.x, py)], color=AXIS, width=0.8)
            self.d.text(self.x - tick - 2 - width, py - 7, width, 14, lab, font=font, align="right")

    def ycats(self, names, positions, font=F, width=170, bold=()):
        for n, p in zip(names, positions):
            self.d.text(self.x - 6 - width, self.Y(p) - 7, width, 14, n, font=font, align="right", bold=n in bold)

    def xlabel(self, text, dy=18, font=F, h=14):
        self.d.text(self.x, self.y + self.h + dy, self.w, h, text, font=font, align="center", valign="top")

    def ylabel(self, text, dx=34, font=F):
        rot_text(self.d, self.x - dx, self.y + self.h / 2, self.h + 40, text, font)

    def title(self, text, font=F_TITLE, bold=False, dy=-17):
        self.d.text(self.x - 20, self.y + dy, self.w + 40, 15, text, font=font, bold=bold, align="center")

    def panel(self, label, dx=-34, dy=-18):
        self.d.text(self.x + dx, self.y + dy, 26, 15, label, font=F_PANEL, bold=True)

    # ---- marks
    def polyline(self, xs, ys, color, width=1.2, dash=None):
        raw = [(self.X(a), self.Y(b)) for a, b in zip(xs, ys)]
        # thin points closer than 2.5 px on the page: the diagrams.net viewer does not draw sub-pixel edges
        pts = [raw[0]]
        for q in raw[1:-1]:
            if math.dist(q, pts[-1]) >= 2.5:
                pts.append(q)
        pts.append(raw[-1])
        if len(pts) <= 4:
            line(self.d, pts, color, width, dash)
            return
        # Long curves as consecutive two-point segments: the diagrams.net viewer silently skips some edges that
        # carry many interior waypoints, while two-point lines always render.
        for a, b in zip(pts[:-1], pts[1:]):
            line(self.d, [a, b], color, width, dash)

    def hline(self, v, color=AXIS, width=0.8, dash=None):
        line(self.d, [(self.x, self.Y(v)), (self.x + self.w, self.Y(v))], color, width, dash)

    def vline(self, v, color=AXIS, width=0.8, dash=None):
        line(self.d, [(self.X(v), self.y), (self.X(v), self.y + self.h)], color, width, dash)

    def marker(self, xv, yv, kind, color, filled=True, size=6.5, edge=None):
        marker(self.d, self.X(xv), self.Y(yv), kind, color, filled, size, edge)

    def errx(self, yv, lo, hi, color, width=0.8, cap=2.5, dy=0):
        py = self.Y(yv) + dy
        line(self.d, [(self.X(lo), py), (self.X(hi), py)], color, width)
        for v in (lo, hi):
            line(self.d, [(self.X(v), py - cap), (self.X(v), py + cap)], color, width)

    def erry(self, xv, lo, hi, color, width=0.8, cap=2.5):
        px = self.X(xv)
        line(self.d, [(px, self.Y(lo)), (px, self.Y(hi))], color, width)
        for v in (lo, hi):
            line(self.d, [(px - cap, self.Y(v)), (px + cap, self.Y(v))], color, width)

    def hbar(self, yv, value, thickness, color, base=None, edge="none", dash=False):
        base = self.xlim[0] if base is None else base
        x0, x1 = sorted((self.X(base), self.X(value)))
        bar(self.d, x0, self.Y(yv) - thickness / 2, x1 - x0, thickness, color, edge, dash)

    def vbar(self, xv, value, thickness, color, base=None, edge="none"):
        base = self.ylim[0] if base is None else base
        y0, y1 = sorted((self.Y(base), self.Y(value)))
        bar(self.d, self.X(xv) - thickness / 2, y0, thickness, y1 - y0, color, edge)

    def label(self, xv, yv, text, dx=3, dy=0, w=60, align="left", font=F, color=INKC, bold=False):
        px, py = self.X(xv) + dx, self.Y(yv) + dy
        x0 = px if align == "left" else (px - w if align == "right" else px - w / 2)
        self.d.text(x0, py - 7, w, 14, text, font=font, align=align, color=color, bold=bold)


# ---- free primitives

def line(d: Diagram, pts, color, width=1.0, dash=None):
    i = d._id(None)
    st = dict(html=1, endArrow="none", startArrow="none", strokeColor=color, strokeWidth=width, rounded=0)
    if dash:
        st.update(dashed=1, dashPattern=dash)
    (x0, y0), (x1, y1), mid = pts[0], pts[-1], pts[1:-1]
    arr = "".join(f'<mxPoint x="{px:.2f}" y="{py:.2f}"/>' for px, py in mid)
    d.cells.append(f'<mxCell id="{i}" value="" style="{style(**st)}" edge="1" parent="1">'
                   f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x0:.2f}" y="{y0:.2f}" as="sourcePoint"/>'
                   f'<mxPoint x="{x1:.2f}" y="{y1:.2f}" as="targetPoint"/>'
                   + (f'<Array as="points">{arr}</Array>' if arr else "") + '</mxGeometry></mxCell>')


def bar(d: Diagram, x, y, w, h, color, edge="none", dash=False):
    i = d._id(None)
    st = dict(rounded=0, html=1, fillColor=color, strokeColor=edge, strokeWidth=0.8)
    if dash:
        st.update(dashed=1, dashPattern="3 2")
    d.cells.append(f'<mxCell id="{i}" value="" style="{style(**st)}" vertex="1" parent="1">'
                   f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{max(w, 0.1):.2f}" height="{h:.2f}" as="geometry"/></mxCell>')


SHAPES = {"o": ("ellipse", None), "s": ("rectangle", None), "^": ("triangle", "north"), "v": ("triangle", "south"),
          "D": ("rhombus", None), "h": ("hexagon", None)}


def marker(d: Diagram, px, py, kind, color, filled=True, size=6.5, edge=None):
    if kind == "x":
        r = size / 2
        line(d, [(px - r, py - r), (px + r, py + r)], color, 1.6)
        line(d, [(px - r, py + r), (px + r, py - r)], color, 1.6)
        return
    shape, direction = SHAPES[kind]
    s = size * (1.15 if kind in "^vDh" else 1.0)
    i = d._id(None)
    st = dict(shape=shape, html=1, fillColor=color if filled else "#ffffff", strokeColor=edge or color,
              strokeWidth=1.0 if not filled else 0.6, perimeter="none")
    if direction:
        st["direction"] = direction
    if shape == "hexagon":
        st["size"] = 0.25
    d.cells.append(f'<mxCell id="{i}" value="" style="{style(**st)}" vertex="1" parent="1">'
                   f'<mxGeometry x="{px - s / 2:.2f}" y="{py - s / 2:.2f}" width="{s:.2f}" height="{s:.2f}" as="geometry"/></mxCell>')


def rot_text(d: Diagram, cx, cy, length, text, font=F):
    """Text rotated by -90 degrees, centred at (cx, cy)."""
    i = d._id(None)
    h = 15 * (text.count("<br>") + 1)
    st = (f"text;html=1;whiteSpace=wrap;fontFamily={FONT};fontSize={font};fontColor={INKC};align=center;"
          f"verticalAlign=middle;strokeColor=none;fillColor=none;spacing=0;rotation=-90;")
    d.cells.append(f'<mxCell id="{i}" value="{escape(label(text))}" style="{st}" vertex="1" parent="1">'
                   f'<mxGeometry x="{cx - length / 2:.2f}" y="{cy - h / 2:.2f}" width="{length:.2f}" height="{h}" as="geometry"/></mxCell>')


def legend(d: Diagram, x, y, items, col_w=150, cols=1, row_h=15, font=F):
    """items: (label, kind, color, filled, line_dash or None, with_line)."""
    for k, (lab, kind, color, filled, dash, with_line) in enumerate(items):
        cx, cy = x + (k % cols) * col_w, y + (k // cols) * row_h
        if with_line:
            line(d, [(cx, cy + 6), (cx + 18, cy + 6)], color, 1.4, dash)
        if kind:
            marker(d, cx + 9, cy + 6, kind, color, filled, 6)
        d.text(cx + 22, cy - 1, col_w - 24, 14, lab, font=font)
