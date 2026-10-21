"""Small graphics embedded in the draw.io schematics (PNG bytes, transparent background).

Two kinds:
- illustrative shapes that carry no measured values (score histograms, prototype clusters, chip icon): no axis
  numbers are drawn, so they cannot be read as data;
- real material: image patches, perturbed copies of the published query image, the Beta(N - k + 1, k) density
  (theory), and the measured per-object gains on VisA (paper_figures/data).
"""

from __future__ import annotations

import io
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402
from scipy import stats  # noqa: E402

from drawio import AQUA, GREY, ORANGE, VIOLET  # noqa: E402

CM = 1 / 2.54
DATA = Path(__file__).resolve().parents[2] / "data" / "ckpt"


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, transparent=True)
    plt.close(fig)
    return buf.getvalue()


def _bare(w_cm, h_cm):
    fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.84])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_yticks([])
    ax.set_xticks([])
    return fig, ax


def score_histogram(color=AQUA, tail=None, w=2.2, h=1.4, seed=1) -> bytes:
    """Illustrative right-skewed score distribution with a threshold line; tail shaded if given."""
    rng = np.random.default_rng(seed)
    v = rng.gamma(3.0, 1.0, 4000)
    thr = np.quantile(v, 0.9 if tail else 0.95)
    fig, ax = _bare(w, h)
    bins = np.linspace(0, 11, 28)
    n, b, patches = ax.hist(v, bins=bins, color=color, alpha=0.55, edgecolor="white", lw=0.3)
    if tail:
        for p, left in zip(patches, b[:-1]):
            if left >= thr:
                p.set_facecolor(tail)
                p.set_alpha(0.9)
    ax.axvline(thr, color="#1a1a1a", lw=0.9, ls=(0, (3, 2)))
    ax.set_xlim(0, 11)
    return _png(fig)


def prototype_scatter(query=True, w=2.4, h=1.6, seed=4, n=4) -> bytes:
    """Illustrative class prototypes (centres) with support points; the query as an orange star near the second
    class (the published prediction of the bottle query is broken_small, the second defect type)."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.75, 0.78], [0.22, 0.68], [0.5, 0.2], [0.85, 0.3]])[:n]
    marks = ["o", "s", "^", "D"][:n]
    shades = ["#2a78d6", "#5b95dd", "#1d5aa3", "#89b3e8"][:n]
    fig = plt.figure(figsize=(w * CM, h * CM))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")
    for c, m, s in zip(centres, marks, shades):
        ax.add_patch(plt.Circle(c, 0.15, fc=s, alpha=0.13, ec=s, lw=0.6))
        pts = c + rng.normal(0, 0.055, (5, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=14, marker=m, color=s, alpha=0.6, lw=0)
        ax.scatter(*c, s=55, marker=m, color=s, edgecolor="#1a1a1a", lw=0.6, zorder=3)
    if query:
        q = np.array([0.34, 0.6])
        ax.plot([q[0], centres[1][0]], [q[1], centres[1][1]], color=ORANGE, lw=0.8, ls=(0, (2, 1.5)))
        ax.scatter(*q, s=110, marker="*", color=ORANGE, edgecolor="#1a1a1a", lw=0.5, zorder=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    return _png(fig)


def patch_mosaic(paths, cols=6, rows=4, tile=28, gap=3, seed=2) -> Image.Image:
    """Real 14 x 14-pixel-grid patches (from 336 px images, 24 x 24 grid) arranged as a bank."""
    rng = np.random.default_rng(seed)
    ims = [Image.open(p).convert("RGB").resize((336, 336)) for p in paths]
    canvas = Image.new("RGBA", (cols * tile + (cols - 1) * gap, rows * tile + (rows - 1) * gap), (255, 255, 255, 0))
    for r in range(rows):
        for c in range(cols):
            im = ims[rng.integers(len(ims))]
            gy, gx = rng.integers(3, 21, 2)
            patch = im.crop((gx * 14, gy * 14, gx * 14 + 14, gy * 14 + 14)).resize((tile, tile), Image.NEAREST)
            canvas.paste(patch, (c * (tile + gap), r * (tile + gap)))
    return canvas


def perturbations(path) -> Image.Image:
    """2 x 2 grid of the query under four influence quantities: brighter, blurred, noisy, rotated."""
    im = Image.open(path).convert("RGB").resize((160, 160))
    bright = ImageEnhance.Brightness(im).enhance(1.6)
    blur = im.filter(ImageFilter.GaussianBlur(3))
    arr = np.asarray(im).astype(float) + np.random.default_rng(0).normal(0, 28, (160, 160, 3))
    noisy = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    from scipy import ndimage   # reflect padding, as the U4 rotation (experiments/p47/common.py)
    rot = Image.fromarray(np.clip(np.rint(ndimage.rotate(np.asarray(im).astype(float), 12, axes=(1, 0), reshape=False,
                                                         order=1, mode="reflect")), 0, 255).astype(np.uint8))
    g = 6
    canvas = Image.new("RGBA", (2 * 160 + g, 2 * 160 + g), (255, 255, 255, 0))
    for k, t in enumerate((bright, blur, noisy, rot)):
        canvas.paste(t, ((k % 2) * (160 + g), (k // 2) * (160 + g)))
    return canvas


def grid(paths, n_cols=2, size=150, gap=6) -> Image.Image:
    ims = [Image.open(p).convert("RGB").resize((size, size)) for p in paths]
    rows = math.ceil(len(ims) / n_cols)
    canvas = Image.new("RGBA", (n_cols * size + (n_cols - 1) * gap, rows * size + (rows - 1) * gap), (255, 255, 255, 0))
    for k, t in enumerate(ims):
        canvas.paste(t, ((k % n_cols) * (size + gap), (k // n_cols) * (size + gap)))
    return canvas


def beta_curves(w=2.6, h=1.6) -> bytes:
    """Theory: false-reject rate of one calibration, alpha = 5 %, N = 50 / 100 / 500 (as in figure 6a)."""
    fig, ax = _bare(w, h)
    x = np.linspace(0, 0.14, 400)
    for n, lw, a in ((50, 0.8, 0.45), (100, 1.0, 0.7), (500, 1.4, 1.0)):
        k = math.ceil((n + 1) * 0.95)
        ax.plot(x, stats.beta(n - k + 1, k).pdf(x), color=AQUA, lw=lw, alpha=a)
    ax.axvline(0.05, color="#1a1a1a", lw=0.7, ls=(0, (3, 2)))
    ax.set_xlim(0, 0.14)
    ax.set_ylim(bottom=0)
    return _png(fig)


def visa_gain_dots(w=2.6, h=1.6) -> bytes:
    """Measured: gain of the proposed method over SimpleShot per VisA object (mean over shots and seeds), with the
    mean; zero line dashed."""
    rows = (json.loads((DATA / "p14_formal/external/p14_external_results.json").read_text(encoding="utf-8"))["raw_metrics"]
            + json.loads((DATA / "p10_external/results/external_results.json").read_text(encoding="utf-8"))["raw_metrics"])
    g = {}
    for r in rows:
        g.setdefault(r["method"], {})[(r["task"], r["K"], r["seed"])] = r["balanced_accuracy"]
    a, b = g["fused_adareg_regres_soft_q99"], g["simpleshot_cl2n"]
    objs = sorted({k[0] for k in a if k[0].startswith("visa__")})
    d = [100 * st.fmean(a[k] - b[k] for k in a if k[0] == o) for o in objs]
    fig, ax = _bare(w, h)
    rng = np.random.default_rng(5)
    ax.scatter(d, 0.45 + rng.uniform(-0.2, 0.2, len(d)), s=22, color=GREY, alpha=0.85, edgecolor="white", lw=0.4)
    m = st.fmean(d)
    ax.plot([m], [0.45], marker="D", color=VIOLET, ms=6, mec="white", mew=0.6)
    ax.axvline(0, color="#1a1a1a", lw=0.6, ls=(0, (3, 2)))
    ax.set_ylim(0, 1)
    ax.set_xlim(min(d) - 4, max(d) + 4)
    return _png(fig)


def chip_icon(w=1.8, h=1.6) -> bytes:
    """Illustrative edge device: a chip with pins and a clock face."""
    fig = plt.figure(figsize=(w * CM, h * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.15, 0.2), 0.55, 0.6, fc="#d9d9d9", ec="#4d4d4d", lw=0.8))
    ax.add_patch(plt.Rectangle((0.27, 0.32), 0.31, 0.36, fc="#7a7a7a", ec="none"))
    for t in np.linspace(0.27, 0.73, 5):
        ax.plot([0.05, 0.15], [t, t], color="#4d4d4d", lw=0.8)
        ax.plot([0.7, 0.8], [t, t], color="#4d4d4d", lw=0.8)
    ax.add_patch(plt.Circle((0.93, 0.78), 0.17, fc="white", ec=ORANGE, lw=1.0))
    ax.plot([0.93, 0.93], [0.78, 0.9], color=ORANGE, lw=0.9)
    ax.plot([0.93, 1.02], [0.78, 0.78], color=ORANGE, lw=0.9)
    return _png(fig)


def to_png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
