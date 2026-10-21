"""P18 adapted baselines (DN4, crop and re-encode) and evidence-purity diagnostics (protocol §B, §P)."""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from experiments.p10 import run_pilot as rp
from experiments.p11 import run_pilot_ab as ab
from experiments.p12.thresholds import EIGHT_CONNECTED, tau_mebin, tau_otsu, tau_top10

DN4_K = 3
CROP_VARIANTS = tuple((c, s) for c in ("largest", "all") for s in (1.0, 1.5, 2.0))
MIN_CELLS = 3
SCHEMES = ("global", "fixed_hard", "fixed_soft", "otsu_soft", "mebin_soft", "ours_hard", "ours_soft")
RHO_BINS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 1.01))
DIST_BINS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10 ** 6))


# --------------------------------------------------------------------------- DN4

def dn4_scores(sp: torch.Tensor, s_lab: list[int], qp: torch.Tensor, n: int, k: int = DN4_K,
               chunk: int = 8) -> torch.Tensor:
    """DN4 image-to-class measure without training: sum over query patches of the k largest
    cosine similarities to the class's support patches. sp [S, P, d], qp [Q, P, d] -> [Q, n]."""
    lab = torch.as_tensor(s_lab, device=sp.device)
    sets = [sp[lab == c].reshape(-1, sp.shape[-1]) for c in range(n)]
    out = []
    for q in qp.split(chunk):
        cols = []
        for patches in sets:
            sim = torch.einsum("qpd,md->qpm", q, patches)
            cols.append(sim.topk(min(k, patches.shape[0]), dim=2).values.sum(dim=2).sum(dim=1))
        out.append(torch.stack(cols, dim=1))
    return torch.cat(out)


# --------------------------------------------------------------------------- crop and re-encode

def _box_1d(lo: float, hi: float, scale: float, side: int) -> tuple[float, float]:
    centre, size = (lo + hi) / 2, max((hi - lo) * scale, MIN_CELLS)
    size = min(size, side)
    a, b = centre - size / 2, centre + size / 2
    if a < 0:
        a, b = 0.0, size
    if b > side:
        a, b = side - size, float(side)
    return a / side, b / side


def crop_boxes(w: torch.Tensor, component: str, scale: float) -> list[tuple[float, float, float, float]]:
    """w [N, P] non-negative region weights (ARR's) -> boxes (y0, x0, y1, x1) as image fractions."""
    side = int(round(w.shape[1] ** 0.5))
    if side * side != w.shape[1]:
        raise ValueError("patch count is not a square grid")
    boxes = []
    for row in w.detach().double().cpu().numpy():
        grid = row.reshape(side, side)
        sel = grid > 0
        if component == "largest":
            lab, count = ndimage.label(sel, structure=EIGHT_CONNECTED)
            if count > 1:
                mass = ndimage.sum(grid, lab, index=np.arange(1, count + 1))
                sel = lab == (int(np.argmax(mass)) + 1)  # first maximum on ties
        elif component != "all":
            raise ValueError(component)
        ys, xs = np.nonzero(sel)
        y0, y1 = _box_1d(float(ys.min()), float(ys.max()) + 1, scale, side)
        x0, x1 = _box_1d(float(xs.min()), float(xs.max()) + 1, scale, side)
        boxes.append((y0, x0, y1, x1))
    return boxes


def apply_box(image: Image.Image, box) -> Image.Image:
    y0, x0, y1, x1 = box
    W, H = image.size
    left, top = int(math.floor(x0 * W)), int(math.floor(y0 * H))
    right, bottom = max(left + 1, int(math.ceil(x1 * W))), max(top + 1, int(math.ceil(y1 * H)))
    return image.crop((left, top, min(right, W), min(bottom, H)))


# --------------------------------------------------------------------------- purity

def scheme_weights(a: torch.Tensor, tau: float) -> dict:
    """Pooling weights [N, P] of the seven schemes for anomalies a [N, P]."""
    top = math.ceil(rp.RHO * a.shape[1])
    hard = torch.zeros_like(a)
    hard.scatter_(1, a.topk(top, dim=1).indices, 1.0)
    return {"global": torch.ones_like(a), "fixed_hard": hard,
            "fixed_soft": ab.region_weights(a.clone(), tau_top10(a), soft=True),
            "otsu_soft": ab.region_weights(a.clone(), tau_otsu(a), soft=True),
            "mebin_soft": ab.region_weights(a.clone(), tau_mebin(a), soft=True),
            "ours_hard": ab.region_weights(a.clone(), tau, soft=False),
            "ours_soft": ab.region_weights(a.clone(), tau, soft=True)}


def purity(w: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(lambda_any, lambda_area) per image; m = mask area fraction per patch."""
    total = w.sum(dim=1).clamp_min(1e-12)
    return (w * (m > 0)).sum(dim=1) / total, (w * m).sum(dim=1) / total


def mask_distance(m: torch.Tensor) -> torch.Tensor:
    """Chebyshev distance (cells) of every patch to the nearest m > 0 patch; 0 inside the mask."""
    side = int(round(m.shape[1] ** 0.5))
    out = []
    for row in (m > 0).cpu().numpy():
        grid = ~row.reshape(side, side)
        out.append(ndimage.distance_transform_cdt(grid, metric="chessboard").reshape(-1))
    return torch.as_tensor(np.stack(out), device=m.device)


def rho_bin(rho: float) -> int:
    return next(i for i, (lo, hi) in enumerate(RHO_BINS) if lo <= rho < hi)
