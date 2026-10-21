"""P12 threshold-source controls (docs/P12_THRESHOLD_CONTROLS_RELIABILITY_FIGURES_PROTOCOL.md, T).

Every function maps patch anomalies ``a`` [N, P] to one threshold per image, shape [N, 1],
computed from that image alone (no normal images, no other queries).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage

from experiments.p10 import run_pilot as rp

MEBIN_STEPS = 64
EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def tau_top10(a: torch.Tensor) -> torch.Tensor:
    """59th largest value: exactly the top ceil(0.1 P) = 58 patches exceed it (barring ties)."""
    top = math.ceil(rp.RHO * a.shape[1])
    return a.topk(top + 1, dim=1).values[:, -1:]


def tau_otsu(a: torch.Tensor) -> torch.Tensor:
    """Exact Otsu on the image's own sorted values; tau = largest value of the lower class."""
    v = a.double().sort(dim=1).values
    p = v.shape[1]
    k = torch.arange(1, p, device=v.device, dtype=v.dtype)
    csum = v.cumsum(dim=1)
    total = csum[:, -1:]
    m0 = csum[:, :-1] / k
    m1 = (total - csum[:, :-1]) / (p - k)
    w0 = k / p
    between = w0 * (1 - w0) * (m0 - m1) ** 2
    idx = between.argmax(dim=1)  # first maximum on ties
    return v.gather(1, idx[:, None]).to(a.dtype)


def _mebin_one(row: np.ndarray, side: int) -> float:
    lo, hi = float(row.min()), float(row.max())
    eps = lo + (hi - lo) * np.arange(MEBIN_STEPS) / (MEBIN_STEPS - 1)
    grid = row.reshape(side, side)
    counts = [ndimage.label(grid > e, structure=EIGHT_CONNECTED)[1] for e in eps]
    nonzero = [c for c in counts if c > 0]
    if not nonzero:
        return hi
    values, freq = np.unique(nonzero, return_counts=True)
    mode = int(values[freq == freq.max()].min())
    best_len, best_start, j = 0, 0, 0
    while j < MEBIN_STEPS:
        if counts[j] == mode:
            start = j
            while j < MEBIN_STEPS and counts[j] == mode:
                j += 1
            if j - start > best_len:
                best_len, best_start = j - start, start
        else:
            j += 1
    return float(eps[best_start])


def tau_mebin(a: torch.Tensor) -> torch.Tensor:
    """MEBin-inspired (AnomalyNCD), simplified per protocol: per-image range, 64 thresholds,
    8-connected components, mode count, lowest threshold of the longest stable run."""
    side = int(round(a.shape[1] ** 0.5))
    if side * side != a.shape[1]:
        raise ValueError("patch count is not a square grid")
    rows = a.detach().double().cpu().numpy()
    return torch.tensor([_mebin_one(r, side) for r in rows], dtype=a.dtype, device=a.device)[:, None]


CONTROLS = {"fused_ctrl_soft_top10": tau_top10, "fused_ctrl_soft_otsu": tau_otsu,
            "fused_ctrl_soft_mebin": tau_mebin}
