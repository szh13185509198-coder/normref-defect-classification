"""Synthetic tests for the P12 threshold-source controls (no model, no real data)."""

import sys
import unittest
from pathlib import Path

import numpy  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p11 import run_pilot_ab as ab
from experiments.p12 import thresholds as th


def _map(blobs, side=24, base=0.05, noise=0.01, seed=0):
    """Anomaly map [1, side*side]: low background plus rectangular high blobs."""
    gen = torch.Generator().manual_seed(seed)
    grid = base + noise * torch.rand(side, side, generator=gen)
    for (r0, r1, c0, c1, v) in blobs:
        grid[r0:r1, c0:c1] = v + noise * torch.rand(r1 - r0, c1 - c0, generator=gen)
    return grid.reshape(1, -1)


class ThresholdTest(unittest.TestCase):
    def test_top10_selects_exactly_58(self):
        a = torch.rand(3, 576, generator=torch.Generator().manual_seed(1))
        tau = th.tau_top10(a)
        self.assertEqual(tau.shape, (3, 1))
        self.assertTrue(bool(((a > tau).sum(dim=1) == 58).all()))

    def test_otsu_separates_bimodal_map(self):
        a = _map([(2, 6, 2, 8, 0.8)])  # 24 high patches
        sel = a > th.tau_otsu(a)
        self.assertEqual(int(sel.sum()), 24)

    def test_otsu_matches_bruteforce(self):
        a = torch.rand(1, 50, generator=torch.Generator().manual_seed(2))
        v = a[0].double().sort().values
        best, best_t = -1.0, None
        for k in range(1, 50):
            lo, hi = v[:k], v[k:]
            var = (k / 50) * (1 - k / 50) * float(lo.mean() - hi.mean()) ** 2
            if var > best:
                best, best_t = var, float(v[k - 1])
        self.assertAlmostEqual(float(th.tau_otsu(a)[0, 0]), best_t, places=6)

    def test_mebin_finds_two_blobs(self):
        a = _map([(2, 5, 2, 5, 0.7), (15, 19, 15, 20, 0.6)])
        tau = th.tau_mebin(a)
        from scipy import ndimage
        grid = (a > tau).reshape(24, 24).numpy()
        self.assertEqual(ndimage.label(grid, structure=th.EIGHT_CONNECTED)[1], 2)
        self.assertEqual(int((a > tau).sum()), 9 + 20)

    def test_mebin_constant_map_falls_back(self):
        a = torch.full((1, 576), 0.3)
        w = ab.region_weights(a, th.tau_mebin(a), soft=True)
        self.assertEqual(int((w > 0).sum()), ab.MIN_REGION)

    def test_per_image_tau_broadcasts_in_region_weights(self):
        a = torch.stack([_map([(0, 3, 0, 3, 0.9)])[0], _map([(10, 20, 10, 20, 0.9)], seed=3)[0]])
        w = ab.region_weights(a, th.tau_otsu(a), soft=True)
        self.assertEqual([int(x) for x in (w > 0).sum(dim=1)], [9, 100])


if __name__ == "__main__":
    unittest.main()
