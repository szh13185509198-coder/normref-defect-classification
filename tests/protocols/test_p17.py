"""Synthetic tests for P17 (no model, no real data)."""

import sys
import unittest
from pathlib import Path

import numpy  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p11 import run_pilot_ab as ab
from experiments.p17 import robustness as rb
from tests.protocols.test_p13 import _images


class P17Test(unittest.TestCase):
    def test_region_weights_matches_p11_at_six(self):
        a = torch.rand(5, 40, generator=torch.Generator().manual_seed(0))
        for tau in (0.2, 0.9, 2.0):
            self.assertTrue(torch.equal(rb.region_weights(a.clone(), tau, 6), ab.region_weights(a.clone(), tau, soft=True)))
        w = rb.region_weights(a.clone(), 5.0, 1)
        self.assertEqual(int((w > 0).sum()), 5)

    def test_signflip(self):
        self.assertAlmostEqual(rb.signflip_p([1.0] * 13), 2 / 2 ** 13)
        self.assertEqual(rb.signflip_p([1.0, -1.0]), 1.0)

    def test_episode_scores_keys(self):
        imgs, _ = _images(patches=36)
        bank = imgs[:, :32].reshape(-1, imgs.shape[-1])[:200].reshape(-1, 20, imgs.shape[-1])[:5]
        views = rb.c1_views(imgs, bank, rb.loo_taus(bank, rb.Q_GRID), rb.C1_KEYS)
        si, qi, s_lab = torch.tensor([0, 3, 6]), torch.tensor([1, 2, 4, 5, 7, 8]), [0, 1, 2]
        cons = rb.cons_views(imgs, si, s_lab, qi, 3)
        sc, zp = rb.episode_scores(torch.ones(9, 16), views, views, cons, si, s_lab, qi, 3, (1.0, 1.0, 1.0))
        for grp in rb.GROUPS.values():
            for m in grp:
                self.assertIn(m, sc)
        self.assertTrue(torch.equal(sc["arr_devweights"], sc[rb.ARR]))
        self.assertTrue(torch.equal(sc["cons_t0.9"], sc[rb.CONS]))

    def test_select_weights_tie_prefers_equal(self):
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        w, _ = rb.select_weights([(z, z, z)], [["a", "b"]], [["a", "b"]], [("p", 1, 1)])
        self.assertEqual(w, (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
