"""Synthetic test for the P14 scorer (no model, no real data)."""

import sys
import unittest
from pathlib import Path

import numpy  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p13 import pilot as p13
from experiments.p14 import final_formal as ff
from tests.protocols.test_p13 import _images


class P14ScorerTest(unittest.TestCase):
    def test_methods_and_classification(self):
        imgs, _ = _images(patches=36)
        bank = imgs[:, :32].reshape(-1, imgs.shape[-1])[:200].reshape(-1, 20, imgs.shape[-1])[:5]  # background-only bank
        tau = ff.fa.tau99(bank)
        per_m = {"": ff.c1_descriptors(imgs, bank, tau)}
        si, qi, s_lab = torch.tensor([0, 3, 6]), torch.tensor([1, 2, 4, 5, 7, 8]), [0, 1, 2]
        g = torch.ones(9, 16)
        xc = ff.rp.local_xclass(imgs[si], s_lab, imgs[qi], 3)
        cons = p13.consensus_scores(imgs, si, s_lab, qi, 3)[ff.CONS]
        sc = ff.episode_scores(g, per_m, si, s_lab, qi, 3, xc, cons)
        self.assertEqual(set(sc), {ff.SS, ff.XC, ff.CONS, ff.NR, ff.A, ff.ARR})
        truth = torch.tensor([0, 0, 1, 1, 2, 2])
        for m in (ff.A, ff.ARR, ff.CONS):
            self.assertTrue(bool((sc[m].argmax(1) == truth).all()), m)


if __name__ == "__main__":
    unittest.main()
