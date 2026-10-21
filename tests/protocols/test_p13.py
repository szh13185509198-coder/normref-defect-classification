"""Synthetic tests for the P13 pilot (no model, no real data)."""

import sys
import unittest
from pathlib import Path

import numpy  # noqa: F401
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p13 import pilot as p13


def _images(n_per_class=3, n_classes=3, patches=36, n_defect=4, dim=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    background = F.normalize(torch.randn(6, dim, generator=gen), dim=-1)
    defect = F.normalize(torch.randn(n_classes, dim, generator=gen), dim=-1)
    imgs, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            bg = background[torch.randint(0, 6, (patches - n_defect,), generator=gen)]
            p = torch.cat([bg, defect[c].expand(n_defect, dim)]) + 0.02 * torch.randn(patches, dim, generator=gen)
            imgs.append(F.normalize(p, dim=-1))
            labels.append(c)
    return torch.stack(imgs), labels


class ConsensusTest(unittest.TestCase):
    def setUp(self):
        self.imgs, labels = _images()
        self.si = torch.tensor([0, 3, 6])
        self.qi = torch.tensor([1, 2, 4, 5, 7, 8])
        self.s_lab = [0, 1, 2]

    def test_defect_patches_are_consensus_anomalous(self):
        a_s, a_q = p13.consensus_anomaly(self.imgs, self.si, self.s_lab, self.qi, 3)
        for a in (a_s, a_q):
            self.assertGreater(float(a[:, -4:].min()), float(a[:, :-4].max()))

    def test_consensus_scores_classify_queries(self):
        scores = p13.consensus_scores(self.imgs, self.si, self.s_lab, self.qi, 3)
        truth = torch.tensor([0, 0, 1, 1, 2, 2])
        self.assertEqual(set(scores), set(p13.IDEA2))
        for sc in scores.values():
            self.assertTrue(bool((sc.argmax(1) == truth).all()))

    def test_two_classes_supported(self):
        imgs, _ = _images(n_classes=2)
        a_s, a_q = p13.consensus_anomaly(imgs, torch.tensor([0, 3]), [0, 1], torch.tensor([1, 2, 4, 5]), 2)
        self.assertGreater(float(a_q[:, -4:].min()), float(a_q[:, :-4].max()))


class RegionResidualTest(unittest.TestCase):
    def test_residual_is_zero_on_bank_patches_and_weighted(self):
        bank = F.normalize(torch.randn(10, 8, generator=torch.Generator().manual_seed(1)), dim=-1)
        clip = torch.stack([torch.cat([bank[:3], F.normalize(torch.ones(1, 8), dim=-1)])])  # 3 normal + 1 odd patch
        w = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        h = p13.region_residual(clip, bank, w)
        odd = clip[0, 3]
        nn = bank[(bank @ odd).argmax()]
        self.assertTrue(torch.allclose(h[0], odd - nn, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
