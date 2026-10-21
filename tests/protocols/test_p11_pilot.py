"""Synthetic tests for the P11 pilot (no model, no real data)."""

import sys
import unittest
from pathlib import Path

import numpy  # noqa: F401
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p11 import run_pilot_ab as ab


def _world(dim=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    background = F.normalize(torch.randn(8, dim, generator=gen), dim=-1)
    defect = F.normalize(torch.randn(3, dim, generator=gen), dim=-1)
    return gen, background, defect


def _image(gen, background, defect_vec=None, n_defect=0, patches=40, noise=0.02):
    bg = background[torch.randint(0, background.shape[0], (patches - n_defect,), generator=gen)]
    parts = [bg] + ([defect_vec.expand(n_defect, -1)] if n_defect else [])
    p = torch.cat(parts) + noise * torch.randn(patches, background.shape[1], generator=gen)
    return F.normalize(p, dim=-1)


class AdaptiveRegionTest(unittest.TestCase):
    def test_threshold_from_normals_separates_defect_patches(self):
        gen, background, defect = _world()
        bank = torch.stack([_image(gen, background) for _ in range(6)])
        tau = ab.loo_threshold(bank, 0.99)
        img = _image(gen, background, defect[0], n_defect=10)
        a = ab.ra.patch_anomaly(img[None], bank.reshape(-1, bank.shape[-1]))
        w = ab.region_weights(a, tau, soft=False)
        self.assertEqual(int(w[0, -10:].sum()), 10)       # every defect patch selected
        self.assertLessEqual(int(w[0, :-10].sum()), 2)    # almost no background

    def test_region_size_follows_defect_size(self):
        gen, background, defect = _world(seed=1)
        bank = torch.stack([_image(gen, background) for _ in range(6)])
        tau = ab.loo_threshold(bank, 0.99)
        flat = bank.reshape(-1, bank.shape[-1])
        small = ab.ra.patch_anomaly(_image(gen, background, defect[0], 8)[None], flat)
        large = ab.ra.patch_anomaly(_image(gen, background, defect[0], 20)[None], flat)
        self.assertLess(int(ab.region_weights(small, tau, False).sum()),
                        int(ab.region_weights(large, tau, False).sum()))

    def test_fallback_uses_top6(self):
        a = torch.tensor([[0.0, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5]])
        for soft in (False, True):
            w = ab.region_weights(a.clone(), tau=5.0, soft=soft)
            self.assertEqual(int((w > 0).sum()), ab.MIN_REGION)
            self.assertTrue(bool((w[0, a[0].topk(6).indices] == 1).all()))

    def test_soft_weights_and_pool(self):
        a = torch.tensor([[0.0] * 4 + [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        w = ab.region_weights(a, tau=0.4, soft=True)
        self.assertTrue(torch.allclose(w[0, 4:], a[0, 4:] - 0.4))
        patches = torch.eye(10)[None]
        pooled = ab.weighted_pool(patches, w)
        self.assertAlmostEqual(float(pooled.sum()), 1.0, places=5)


class GuidedMatchingTest(unittest.TestCase):
    def test_guided_xclass_classifies_by_defect_region(self):
        gen, background, defect = _world(seed=2)
        imgs, labels = [], []
        for c in range(3):
            for _ in range(3):
                imgs.append(_image(gen, background, defect[c], n_defect=4))
                labels.append(c)
        region = torch.stack([im[-4:] for im in imgs])  # pretend T(x) = the defect patches
        si = torch.tensor([0, 3, 6])
        qi = torch.tensor([1, 2, 4, 5, 7, 8])
        plain, xclass = ab.guided_scores(region, [0, 1, 2], si, qi, 3)
        truth = torch.tensor([0, 0, 1, 1, 2, 2])
        self.assertTrue(bool((plain.argmax(1) == truth).all()))
        self.assertTrue(bool((xclass.argmax(1) == truth).all()))
        self.assertTrue(bool((xclass.gather(1, truth[:, None]) > 0).all()))

    def test_region_agreement(self):
        sel = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        r = ab.region_agreement(sel, mask)[0]
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["recall"], 0.5)
        self.assertEqual(ab.region_agreement(sel, torch.zeros(1, 4)), [])


class FormalScorerTest(unittest.TestCase):
    def test_episode_scores_methods_and_descriptors(self):
        from experiments.p11 import formal_ab as fa
        gen, background, defect = _world(seed=3)
        imgs = torch.stack([_image(gen, background, defect[i % 3], n_defect=6, patches=60) for i in range(9)])
        bank = torch.stack([_image(gen, background, patches=60) for _ in range(5)])
        tau = fa.tau99(bank)
        pooled, soft, region = fa.descriptors(imgs, bank, tau)
        self.assertEqual(tuple(region.shape), (9, 6, 32))
        self.assertEqual(tuple(soft.shape), (9, 32))
        g = torch.ones(9, 16)  # identical global features: the z-scored global term is 0
        scores = fa.episode_scores(g, {"": (pooled, soft, region)}, torch.tensor([0, 1, 2]), [0, 1, 2],
                                   torch.tensor([3, 4, 5, 6, 7, 8]), 3)
        self.assertEqual(set(scores), {fa.SS, fa.NR, fa.A, fa.B})
        truth = torch.tensor([0, 1, 2, 0, 1, 2])
        self.assertTrue(bool((scores[fa.A].argmax(1) == truth).all()))


if __name__ == "__main__":
    unittest.main()
