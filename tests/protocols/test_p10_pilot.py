"""Synthetic tests for the P10 pilot (no model, no real data)."""

import sys
import unittest
from pathlib import Path

import numpy  # noqa: F401
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p10 import run_pilot as rp


def _images(n_per_class, n_classes, patches=40, dim=32, seed=0):
    """Each image: shared background patches + a few class-specific defect patches."""
    gen = torch.Generator().manual_seed(seed)
    background = F.normalize(torch.randn(8, dim, generator=gen), dim=-1)
    defect = F.normalize(torch.randn(n_classes, dim, generator=gen), dim=-1)
    imgs, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            bg = background[torch.randint(0, 8, (patches - 3,), generator=gen)]
            df = defect[c].expand(3, dim)
            p = torch.cat([bg, df]) + 0.02 * torch.randn(patches, dim, generator=gen)
            imgs.append(F.normalize(p, dim=-1))
            labels.append(c)
    return torch.stack(imgs), labels


class EpisodeTest(unittest.TestCase):
    def test_nested_and_disjoint(self):
        labels = ["a"] * 6 + ["b"] * 5
        eps = {k: (s, q) for k, _, s, _, q, _ in rp.episodes(labels, 9001, "x")}
        s1, _ = eps[1]
        s3, q3 = eps[3]
        self.assertTrue(set(s1) <= set(s3))
        self.assertFalse(set(s3) & set(q3))
        self.assertEqual(len(s3) + len(q3), len(labels))

    def test_too_small_class_fails(self):
        with self.assertRaises(Exception):
            list(rp.episodes(["a"] * 3 + ["b"] * 1, 9001, "x"))


class XClassTest(unittest.TestCase):
    def test_background_only_query_has_no_margin(self):
        sp, sl = _images(1, 3, seed=3)
        bg_only = sp[:, :-3][:1]  # a query made only of shared background patches
        scores = rp.local_xclass(sp, sl, bg_only, 3)
        self.assertLess(float(scores.abs().max()), 0.1)

    def test_local_xclass_classifies(self):
        sp, sl = _images(1, 4, seed=2)
        qp, ql = _images(3, 4, seed=2)
        pred = rp.local_xclass(sp, sl, qp, 4).argmax(dim=1).tolist()
        self.assertEqual(pred, ql)

    def test_zscore_constant_row(self):
        z = rp._zscore(torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]))
        self.assertTrue(torch.all(z[0] == 0))
        self.assertAlmostEqual(float(z[1].mean()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()


class NormalReferenceTest(unittest.TestCase):
    def test_anomaly_selects_defect_patches(self):
        from experiments.p10 import run_pilot_a as ra
        imgs, _ = _images(2, 2, patches=40, seed=4)       # last 3 patches are defects
        bank = imgs[:, :-3].reshape(-1, imgs.shape[-1])    # normal bank = background only
        anomaly = ra.patch_anomaly(imgs, bank)
        for i in range(imgs.shape[0]):
            self.assertEqual(set(anomaly[i].topk(3).indices.tolist()), {37, 38, 39})
        pooled = ra.anomaly_pool(imgs, anomaly)
        self.assertEqual(tuple(pooled.shape), (4, imgs.shape[-1]))

    def test_patch_auroc(self):
        from experiments.p10 import run_pilot_a as ra
        scores = torch.tensor([0.1, 0.2, 0.9, 0.8])
        self.assertAlmostEqual(ra.patch_auroc(scores, torch.tensor([False, False, True, True])), 1.0)
        self.assertAlmostEqual(ra.patch_auroc(scores, torch.tensor([True, True, False, False])), 0.0)
        self.assertIsNone(ra.patch_auroc(scores, torch.tensor([False] * 4)))


class ControlsTest(unittest.TestCase):
    def test_centered_proto_matches_simpleshot_with_support_mean(self):
        from experiments.p10 import run_controls_a as rca
        gs = torch.randn(6, 16, generator=torch.Generator().manual_seed(5))
        sl = [0, 0, 1, 1, 2, 2]
        gq = gs + 0.01
        mu = gs.mean(0, keepdim=True)
        torch.testing.assert_close(rca.centered_proto(gs, sl, gq, 3, mu), rp.simpleshot(gs, sl, gq, 3))

    def test_random_pool_deterministic(self):
        from experiments.p10 import run_controls_a as rca
        imgs, _ = _images(1, 2, seed=6)
        torch.testing.assert_close(rca.random_pool(imgs, 9001), rca.random_pool(imgs, 9001))
        self.assertFalse(torch.equal(rca.random_pool(imgs, 9001), rca.random_pool(imgs, 9002)))


class SupplementSTest(unittest.TestCase):
    def test_residual_zero_for_normal_patches(self):
        from experiments.p10 import supplement_s as ss
        imgs, _ = _images(1, 2, patches=40, seed=7)
        bank = imgs.reshape(-1, imgs.shape[-1])            # every patch is in the bank
        pooled, residual = ss.normal_descriptors(imgs, bank, torch.device("cpu"))
        self.assertEqual(tuple(pooled.shape), (2, imgs.shape[-1]))
        self.assertLess(float(residual.abs().max()), 1e-6)

    def test_mask_pool_weights_and_empty_fallback(self):
        from experiments.p10 import supplement_s as ss
        imgs, _ = _images(1, 2, patches=40, seed=8)
        masks = torch.zeros(2, 40)
        masks[0, 37:] = 1.0                                # image 0: defect patches only
        pooled = ss.mask_pool(imgs, masks, torch.device("cpu"))
        torch.testing.assert_close(pooled[0], imgs[0, 37:].mean(0))
        torch.testing.assert_close(pooled[1], imgs[1].mean(0))   # empty mask -> plain mean


class IndustrialMetricsTest(unittest.TestCase):
    def test_normal_aware_counts(self):
        from experiments.p10.industrial_metrics import normal_aware, per_thousand
        truth = ["good", "good", "good", "good", "a", "a", "b", "b"]
        pred = ["good", "good", "good", "a", "a", "b", "good", "b"]
        m = normal_aware(truth, pred, "good")
        self.assertAlmostEqual(m["good_recall"], 0.75)
        self.assertAlmostEqual(m["defect_recall"], 0.75)        # 3 of 4 defects flagged
        self.assertAlmostEqual(m["defect_type_acc_detected"], 2 / 3)
        self.assertAlmostEqual(m["binary_ba"], 0.75)
        p = per_thousand(m["false_reject_rate"], m["escape_rate"])
        self.assertAlmostEqual(p["false_rejects_per_1000"], 237.5)
        self.assertAlmostEqual(p["escapes_per_1000"], 12.5)

    def test_requires_both_groups(self):
        from experiments.p10.industrial_metrics import normal_aware
        with self.assertRaises(ValueError):
            normal_aware(["a", "b"], ["a", "b"], "good")
