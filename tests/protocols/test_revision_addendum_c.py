"""Synthetic tests for Addendum C (no model, no real data)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.revision import revision_common as rc
from experiments.revision.c1_paired_bootstrap import contrast
from experiments.revision.c2_frozen_feature_baselines import (
    _first_argmax, logreg_rfs, simpleshot_cl2n,
)


def _clustered(n_classes, per_class, dim=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=gen) * 3 + 5.0  # shared offset
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(centers[c] + 0.1 * torch.randn(per_class, dim, generator=gen))
        ys += [c] * per_class
    return torch.cat(xs), ys


class SimpleShotTest(unittest.TestCase):
    def test_separable_clusters_classified(self):
        support, labels = _clustered(4, 3)
        query, truth = _clustered(4, 5, seed=0)
        pred = _first_argmax(simpleshot_cl2n(support, labels, query, 4))
        self.assertEqual(list(pred), truth)

    def test_centering_uses_support_only(self):
        support, labels = _clustered(3, 2)
        query, _ = _clustered(3, 4, seed=1)
        base = simpleshot_cl2n(support, labels, query, 3)
        shifted = simpleshot_cl2n(support, labels, torch.cat([query, query + 100]), 3)
        # adding extra queries must not change scores of the original queries
        np.testing.assert_allclose(base, shifted[: len(query)], rtol=0, atol=1e-6)

    def test_one_shot_supported(self):
        support, labels = _clustered(5, 1)
        scores = simpleshot_cl2n(support, labels, support, 5)
        self.assertEqual(list(_first_argmax(scores)), labels)


class LogRegTest(unittest.TestCase):
    def test_separable_and_shape(self):
        support, labels = _clustered(3, 3)
        query, truth = _clustered(3, 4, seed=0)
        scores = logreg_rfs(support, labels, query, 3)
        self.assertEqual(scores.shape, (12, 3))
        self.assertEqual(list(_first_argmax(scores)), truth)

    def test_deterministic(self):
        support, labels = _clustered(4, 2)
        a = logreg_rfs(support, labels, support, 4)
        b = logreg_rfs(support, labels, support, 4)
        np.testing.assert_array_equal(a, b)

    def test_tie_break_first_index(self):
        self.assertEqual(list(_first_argmax(np.array([[0.5, 0.5, 0.1]]))), [0])


class BootstrapContrastTest(unittest.TestCase):
    def _grid(self, offset):
        ours, ref = {}, {}
        for i, p in enumerate(rc.PRODUCTS):
            for k in rc.K_VALUES:
                for s in rc.SEEDS:
                    base = 0.5 + 0.01 * i
                    ref[(p, k, s)] = {m: base for m in rc.METRICS}
                    ours[(p, k, s)] = {m: base + offset for m in rc.METRICS}
        return {"ours": ours, "ref": ref}

    def test_constant_positive_offset(self):
        result = contrast(self._grid(0.02), "ref", "balanced_accuracy")
        for k in rc.K_VALUES:
            r = result["per_K"][str(k)]
            self.assertAlmostEqual(r["D"], 0.02)
            self.assertAlmostEqual(r["ci95"][0], 0.02)
            self.assertEqual(r["positive_products"], 13)
            self.assertTrue(r["ci_excludes_zero"])
        self.assertAlmostEqual(result["Dbar"], 0.02)

    def test_zero_offset_ci_contains_zero(self):
        result = contrast(self._grid(0.0), "ref", "macro_f1")
        self.assertFalse(result["Dbar_ci_excludes_zero"])
        self.assertEqual(result["Dbar_positive_products"], 0)


if __name__ == "__main__":
    unittest.main()
