"""Synthetic tests for the P15 decision layer (no model, no real data)."""

import random
import sys
import unittest
from pathlib import Path

import numpy  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p15 import screening as sc


class ScreeningTest(unittest.TestCase):
    def test_image_scores(self):
        a = torch.arange(600, dtype=torch.float32).reshape(1, 600) / 600
        self.assertAlmostEqual(float(sc.image_score(a, "max")), 599 / 600, places=6)
        self.assertAlmostEqual(float(sc.image_score(a, "top1pct")), sum(range(594, 600)) / 6 / 600, places=6)

    def test_conformal_threshold_and_guarantee(self):
        cal = list(range(1, 101))  # 100 calibration scores
        self.assertEqual(sc.conformal_threshold(cal, 0.05), 96)   # ceil(101*0.95)=96
        self.assertEqual(sc.conformal_threshold(cal, 0.01), 100)  # ceil(99.99)=100
        self.assertEqual(sc.conformal_threshold(cal[:50], 0.01), float("inf"))
        rng = random.Random(0)
        frr = []
        for _ in range(400):  # exchangeable normals: mean FRR <= alpha
            c = [rng.random() for _ in range(100)]
            t = sc.conformal_threshold(c, 0.05)
            frr.append(sum(rng.random() > t for _ in range(200)) / 200)
        self.assertLessEqual(sum(frr) / len(frr), 0.05 + 0.01)

    def test_auroc(self):
        self.assertEqual(sc.auroc([0.1, 0.2], [0.3, 0.4]), 1.0)
        self.assertEqual(sc.auroc([0.3], [0.3]), 0.5)

    def test_restricted_type_skips_normal(self):
        scores = torch.tensor([[9.0, 1.0, 2.0], [9.0, 3.0, 1.0]])
        self.assertEqual(sc.restricted_type(scores, ["good", "a", "b"], "good"), ["b", "a"])

    def test_two_level_and_three_way(self):
        s = [0.1, 0.2, 0.9, 0.5, 0.8, 0.3]
        truth = ["good", "good", "good", "x", "y", "x"]
        types = ["x", "x", "x", "x", "x", "x"]
        r = sc.two_level(s, truth, types, "good", tau=0.4)
        self.assertAlmostEqual(r["false_reject_rate"], 1 / 3)
        self.assertAlmostEqual(r["escape_rate"], 1 / 3)
        self.assertAlmostEqual(r["type_acc_rejected_defects"], 0.5)
        t = sc.three_way(s, truth, "good", tau_reject=0.85, tau_pass=0.25)
        self.assertAlmostEqual(t["auto_false_reject_rate"], 1 / 3)
        self.assertEqual(t["escape_rate"], 0.0)
        self.assertAlmostEqual(t["review_rate_good"], 0.0)
        self.assertAlmostEqual(t["review_rate_defect"], 1.0)  # 0.5, 0.8, 0.3 all in (0.25, 0.85]

    def test_oracle_review_meets_escape_target(self):
        rng = random.Random(1)
        s = [rng.random() * 0.6 for _ in range(300)] + [0.3 + rng.random() * 0.7 for _ in range(100)]
        truth = ["good"] * 300 + ["d"] * 100
        r = sc.oracle_min_review(s, truth, "good", tau_reject=0.9)
        self.assertLessEqual(r["escape_rate"], 0.01)


class DecisionMetricsTest(unittest.TestCase):
    def test_decision_metrics_and_report(self):
        from experiments.p15 import two_stage as ts
        rng = random.Random(2)
        s_q = [rng.random() * 0.5 for _ in range(50)] + [0.4 + rng.random() * 0.6 for _ in range(20)]
        truth = ["good"] * 50 + ["a", "b"] * 10
        types = {m: ["a"] * 70 for m in ts.TYPE_METHODS}
        unit = ts.decision_metrics(s_q, [rng.random() * 0.5 for _ in range(100)], truth, types, "good")
        flat = ts.flatten(unit)
        agg = {"5": {k: ts.mean_sd([v, v]) for k, v in flat.items()}}
        lines = ts.report_lines("synthetic", agg)
        self.assertTrue(any("oracle" in line for line in lines))
        self.assertGreater(unit["auroc"], 0.8)


if __name__ == "__main__":
    unittest.main()
