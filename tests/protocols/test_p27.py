"""Synthetic tests for P27 (no model; masks are written to a temp directory)."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p27 import external_fill as p27


class P27Test(unittest.TestCase):
    def test_mask_paths(self):
        root = Path("R")
        self.assertEqual(p27.mask_path("visa", root, "pcb1/Data/Images/Anomaly/000.JPG"),
                         root / "pcb1/Data/Masks/Anomaly/000.png")
        self.assertEqual(p27.mask_path("mtd", root, "MT_Crack/Imgs/exp1_num_1.jpg"),
                         root / "MT_Crack/Imgs/exp1_num_1.png")

    def test_mask_binarization_per_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = np.zeros((48, 48), np.uint8)
            a[:24, :24] = 1                      # VisA-style {0, 1}
            p = Path(tmp) / "v.png"
            Image.fromarray(a).save(p)
            cells = p27.mask_cells(p, "visa")
            self.assertAlmostEqual(float(cells.sum()), 144.0)   # a quarter of 24 x 24
            b = np.full((48, 48), 9, np.uint8)   # MTD compression noise only
            b[24:, 24:] = 255
            q = Path(tmp) / "m.png"
            Image.fromarray(b).save(q)
            cells = p27.mask_cells(q, "mtd")
            self.assertAlmostEqual(float(cells.sum()), 144.0)
            self.assertEqual(float(cells[0]), 0.0)

    def test_empty_mask_pools_uniformly(self):
        clip = torch.randn(3, 576, 8)
        masks = torch.zeros(3, 576)
        masks[1, :10] = 1.0
        out = p27.pool_by_mask(clip, masks, "cpu")
        self.assertTrue(torch.allclose(out[0], clip[0].mean(dim=0), atol=1e-6))
        self.assertTrue(torch.allclose(out[1], clip[1, :10].mean(dim=0), atol=1e-6))

    def test_residual_matches_supplement_formula(self):
        from experiments.p10 import supplement_s
        clip = torch.nn.functional.normalize(torch.randn(5, 576, 8), dim=-1)
        bank = torch.nn.functional.normalize(torch.randn(40, 8), dim=-1)
        _, ref = supplement_s.normal_descriptors(clip, bank, "cpu")
        self.assertTrue(torch.allclose(p27.residual_mean(clip, bank, "cpu"), ref, atol=1e-6))

    def test_exact_signflip(self):
        self.assertAlmostEqual(p27.exact_signflip([1.0] * 11), 2 / 2048)
        self.assertEqual(p27.exact_signflip([0.0] * 5), 1.0)


if __name__ == "__main__":
    unittest.main()
