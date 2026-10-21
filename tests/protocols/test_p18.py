"""Synthetic tests for P18 (no model weights, no real data)."""

import sys
import unittest
from pathlib import Path

import numpy  # noqa: F401
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p11 import run_pilot_ab as ab
from experiments.p14 import final_formal as ff
from experiments.p18 import backbones as bb
from experiments.p18 import baselines as bl
from experiments.p18 import run
from tests.protocols.test_p13 import _images


class DN4Test(unittest.TestCase):
    def test_own_class_wins_and_k_capped(self):
        imgs, labels = _images(n_per_class=2, patches=16)
        si, qi = torch.tensor([0, 2, 4]), torch.tensor([1, 3, 5])
        sc = bl.dn4_scores(imgs[si], [0, 1, 2], imgs[qi], 3)
        self.assertEqual(sc.argmax(dim=1).tolist(), [0, 1, 2])
        one = torch.nn.functional.normalize(torch.randn(1, 2, 8), dim=-1)
        self.assertEqual(bl.dn4_scores(one, [0], one, 1, k=3).shape, (1, 1))  # only 2 support patches


class CropTest(unittest.TestCase):
    def _w(self, cells):
        w = torch.zeros(1, 24 * 24)
        for r, c, v in cells:
            w[0, r * 24 + c] = v
        return w

    def test_largest_component_and_min_size(self):
        w = self._w([(5, 5, 1.0), (5, 6, 1.0), (6, 5, 1.0), (20, 20, 0.5)])
        (y0, x0, y1, x1), = bl.crop_boxes(w, "largest", 1.0)
        self.assertAlmostEqual(y0 * 24, 4.5)
        self.assertAlmostEqual(y1 * 24, 7.5)  # 2 cells enlarged to the 3-cell minimum
        (y0, x0, y1, x1), = bl.crop_boxes(w, "all", 1.0)
        self.assertAlmostEqual(y0 * 24, 5.0)
        self.assertAlmostEqual(y1 * 24, 21.0)

    def test_scale_and_border_shift(self):
        w = self._w([(r, c, 1.0) for r in range(0, 4) for c in range(0, 4)])
        (y0, x0, y1, x1), = bl.crop_boxes(w, "largest", 2.0)
        self.assertAlmostEqual(y0, 0.0)
        self.assertAlmostEqual(y1 * 24, 8.0)  # 8 cells, shifted inside the image

    def test_apply_box(self):
        img = Image.new("RGB", (100, 50))
        self.assertEqual(bl.apply_box(img, (0.2, 0.1, 0.6, 0.5)).size, (40, 20))


class PurityTest(unittest.TestCase):
    def test_purity_and_distance(self):
        m = torch.zeros(1, 36)
        m[0, 14] = 1.0  # cell (2, 2) of a 6x6 grid
        m[0, 15] = 0.5
        w = torch.zeros(1, 36)
        w[0, 14], w[0, 0] = 3.0, 1.0
        lam_any, lam_area = bl.purity(w, m)
        self.assertAlmostEqual(float(lam_any), 0.75)
        self.assertAlmostEqual(float(lam_area), 0.75)
        d = bl.mask_distance(m)
        self.assertEqual(int(d[0, 14]), 0)
        self.assertEqual(int(d[0, 0]), 2)
        self.assertEqual(int(d[0, 35]), 3)

    def test_schemes_and_rho_bins(self):
        a = torch.rand(3, 64, generator=torch.Generator().manual_seed(1))
        sw = bl.scheme_weights(a, 0.5)
        self.assertEqual(set(sw), set(bl.SCHEMES))
        self.assertTrue(torch.equal(sw["ours_soft"], ab.region_weights(a, 0.5, soft=True)))
        self.assertEqual(int(sw["fixed_hard"][0].sum()), 7)
        self.assertEqual([bl.rho_bin(x) for x in (0.0, 0.03, 0.1, 1.0)], [0, 1, 2, 3])


class RunHelpersTest(unittest.TestCase):
    def test_c1_parts_matches_p14(self):
        imgs, _ = _images(patches=36)
        bank_imgs = imgs[:5]
        tau = 0.1
        mine, _, _ = run.c1_parts(imgs, bank_imgs, tau)
        ref = ff.c1_descriptors(imgs, bank_imgs, tau)
        for x, y in zip(mine, ref):
            self.assertTrue(torch.equal(x, y))

    def test_whole_scores_keys_and_rename(self):
        imgs, _ = _images(patches=36)
        parts, _, _ = run.c1_parts(imgs, imgs[:5], 0.1)
        si, qi = torch.tensor([0, 3, 6]), torch.tensor([1, 2, 4, 5, 7, 8])
        sc = run.whole_scores(torch.randn(9, 16), imgs, parts, si, [0, 1, 2], qi, 3)
        self.assertEqual(set(sc), set(run.MAIN))
        self.assertEqual(set(run.renamed(sc, "dinov2_b")), {f"{s}_dinov2_b" for s in ("ss", "nr", "xc", "a", "arr", "cons")})

    def test_selection_rules(self):
        ba = {f"crop_{run.tag(v)}_fused": 0.5 for v in bl.CROP_VARIANTS}
        self.assertEqual(run.select_crop(ba), ("largest", 1.0))
        ba[f"crop_{run.tag(('all', 2.0))}_fused"] = 0.6
        self.assertEqual(run.select_crop(ba), ("all", 2.0))
        ba = {f"arr_{c}": 0.7 for c in run.dev_configs()}
        self.assertEqual(run.select_config(ba, "clip_b16"), "clip_b16_r224_L6")
        ba["arr_dinov2_b_L10"] = 0.8
        self.assertEqual(run.select_config(ba, "dinov2_b"), "dinov2_b_L10")
        self.assertEqual(run.parse_config("clip_b16_r336_L9"), ("clip_b16", 336, 9))
        self.assertEqual(run.parse_config("dinov2_b_L11"), ("dinov2_b", 336, 11))

    def test_contrast_pairs_cover_protocol(self):
        pairs = run.contrast_pairs()
        self.assertIn((run.ARR, "crop_fused"), pairs)
        self.assertIn((run.CONS, "dn4_fused"), pairs)
        self.assertIn(("arr_clip_b16", "ss_clip_b16"), pairs)
        self.assertEqual(len(pairs), 7 + 7 * len(run.ALT))

    def test_eval_tensor_sizes(self):
        img = Image.new("RGB", (90, 60))
        self.assertEqual(tuple(bb.eval_tensor(img, 224).shape), (1, 3, 224, 224))
        self.assertEqual(tuple(bb.eval_tensor(img, 336).shape), (1, 3, 336, 336))


if __name__ == "__main__":
    unittest.main()
