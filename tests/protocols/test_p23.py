"""P23 tests: the resolution switch reaches every place that carries it (no model, no image)."""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p23 import mvrec_rgb_res as p23


class ConfigureTest(unittest.TestCase):
    def tearDown(self):
        p23.configure(224, Path("unused"))

    def test_336_reaches_param_dataset_model_and_query_loader(self):
        p23.configure(336, Path("x.pth"))
        param = p23.rgb_param.build_rgb_param(["a", "b"], k_shot=1)
        self.assertEqual(param.data.input_shape, 336)
        self.assertEqual(param.ClipModel.input_shape, 336)
        self.assertEqual(param.ClipModel.backbone_name, "ViT-L/14@336px")
        self.assertEqual(p23.rgb_model.OFFICIAL_INPUT_SHAPE, 336)
        self.assertEqual(p23.rgb_model.OFFICIAL_BACKBONE, "ViT-L/14@336px")
        self.assertEqual(p23.rgb_dataset.OFFICIAL_INPUT_SHAPE, 336)
        self.assertEqual(p23.qe.OFFICIAL_INPUT_SHAPE, 336)
        self.assertTrue(os.environ["MVREC_ALPHA_L14_336_CKPT"].endswith("x.pth"))

    def test_224_is_the_untouched_p6_setting(self):
        p23.configure(224, Path("unused"))
        param = p23.rgb_param.build_rgb_param(["a", "b"], k_shot=1)
        self.assertEqual((param.data.input_shape, param.ClipModel.backbone_name), (224, "ViT-L/14"))
        self.assertEqual(p23.qe.OFFICIAL_INPUT_SHAPE, 224)

    def test_grid_is_the_frozen_195_units(self):
        units = list(p23.expected_units())
        self.assertEqual(len(units), 195)
        self.assertIn(p23.REPRO_UNIT.key, {u.key for u in units})


if __name__ == "__main__":
    unittest.main()
