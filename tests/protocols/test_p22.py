"""Synthetic tests for P22 (no model, no image)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p22 import lightweight as lw


class P22Test(unittest.TestCase):
    def test_grid(self):
        names = lw.config_names()
        self.assertEqual(len(names), 12)
        self.assertIn(lw.ANCHOR, names)

    def test_layer_selection_prefers_best_then_shallower(self):
        ba = {f"arr_{c}_L{L}": 50.0 for c in lw.config_names() for L in lw.LAYERS}
        ba[f"arr_{lw.ANCHOR}_L11"] = 60.0
        ba[f"arr_{lw.ANCHOR}_L12"] = 60.0
        self.assertEqual(lw.select(ba)[lw.ANCHOR], 11)
        self.assertEqual(lw.select(ba)["dinov2_s_r224_fp16"], lw.LAYERS[0])

    def test_integrity_maps_anchor_rows_to_p18_names(self):
        metric = {"balanced_accuracy": 0.5, "macro_f1": 0.4, "top1_accuracy": 0.6}
        rows = [{"method": f"{m}_{lw.ANCHOR}_L{L}", "product": "p", "K": 1, "seed": 9001, **metric}
                for m in lw.METHODS for L in lw.LAYERS]
        rows.append({"method": f"arr_dinov2_s_r224_fp32_L6", "product": "p", "K": 1, "seed": 9001, **metric})
        ref = [{"method": f"{m}_dinov2_b_L{L}", "product": "p", "K": 1, "seed": 9001, **metric}
               for m in lw.METHODS for L in lw.LAYERS]
        ref.append({"method": "xc_dinov2_b_L6", "product": "p", "K": 1, "seed": 9001, **metric})
        with mock.patch.object(lw.rc, "load_json", return_value={"rows": ref}):
            self.assertEqual(lw.p18_integrity(rows), 0.0)
            ref[0] = {**ref[0], "balanced_accuracy": 0.7}
            self.assertAlmostEqual(lw.p18_integrity(rows), 0.2)

    def test_formal_contrasts_name_methods_that_are_produced(self):
        from experiments.p22 import formal
        produced = {f"{s}_{t}" for s in formal.SHORT.values() for t in (formal.S32, formal.S16, formal.B)}
        produced |= {formal.ARR, formal.SS}   # the main rows come from the P14 results
        for name, (t, r) in formal.DECLARED.items():
            self.assertIn(t, produced, name)
            self.assertIn(r, produced, name)
        self.assertEqual(formal.B, "dinov2_b")   # the P18 formal names, needed by the integrity check

    def test_resume_flag_reaches_both_stages(self):
        for stage, fn in (("dev", "run_dev"), ("profile", "run_profile")):
            with mock.patch.object(lw, fn, return_value=0) as run:
                lw.main(["--stage", stage, "--mvtec-root", "x", "--device", "cpu", "--resume"])
            self.assertIs(run.call_args.args[-1], True)

    def test_single_shot_needs_resume_after_a_stop(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "dev"
            clean = mock.Mock(stdout="")
            with mock.patch.object(lw.subprocess, "run", return_value=clean), \
                    mock.patch.object(lw.rc, "git_provenance", return_value={}):
                with self.assertRaises(RuntimeError):
                    with lw.single_shot(out, False) as staging:
                        (staging / "products").mkdir()
                        raise RuntimeError("stopped")
                with self.assertRaises(lw.rc.P6BFailClosedError):
                    with lw.single_shot(out, False):
                        pass
                with lw.single_shot(out, True) as staging:
                    self.assertTrue((staging / "products").is_dir())
            self.assertTrue(out.is_dir())
            self.assertEqual(len(lw.json.loads((out / "launches.json").read_text(encoding="utf-8"))), 2)


if __name__ == "__main__":
    unittest.main()
