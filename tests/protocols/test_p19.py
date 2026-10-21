"""Synthetic tests for P19 (no Real-IAD image is read)."""

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

from experiments.p19 import manifest as mf
from experiments.p19 import run


def _entry(cls, sample, view, split_folder="NG"):
    folder = f"NG/{cls}/S{sample:04d}" if cls != "OK" else f"{split_folder}/S{sample:04d}"
    path = f"{folder}/x_{sample:04d}_{cls}_C{view}_1.jpg"
    return {"category": "x", "anomaly_class": cls, "image_path": path,
            "mask_path": path[:-4] + ".png" if cls != "OK" else None}


class ManifestTest(unittest.TestCase):
    def setUp(self):
        test = [_entry(c, i, v) for c in ("AA", "BB") for i in range(60) for v in (1, 2)]
        test += [_entry("AA", 100 + i, 2) for i in range(5)]           # view 2 has more defects
        test += [_entry("OK", 200 + i, 2, "OK") for i in range(60)]     # genuine normal parts
        test += [_entry("OK", 300 + i, 2, "NG") for i in range(30)]     # defect not visible: never normal
        train = [_entry("OK", 400 + i, 2, "OK") for i in range(40)]
        self.task = mf.build_object("x", {"test": test, "train": train})

    def test_view_classes_and_split(self):
        t = self.task
        self.assertEqual(t["view"], 2)
        self.assertEqual(sorted(t["classes"]), ["AA", "BB"])
        for c in t["classes"].values():
            self.assertEqual(len(c["pool"]), 10)
            self.assertEqual(len(c["queries"]), 40)
            self.assertFalse(set(c["pool"]) & set(c["queries"]))
        self.assertTrue(all(p.startswith("OK/") for p in t["normal"]["pool"] + t["normal"]["queries"]))
        self.assertEqual(t["normal"]["available"], 60)

    def test_supports_nested_and_banks_disjoint(self):
        t = self.task
        for s, per in t["supports"].items():
            self.assertEqual(sorted(per["AA"]), sorted(t["classes"]["AA"]["pool"]))
            self.assertEqual(len(t["banks"][s]), 20)
            self.assertTrue(all("/S04" in p for p in t["banks"][s]))  # train normals only

    def test_deterministic(self):
        again = mf.build_object("x", {"test": [_entry(c, i, v) for c in ("AA", "BB") for i in range(60) for v in (1, 2)]
                                      + [_entry("AA", 100 + i, 2) for i in range(5)]
                                      + [_entry("OK", 200 + i, 2, "OK") for i in range(60)]
                                      + [_entry("OK", 300 + i, 2, "NG") for i in range(30)],
                                      "train": [_entry("OK", 400 + i, 2, "OK") for i in range(40)]})
        self.assertEqual(again, self.task)


class StatsTest(unittest.TestCase):
    def test_holm(self):
        adj = run.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)  # monotone

    def test_signflip_mc(self):
        self.assertLess(run.signflip_mc([1.0] * 30, draws=20000), 0.001)
        self.assertGreater(run.signflip_mc([1.0, -1.0] * 15, draws=20000), 0.5)

    def test_contrast(self):
        rows = []
        for o in ("o1", "o2", "o3"):
            for k in run.K_VALUES:
                for s in (1, 2):
                    rows.append({"method": "t", "object": o, "setting": "defect", "K": k, "seed": s, "balanced_accuracy": 0.6})
                    rows.append({"method": "r", "object": o, "setting": "defect", "K": k, "seed": s, "balanced_accuracy": 0.5})
        c = run.contrast(rows, "defect", "t", "r")
        self.assertAlmostEqual(c["delta_pp"], 10.0)
        self.assertEqual(c["positive_objects"], 3)
        self.assertAlmostEqual(c["ci95_pp"][0], 10.0)

    def test_mask_cells(self):
        with tempfile.TemporaryDirectory() as d:
            arr = np.zeros((48, 48), np.uint8)
            arr[:24, :24] = 255
            Image.fromarray(arr).save(Path(d) / "m.png")
            cells = run.mask_cells(Path(d) / "m.png", grid=2)
        self.assertTrue(torch.allclose(cells, torch.tensor([1.0, 0.0, 0.0, 0.0])))


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self._guarded = run.GUARDED
        run.GUARDED = ("experiments/p19_no_such_path",)  # git status of a clean, empty path set

    def tearDown(self):
        run.GUARDED = self._guarded

    def test_interrupted_then_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "blind"
            with self.assertRaises(RuntimeError):
                with run.single_shot(out) as staging:
                    run.save_checkpoint(staging, "objA", "h", "m", [{"x": 1}])
                    raise RuntimeError("simulated silent stop")
            self.assertTrue((Path(d) / "blind.lock").is_dir())
            with self.assertRaises(run.rc.P6BFailClosedError):  # a plain relaunch still fails closed
                with run.single_shot(out):
                    pass
            with run.single_shot(out, resume=True) as staging:
                self.assertEqual(run.load_checkpoint(staging, "objA", "h", "m"), [{"x": 1}])
                self.assertIsNone(run.load_checkpoint(staging, "objB", "h", "m"))
                with self.assertRaises(run.rc.P6BFailClosedError):
                    run.load_checkpoint(staging, "objA", "other-commit", "m")
            self.assertTrue(out.is_dir())
            launches = __import__("json").loads((out / "launches.json").read_text(encoding="utf-8"))
            self.assertEqual([x["resume"] for x in launches], [False, True])

    def test_resume_needs_interrupted_launch(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(run.rc.P6BFailClosedError):
                with run.single_shot(Path(d) / "blind", resume=True):
                    pass


if __name__ == "__main__":
    unittest.main()
