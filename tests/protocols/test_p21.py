"""Tests for P21 (CLIP-LoRA on MVTec-FS). No model is loaded; the scorer is fed synthetic records."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p21 import cliplora_runner as runner
from experiments.p21 import score

EPISODES = PROJECT_ROOT / "docs" / "p21" / "P21_MVTEC_FS_EPISODES.json"


class EpisodeFileTest(unittest.TestCase):
    def setUp(self):
        self.doc = runner.load_episodes(EPISODES)

    def test_frozen_grid_and_no_query_labels(self):
        eps = self.doc["episodes"]
        self.assertEqual(len(eps), 13 * 3 * 5)
        self.assertEqual(sum(len(v) for v in self.doc["queries"].values()), 530)
        self.assertEqual(len({runner.episode_key(e) for e in eps}), len(eps))
        for qs in self.doc["queries"].values():
            for q in qs:
                self.assertEqual(set(q), {"rel_path", "sha256"})
        for e in eps:
            counts = [sum(s["label"] == c for s in e["supports"]) for c in range(len(e["class_order"]))]
            self.assertEqual(counts, [e["K"]] * len(e["class_order"]))

    def test_shards_partition_the_episodes(self):
        eps = self.doc["episodes"]
        for n in (1, 2, 3, 4):
            keys = [runner.episode_key(e) for i in range(n) for e in runner.shard_of(eps, i, n)]
            self.assertEqual(sorted(keys), sorted(runner.episode_key(e) for e in eps))
        self.assertEqual(runner.heaviest(eps)["K"], 5)

    def test_shard_count_divisible_by_three_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            argv = ["--episodes", str(EPISODES), "--data-root", d, "--cliplora", d, "--out", d, "--shard", "0/6"]
            with mock.patch.object(runner, "verify_images", return_value=0),                     mock.patch.object(runner, "environment", return_value={"cliplora_dirty": False}),                     mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}),                     mock.patch("os.chdir"):
                with self.assertRaises(SystemExit) as cm:
                    runner.main(argv)
            self.assertIn("not divisible", str(cm.exception))
            self.assertFalse((Path(d) / "locks").exists())

    def test_tampered_episode_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.json"
            p.write_bytes(EPISODES.read_bytes() + b" ")
            with self.assertRaises(SystemExit):
                runner.load_episodes(p)

    def test_verify_images_detects_a_changed_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.png").write_bytes(b"abc")
            doc = {"episodes": [{"supports": [{"rel_path": "a.png", "sha256": hashlib.sha256(b"abc").hexdigest()}]}],
                   "queries": {"x": [{"rel_path": "b.png", "sha256": hashlib.sha256(b"xyz").hexdigest()}]}}
            with self.assertRaises(SystemExit):
                runner.verify_images(doc, root)
            (root / "b.png").write_bytes(b"xyz")
            self.assertEqual(runner.verify_images(doc, root), 2)


class CheckoutTest(unittest.TestCase):
    def test_untracked_cache_is_clean_but_edits_are_dirty(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            git = ["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t"]
            subprocess.run(["git", "init", "-q", d], check=True)
            (Path(d) / "lora.py").write_text("x = 1" + chr(10), encoding="utf-8")
            subprocess.run(git + ["add", "lora.py"], check=True)
            subprocess.run(git + ["commit", "-q", "-m", "c"], check=True)
            (Path(d) / "__pycache__").mkdir()
            (Path(d) / "__pycache__" / "lora.cpython-310.pyc").write_bytes(b"0")
            self.assertFalse(runner.checkout_dirty(Path(d)))
            (Path(d) / "lora.py").write_text("x = 2" + chr(10), encoding="utf-8")
            self.assertTrue(runner.checkout_dirty(Path(d)))


class ResolutionTest(unittest.TestCase):
    def test_positional_embedding_interpolation_keeps_cls_and_grid_corners(self):
        import torch
        from types import SimpleNamespace
        pe = torch.randn(14 * 14 + 1, 8)
        visual = SimpleNamespace(conv1=SimpleNamespace(kernel_size=(16, 16)),
                                 positional_embedding=torch.nn.Parameter(pe.clone()), input_resolution=224)
        model = SimpleNamespace(visual=visual)
        runner.resize_positional_embedding(model, 336)
        new = visual.positional_embedding.data
        self.assertEqual(tuple(new.shape), (21 * 21 + 1, 8))
        self.assertTrue(torch.equal(new[0], pe[0]))
        self.assertEqual(visual.input_resolution, 336)
        runner.resize_positional_embedding(model, 336)          # idempotent at the target size
        self.assertEqual(tuple(visual.positional_embedding.shape), (442, 8))


class BackboneTest(unittest.TestCase):
    """P29: ViT-L/14@336px option; the ViT-B/16 paths of P21 / P25 must stay as they were."""

    def _argv(self, d, *extra):
        return ["--episodes", str(EPISODES), "--data-root", d, "--cliplora", d, "--out", d, *extra]

    def test_official_args_carry_the_backbone(self):
        sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "CLIP-LoRA"))
        try:
            self.assertEqual(runner.official_args(5, 111).backbone, "ViT-B/16")
            a = runner.official_args(5, 111, runner.L14_336)
            self.assertEqual((a.backbone, a.shots, a.seed), (runner.L14_336, 5, 111))
        finally:
            sys.path.remove(str(PROJECT_ROOT / "third_party" / "CLIP-LoRA"))

    def test_l14_only_at_its_native_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                runner.main(self._argv(d, "--backbone", runner.L14_336, "--verify-only"))
            self.assertIn("native 336", str(cm.exception))

    def test_a_b16_directory_refuses_l14(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "resolution.txt").write_text("336")          # a P25-like directory, no backbone marker
            with mock.patch.object(runner, "verify_images", return_value=0):
                with self.assertRaises(SystemExit) as cm:
                    runner.main(self._argv(d, "--backbone", runner.L14_336, "--resolution", "336", "--smoke"))
            self.assertIn("ViT-B/16", str(cm.exception))
            self.assertFalse((Path(d) / "backbone.txt").exists())

    def test_l14_file_is_hash_checked(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "ViT-L-14-336px.pt"
            fake.write_bytes(b"not the OpenAI file")
            clip = mock.MagicMock()
            with mock.patch.object(runner, "L14_336_FILE", fake), mock.patch.object(runner, "_VERIFIED", set()):
                with self.assertRaises(SystemExit):
                    runner.load_clip(clip, runner.L14_336)
            clip.load.assert_not_called()

    def test_b16_is_the_official_load(self):
        clip = mock.MagicMock()
        clip.load.return_value = ("model", "preprocess")
        self.assertEqual(runner.load_clip(clip, "ViT-B/16"), ("model", "preprocess", "ViT-B/16"))
        clip.load.assert_called_once_with("ViT-B/16")


class ScoreTest(unittest.TestCase):
    def test_lora_rows_join_local_labels(self):
        e = {"product": "p", "K": 1, "seed": 111, "class_order": ["a", "b"],
             "supports": [{"rel_path": "s0", "sha256": "0", "label": 0}, {"rel_path": "s1", "sha256": "1", "label": 1}]}
        doc = {"episodes": [e], "queries": {"p": [{"rel_path": "q0", "sha256": "h0"}, {"rel_path": "q1", "sha256": "h1"}]}}
        rec = {"schema_version": runner.SCHEMA, "class_order": ["a", "b"],
               "fidelity_share_class0": {"match": True},
               "queries": [{"sha256": "h0", "cliplora_pred": "a", "zeroshot_pred": "b"},
                           {"sha256": "h1", "cliplora_pred": "b", "zeroshot_pred": "b"}]}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "episodes").mkdir()
            (Path(d) / "episodes" / "p_k1_s111.json").write_text(json.dumps(rec), encoding="utf-8")
            rows = score.lora_rows(Path(d), doc, {("p", "h0"): "a", ("p", "h1"): "b"})
            by = {r["method"]: r for r in rows}
            self.assertAlmostEqual(by[score.LORA]["balanced_accuracy"], 1.0)
            self.assertAlmostEqual(by[score.ZS]["balanced_accuracy"], 0.5)
            rec["fidelity_share_class0"]["match"] = False
            (Path(d) / "episodes" / "p_k1_s111.json").write_text(json.dumps(rec), encoding="utf-8")
            with self.assertRaises(Exception):
                score.lora_rows(Path(d), doc, {("p", "h0"): "a", ("p", "h1"): "b"})

    def test_runner_is_standalone(self):
        text = (PROJECT_ROOT / "experiments" / "p21" / "cliplora_runner.py").read_text(encoding="utf-8")
        self.assertNotIn("from experiments", text)
        self.assertNotIn("import experiments", text)


if __name__ == "__main__":
    unittest.main()
