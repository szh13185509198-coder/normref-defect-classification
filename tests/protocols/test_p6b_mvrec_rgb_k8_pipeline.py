"""MVREC-RGB / Protocol B K8 nested shot-scaling infrastructure tests.

Synthetic K8 manifests, synthetic metric grids and fail-closed tamper
variants only; real AlphaCLIP weights are never loaded.

Covers:
1. K8 execution manifest: 135 units (3 methods x 9 products x 5 seeds),
   deterministic, K8 manifest self-SHA verification, tamper fail-closed;
2. K8 scaling evaluator: all four Protocol B decision states plus
   fail-closed completeness;
3. Stage 2 runner K8 gate: non-K8 product rejected, missing K8 manifest
   fail-closed (both before any model construction);
4. MVREC-RGB support runner K8 flag and command construction.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import P6BFailClosedError
from experiments.p6.mvrec_rgb_formal import k8_builder
from experiments.p6.mvrec_rgb_formal import k8_execution
from experiments.p6.mvrec_rgb_formal import k8_scaling_evaluator as k8scale

K8_PRODUCTS = k8_builder.K8_PRODUCTS
SEEDS = (111, 222, 333, 444, 555)


def make_records(n_per_class=12):
    records = []
    for product in K8_PRODUCTS:
        for cls in ("clsA", "clsB", "clsC"):
            for i in range(n_per_class):
                rel = f"image/{product}/training/{cls}/{i:03d}.png"
                sha = __import__("hashlib").sha256(
                    f"{product}|{cls}|{i}".encode()).hexdigest()
                records.append({
                    "schema_version": "p6b_mvtec_fs_rgb_image_record_v2",
                    "product": product,
                    "rgb_sha256": sha,
                    "canonical_relative_path": rel,
                    "source_pools": ["support"],
                    "unique_defect_types": [cls],
                    "multi_type_removed": False,
                })
    return records


def make_k8_bundle(base):
    records = make_records()
    records_path = base / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    # Minimal stage1-shaped bundle is not needed for k8_execution tests;
    # k8_builder.build_k8_bundle only requires the records JSONL.
    out = base / "k8_bundle"
    k8_builder.build_k8_bundle(records_path, out, base / "unused_stage1")
    return records_path, out


class TestK8ExecutionManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_k8x_"))
        self.records_path, self.bundle = make_k8_bundle(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_01_135_units_deterministic(self):
        m1 = k8_execution.build_k8_execution_manifest(self.bundle)
        m2 = k8_execution.build_k8_execution_manifest(self.bundle)
        self.assertEqual(m1["unit_count"], 135)
        self.assertEqual(len(m1["units"]), 135)
        keys = [u["key"] for u in m1["units"]]
        self.assertEqual(len(set(keys)), 135)
        self.assertEqual(m1, m2)
        self.assertEqual(set(m1["methods"]),
                         {"ours", "mvrec_rgb", "frozen_fusion"})
        self.assertEqual(m1["K"], 8)
        self.assertTrue(m1["K1_K3_K5_not_rerun"])
        # Every unit points at a K8 manifest with exact K support identity.
        for unit in m1["units"]:
            self.assertEqual(unit["K"], 8)
            self.assertEqual(len(unit["support_identities"]),
                             3 * 8)  # 3 classes x K=8

    def test_02_k8_manifest_tamper_fails_closed(self):
        target = self.bundle / "p6b_k8_support_manifests/bottle/k8_s111.json"
        manifest = json.loads(target.read_text(encoding="utf-8"))
        manifest["classes"]["clsA"]["items"][0]["rgb_sha256"] = "0" * 64
        target.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(P6BFailClosedError):
            k8_execution.build_k8_execution_manifest(self.bundle)

    def test_03_missing_k8_manifest_fails_closed(self):
        (self.bundle / "p6b_k8_support_manifests/capsule/k8_s222.json").unlink()
        with self.assertRaises(P6BFailClosedError):
            k8_execution.build_k8_execution_manifest(self.bundle)

    def test_04_mvrec_rgb_k8_command_carries_k8_flag(self):
        unit = k8_execution.K8MethodUnit("mvrec_rgb", "bottle", 111)
        command = k8_execution.k8_mvrec_rgb_command(
            "python", self.bundle, "data/mvtec-fs", "model/x.pth", unit,
            "out", "cuda:0")
        self.assertIn("--k8", command)
        self.assertIn("--k", command)
        self.assertIn("8", command)
        stage2_command = k8_execution.k8_stage2_command(
            "python", self.bundle, "data/mvtec-fs",
            k8_execution.K8MethodUnit("ours", "bottle", 111), "out", None)
        self.assertEqual(stage2_command[2], "experiments.p6.run_p6b_stage2_support")


class TestK8ScalingEvaluator(unittest.TestCase):
    def _grid(self, diff_fn):
        """Synthetic 3-method metric grid; diff_fn(product, k, seed) -> d_ba."""
        grid = {method: {} for method in k8scale.METHODS}
        rng = np.random.RandomState(11)
        for product in K8_PRODUCTS:
            for k in (1, 3, 5, 8):
                for seed in SEEDS:
                    base = 0.50 + 0.02 * k + rng.uniform(0, 0.02)
                    d = diff_fn(product, k, seed)
                    grid["mvrec_rgb"][(product, k, seed)] = {
                        "balanced_accuracy": base,
                        "macro_f1": base - 0.01,
                    }
                    grid["ours"][(product, k, seed)] = {
                        "balanced_accuracy": base + d,
                        "macro_f1": base - 0.01 + d,
                    }
                    grid["frozen_fusion"][(product, k, seed)] = {
                        "balanced_accuracy": base + d * 0.5,
                        "macro_f1": base - 0.01 + d * 0.5,
                    }
        return grid

    def test_05_monotonic_scaling_confirmed(self):
        def d_fn(p, k, s):
            return 0.05 + 0.015 * k  # strictly increasing, C = 0.105 > 0
        grid = self._grid(d_fn)
        result = k8scale.final_scaling_decision(grid)
        primary = result["primary_contrast"]
        self.assertEqual(primary["state"], "CONFIRMED_MONOTONIC_SHOT_SCALING")
        self.assertTrue(primary["gates"]["delta_monotonic_1_3_5_8"])
        self.assertEqual(primary["c_product_positive_count"], 9)
        self.assertEqual(primary["c_seed_positive_count"], 5)
        self.assertGreater(primary["C"], 0)
        self.assertGreater(primary["C_ci95"][0], 0)

    def test_06_k8_below_k1_no_positive_scaling(self):
        def d_fn(p, k, s):
            return 0.05 + 0.01 * k if k <= 5 else 0.03  # C = 0.03-0.06 < 0
        grid = self._grid(d_fn)
        primary = k8scale.final_scaling_decision(grid)["primary_contrast"]
        self.assertEqual(primary["state"], "NO_POSITIVE_K8_VS_K1_SCALING")
        self.assertLess(primary["C"], 0)

    def test_07_k8_trajectory_break_locked_text(self):
        def d_fn(p, k, s):
            return {1: 0.02, 3: 0.04, 5: 0.06, 8: 0.055}[k]
        grid = self._grid(d_fn)
        primary = k8scale.final_scaling_decision(grid)["primary_contrast"]
        # Delta[8]=0.055 <= Delta[5]=0.06 with C>0: fixed decision text.
        self.assertEqual(primary["state"], "NO_MONOTONIC_ADVANTAGE_AT_K8")
        self.assertEqual(
            primary["required_decision_text"],
            "Positive adaptation benefit exists, but no monotonic "
            "shot-scaling advantage is confirmed.",
        )
        self.assertGreater(primary["C"], 0)

    def test_08_missing_unit_fails_closed(self):
        def d_fn(p, k, s):
            return 0.05 + 0.01 * k
        grid = self._grid(d_fn)
        del grid["ours"][("bottle", 8, 111)]
        with self.assertRaises(P6BFailClosedError):
            k8scale.final_scaling_decision(grid)

    def test_09_non_finite_metric_fails_closed(self):
        def d_fn(p, k, s):
            return 0.05 + 0.01 * k
        grid = self._grid(d_fn)
        grid["ours"][("bottle", 8, 111)]["balanced_accuracy"] = float("nan")
        with self.assertRaises(P6BFailClosedError):
            k8scale.final_scaling_decision(grid)

    def test_10_bootstrap_deterministic_and_paired(self):
        def d_fn(p, k, s):
            return 0.05 + 0.01 * k + 0.001 * ((s % 17) % 3 - 1)
        grid = self._grid(d_fn)
        r1 = k8scale.final_scaling_decision(grid)
        r2 = k8scale.final_scaling_decision(grid)
        self.assertEqual(r1["primary_contrast"]["C_ci95"],
                         r2["primary_contrast"]["C_ci95"])
        self.assertEqual(r1["bootstrap"]["seed"], 20260813)
        self.assertEqual(r1["bootstrap"]["resamples"], 10000)
        self.assertEqual(
            r1["bootstrap"]["sampling_unit"],
            "product cluster (five paired seeds together)")


class TestStage2K8Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_k8s2_"))
        self.records_path, self.bundle = make_k8_bundle(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _call(self, product, method="ours", bundle=None, k=8):
        from experiments.p6 import run_p6b_stage2_support as stage2_support

        return stage2_support.execute_run(
            bundle_dir=str(bundle or self.bundle),
            image_root=str(self.tmp / "images"),
            product=product, k_shot=k, seed=111, method=method,
            output_dir=str(self.tmp / "out" / product),
            allow_dirty_diagnostic=True,
        )

    def test_11_non_k8_product_rejected_before_model(self):
        # grid is not in the nine-product K8 set: fail-closed before any
        # model construction.
        with self.assertRaises(P6BFailClosedError):
            self._call("grid")

    def test_12_missing_k8_manifest_fails_closed(self):
        (self.bundle / "p6b_k8_support_manifests/bottle/k8_s111.json").unlink()
        with self.assertRaises(P6BFailClosedError):
            self._call("bottle")

    def test_13_tampered_k8_manifest_fails_closed(self):
        target = self.bundle / "p6b_k8_support_manifests/bottle/k8_s111.json"
        manifest = json.loads(target.read_text(encoding="utf-8"))
        manifest["classes"]["clsA"]["items"][0]["rgb_sha256"] = "f" * 64
        target.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(P6BFailClosedError):
            self._call("bottle")

    def test_14_k1_3_5_still_reject_k8_bundle(self):
        # K=1/3/5 keep the frozen Stage 1 bundle gate; the K8 bundle lacks
        # the Stage 1 completion, so it must fail-closed.
        with self.assertRaises(P6BFailClosedError):
            self._call("bottle", k=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
