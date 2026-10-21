"""MVREC-RGB formal pipeline focused regression tests.

Covers the full pre-cloud infrastructure with synthetic manifests, tiny
PNGs and a fake official-pipeline model (real AlphaCLIP weights are never
loaded here):

1. 195-unit execution manifest / job list (deterministic, fail-closed);
2. formal support-grid runner (checkpoint + provenance hashing, atomic
   publish, gradient/step gates);
3. support-grid audit (unit and global validation, tamper fail-closed);
4. query firewall (gate raise, no query imports in support modules);
5. V1 GO/NO-GO decision evaluator (all three states, bootstrap determinism);
6. K8 manifest builder + nested K1/K3/K5/K8 identity audit.
"""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import (
    LOCKED_K,
    LOCKED_PRODUCTS,
    LOCKED_SEEDS,
    P6BFailClosedError,
    build_support_manifest,
    finalize_manifest,
)
from experiments.p6.mvrec_rgb_formal import (
    decision_evaluator as decision,
    execution_manifest as execman,
    grid_audit,
    k8_builder,
    query_evaluator,
    support_runner,
)

PKG_DIR = PROJECT_ROOT / "experiments" / "p6" / "mvrec_rgb_formal"


# ---------------------------------------------------------------------------
# Synthetic assets
# ---------------------------------------------------------------------------

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path, format="PNG")
    return path


def _image_color(product, cls, i):
    pi = LOCKED_PRODUCTS.index(product)
    ci = {"clsA": 0, "clsB": 1, "clsC": 2}[cls]
    return ((pi * 13 + i * 7) % 256, (ci * 51 + i * 11) % 256, (i * 29) % 256)


def _png_sha256(color):
    import io

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def make_records(n_per_class=6):
    """Synthetic P6.B image records: 13 products x 3 classes x N candidates
    with RGB SHA256 derived from the deterministic image content."""
    records = []
    for product in LOCKED_PRODUCTS:
        for cls in ("clsA", "clsB", "clsC"):
            for i in range(n_per_class):
                rel = f"image/{product}/training/{cls}/{i:03d}.png"
                sha = _png_sha256(_image_color(product, cls, i))
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


def make_bundle(base, records):
    """Frozen-bundle-shaped directory with K1/K3/K5 manifests per product."""
    bundle = base / "bundle"
    image_root = base / "mvtec-fs"
    for product in LOCKED_PRODUCTS:
        for k_shot in LOCKED_K:
            for seed in LOCKED_SEEDS:
                manifest = finalize_manifest(
                    build_support_manifest(records, product, k_shot, seed)
                )
                rel = Path("p6b_support_manifests") / product / f"k{k_shot}_s{seed}.json"
                path = bundle / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(manifest), encoding="utf-8")
                for cls in manifest["class_order"]:
                    for item in manifest["classes"][cls]["items"]:
                        rel_path = item["rel_path"]
                        parts = rel_path.split("/")
                        idx = int(parts[-1].split(".")[0])
                        write_image(image_root / rel_path,
                                    _image_color(product, cls, idx))
    return bundle, image_root


def make_single_manifest(base, product="bottle", k=1, seed=111, records=None):
    records = records or make_records()
    bundle, image_root = make_bundle(base, records)
    manifest = finalize_manifest(build_support_manifest(records, product, k, seed))
    path = bundle / f"p6b_support_manifests/{product}/k{k}_s{seed}.json"
    return manifest, path, image_root


# ---------------------------------------------------------------------------
# Fake official-pipeline model (EchoClassfierF-shaped head, real AdamW steps)
# ---------------------------------------------------------------------------

class _FakeZifa(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(8, 8))

    def forward(self, x):
        return self.adapter(x) + x


class _FakeHead(nn.Module):
    def __init__(self, n_classes, extra_param=False):
        super().__init__()
        self.zifa = _FakeZifa()
        self.support_key = None  # set during set_img_prototype / _init_weight
        self.tau = torch.tensor(0.11)
        self.frozen = nn.Parameter(torch.zeros(5), requires_grad=False)
        self.extra = nn.Parameter(torch.zeros(3)) if extra_param else None

    def _init_weight(self, cache_keys, cache_vals, proxy_style):
        # cache_keys [N, V, L, C]; cache_vals [N, n_classes] one-hot
        labels = cache_vals.argmax(dim=-1)
        flat = cache_keys.mean(dim=1).mean(dim=1)  # [N, C]
        self.support_key = nn.Parameter(torch.cat([
            flat[labels == c].mean(dim=0, keepdim=True)
            for c in range(cache_vals.shape[1])
        ], dim=0))  # [n_classes, C]

    def forward(self, x):
        emb = self.zifa(x)
        logits = emb @ self.support_key.t() / self.tau
        return {"predicts": logits.softmax(-1), "logits": logits}


class FakeMvrecModel(nn.Module):
    def __init__(self, class_names, extra_param=False, short_steps=False):
        super().__init__()
        self.class_names = list(class_names)
        self.text_features = torch.randn(len(class_names), 8)
        self.head = _FakeHead(len(class_names), extra_param=extra_param)
        self.mode = "eval"
        self.proto = None
        self.short_steps = short_steps
        self._steps = 499 if short_steps else 500
        self._forward_called = False
        self._forward_inputs = None

    def set_img_prototype(self, k_shot, dataset):
        weight_x, weight_y = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            feat = torch.randn(1, 1, 8)  # per-image [V=1, L=1, C]
            weight_x.append(feat)
            weight_y.append(
                F.one_hot(sample["y"], len(self.class_names)).float().squeeze(0)
            )
        weight_x = torch.stack(weight_x)
        weight_y = torch.stack(weight_y)
        self.proto = (weight_x, weight_y)
        self.head._init_weight(weight_x, weight_y, "onehot")
        for n, p in self.named_parameters():
            p.requires_grad = ("support_key" in n) or ("zifa" in n) or ("extra" in n)
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, eps=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self._steps, eta_min=1e-5
        )
        feats = weight_x.reshape(len(dataset), 8)
        labels = torch.arange(len(dataset)) % len(self.class_names)
        for step in range(self._steps):
            optimizer.zero_grad()
            logits = self.head(feats)["logits"]
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        self.proto = (weight_x, weight_y)

    def forward(self, input_dict, mulit_view):
        """Official-shape forward used by the query evaluator tests.

        Mirrors the official assemble_uncertainty contract: the aggregated
        predicts are squeezed, so batch=1 returns [C] and batch>1 returns
        [B, C] (see MVREC-main/modules/model.py forward).
        """
        x = input_dict["x"]
        assert input_dict["masks"].shape[0] == x.shape[0]
        self._forward_called = True
        self._forward_inputs = input_dict
        b = x.shape[0]
        emb = self.head.zifa(torch.randn(b, 8))
        logits = emb @ self.head.support_key.t() / self.head.tau
        predicts = logits.softmax(-1)
        if b == 1:
            predicts = predicts.squeeze(0)  # official squeeze: [C]
        return {"predicts": predicts, "logits": logits}

    def set_mode(self, mode):
        self.mode = mode
        if mode == "eval":
            for p in self.parameters():
                p.requires_grad = False


# ---------------------------------------------------------------------------
# 1. Execution manifest
# ---------------------------------------------------------------------------

class TestExecutionManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_f_"))
        self.bundle, self.image_root = make_bundle(self.tmp, make_records())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_01_195_units_deterministic(self):
        m1 = execman.build_execution_manifest(self.bundle)
        m2 = execman.build_execution_manifest(self.bundle)
        self.assertEqual(m1["unit_count"], 195)
        self.assertEqual(len(m1["units"]), 195)
        keys = [u["key"] for u in m1["units"]]
        self.assertEqual(len(set(keys)), 195)
        self.assertEqual(m1, m2)
        self.assertEqual(m1["products"], list(LOCKED_PRODUCTS))
        self.assertEqual(m1["K_values"], [1, 3, 5])
        self.assertEqual(m1["seeds"], [111, 222, 333, 444, 555])

    def test_02_manifest_sha_mismatch_fails_closed(self):
        target = self.bundle / "p6b_support_manifests/bottle/k1_s111.json"
        manifest = json.loads(target.read_text(encoding="utf-8"))
        manifest["classes"]["clsA"]["items"][0]["rgb_sha256"] = "0" * 64
        target.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(P6BFailClosedError):
            execman.build_execution_manifest(self.bundle)

    def test_03_missing_manifest_fails_closed(self):
        (self.bundle / "p6b_support_manifests/bottle/k1_s111.json").unlink()
        with self.assertRaises(P6BFailClosedError):
            execman.build_execution_manifest(self.bundle)


# ---------------------------------------------------------------------------
# 2. Support runner
# ---------------------------------------------------------------------------

class TestSupportRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_sr_"))
        self.records = make_records()
        self.manifest, self.manifest_path, self.image_root = make_single_manifest(
            self.tmp, records=self.records
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, extra_param=False, short_steps=False, tamper=None):
        unit = execman.MvrecRgbUnit("bottle", 1, 111)
        output_dir = self.tmp / "out" / "mvrec_rgb" / "bottle" / "k1_s111"
        execution = {"commit": "a" * 40, "dirty": False,
                     "allow_dirty_diagnostic": False}
        fake_alpha = self.tmp / "fake_alpha.pth"
        fake_alpha.write_bytes(b"fake-alpha-clip-weights")
        if tamper == "image":
            rel = self.manifest["classes"]["clsA"]["items"][0]["rel_path"]
            write_image(self.image_root / rel, (250, 250, 250))
        return support_runner.run_support_unit(
            unit=unit,
            manifest=self.manifest,
            manifest_path=self.manifest_path,
            image_root=self.image_root,
            alpha_ckpt=self.tmp / "fake_alpha.pth",
            device="cpu",
            output_dir=output_dir,
            execution=execution,
            model_factory=lambda class_names: FakeMvrecModel(
                class_names, extra_param=extra_param, short_steps=short_steps
            ),
        )

    def test_04_unit_completes_with_full_provenance(self):
        completion = self._run()
        self.assertEqual(completion["status"], "completed")
        self.assertEqual(completion["formal_eligible"], True)
        self.assertEqual(completion["query_pixels_read"], False)
        self.assertEqual(completion["query_labels_read"], False)
        self.assertEqual(completion["performance_metrics_generated"], False)
        self.assertEqual(completion["method"], "mvrec_rgb")
        self.assertEqual(completion["product"], "bottle")
        self.assertEqual(completion["K"], 1)
        self.assertEqual(completion["seed"], 111)
        self.assertEqual(
            set(completion["model_provenance"]["trainable_parameter_names"]),
            {"head.support_key", "head.zifa.adapter.0.weight",
             "head.zifa.adapter.0.bias"},
        )
        # protocol bundle pinned canonical-LF hashes
        self.assertEqual(completion["protocol_bundle"]["hash_mode"], "canonical-lf")
        self.assertTrue(completion["protocol_bundle"]["protocol_A_canonical_lf_sha256"])
        # artifact registry present with self-consistent sha
        unit_dir = self.tmp / "out" / "mvrec_rgb" / "bottle" / "k1_s111"
        for record in completion["artifacts"]:
            path = unit_dir / record["relative_path"]
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            self.assertEqual(file_sha256(path), record["sha256"])
        # checkpoint + support proto present
        self.assertTrue((unit_dir / support_runner.HEAD_CHECKPOINT).is_file())
        self.assertTrue((unit_dir / support_runner.SUPPORT_PROTO).is_file())
        self.assertFalse(
            unit_dir.with_name(unit_dir.name + support_runner.STAGING_SUFFIX).exists()
        )

    def test_05_checkpoint_head_weights_persist(self):
        completion = self._run()
        unit_dir = self.tmp / "out" / "mvrec_rgb" / "bottle" / "k1_s111"
        state = torch.load(str(unit_dir / support_runner.HEAD_CHECKPOINT),
                           map_location="cpu", weights_only=True)
        self.assertIn("head.support_key", state)
        self.assertIn("head.zifa.adapter.0.weight", state)
        step_audit = json.loads(
            (unit_dir / support_runner.STEP_AUDIT).read_text(encoding="utf-8")
        )
        self.assertEqual(step_audit["expected_steps"], 500)
        self.assertEqual(step_audit["actual_steps"], 500)
        self.assertTrue(step_audit["optimizer_ownership_ok"])
        self.assertTrue(step_audit["ce_loss_finite"])

    def test_06_trainable_whitelist_drift_fails_closed(self):
        with self.assertRaises(P6BFailClosedError):
            self._run(extra_param=True)

    def test_07_short_steps_fails_closed(self):
        with self.assertRaises(P6BFailClosedError):
            self._run(short_steps=True)

    def test_08_support_identity_tamper_fails_closed(self):
        with self.assertRaises(P6BFailClosedError):
            self._run(tamper="image")

    def test_09_failure_leaves_no_output(self):
        output_dir = self.tmp / "out2" / "mvrec_rgb" / "bottle" / "k1_s111"
        unit = execman.MvrecRgbUnit("bottle", 1, 111)
        fake_alpha = self.tmp / "fake_alpha.pth"
        fake_alpha.write_bytes(b"fake-alpha-clip-weights")
        with self.assertRaises(P6BFailClosedError):
            support_runner.run_support_unit(
                unit=unit, manifest=self.manifest,
                manifest_path=self.manifest_path,
                image_root=self.image_root,
                alpha_ckpt=fake_alpha,
                device="cpu", output_dir=output_dir,
                execution={"commit": "a" * 40, "dirty": False,
                           "allow_dirty_diagnostic": False},
                model_factory=lambda class_names: FakeMvrecModel(
                    class_names, short_steps=True
                ),
            )
        self.assertFalse(output_dir.exists())


# ---------------------------------------------------------------------------
# 3. Support-grid audit
# ---------------------------------------------------------------------------

class TestGridAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_ga_"))
        self.records = make_records()
        self.manifest, self.manifest_path, self.image_root = make_single_manifest(
            self.tmp, records=self.records
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_unit(self, commit="a" * 40):
        unit = execman.MvrecRgbUnit("bottle", 1, 111)
        output_dir = self.tmp / "grid" / "mvrec_rgb" / "bottle" / "k1_s111"
        fake_alpha = self.tmp / "fake_alpha.pth"
        fake_alpha.write_bytes(b"fake-alpha-clip-weights")
        support_runner.run_support_unit(
            unit=unit, manifest=self.manifest,
            manifest_path=self.manifest_path,
            image_root=self.image_root,
            alpha_ckpt=fake_alpha,
            device="cpu", output_dir=output_dir,
            execution={"commit": commit, "dirty": False,
                       "allow_dirty_diagnostic": False},
            model_factory=lambda class_names: FakeMvrecModel(class_names),
        )
        return unit, output_dir

    def test_10_valid_unit_passes(self):
        unit, unit_dir = self._make_unit()
        completion = grid_audit.validate_support_unit(unit, unit_dir, "a" * 40)
        self.assertEqual(completion["status"], "completed")

    def test_11_commit_mismatch_fails(self):
        unit, unit_dir = self._make_unit(commit="a" * 40)
        with self.assertRaises(P6BFailClosedError):
            grid_audit.validate_support_unit(unit, unit_dir, "b" * 40)

    def test_12_tampered_artifact_fails(self):
        unit, unit_dir = self._make_unit()
        checkpoint = unit_dir / support_runner.HEAD_CHECKPOINT
        checkpoint.write_bytes(b"tampered")
        with self.assertRaises(P6BFailClosedError):
            grid_audit.validate_support_unit(unit, unit_dir, "a" * 40)

    def test_13_tampered_completion_flags_fail(self):
        unit, unit_dir = self._make_unit()
        completion_path = unit_dir / support_runner.COMPLETION
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["query_pixels_read"] = True
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        with self.assertRaises(P6BFailClosedError):
            grid_audit.validate_support_unit(unit, unit_dir, "a" * 40)

    def test_14_global_grid_requires_all_195(self):
        unit, unit_dir = self._make_unit()
        grid_root = self.tmp / "grid"
        with self.assertRaises(P6BFailClosedError):
            grid_audit.validate_formal_grid(grid_root,
                                            require_global_completion=False)


# ---------------------------------------------------------------------------
# 4. Query firewall
# ---------------------------------------------------------------------------

class TestQueryFirewall(unittest.TestCase):
    def test_15_query_evaluator_gated(self):
        with self.assertRaises(P6BFailClosedError):
            query_evaluator.evaluate_query_grid(
                grid_root=Path("."), records_path=Path("."),
                dataset_root=Path("."), stage1_bundle=Path("."),
                output_dir=Path("."), alpha_ckpt=Path("."),
            )
        # With the gate flag the evaluator proceeds into the grid
        # validation, which fail-closes on this branch (no real query is
        # executed anywhere in these tests).
        with self.assertRaises(P6BFailClosedError):
            query_evaluator.evaluate_query_grid(
                grid_root=Path("."), records_path=Path("."),
                dataset_root=Path("."), stage1_bundle=Path("."),
                output_dir=Path("."), alpha_ckpt=Path("."),
                open_query_gate=True,
            )

    def test_16_support_modules_never_import_query_paths(self):
        import ast

        forbidden = {
            "query_evaluator", "evaluate_p6b_formal_grid",
            "p6b_stage3_common", "audit_p6b_formal_grid",
            "package_p6b_formal_grid",
        }
        support_files = ["support_runner.py", "support_grid.py",
                         "grid_audit.py", "execution_manifest.py",
                         "k8_builder.py", "decision_evaluator.py"]
        violations = []
        for filename in support_files:
            tree = ast.parse((PKG_DIR / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[-1] in forbidden:
                            violations.append((filename, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    tail = (node.module or "").split(".")[-1]
                    if tail in forbidden:
                        violations.append((filename, node.module))
                    for alias in node.names:
                        if alias.name in forbidden:
                            violations.append((filename, alias.name))
        self.assertEqual(violations, [])

    def test_17_decision_evaluator_imports_no_query_module(self):
        import ast

        tree = ast.parse(
            (PKG_DIR / "decision_evaluator.py").read_text(encoding="utf-8")
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn("query", node.module)
                self.assertNotIn("stage3", node.module)


# ---------------------------------------------------------------------------
# 5. V1 GO/NO-GO decision evaluator
# ---------------------------------------------------------------------------

def make_metric_grid(diff_fn):
    """Synthetic per-unit metrics for both methods."""
    grid = {method: {} for method in ("ours", "mvrec_rgb")}
    rng = np.random.RandomState(7)
    for product in LOCKED_PRODUCTS:
        for k_shot in LOCKED_K:
            for seed in LOCKED_SEEDS:
                base_ba = rng.uniform(0.55, 0.75)
                base_f1 = rng.uniform(0.50, 0.70)
                d_ba, d_f1 = diff_fn(product, k_shot, seed, base_ba, base_f1)
                grid["mvrec_rgb"][(product, k_shot, seed)] = {
                    "top1_accuracy": base_ba - 0.02,
                    "balanced_accuracy": base_ba,
                    "macro_f1": base_f1,
                }
                grid["ours"][(product, k_shot, seed)] = {
                    "top1_accuracy": base_ba - 0.02 + d_ba,
                    "balanced_accuracy": base_ba + d_ba,
                    "macro_f1": base_f1 + d_f1,
                }
    return grid


class TestDecisionEvaluator(unittest.TestCase):
    def test_18_performance_positive(self):
        grid = make_metric_grid(
            lambda p, k, s, ba, f1: (0.04 + 0.002 * (k - 1), 0.02)
        )
        result = decision.evaluate_v1_decision(grid)
        self.assertEqual(result["state"], "GO-PERFORMANCE_POSITIVE")
        self.assertTrue(result["gates"]["g1_dbar_ba_ci_lower_positive"])
        self.assertTrue(result["gates"]["g2_ba_two_k_positive_no_below_margin"])
        self.assertTrue(result["gates"]["g3_at_least_7_products_positive"])
        self.assertTrue(result["gates"]["g4_at_least_4_seeds_positive"])
        self.assertTrue(result["gates"]["g5_macro_f1_nonnegative"])

    def test_19_practical_noninferior_with_resource_gate(self):
        # BA diffs: only K5 positive, all inside the 1pp margin; with the
        # complete five-ratio resource block this must resolve to practical
        # non-inferiority rather than performance-positive.
        def d_fn(p, k, s, ba, f1):
            ba_d = 0.003 if k == 5 else -0.002
            return ba_d, 0.001

        grid = make_metric_grid(d_fn)
        full_resource = {
            "trainable_ratio": 0.30, "latency_ratio": 0.75,
            "throughput_ratio": 0.95, "gpu_time_ratio": 0.67,
            "memory_ratio": 1.05,
        }
        result = decision.evaluate_v1_decision(grid, resource=full_resource)
        self.assertEqual(result["state"], "GO-PRACTICAL_NONINFERIOR")
        self.assertTrue(
            result["gates"]["resource_advantage_gate"]["clear_resource_advantage"]
        )
        # Without the complete resource block the same data must NOT GO.
        result_no_resource = decision.evaluate_v1_decision(grid, resource={})
        self.assertEqual(result_no_resource["state"], "NO-GO-V1_STOP")

    def test_20_stable_mvrec_dominance_stop(self):
        grid = make_metric_grid(
            lambda p, k, s, ba, f1: (-0.05, -0.04)
        )
        result = decision.evaluate_v1_decision(grid)
        self.assertEqual(result["state"], "NO-GO-V1_STOP")
        self.assertTrue(result["stable_mvrec_dominance"])

    def test_21_bootstrap_deterministic(self):
        grid = make_metric_grid(lambda p, k, s, ba, f1: (0.01, 0.005))
        r1 = decision.evaluate_v1_decision(grid)
        r2 = decision.evaluate_v1_decision(grid)
        self.assertEqual(r1["bootstrap_results"], r2["bootstrap_results"])
        self.assertEqual(r1["state"], r2["state"])
        self.assertEqual(r1["bootstrap"]["seed"], 20260813)
        self.assertEqual(r1["bootstrap"]["resamples"], 10000)
        self.assertEqual(r1["non_inferiority_margin"], 0.010)

    def test_22_missing_unit_fails_closed(self):
        grid = make_metric_grid(lambda p, k, s, ba, f1: (0.01, 0.005))
        del grid["ours"][("bottle", 1, 111)]
        with self.assertRaises(P6BFailClosedError):
            decision.evaluate_v1_decision(grid)

    def test_23_non_finite_metric_fails_closed(self):
        grid = make_metric_grid(lambda p, k, s, ba, f1: (0.01, 0.005))
        grid["ours"][("bottle", 1, 111)]["balanced_accuracy"] = float("nan")
        with self.assertRaises(P6BFailClosedError):
            decision.evaluate_v1_decision(grid)

    def test_24_throughput_gate_direction(self):
        # Throughput ratio is Ours / MVREC-RGB and HIGHER-is-better:
        # "not worse by more than 10%" == throughput_ratio >= 0.90, and a
        # better throughput must never be rejected by the 1.25 upper-bound
        # gate.  All five ratios are present; the other conditions are set
        # to fail so the throughput condition is tested in isolation.
        def gate(throughput_ratio):
            return decision.resource_advantage_gate({
                "trainable_ratio": 0.80,   # cond1 false (>0.50)
                "latency_ratio": 0.75,
                "throughput_ratio": throughput_ratio,
                "gpu_time_ratio": 0.80,   # cond3 false (>0.67)
                "memory_ratio": 1.00,
            })

        self.assertFalse(gate(0.89)["condition2_latency"])  # worse than 10%
        self.assertFalse(gate(0.89)["clear_resource_advantage"])
        self.assertTrue(gate(0.90)["condition2_latency"])   # exactly at bound
        self.assertTrue(gate(0.90)["clear_resource_advantage"])
        self.assertTrue(gate(1.00)["condition2_latency"])
        self.assertTrue(gate(1.00)["clear_resource_advantage"])
        # 30% BETTER throughput must not fail via any upper-bound gate.
        result = gate(1.30)
        self.assertTrue(result["condition2_latency"])
        self.assertTrue(result["no_measured_ratio_exceeds_1_25"])
        self.assertTrue(result["clear_resource_advantage"])

    def test_30_resource_incomplete_fails_closed(self):
        # A partial resource block must never open the gate.
        partial = decision.resource_advantage_gate(
            {"trainable_ratio": 0.30}
        )
        self.assertFalse(partial["clear_resource_advantage"])
        self.assertTrue(partial["resource_measurement_problems"])
        self.assertTrue(any("missing" in p
                            for p in partial["resource_measurement_problems"]))
        # NaN / negative / zero ratios are fail-closed per field.
        base = {
            "trainable_ratio": 0.5, "latency_ratio": 0.7,
            "throughput_ratio": 1.0, "gpu_time_ratio": 0.6,
            "memory_ratio": 1.0,
        }
        for key in base:
            for bad_value in (float("nan"), -1.0, 0.0):
                candidate = dict(base)
                candidate[key] = bad_value
                result = decision.resource_advantage_gate(candidate)
                self.assertFalse(result["clear_resource_advantage"],
                                 f"{key}={bad_value} must fail closed")
                self.assertTrue(result["resource_measurement_problems"])
        # Decision-level: non-inferiority metrics plus an incomplete
        # resource block must stay NO-GO.
        grid = make_metric_grid(
            lambda p, k, s, ba, f1: (0.003 if k == 5 else -0.002, 0.001)
        )
        state = decision.evaluate_v1_decision(
            grid, resource={"trainable_ratio": 0.30}
        )["state"]
        self.assertEqual(state, "NO-GO-V1_STOP")


# ---------------------------------------------------------------------------
# 6. K8 builder + nested identity audit
# ---------------------------------------------------------------------------

class TestK8Nested(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_k8_"))
        self.records = make_records(n_per_class=12)
        self.records_path = self.tmp / "records.jsonl"
        with self.records_path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record) + "\n")
        self.bundle, _ = make_bundle(self.tmp, self.records)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_24_k8_bundle_45_units(self):
        out = self.tmp / "k8_bundle"
        completion = k8_builder.build_k8_bundle(
            self.records_path, out, self.bundle
        )
        self.assertEqual(completion["unit_count"], 45)
        self.assertEqual(len(completion["units"]), 45)
        self.assertEqual(completion["K"], 8)
        self.assertEqual(set(completion["products"]), set(k8_builder.K8_PRODUCTS))
        self.assertEqual(
            set(completion["excluded_by_availability"]),
            set(k8_builder.EXCLUDED_BY_AVAILABILITY),
        )
        for unit in completion["units"]:
            self.assertEqual(unit["seed"] in LOCKED_SEEDS, True)

    def test_25_nested_identity_passes(self):
        out = self.tmp / "k8_bundle"
        k8_builder.build_k8_bundle(self.records_path, out, self.bundle)
        audit = k8_builder.audit_nested_identity(self.bundle, out)
        self.assertTrue(audit["nested_pass"])
        self.assertEqual(audit["nested_chain"], "K1 subset K3 subset K5 subset K8")
        self.assertEqual(len(audit["checks"]), 45)

    def test_26_nested_violation_fails_closed(self):
        out = self.tmp / "k8_bundle"
        k8_builder.build_k8_bundle(self.records_path, out, self.bundle)
        # Corrupt one K8 class item identity: K1 no longer subsets K8.
        target = (out / "p6b_k8_support_manifests" / "bottle"
                  / "k8_s111.json")
        manifest = json.loads(target.read_text(encoding="utf-8"))
        manifest["classes"]["clsA"]["items"][0]["rgb_sha256"] = "f" * 64
        target.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(P6BFailClosedError):
            k8_builder.audit_nested_identity(self.bundle, out)

    def test_27_availability_minimum_fails_closed(self):
        # Rebuild records with fewer candidates for one product (screw
        # minimum is 11): with 6 per class the builder must refuse.
        records = make_records(n_per_class=6)
        records_path = self.tmp / "records_small.jsonl"
        with records_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        with self.assertRaises(P6BFailClosedError):
            k8_builder.build_k8_bundle(records_path, self.tmp / "k8_small",
                                       self.bundle)


# ---------------------------------------------------------------------------
# 7. Locked resource measurements (pre-cloud hardening)
# ---------------------------------------------------------------------------

class TestResourceMeasurements(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_rm_"))
        self.records = make_records()
        self.manifest, self.manifest_path, self.image_root = make_single_manifest(
            self.tmp, records=self.records
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_25_resource_block_present_with_scalar_counts(self):
        unit = execman.MvrecRgbUnit("bottle", 1, 111)
        output_dir = self.tmp / "out" / "mvrec_rgb" / "bottle" / "k1_s111"
        fake_alpha = self.tmp / "fake_alpha.pth"
        fake_alpha.write_bytes(b"fake-alpha-clip-weights")
        completion = support_runner.run_support_unit(
            unit=unit, manifest=self.manifest,
            manifest_path=self.manifest_path,
            image_root=self.image_root,
            alpha_ckpt=fake_alpha,
            device="cpu", output_dir=output_dir,
            execution={"commit": "a" * 40, "dirty": False,
                       "allow_dirty_diagnostic": False},
            model_factory=lambda class_names: FakeMvrecModel(class_names),
        )
        resource = completion["resource_measurements"]
        # Fake head: support_key [3,8]=24 + zifa 8x8+8=72 => 96 trainable
        # scalars in exactly 3 tensors; frozen [5] makes total 101.
        self.assertEqual(resource["trainable_parameter_tensor_count"], 3)
        self.assertEqual(resource["trainable_parameter_scalar_count"], 96)
        self.assertEqual(resource["total_parameter_scalar_count"], 101)
        self.assertGreaterEqual(resource["support_training_gpu_seconds"], 0)
        peak = resource["peak_allocated_gpu_memory_bytes"]
        if peak is not None:
            self.assertIsInstance(peak, int)
            self.assertGreaterEqual(peak, 0)
        for key in ("device", "torch_version", "precision", "gpu_name",
                    "cuda_version"):
            self.assertIn(key, resource)


# ---------------------------------------------------------------------------
# 8. Query evaluator implementation (gate closed, synthetic only)
# ---------------------------------------------------------------------------

class TestQueryEvaluatorImplementation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_qe_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_unit_dir(self, k=1):
        unit_dir = self.tmp / f"unit_k{k}"
        unit_dir.mkdir(parents=True)
        run_config = {
            "schema_version": support_runner.RUN_CONFIG_SCHEMA,
            "method": "mvrec_rgb", "product": "bottle", "K": k, "seed": 111,
            "class_order": ["clsA", "clsB", "clsC"], "ft_epo": 500,
            "zip_config_index": 6,
        }
        (unit_dir / support_runner.RUN_CONFIG).write_text(
            json.dumps(run_config), encoding="utf-8"
        )
        wx = torch.randn(3 * k, 1, 1, 8)
        wy = F.one_hot(torch.arange(3 * k) % 3, 3).float()
        torch.save((wx, wy), str(unit_dir / support_runner.SUPPORT_PROTO))
        # Real support-runner serialization: model.state_dict() filtered by
        # k.startswith('head.'), so every key carries the 'head.' prefix.
        head_state = {
            "head.support_key": torch.randn(3, 8),
            "head.zifa.adapter.0.weight": torch.randn(8, 8),
            "head.zifa.adapter.0.bias": torch.randn(8),
            "head.frozen": torch.zeros(5),
        }
        torch.save(head_state, str(unit_dir / support_runner.HEAD_CHECKPOINT))
        return unit_dir

    def test_26_rebuild_recovers_k_from_run_config(self):
        unit_dir = self._make_unit_dir(k=1)
        captured = {}

        def fake_factory(class_names, k_shot):
            captured["k_shot"] = k_shot
            return FakeMvrecModel(class_names)

        model = query_evaluator.rebuild_model_for_unit(
            unit_dir, None, "cpu", model_factory=fake_factory
        )
        self.assertEqual(captured["k_shot"], 1)
        self.assertEqual(model.head.support_key.shape, (3, 8))
        # strict head load: trained weights must match the checkpoint
        # (head.-prefixed keys, exactly as the support runner serializes).
        loaded = model.head.state_dict()
        checkpoint = torch.load(
            str(unit_dir / support_runner.HEAD_CHECKPOINT),
            map_location="cpu", weights_only=True,
        )
        self.assertTrue(
            torch.equal(loaded["support_key"], checkpoint["head.support_key"])
        )

    def test_34_strip_head_namespace_real_semantics(self):
        # Real support-runner key names map exactly to head-module names.
        tensor = torch.zeros(2)
        stripped = query_evaluator._strip_head_namespace({
            "head.logit_scale": tensor,
            "head.support_key": tensor,
            "head.zifa.adapter.0.weight": tensor,
        })
        self.assertEqual(set(stripped.keys()),
                         {"logit_scale", "support_key",
                          "zifa.adapter.0.weight"})
        self.assertIs(stripped["logit_scale"], tensor)
        self.assertIs(stripped["support_key"], tensor)
        self.assertIs(stripped["zifa.adapter.0.weight"], tensor)
        # Exactly one prefix strip: a double-prefixed key keeps the second.
        self.assertIn("head.zifa",
                      query_evaluator._strip_head_namespace(
                          {"head.head.zifa": tensor}
                      ))

    def test_35_non_head_key_fails_closed(self):
        with self.assertRaises(P6BFailClosedError):
            query_evaluator._strip_head_namespace(
                {"support_key": torch.zeros(1)}
            )
        # Empty dict is fail-closed too.
        with self.assertRaises(P6BFailClosedError):
            query_evaluator._strip_head_namespace({})

    def test_36_illegal_namespace_fails_closed(self):
        # Non-string key.
        with self.assertRaises(P6BFailClosedError):
            query_evaluator._strip_head_namespace({7: torch.zeros(1)})
        # Degenerate 'head.' key with empty remainder.
        with self.assertRaises(P6BFailClosedError):
            query_evaluator._strip_head_namespace(
                {"head.": torch.zeros(1)}
            )

    def test_37_unknown_key_strict_load_stays_enabled(self):
        # A well-formed head.-namespaced checkpoint containing an unknown
        # head parameter must still fail strict=True at load time.
        unit_dir = self._make_unit_dir(k=1)
        checkpoint = torch.load(
            str(unit_dir / support_runner.HEAD_CHECKPOINT),
            map_location="cpu", weights_only=True,
        )
        checkpoint["head.no_such_parameter"] = torch.zeros(3)
        torch.save(checkpoint, str(unit_dir / support_runner.HEAD_CHECKPOINT))
        with self.assertRaises(RuntimeError):
            query_evaluator.rebuild_model_for_unit(
                unit_dir, None, "cpu",
                model_factory=lambda class_names, k_shot:
                    FakeMvrecModel(class_names),
            )

    def test_27_predict_batch_uses_official_forward_constant_mask(self):
        model = FakeMvrecModel(["clsA", "clsB", "clsC"])
        wx = torch.randn(3, 1, 1, 8)
        wy = F.one_hot(torch.tensor([0, 1, 2]), 3).float()
        model.head._init_weight(wx, wy, "onehot")
        views = torch.randn(1, 1, 3, 16, 16)
        predicts = query_evaluator._predict_batch(model, views, "cpu")
        self.assertTrue(model._forward_called)
        # Official forward squeezes batch=1: the output is [C], not [1, C].
        self.assertEqual(predicts.shape, (3,))
        indices = query_evaluator._predicted_indices(predicts, 3)
        self.assertEqual(len(indices), 1)
        self.assertIn(indices[0], (0, 1, 2))
        masks = model._forward_inputs["masks"]
        self.assertTrue(torch.equal(masks, masks.flatten()[0].expand_as(masks)))
        expected = (1.0 - 0.5) / 0.26
        self.assertAlmostEqual(float(masks.flatten()[0]), expected, places=6)
        self.assertEqual(model._forward_inputs["x"].shape[1], 1)  # V=1

    def test_31_predicted_indices_shape_contract(self):
        # [C] (official batch=1 squeeze) and [B, C] are both accepted.
        self.assertEqual(
            query_evaluator._predicted_indices(
                torch.tensor([0.1, 0.7, 0.2]), 3), [1]
        )
        self.assertEqual(
            query_evaluator._predicted_indices(
                torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]), 3),
            [1, 0],
        )
        # Wrong class count must fail closed, never degrade to a constant.
        with self.assertRaises(P6BFailClosedError):
            query_evaluator._predicted_indices(torch.tensor([0.5, 0.5]), 3)
        # Scalar output must fail closed.
        with self.assertRaises(P6BFailClosedError):
            query_evaluator._predicted_indices(torch.tensor(0.5), 3)

    def test_28_evaluator_end_to_end_synthetic(self):
        from experiments.p6 import p6b_stage3_common as stage3_mod

        # Three queries per product exercise the real batch path
        # (batch_size=4 -> mixed chunks of B=3 and B=1).
        queries = []
        for product in LOCKED_PRODUCTS:
            for q in range(3):
                png = self.tmp / f"{product}_q{q}.png"
                write_image(png, (60 + q * 17, 70, 80))
                queries.append({
                    "product": product,
                    "rgb_sha256": hashlib.sha256(
                        f"{product}|q{q}".encode()).hexdigest(),
                    "relative_path": f"image/{product}/testing/{q:03d}.png",
                    "image_path": str(png),
                    "true_label": "clsA",
                    "class_order": ["clsA", "clsB", "clsC"],
                })
        contract = {"raw_query_count": 39, "multi_type_excluded_count": 0,
                    "formal_query_count": 39, "excluded": []}
        fake_model = FakeMvrecModel(["clsA", "clsB", "clsC"])
        wx = torch.randn(3, 1, 1, 8)
        wy = F.one_hot(torch.tensor([0, 1, 2]), 3).float()
        fake_model.head._init_weight(wx, wy, "onehot")
        grid_root = self.tmp / "grid"
        grid_root.mkdir(parents=True)
        (grid_root / execman.GRID_COMPLETION).write_text(
            json.dumps({"execution_commit": "a" * 40}), encoding="utf-8"
        )
        records_path = self.tmp / "records.jsonl"
        records_path.write_text("{}\n", encoding="utf-8")
        alpha_ckpt = self.tmp / "alpha.pth"
        alpha_ckpt.write_bytes(b"alpha-clip-weights")
        alpha_sha = file_sha256(alpha_ckpt)
        validated = {
            "backbone_identities": {"alpha_clip": {"weight_sha256": alpha_sha}},
        }
        out_dir = self.tmp / "query_out"
        with mock.patch.object(query_evaluator, "_git_head_and_dirty",
                               return_value=("a" * 40, False)), \
                mock.patch.object(query_evaluator, "validate_formal_grid",
                                  return_value=validated), \
                mock.patch.object(stage3_mod, "load_query_contract",
                                  return_value=(queries, contract)), \
                mock.patch.object(query_evaluator, "rebuild_model_for_unit",
                                  return_value=fake_model):
            result = query_evaluator.evaluate_query_grid(
                grid_root=grid_root,
                records_path=records_path,
                dataset_root=self.tmp,
                stage1_bundle=self.tmp,
                output_dir=out_dir,
                alpha_ckpt=alpha_ckpt,
                device="cpu",
                batch_size=4,
                open_query_gate=True,
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["prediction_count"], 195 * 3)
        self.assertTrue((out_dir / query_evaluator.PREDICTIONS_NAME).is_file())
        self.assertTrue(
            (out_dir / query_evaluator.PREDICTION_MANIFEST_NAME).is_file()
        )
        self.assertTrue((out_dir / query_evaluator.AUDIT_NAME).is_file())
        # Directory-level atomic publish: no staging sibling left behind.
        self.assertFalse(
            out_dir.with_name(out_dir.name + ".mvrec_rgb_query_staging").exists()
        )
        manifest = json.loads(
            (out_dir / query_evaluator.PREDICTION_MANIFEST_NAME).read_text(
                encoding="utf-8")
        )
        self.assertEqual(manifest["prediction_count"], 195 * 3)
        self.assertEqual(len(manifest["metrics"]["raw"]), 195)
        self.assertEqual(len(manifest["metrics"]["product_macro_per_seed"]), 15)
        # Provenance: evaluator commit, grid completion SHA, records SHA,
        # the locked AlphaCLIP weight SHA and the canonical-LF protocol
        # hashes must all be recorded.
        self.assertEqual(manifest["evaluator_commit"], "a" * 40)
        self.assertEqual(len(manifest["support_grid_completion_sha256"]), 64)
        self.assertEqual(len(manifest["records_jsonl_sha256"]), 64)
        self.assertEqual(manifest["alpha_clip_weight_sha256"], alpha_sha)
        self.assertEqual(manifest["protocol_bundle"]["hash_mode"], "canonical-lf")
        self.assertTrue(
            manifest["protocol_bundle"]["protocol_A_canonical_lf_sha256"]
        )
        audit = json.loads(
            (out_dir / query_evaluator.AUDIT_NAME).read_text(encoding="utf-8")
        )
        self.assertTrue(audit["query_pixels_read"])   # honest evaluator record
        self.assertTrue(audit["query_labels_read"])
        self.assertEqual(audit["evaluator_commit"], "a" * 40)
        self.assertFalse(audit["evaluator_dirty"])
        self.assertEqual(audit["alpha_clip_weight_sha256"], alpha_sha)

    def test_32_wrong_alpha_weights_rejected_before_query_contract(self):
        from experiments.p6 import p6b_stage3_common as stage3_mod

        grid_root = self.tmp / "grid"
        grid_root.mkdir(parents=True)
        (grid_root / execman.GRID_COMPLETION).write_text(
            json.dumps({"execution_commit": "a" * 40}), encoding="utf-8"
        )
        records_path = self.tmp / "records.jsonl"
        records_path.write_text("{}\n", encoding="utf-8")
        alpha_ckpt = self.tmp / "wrong_alpha.pth"
        alpha_ckpt.write_bytes(b"wrong-alpha-clip-weights")
        validated = {
            "backbone_identities": {
                "alpha_clip": {"weight_sha256": "9" * 64},
            },
        }
        out_dir = self.tmp / "query_out_wrong"
        called = {"contract": False}

        def fail_if_called(*args, **kwargs):
            called["contract"] = True
            raise AssertionError("query contract must not be opened")

        with mock.patch.object(query_evaluator, "_git_head_and_dirty",
                               return_value=("a" * 40, False)), \
                mock.patch.object(query_evaluator, "validate_formal_grid",
                                  return_value=validated), \
                mock.patch.object(stage3_mod, "load_query_contract",
                                  side_effect=fail_if_called):
            with self.assertRaises(P6BFailClosedError):
                query_evaluator.evaluate_query_grid(
                    grid_root=grid_root,
                    records_path=records_path,
                    dataset_root=self.tmp,
                    stage1_bundle=self.tmp,
                    output_dir=out_dir,
                    alpha_ckpt=alpha_ckpt,
                    device="cpu",
                    open_query_gate=True,
                )
        self.assertFalse(called["contract"])
        self.assertFalse(out_dir.exists())

    def test_33_batch_size_below_one_fails_closed(self):
        with self.assertRaises(P6BFailClosedError):
            query_evaluator.evaluate_query_grid(
                grid_root=self.tmp / "grid",
                records_path=self.tmp / "records.jsonl",
                dataset_root=self.tmp,
                stage1_bundle=self.tmp,
                output_dir=self.tmp / "out",
                alpha_ckpt=self.tmp / "alpha.pth",
                batch_size=0,
                open_query_gate=True,
            )

    def test_29_inference_path_is_official_forward_not_mean_embedding(self):
        import ast

        source = (PKG_DIR / "query_evaluator.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = [n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "_predict_batch"][0]
        attribute_calls = [
            n.func.attr for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        self.assertIn("forward", attribute_calls)  # official ClipModel.forward
        self.assertNotIn("mean", attribute_calls)  # no mean-embedding aggregation


if __name__ == "__main__":
    unittest.main(verbosity=2)
