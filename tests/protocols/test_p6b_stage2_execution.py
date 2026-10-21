import dataclasses
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
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

from experiments.p6 import p6b_execution_common as common
from experiments.p6 import p6b_stage2_execution as stage2


def write_image(path: Path, color) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path, format="PNG")
    return stage2.file_sha256(path)


def support_record(product, rel_path, sha, cls):
    return {
        "schema_version": common.RECORD_SCHEMA_VERSION,
        "product": product,
        "rgb_sha256": sha,
        "canonical_relative_path": rel_path,
        "source_pools": ["support"],
        "unique_defect_types": [cls],
        "multi_type_removed": False,
    }


def make_formal_bundle(base: Path, per_class: int = 6):
    """Build a full formal-shaped synthetic bundle (13 products x 3 K x 5
    seeds = 195 manifests) plus an image root of tiny PNGs. No real models,
    no feasibility JSONL, no query pixels."""
    bundle = base / "bundle"
    image_root = base / "dataset"
    records_by_product = {}
    for p_index, product in enumerate(common.LOCKED_PRODUCTS):
        records = []
        for c_index, cls in enumerate(("clsA", "clsB")):
            for i in range(per_class):
                rel = f"image/{product}/training/{cls}/{i:03d}.png"
                sha = write_image(image_root / rel,
                                  (p_index * 15 + i, c_index * 90, i * 37))
                records.append(support_record(product, rel, sha, cls))
        records_by_product[product] = records

    per_product = {}
    manifest_count = 0
    for product in common.LOCKED_PRODUCTS:
        records = records_by_product[product]
        candidates = common.support_candidates_by_class(records, product)
        manifests_entries = {}
        for k_shot in common.LOCKED_K:
            for seed in common.LOCKED_SEEDS:
                manifest = common.finalize_manifest(
                    common.build_support_manifest(records, product, k_shot, seed))
                relative = Path("p6b_support_manifests") / product / f"k{k_shot}_s{seed}.json"
                common.atomic_write_json(bundle / relative, manifest)
                manifests_entries[f"k{k_shot}_s{seed}"] = {
                    "relative_path": str(relative).replace("\\", "/"),
                    "manifest_sha256": manifest["manifest_sha256"],
                }
                manifest_count += 1
        per_product[product] = {
            "class_universe": sorted(candidates.keys()),
            "support_candidate_count": {
                cls: len(items) for cls, items in sorted(candidates.items())},
            "support_query_sha_overlap": 0,
            "query_sha_metadata_count": 0,
            "manifests": manifests_entries,
        }
    completion = {
        "schema_version": "p6b_support_manifest_build_v1",
        "status": "completed",
        "formal_eligible": True,
        "allow_dirty": False,
        "formal_test_used": False,
        "query_pixels_read": False,
        "query_labels_read": False,
        "products": list(common.LOCKED_PRODUCTS),
        "K_values": list(common.LOCKED_K),
        "seeds": list(common.LOCKED_SEEDS),
        "manifest_count": manifest_count,
        "feasibility_dir": "synthetic",
        "feasibility_json_sha256": "0" * 64,
        "records_jsonl_sha256": "0" * 64,
        "project_provenance": {"commit": "f" * 40, "dirty": False},
        "per_product": per_product,
    }
    common.atomic_write_json(bundle / "p6b_manifest_build_complete.json", completion)
    return bundle, image_root


class FakeClipExtractor(stage2.ImageFeatureExtractor):
    def __init__(self, dim=8, salt=b"clip"):
        self.dim = dim
        self.salt = salt

    def embed_image(self, image):
        digest = hashlib.sha256(self.salt + image.convert("RGB").tobytes()).digest()
        vec = torch.tensor(list(digest[:self.dim]), dtype=torch.float32)
        return stage2.l2_normalize(vec)


class FakeDinoExtractor(stage2.ImageFeatureExtractor):
    def __init__(self, dim=6, salt=b"dino"):
        self.dim = dim
        self.salt = salt

    def embed_image(self, image):
        digest = hashlib.sha256(self.salt + image.convert("RGB").tobytes()).digest()
        vec = torch.tensor(list(digest[:self.dim]), dtype=torch.float32)
        return stage2.l2_normalize(vec)


class TestP6BStage2Execution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.class_temp = Path(tempfile.mkdtemp())
        cls.bundle, cls.image_root = make_formal_bundle(cls.class_temp)
        # The real pin is asserted separately (test_00); the synthetic bundle
        # is pinned to its own completion sha via an explicit test seam.
        cls._original_pin = stage2.STAGE1_COMPLETION_SHA256
        cls.real_pin = cls._original_pin
        cls.synthetic_pin = stage2.file_sha256(
            cls.bundle / "p6b_manifest_build_complete.json")
        stage2.STAGE1_COMPLETION_SHA256 = cls.synthetic_pin

    @classmethod
    def tearDownClass(cls):
        stage2.STAGE1_COMPLETION_SHA256 = cls._original_pin
        shutil.rmtree(cls.class_temp, ignore_errors=True)

    # 0. locked constant values + real pin format ---------------------------------
    def test_00_locked_constants(self):
        # the genuine module constant (setUpClass patches the live attribute
        # to a synthetic seam; the real pin is captured before patching)
        self.assertEqual(self.real_pin,
                         "5da9eebb5463ffba11a92ba6f7ac4125fcf4f85d0b8fa2dcd4cbc79d2d901a73")
        self.assertEqual(stage2.ZERO_SHOT_LOGIT_SCALE, 100.0)
        self.assertEqual((stage2.TIP_ALPHA, stage2.TIP_BETA), (1.0, 5.5))
        spec = stage2.LinearProbeSpec()
        self.assertEqual(
            (spec.epochs, spec.repeat_per_epoch, spec.batch_size, spec.grad_accum),
            (20, 4, 8, 2))
        self.assertAlmostEqual(spec.lr, 1e-4)
        self.assertAlmostEqual(spec.weight_decay, 0.01)
        self.assertAlmostEqual(spec.label_smoothing, 0.1)

    # 1. formal bundle gate ------------------------------------------------------
    def test_01_formal_bundle_gate_pass(self):
        bundle = stage2.load_formal_bundle(self.bundle)
        self.assertEqual(len(bundle["manifests"]), 195)
        manifest = stage2.select_manifest(bundle, "bottle", 3, 222)
        self.assertEqual(manifest["product"], "bottle")
        self.assertEqual(manifest["K"], 3)
        self.assertEqual(manifest["seed"], 222)
        dataset = stage2.P6BSupportDataset(manifest, self.image_root)
        self.assertEqual(len(dataset), 2 * 3)  # 2 classes x K
        with self.assertRaises(common.P6BFailClosedError):
            stage2.select_manifest(bundle, "toothbrush", 1, 111)
        with self.assertRaises(common.P6BFailClosedError):
            stage2.select_manifest(bundle, "bottle", 2, 111)

    def test_02_formal_gate_fail_closed_variants(self):
        def load_variant(mutate):
            variant = self.class_temp / f"variant_{mutate.__name__}"
            shutil.copytree(self.bundle, variant)
            mutate(variant)
            try:
                with self.assertRaises(common.P6BFailClosedError):
                    stage2.load_formal_bundle(variant)
            finally:
                shutil.rmtree(variant, ignore_errors=True)

        def status_diagnostic(bundle):
            path = bundle / "p6b_manifest_build_complete.json"
            completion = json.loads(path.read_text(encoding="utf-8"))
            completion["status"] = "diagnostic"
            common.atomic_write_json(path, completion)

        def not_formal_eligible(bundle):
            path = bundle / "p6b_manifest_build_complete.json"
            completion = json.loads(path.read_text(encoding="utf-8"))
            completion["formal_eligible"] = False
            common.atomic_write_json(path, completion)

        def manifest_tampered_bytes(bundle):
            path = bundle / "p6b_support_manifests" / "bottle" / "k1_s111.json"
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

        def manifest_missing(bundle):
            (bundle / "p6b_support_manifests" / "bottle" / "k1_s111.json").unlink()

        def manifest_self_hash_broken(bundle):
            # Hit the SELF-HASH branch exactly: the manifest's recorded
            # manifest_sha256 field stays equal to the completion entry, but
            # the manifest CONTENT changed, so the recomputed self-hash must
            # disagree with the recorded field.
            path = bundle / "p6b_support_manifests" / "bottle" / "k1_s111.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["classes"]["clsA"]["candidate_count"] = 999
            common.atomic_write_json(path, manifest)

        def completion_provenance_dirty(bundle):
            path = bundle / "p6b_manifest_build_complete.json"
            completion = json.loads(path.read_text(encoding="utf-8"))
            completion["project_provenance"]["dirty"] = True
            common.atomic_write_json(path, completion)

        def completion_commit_shortened(bundle):
            path = bundle / "p6b_manifest_build_complete.json"
            completion = json.loads(path.read_text(encoding="utf-8"))
            completion["project_provenance"]["commit"] = "abc123"
            common.atomic_write_json(path, completion)

        def completion_k_values_tampered(bundle):
            path = bundle / "p6b_manifest_build_complete.json"
            completion = json.loads(path.read_text(encoding="utf-8"))
            completion["K_values"] = [1, 3]
            common.atomic_write_json(path, completion)

        def manifest_path_escape(bundle):
            path = bundle / "p6b_manifest_build_complete.json"
            completion = json.loads(path.read_text(encoding="utf-8"))
            entry = completion["per_product"]["bottle"]["manifests"]["k1_s111"]
            entry["relative_path"] = "../escape.json"
            common.atomic_write_json(path, completion)

        load_variant(status_diagnostic)
        load_variant(not_formal_eligible)
        load_variant(manifest_tampered_bytes)
        load_variant(manifest_missing)
        load_variant(manifest_self_hash_broken)
        load_variant(completion_provenance_dirty)
        load_variant(completion_commit_shortened)
        load_variant(completion_k_values_tampered)
        load_variant(manifest_path_escape)

    def test_02b_pinned_completion_sha_enforced(self):
        # wrong pin -> fail-closed before any content is trusted
        stage2.STAGE1_COMPLETION_SHA256 = "0" * 64
        try:
            with self.assertRaises(common.P6BFailClosedError):
                stage2.load_formal_bundle(self.bundle)
        finally:
            stage2.STAGE1_COMPLETION_SHA256 = self.synthetic_pin
        # completion bytes tamper changes the sha -> pinned check catches it
        variant = self.class_temp / "variant_sha_pin"
        shutil.copytree(self.bundle, variant)
        try:
            path = variant / "p6b_manifest_build_complete.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(common.P6BFailClosedError):
                stage2.load_formal_bundle(variant)
        finally:
            shutil.rmtree(variant, ignore_errors=True)

    # 2. fixed class order --------------------------------------------------------
    def test_03_fixed_class_order(self):
        bundle = stage2.load_formal_bundle(self.bundle)
        manifest = stage2.select_manifest(bundle, "capsule", 5, 111)
        self.assertEqual(manifest["class_order"], ["clsA", "clsB"])
        dataset = stage2.P6BSupportDataset(manifest, self.image_root)
        self.assertEqual(dataset.class_order, ["clsA", "clsB"])
        labels = [dataset.items[i][0] for i in range(len(dataset))]
        self.assertEqual(labels[:5], ["clsA"] * 5)
        self.assertEqual(labels[5:], ["clsB"] * 5)
        self.assertEqual(dataset[0]["label"], 0)
        self.assertEqual(dataset[5]["label"], 1)

    # 3. dataset path / existence / SHA fail-closed (local bundle copy to avoid
    #    polluting the class-level shared bundle) -------------------------------
    def test_04_dataset_fail_closed(self):
        local = self.class_temp / "bundle_t04"
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(self.bundle, local)
        try:
            bundle = stage2.load_formal_bundle(local)
            manifest = stage2.select_manifest(bundle, "carpet", 1, 111)
            dataset = stage2.P6BSupportDataset(manifest, self.image_root)
            self.assertEqual(dataset[0]["image"].size, (8, 8))
            # (a) content tamper -> RGB sha mismatch
            rel = dataset.items[0][1]
            target = self.image_root / rel
            original = target.read_bytes()
            Image.new("RGB", (8, 8), (255, 255, 255)).save(target, format="PNG")
            try:
                with self.assertRaises(common.P6BFailClosedError):
                    dataset[0]
            finally:
                target.write_bytes(original)
            # (b) missing file
            missing_manifest = json.loads(json.dumps(manifest))
            missing_manifest["classes"]["clsA"]["items"][0]["rel_path"] = (
                "image/carpet/training/clsA/missing.png")
            dataset_missing = stage2.P6BSupportDataset(missing_manifest, self.image_root)
            with self.assertRaises(common.P6BFailClosedError):
                dataset_missing[0]
            # (c) path escape containment
            escape_manifest = json.loads(json.dumps(manifest))
            escape_manifest["classes"]["clsA"]["items"][0]["rel_path"] = "../outside.png"
            dataset_escape = stage2.P6BSupportDataset(escape_manifest, self.image_root)
            with self.assertRaises(common.P6BFailClosedError):
                dataset_escape[0]
        finally:
            shutil.rmtree(local, ignore_errors=True)

    # 4. prototype math (Addendum A §1.2) ---------------------------------------------
    def test_05_prototype_math(self):
        embs = [torch.tensor([3.0, 4.0]), torch.tensor([6.0, 8.0]),
                torch.tensor([0.0, 2.0])]
        labels = [0, 0, 1]
        protos = stage2.build_prototypes(embs, labels, ["c0", "c1"])
        n1 = torch.nn.functional.normalize(torch.tensor([3.0, 4.0]), dim=0)
        n2 = torch.nn.functional.normalize(torch.tensor([6.0, 8.0]), dim=0)
        expected0 = torch.nn.functional.normalize((n1 + n2) / 2, dim=0)
        expected1 = torch.nn.functional.normalize(torch.tensor([0.0, 2.0]), dim=0)
        self.assertTrue(torch.allclose(protos[0], expected0, atol=1e-6))
        self.assertTrue(torch.allclose(protos[1], expected1, atol=1e-6))
        norms = protos.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-6))
        with self.assertRaises(common.P6BFailClosedError):
            stage2.build_prototypes(embs, labels, ["c0", "c1", "c_missing"])

    # 5. frozen fusion math (Addendum A §1.1) -------------------------------------------
    def test_06_fusion_math(self):
        clip = torch.tensor([[3.0, 4.0]])
        dino = torch.tensor([[0.0, 5.0, 12.0]])
        fused = stage2.fuse_frozen_embeddings(clip, dino)
        expected = torch.cat([
            torch.nn.functional.normalize(clip, dim=1),
            torch.nn.functional.normalize(dino, dim=1)], dim=1)
        expected = torch.nn.functional.normalize(expected, dim=1)
        self.assertEqual(fused.shape, (1, 5))
        self.assertTrue(torch.allclose(fused, expected, atol=1e-6))

    # 6. cosine argmax + tie-break by class_order -----------------------------------------
    def test_07_classify_tie_break_class_order(self):
        protos = torch.eye(3)
        equal = torch.tensor([[1.0, 1.0, 1.0]])
        self.assertEqual(stage2.classify_by_prototypes(equal, protos), [0])
        query = torch.tensor([[0.0, 2.0, 0.0]])
        self.assertEqual(stage2.classify_by_prototypes(query, protos), [1])

    # 7. deterministic head construction + locked training spec ----------------------
    def test_08_linear_head_and_locked_spec(self):
        # deterministic head: Linear(dim, N, bias=True), the ONLY trainable
        # module; identical under the same seed
        head_a = stage2.make_linear_head(4, 2, seed=111)
        head_b = stage2.make_linear_head(4, 2, seed=111)
        self.assertTrue(torch.equal(head_a.weight, head_b.weight))
        self.assertTrue(torch.equal(head_a.bias, head_b.bias))
        self.assertEqual(head_a.weight.shape, (2, 4))
        self.assertIsNotNone(head_a.bias)
        trainable = sorted(name for name, p in head_a.named_parameters()
                           if p.requires_grad)
        self.assertEqual(trainable, ["bias", "weight"])
        with self.assertRaises(common.P6BFailClosedError):
            stage2.make_linear_head(0, 2, seed=111)
        # the augmentation-aware training loop is deliberately absent in 2A
        self.assertFalse(hasattr(stage2, "train_linear_probe"))
        # locked spec validates; ANY single-field override must fail-closed
        spec = stage2.LinearProbeSpec()
        self.assertIs(stage2.validate_linear_probe_spec(spec), spec)
        overrides = {
            "epochs": 19, "repeat_per_epoch": 3, "batch_size": 16,
            "grad_accum": 1, "lr": 5e-5, "weight_decay": 0.0,
            "label_smoothing": 0.0,
        }
        for field, wrong_value in overrides.items():
            with self.assertRaises(common.P6BFailClosedError, msg=field):
                stage2.validate_linear_probe_spec(
                    dataclasses.replace(stage2.LinearProbeSpec(),
                                        **{field: wrong_value}))

    # 8. zero-shot CLIP semantics -------------------------------------------------------------
    def test_09_zero_shot_clip(self):
        self.assertEqual(stage2.format_zero_shot_prompt("broken_small"),
                         "a photo of a broken small.")
        queries = torch.tensor([[1.0, 0.0]])
        texts = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        logits = stage2.zero_shot_clip_logits(queries, texts)
        self.assertTrue(torch.allclose(logits, torch.tensor([[100.0, 0.0]]), atol=1e-5))
        self.assertEqual(stage2.ZERO_SHOT_LOGIT_SCALE, 100.0)
        # locked constant: any non-exact scale override fails
        for wrong in (99.0, 100.5, 1.0):
            with self.assertRaises(common.P6BFailClosedError):
                stage2.zero_shot_clip_logits(queries, texts, logit_scale=wrong)
        with self.assertRaises(common.P6BFailClosedError):
            stage2.zero_shot_clip_logits(queries, torch.tensor([[1.0, 0.0, 0.0]]))

    # 9. Tip-Adapter locked formula --------------------------------------------------------------
    def test_10_tip_adapter_formula(self):
        keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        cache = stage2.TipAdapterCache(keys, [0, 1], num_classes=2)
        query = torch.tensor([[1.0, 0.0]])
        affinity = cache.affinity(query)
        expected_affinity = np.exp(-5.5 * np.array([[1.0 - 1.0, 1.0 - 0.0]]))
        self.assertTrue(np.allclose(affinity.numpy(), expected_affinity, atol=1e-6))
        clip_logits = torch.tensor([[10.0, 5.0]])
        logits = cache.logits(clip_logits, query)
        expected = np.array([[10.0, 5.0]]) + 1.0 * (expected_affinity @ np.eye(2))
        self.assertTrue(np.allclose(logits.numpy(), expected, atol=1e-6))
        self.assertEqual((stage2.TIP_ALPHA, stage2.TIP_BETA), (1.0, 5.5))
        self.assertEqual(cache.values.tolist(), [[1.0, 0.0], [0.0, 1.0]])
        # locked constants: any non-exact alpha/beta override fails
        with self.assertRaises(common.P6BFailClosedError):
            stage2.TipAdapterCache(keys, [0, 1], num_classes=2, beta=5.4)
        with self.assertRaises(common.P6BFailClosedError):
            stage2.TipAdapterCache(keys, [0, 1], num_classes=2, alpha=0.9)

    # 10. query firewall at the API level ----------------------------------------------------------
    def test_11_query_firewall_api(self):
        import inspect
        for func in (stage2.P6BSupportDataset.__init__,
                     stage2.make_linear_head, stage2.build_prototypes,
                     stage2.TipAdapterCache.__init__,
                     stage2.validate_linear_probe_spec):
            params = set(inspect.signature(func).parameters)
            self.assertFalse(params & {"query", "query_path", "query_meta",
                                       "query_metadata", "feasibility_jsonl",
                                       "records_path"},
                             f"{func} exposes query/feasibility inputs: {params}")
        # query pixels present in the image root are never surfaced: the
        # manifest-driven dataset yields exactly the support items only
        query_dir = self.image_root / "image" / "carpet" / "testing" / "clsA"
        query_dir.mkdir(parents=True, exist_ok=True)
        write_image(query_dir / "q000.png", (1, 2, 3))
        bundle = stage2.load_formal_bundle(self.bundle)
        manifest = stage2.select_manifest(bundle, "carpet", 1, 111)
        dataset = stage2.P6BSupportDataset(manifest, self.image_root)
        self.assertEqual(len(dataset), 2)
        for idx in range(len(dataset)):
            self.assertIn("/training/", dataset.items[idx][1])
            self.assertNotIn("/testing/", dataset.items[idx][1])

    # 11. determinism of the full training-free pipeline ----------------------------------------------
    def test_12_pipeline_determinism(self):
        bundle = stage2.load_formal_bundle(self.bundle)
        clip_ext, dino_ext = FakeClipExtractor(), FakeDinoExtractor()

        def run_once():
            manifest = stage2.select_manifest(bundle, "grid", 5, 333)
            dataset = stage2.P6BSupportDataset(manifest, self.image_root)
            clip_embs, dino_embs, labels = [], [], []
            for idx in range(len(dataset)):
                item = dataset[idx]
                clip_embs.append(clip_ext.embed_image(item["image"]))
                dino_embs.append(dino_ext.embed_image(item["image"]))
                labels.append(item["label"])
            protos = stage2.build_prototypes(
                list(stage2.fuse_frozen_embeddings(torch.stack(clip_embs),
                                                   torch.stack(dino_embs))),
                labels, dataset.class_order)
            fusion = stage2.fuse_frozen_embeddings(torch.stack(clip_embs),
                                                   torch.stack(dino_embs))
            preds = stage2.classify_by_prototypes(fusion, protos)
            self.assertFalse(any(e.requires_grad for e in clip_embs + dino_embs))
            return preds, fusion, protos

        preds_a, fusion_a, protos_a = run_once()
        preds_b, fusion_b, protos_b = run_once()
        self.assertEqual(preds_a, preds_b)
        self.assertTrue(torch.equal(fusion_a, fusion_b))
        self.assertTrue(torch.equal(protos_a, protos_b))

    # 12. input validation fail-closed (shape/length/labels/finite/non-zero) -------
    def test_13_input_validation_fail_closed(self):
        embs = [torch.tensor([3.0, 4.0]), torch.tensor([6.0, 8.0])]
        labels = [0, 1]
        universe = ["c0", "c1"]
        # length mismatch -> no silent zip truncation
        with self.assertRaises(common.P6BFailClosedError):
            stage2.build_prototypes(embs, [0], universe)
        # label out of range
        with self.assertRaises(common.P6BFailClosedError):
            stage2.build_prototypes(embs, [0, 7], universe)
        with self.assertRaises(common.P6BFailClosedError):
            stage2.build_prototypes(embs, [0, -1], universe)
        # non-finite / zero-norm rows
        with self.assertRaises(common.P6BFailClosedError):
            stage2.build_prototypes(
                [torch.tensor([float("nan"), 1.0]), embs[1]], labels, universe)
        with self.assertRaises(common.P6BFailClosedError):
            stage2.build_prototypes(
                [torch.tensor([0.0, 0.0]), embs[1]], labels, universe)
        # wrong dimensionality
        with self.assertRaises(common.P6BFailClosedError):
            stage2.build_prototypes(torch.tensor([1.0, 2.0]), labels, universe)
        # classification shape mismatches
        protos = stage2.build_prototypes(embs, labels, universe)
        with self.assertRaises(common.P6BFailClosedError):
            stage2.classify_by_prototypes(torch.tensor([[1.0, 0.0, 0.0]]), protos)
        with self.assertRaises(common.P6BFailClosedError):
            stage2.classify_by_prototypes(torch.tensor([[float("inf"), 0.0]]), protos)
        # fusion row-count mismatch -> no silent truncation
        with self.assertRaises(common.P6BFailClosedError):
            stage2.fuse_frozen_embeddings(torch.tensor([[1.0, 0.0]]),
                                          torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        # Tip cache: label count vs keys, label range
        with self.assertRaises(common.P6BFailClosedError):
            stage2.TipAdapterCache(torch.tensor([[1.0, 0.0]]), [0, 1], num_classes=2)
        with self.assertRaises(common.P6BFailClosedError):
            stage2.TipAdapterCache(torch.tensor([[1.0, 0.0]]), [2], num_classes=2)
        cache = stage2.TipAdapterCache(torch.eye(2), [0, 1], num_classes=2)
        with self.assertRaises(common.P6BFailClosedError):
            cache.affinity(torch.tensor([[1.0, 0.0, 0.0]]))
        with self.assertRaises(common.P6BFailClosedError):
            cache.logits(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0]]))
        with self.assertRaises(common.P6BFailClosedError):
            cache.logits(torch.tensor([[float("nan"), 0.0]]),
                         torch.tensor([[1.0, 0.0]]))
        # broadcasting forbidden: 1 clip-logit row vs 2 query rows
        with self.assertRaises(common.P6BFailClosedError):
            cache.logits(torch.tensor([[1.0, 0.0]]),
                         torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    # 13. standalone import in a fresh subprocess (cole/OpenMP safe order) ----
    def test_14_standalone_import(self):
        # Fresh interpreter: the test file's own numpy/torch imports do NOT
        # precede it. The module must import cleanly on its own, relying on
        # its numpy-before-torch order; KMP_DUPLICATE_LIB_OK must NOT be set.
        env = os.environ.copy()
        env.pop("KMP_DUPLICATE_LIB_OK", None)
        code = (
            "import sys; "
            f"sys.path.insert(0, r'{PROJECT_ROOT}'); "
            "import experiments.p6.p6b_stage2_execution as m; "
            "import numpy, torch; "
            "assert sys.modules['numpy'] is numpy; "
            "assert len(m.STAGE1_COMPLETION_SHA256) == 64; "
            "print('import-ok')"
        )
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True,
                                cwd=str(PROJECT_ROOT), env=env)
        self.assertEqual(result.returncode, 0,
                         f"stdout={result.stdout!r} stderr={result.stderr!r}")
        self.assertIn("import-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
