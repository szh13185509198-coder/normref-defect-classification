"""P6.B Stage 2B support-side runner tests.

Synthetic formal-shaped bundle + tiny PNGs + fake model factory only; real
CLIP/DINO weights are never loaded. The formal git gate is exercised inside
a throwaway temporary git repository (the main repo is never touched).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6 import p6b_execution_common as common
from experiments.p6 import p6b_stage2_execution as stage2
from experiments.p6 import p6b_stage2_models as models
from experiments.p6 import run_p6b_stage2_support as runner


# ---------------------------------------------------------------------------
# Synthetic formal bundle + images (same shape as Stage 1, 13 products)
# ---------------------------------------------------------------------------

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
    bundle = base / "bundle"
    image_root = base / "dataset"
    records_by_product = {}
    for p_index, product in enumerate(common.LOCKED_PRODUCTS):
        records = []
        for c_index, cls in enumerate(("clsA", "clsB")):
            for i in range(per_class):
                rel = f"image/{product}/training/{cls}/{i:03d}.png"
                sha = write_image(image_root / rel,
                                  (p_index * 15 + i * 7, c_index * 90, i * 37))
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
                relative = Path("p6b_support_manifests") / product / \
                    f"k{k_shot}_s{seed}.json"
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
    common.atomic_write_json(bundle / "p6b_manifest_build_complete.json",
                             completion)
    return bundle, image_root


def make_temp_project_repo(base: Path) -> Path:
    """Throwaway clean git repo containing the two locked documents."""
    repo = base / "project_repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "p6b@test.local"],
                   cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "P6B Test"],
                   cwd=repo, check=True)
    docs = repo / "docs"
    docs.mkdir()
    for _key, name in runner.LOCKED_DOCUMENTS:
        shutil.copy(PROJECT_ROOT / name, docs / Path(name).name)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


# ---------------------------------------------------------------------------
# Fake factory (implements the Stage 2A interfaces; no real weights)
# ---------------------------------------------------------------------------

def _sha_vector(salt: bytes, payload: bytes, dim: int) -> torch.Tensor:
    digest = b""
    counter = 0
    while len(digest) < dim:
        digest += hashlib.sha256(salt + payload + counter.to_bytes(4, "big")).digest()
        counter += 1
    vec = torch.tensor(list(digest[:dim]), dtype=torch.float32)
    return stage2.l2_normalize(vec)


def _image_payload(image) -> bytes:
    if isinstance(image, torch.Tensor):
        return bytes(image.detach().cpu().numpy().tobytes())
    return image.convert("RGB").tobytes()


class FakeClipExtractor(stage2.ImageFeatureExtractor):
    dim = models.EXPECTED_CLIP_DIM

    def embed_image(self, image):
        return _sha_vector(b"clip", _image_payload(image), self.dim)

    def embed_images(self, images):
        batch = images if isinstance(images, torch.Tensor) else \
            torch.stack([_as_tensor(i) for i in images])
        return torch.stack([self.embed_image(batch[idx])
                            for idx in range(batch.shape[0])])


class FakeDinoExtractor(stage2.ImageFeatureExtractor):
    dim = models.EXPECTED_DINO_DIM

    def embed_image(self, image):
        return _sha_vector(b"dino", _image_payload(image), self.dim)

    def embed_images(self, images):
        batch = images if isinstance(images, torch.Tensor) else \
            torch.stack([_as_tensor(i) for i in images])
        return torch.stack([self.embed_image(batch[idx])
                            for idx in range(batch.shape[0])])


class FakeTextExtractor(stage2.TextFeatureExtractor):
    def embed_text(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        return torch.stack([
            _sha_vector(b"text", prompt.encode("utf-8"),
                        models.EXPECTED_CLIP_DIM)
            for prompt in prompts])


def _as_tensor(image):
    return image if isinstance(image, torch.Tensor) else \
        models.deterministic_eval_tensor(image)[0]


class FakeFusion(models.FrozenFusion):
    pass


class ConstantFusion(models.FrozenFusion):
    """Constant output used to prove re-extraction is structural, not numeric."""

    def fuse(self, clip_embeddings, dino_embeddings):
        rows = clip_embeddings.shape[0]
        fixed = _sha_vector(b"fixed", b"embedding",
                            models.EXPECTED_FUSION_DIM)
        return fixed.unsqueeze(0).repeat(rows, 1)


class FakeFactory:
    def __init__(self, fusion=None):
        self.clip_extractor = FakeClipExtractor()
        self.dino_extractor_interface = FakeDinoExtractor()
        self.text_extractor = FakeTextExtractor()
        self.fusion = fusion or FakeFusion()
        self.provenance = {"note": "synthetic-fake-factory"}


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestP6BStage2SupportRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.class_temp = Path(tempfile.mkdtemp())
        cls.bundle, cls.image_root = make_formal_bundle(cls.class_temp)
        cls.repo = make_temp_project_repo(cls.class_temp)
        # Real pin sanity (captured before patching) + synthetic seam.
        cls.real_pin = stage2.STAGE1_COMPLETION_SHA256
        cls._original_pin = stage2.STAGE1_COMPLETION_SHA256
        cls.synthetic_pin = stage2.file_sha256(
            cls.bundle / "p6b_manifest_build_complete.json")
        stage2.STAGE1_COMPLETION_SHA256 = cls.synthetic_pin
        cls.factory = FakeFactory()

    @classmethod
    def tearDownClass(cls):
        stage2.STAGE1_COMPLETION_SHA256 = cls._original_pin
        shutil.rmtree(cls.class_temp, ignore_errors=True)

    def run_method(self, method, suffix, factory=None, diagnostic=False, **kwargs):
        output = self.class_temp / f"out_{method}_{suffix}"
        completion = runner.execute_run(
            project_root=self.repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1, seed=111,
            method=method, output_dir=output, device="cpu",
            factory_bundle=factory or self.factory if diagnostic else None,
            allow_dirty_diagnostic=diagnostic, **kwargs)
        return output, completion

    def make_fake_ours_overrides(self, step_count, prototype_mem=None):
        captured = {}

        class FakeAdaptedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fusion_dim = models.EXPECTED_FUSION_DIM
                self.image_adapter = nn.ModuleDict(
                    {"det_proj": nn.Linear(4, 4)})
                self.prototype_text = nn.Linear(4, 4)

        from model.faiss_memory import FaissPrototypeMemory
        if prototype_mem is None:
            prototype_mem = FaissPrototypeMemory(
                num_classes=2, feat_dim=models.EXPECTED_FUSION_DIM,
                orth_margin=0.3, init_logit_scale=10.0,
                prototype_init_mode="zero", prototypes_per_class=1)

        def fake_init(model, loader, memory, device, class_to_id, **kwargs):
            return memory

        def fake_train(**kwargs):
            captured["train"] = kwargs
            for _ in range(step_count):
                kwargs["optimizer"].step()
            torch.save({"marker": "fake-final"},
                       Path(kwargs["save_path"]) / "multiclass_epoch_19.pth")

        return {
            "clip_model": object(),
            "model": FakeAdaptedModel(),
            "prototype_mem": prototype_mem,
            "init_prototypes": fake_init,
            "train_multiclass": fake_train,
        }, captured

    # 1. real pin + CLI contract ------------------------------------------------
    def test_01_cli_contract_and_real_pin(self):
        self.assertEqual(self.real_pin,
                         "5da9eebb5463ffba11a92ba6f7ac4125fcf4f85d0b8fa2dcd4cbc79d2d901a73")
        parser = runner.build_arg_parser()
        options = set(parser._option_string_actions.keys())
        required = {"--bundle-dir", "--image-root", "--product", "--k",
                    "--seed", "--method", "--output-dir", "--device",
                    "--allow-dirty-diagnostic"}
        self.assertTrue(required <= options)
        forbidden = {"--epochs", "--num_epochs", "--lr", "--proto_lr",
                     "--beta", "--alpha", "--prompt", "--batch_size",
                     "--batch-size", "--repeat_per_epoch", "--repeat-per-epoch",
                     "--grad_accum", "--grad-accum", "--weight_decay",
                     "--weight-decay", "--label_smoothing", "--label-smoothing",
                     "--augmentation", "--gradient_accumulation_steps"}
        self.assertEqual(options & forbidden, set())
        self.assertEqual(set(runner.METHODS), {
            "clip_proto", "dino_proto", "fusion_proto", "linear_probe",
            "ours", "zero_shot_clip", "tip_adapter",
            "p30_frozen_proto", "tip_adapter_f"})
        self.assertEqual(len(runner.METHODS), 9)
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="not_a_method",
                output_dir=self.class_temp / "x1", device="cpu",
                factory_bundle=self.factory)
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="toothbrush", k_shot=1,
                seed=111, method="clip_proto",
                output_dir=self.class_temp / "x2", device="cpu",
                factory_bundle=self.factory)
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=2,
                seed=111, method="clip_proto",
                output_dir=self.class_temp / "x3", device="cpu",
                factory_bundle=self.factory)

    # 2. end-to-end formal runs for the five support-side baselines -------------
    def test_02_end_to_end_support_baselines(self):
        expected_class_order = ["clsA", "clsB"]
        for method in ("clip_proto", "dino_proto", "fusion_proto",
                       "tip_adapter", "zero_shot_clip"):
            # Use diagnostic mode with synthetic fake factory
            output, completion = self.run_method(method, "e2e", diagnostic=True)
            self.assertEqual(completion["status"], "diagnostic")
            self.assertFalse(completion["formal_eligible"])
            self.assertFalse(completion["formal_test_used"])
            self.assertFalse(completion["query_pixels_read"])
            self.assertFalse(completion["query_labels_read"])
            self.assertEqual(completion["class_order"], expected_class_order)
            self.assertEqual(completion["method"], method)
            self.assertEqual(completion["product"], "bottle")
            self.assertEqual(completion["K"], 1)
            self.assertEqual(completion["seed"], 111)
            self.assertEqual(completion["execution"]["dirty"], False)
            self.assertEqual(len(completion["execution"]["commit"]), 40)
            self.assertEqual(completion["stage1"]["completion_sha256"],
                             self.synthetic_pin)
            self.assertEqual(
                completion["stage1"]["selected_manifest_relative_path"],
                "p6b_support_manifests/bottle/k1_s111.json")
            self.assertEqual(
                completion["protocol"]["sha256"],
                runner.EXPECTED_PROTOCOL_SHA256_CANONICAL_LF)
            self.assertEqual(
                completion["addendum_a"]["sha256"],
                runner.EXPECTED_ADDENDUM_A_SHA256_CANONICAL_LF)
            self.assertEqual(completion["protocol"]["hash_mode"],
                             "canonical-lf")
            self.assertEqual(completion["addendum_a"]["hash_mode"],
                             "canonical-lf")
            self.assertEqual(
                completion["locked_constants"]["zero_shot_logit_scale"], 100.0)
            self.assertEqual(completion["locked_constants"]["tip_alpha"], 1.0)
            self.assertEqual(completion["locked_constants"]["tip_beta"], 5.5)
            # no performance or query-derived field may exist (key-level)
            def all_keys(obj):
                keys = set()
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        keys.add(str(key).lower())
                        keys |= all_keys(value)
                elif isinstance(obj, list):
                    for value in obj:
                        keys |= all_keys(value)
                return keys
            forbidden = {"accuracy", "balanced_accuracy", "macro_f1", "f1",
                         "query_predictions", "ba", "top1"}
            self.assertEqual(all_keys(completion) & forbidden, set())
            # artifact hashes self-consistent and unique
            names = [record["relative_path"] for record in
                     completion["artifacts"]]
            self.assertEqual(len(names), len(set(names)),
                             f"{method}: duplicate artifact registration")
            for record in completion["artifacts"]:
                path = output / record["relative_path"]
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, record["size_bytes"])
                self.assertEqual(stage2.file_sha256(path), record["sha256"])
            expected_pixels = method != "zero_shot_clip"
            self.assertEqual(completion["support_pixels_read"],
                             expected_pixels)

    # 3. dirty tree: formal rejected, diagnostic downgraded ---------------------
    def test_03_dirty_formal_rejected_diagnostic_downgrade(self):
        dirty_repo = make_temp_project_repo(self.class_temp / "dirty_case")
        (dirty_repo / "untracked_noise.txt").write_text("noise\n",
                                                        encoding="utf-8")
        output = self.class_temp / "out_dirty_formal"
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=dirty_repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="clip_proto", output_dir=output,
                device="cpu", factory_bundle=self.factory)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name + ".p6b_staging").exists())

        diag_output = self.class_temp / "out_dirty_diagnostic"
        completion = runner.execute_run(
            project_root=dirty_repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1, seed=111,
            method="clip_proto", output_dir=diag_output, device="cpu",
            factory_bundle=self.factory, allow_dirty_diagnostic=True)
        self.assertEqual(completion["status"], "diagnostic")
        self.assertFalse(completion["formal_eligible"])
        self.assertTrue(completion["execution"]["dirty"])

    # 4. locked document SHA gate ------------------------------------------------
    def test_04_document_sha_fail_closed(self):
        _, completion = self.run_method(
            "zero_shot_clip", "doc_ok", diagnostic=True)
        self.assertEqual(completion["status"], "diagnostic")

    # 5. Stage 1 gate tamper variants fail-closed through the runner ------------
    def test_05_stage1_gate_fail_closed(self):
        def tampered_bundle(mutate):
            variant_dir = self.class_temp / f"variant_{mutate.__name__}"
            shutil.copytree(self.bundle, variant_dir)
            mutate(variant_dir)
            return variant_dir

        def status_diagnostic(bundle):
            path = bundle / "p6b_manifest_build_complete.json"
            completion = json.loads(path.read_text(encoding="utf-8"))
            completion["status"] = "diagnostic"
            common.atomic_write_json(path, completion)

        def manifest_tampered(bundle):
            path = bundle / "p6b_support_manifests" / "bottle" / "k1_s111.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["K"] = 99
            common.atomic_write_json(path, manifest)

        for mutate in (status_diagnostic, manifest_tampered):
            variant = tampered_bundle(mutate)
            output = self.class_temp / f"out_{mutate.__name__}"
            with self.assertRaises(common.P6BFailClosedError):
                runner.execute_run(
                    project_root=self.repo, bundle_dir=variant,
                    image_root=self.image_root, product="bottle", k_shot=1,
                    seed=111, method="clip_proto", output_dir=output,
                    device="cpu", factory_bundle=None)  # Use None to pass formal gate
            self.assertFalse(output.exists())

    # 6. support asset gate: containment / sha / missing file --------------------
    def _rebuilt_variant_bundle(self, mutate_item, name):
        """Tamper one manifest item, recompute manifest + completion hashes,
        and re-pin the synthetic seam so only the runner-side asset gate is
        exercised."""
        variant_dir = self.class_temp / f"variant_{name}"
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        shutil.copytree(self.bundle, variant_dir)
        manifest_path = variant_dir / "p6b_support_manifests" / "bottle" / \
            "k1_s111.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_class = manifest["class_order"][0]
        mutate_item(manifest["classes"][first_class]["items"][0])
        manifest.pop("manifest_sha256", None)
        manifest = common.finalize_manifest(manifest)
        common.atomic_write_json(manifest_path, manifest)
        completion_path = variant_dir / "p6b_manifest_build_complete.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["per_product"]["bottle"]["manifests"]["k1_s111"][
            "manifest_sha256"] = manifest["manifest_sha256"]
        common.atomic_write_json(completion_path, completion)
        original_pin = stage2.STAGE1_COMPLETION_SHA256
        stage2.STAGE1_COMPLETION_SHA256 = stage2.file_sha256(completion_path)
        self.addCleanup(setattr, stage2, "STAGE1_COMPLETION_SHA256",
                        original_pin)
        return variant_dir

    def test_06_support_asset_gate(self):
        def escape_root(item):
            item["rel_path"] = "../evil.png"

        def wrong_sha(item):
            item["rgb_sha256"] = "0" * 64

        for mutate, name in ((escape_root, "escape_root"),
                             (wrong_sha, "wrong_sha")):
            variant = self._rebuilt_variant_bundle(mutate, name)
            output = self.class_temp / f"out_asset_{name}"
            with self.assertRaises(common.P6BFailClosedError):
                runner.execute_run(
                    project_root=self.repo, bundle_dir=variant,
                    image_root=self.image_root, product="bottle", k_shot=1,
                    seed=111, method="clip_proto", output_dir=output,
                    device="cpu", factory_bundle=None)
            self.assertFalse(output.exists())

        # missing file: point the manifest at a contained path that does not
        # exist on disk (no surgery on the shared image root)
        def missing_file(item):
            item["rel_path"] = "image/bottle/training/clsA/zz_missing.png"

        variant = self._rebuilt_variant_bundle(missing_file, "missing")
        output = self.class_temp / "out_asset_missing"
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=variant,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="clip_proto", output_dir=output,
                device="cpu", factory_bundle=None)
        self.assertFalse(output.exists())

    # 7. zero-shot consumes no support pixels ------------------------------------
    def test_07_zero_shot_no_support_pixels(self):
        real_open = Image.open
        with mock.patch.object(Image, "open",
                               side_effect=RuntimeError("pixels forbidden")):
            output, completion = self.run_method("zero_shot_clip", "nopixels", diagnostic=True)
        self.assertFalse(completion["support_pixels_read"])
        artifact = torch.load(output / "zero_shot_clip_text_weights.pt",
                              weights_only=False)
        self.assertEqual(artifact["logit_scale"], 100.0)
        self.assertEqual(artifact["prompt_template"], "a photo of a {}.")
        self.assertEqual(artifact["prompts"],
                         ["a photo of a clsA.", "a photo of a clsB."])
        embeddings = artifact["text_embeddings"]
        self.assertEqual(embeddings.shape, (2, models.EXPECTED_CLIP_DIM))
        norms = embeddings.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms),
                                       atol=1e-5))
        # a pixel-consuming method must fail under the same pixel ban
        with mock.patch.object(Image, "open",
                               side_effect=RuntimeError("pixels forbidden")):
            with self.assertRaises(Exception):
                self.run_method("clip_proto", "nopixels_fail", diagnostic=True)
        self.assertFalse((self.class_temp /
                          "out_clip_proto_nopixels_fail").exists())

    # 8. CLIP/DINO/Fusion contracts on the models module -------------------------
    def test_08_extractor_contracts(self):
        class FakeVisualBackbone:
            def __init__(self, dim):
                self.dim = dim

            def visual(self, batch, extra):
                payload = bytes(batch.detach().cpu().numpy().tobytes())
                rows = torch.stack([
                    _sha_vector(b"vis", payload + idx.to_bytes(2, "big"),
                                self.dim)
                    for idx in range(batch.shape[0])])
                return rows, None

        extractor = models.FrozenClipImageExtractor(
            FakeVisualBackbone(models.EXPECTED_CLIP_DIM), "cpu")
        image = Image.new("RGB", (16, 16), color=(10, 20, 30))
        emb = extractor.embed_image(image)
        self.assertEqual(tuple(emb.shape), (models.EXPECTED_CLIP_DIM,))
        self.assertAlmostEqual(float(emb.norm()), 1.0, places=5)
        self.assertFalse(emb.requires_grad)
        batch = extractor.embed_images([image, image])
        self.assertEqual(tuple(batch.shape), (2, models.EXPECTED_CLIP_DIM))
        # dimension contract fail-closed
        bad = models.FrozenClipImageExtractor(FakeVisualBackbone(512), "cpu")
        with self.assertRaises(common.P6BFailClosedError):
            bad.embed_image(image)

        class FakeDinoBackbone:
            def forward_patches(self, batch):
                rows = []
                payload = bytes(batch.detach().cpu().numpy().tobytes())
                for idx in range(batch.shape[0]):
                    tokens = torch.stack([
                        _sha_vector(b"dinop", payload + t.to_bytes(2, "big"),
                                    models.EXPECTED_DINO_DIM)
                        for t in range(4)])
                    rows.append(tokens)
                return torch.stack(rows)

        dino = models.FrozenDinoImageExtractor(FakeDinoBackbone(), "cpu")
        demb = dino.embed_image(image)
        self.assertEqual(tuple(demb.shape), (models.EXPECTED_DINO_DIM,))
        self.assertAlmostEqual(float(demb.norm()), 1.0, places=5)

        fusion = models.FrozenFusion()
        fused = fusion.fuse(emb.unsqueeze(0), demb.unsqueeze(0))
        self.assertEqual(tuple(fused.shape),
                         (1, models.EXPECTED_FUSION_DIM))
        self.assertAlmostEqual(float(fused.norm()), 1.0, places=5)
        with self.assertRaises(common.P6BFailClosedError):
            fusion.fuse(torch.ones(1, 10), torch.ones(1, 10))

    # 9. prototype / tip artifacts mathematically correct ------------------------
    def test_09_prototype_and_tip_artifacts(self):
        for method in ("clip_proto", "dino_proto", "fusion_proto"):
            output, completion = self.run_method(method, "art", diagnostic=True)
            artifact = torch.load(output / f"{method}_prototypes.pt",
                                  weights_only=False)
            protos = artifact["prototypes"]
            expected = stage2.build_prototypes(
                artifact["support_embeddings"], artifact["support_labels"],
                artifact["class_order"])
            self.assertTrue(torch.equal(protos, expected))
            self.assertEqual(protos.dtype, torch.float32)
            self.assertEqual(tuple(protos.shape),
                             (2, {"clip_proto": models.EXPECTED_CLIP_DIM,
                                  "dino_proto": models.EXPECTED_DINO_DIM,
                                  "fusion_proto":
                                      models.EXPECTED_FUSION_DIM}[method]))
            norms = protos.norm(dim=1)
            self.assertTrue(torch.allclose(norms, torch.ones(2), atol=1e-5))

        output, completion = self.run_method("tip_adapter", "art", diagnostic=True)
        artifact = torch.load(output / "tip_adapter_cache.pt",
                              weights_only=False)
        self.assertEqual(artifact["alpha"], 1.0)
        self.assertEqual(artifact["beta"], 5.5)
        keys = artifact["cache_keys"]
        values = artifact["cache_values"]
        self.assertEqual(tuple(values.shape), (keys.shape[0], 2))
        self.assertTrue(torch.equal(values.sum(dim=1),
                                    torch.ones(keys.shape[0])))
        for row, label in enumerate(artifact["cache_labels"]):
            self.assertEqual(int(values[row, label].item()), 1)
        norms = keys.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms),
                                       atol=1e-5))
        # reconstruction through the locked Stage 2A cache passes validation
        cache = stage2.TipAdapterCache(keys, [0, 1],
                                       num_classes=2)
        self.assertEqual(cache.values.tolist(), values.tolist())

    # 10. linear probe: re-extraction, step audit, head-only trainable ----------
    def test_10_linear_probe_locked_loop(self):
        output, completion = self.run_method("linear_probe", "loop", diagnostic=True)
        audit = json.loads((output / "linear_probe_step_audit.json")
                           .read_text(encoding="utf-8"))
        spec = stage2.LinearProbeSpec()
        # K=1, 2 classes -> 2 support samples -> 1 batch per pass
        self.assertEqual(len(audit["epochs"]), spec.epochs)
        for epoch in audit["epochs"]:
            self.assertEqual(epoch["batches"], spec.repeat_per_epoch)
            self.assertEqual(epoch["features_extracted"],
                             spec.repeat_per_epoch * 2)
        self.assertEqual(audit["total_forward_batches"],
                         spec.epochs * spec.repeat_per_epoch)
        self.assertEqual(audit["total_optimizer_steps"],
                         spec.epochs * spec.repeat_per_epoch *
                         runner.steps_per_pass(1, spec.grad_accum))
        self.assertTrue(audit["feature_reextraction_per_pass"])
        self.assertEqual(completion["trainable_parameter_whitelist"],
                         ["bias", "weight"])
        head = torch.load(output / "linear_probe_head.pt",
                          weights_only=False)
        self.assertEqual(set(head["head_state_dict"].keys()),
                         {"weight", "bias"})
        self.assertEqual(tuple(head["head_state_dict"]["weight"].shape),
                         (2, models.EXPECTED_FUSION_DIM))
        self.assertEqual(head["final_epoch"], spec.epochs - 1)
        self.assertEqual(head["locked_spec"]["epochs"], 20)
        self.assertEqual(head["locked_spec"]["repeat_per_epoch"], 4)

    def test_11_linear_probe_constant_output_still_reextracts(self):
        output = self.class_temp / "out_linear_constant"
        completion = runner.execute_run(
            project_root=self.repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1,
            seed=111, method="linear_probe", output_dir=output,
            device="cpu", allow_dirty_diagnostic=True,
            factory_bundle=FakeFactory(fusion=ConstantFusion()))
        self.assertFalse(completion["formal_eligible"])
        audit = json.loads((output / "linear_probe_step_audit.json")
                           .read_text(encoding="utf-8"))
        self.assertEqual(audit["per_pass_audit"]["extract_call_count"], 80)
        self.assertEqual(audit["per_pass_audit"]["expected_extract_count"], 80)
        self.assertEqual(audit["per_pass_audit"]["epoch_repeat_contexts_size"], 80)

    # 12. Ours: exact args + whitelist + production objects reused ---------------
    def test_12_ours_exact_args_and_whitelist(self):
        captured = {}

        class FakeAdaptedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fusion_dim = models.EXPECTED_FUSION_DIM
                self.image_adapter = nn.ModuleDict(
                    {"det_proj": nn.Linear(4, 4)})
                self.prototype_text = nn.Linear(4, 4)  # must stay frozen

        from model.faiss_memory import FaissPrototypeMemory

        def fake_init(model, loader, prototype_mem, device, class_to_id,
                      **kwargs):
            captured["init"] = {"class_to_id": dict(class_to_id),
                                "kwargs": kwargs}
            return prototype_mem

        def fake_train(**kwargs):
            captured["train"] = kwargs
            for _ in range(80):
                kwargs["optimizer"].step()
            save_path = Path(kwargs["save_path"])
            torch.save({"marker": "fake-final"},
                       save_path / "multiclass_epoch_19.pth")
            return kwargs["model"], kwargs["prototype_mem"]

        prototype_mem = FaissPrototypeMemory(
            num_classes=2, feat_dim=models.EXPECTED_FUSION_DIM,
            orth_margin=0.3, init_logit_scale=10.0,
            prototype_init_mode="zero", prototypes_per_class=1)
        overrides = {
            "clip_model": object(),
            "model": FakeAdaptedModel(),
            "prototype_mem": prototype_mem,
            "init_prototypes": fake_init,
            "train_multiclass": fake_train,
        }
        output = self.class_temp / "out_ours_args"
        completion = runner.execute_run(
            project_root=self.repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1, seed=222,
            method="ours", output_dir=output, device="cpu",
            allow_dirty_diagnostic=True,
            ours_overrides=overrides)  # Use diagnostic mode for injected overrides

        self.assertEqual(captured["init"]["class_to_id"],
                         {"clsA": 0, "clsB": 1})
        self.assertEqual(captured["init"]["kwargs"]["num_views"], 1)
        self.assertFalse(captured["init"]["kwargs"]["use_text_guided"])
        train = captured["train"]
        self.assertEqual(train["num_epochs"], 20)
        self.assertEqual(train["repeat_per_epoch"], 4)
        self.assertEqual(train["gradient_accumulation_steps"], 2)
        self.assertEqual(train["label_smoothing"], 0.1)
        self.assertEqual(train["supcon_weight"], 0.0)
        self.assertEqual(train["orth_weight"], 0.0)
        self.assertEqual(train["itc_weight"], 0.0)
        self.assertEqual(train["arc_margin"], 0.0)
        self.assertEqual(train["save_every"], 0)
        self.assertEqual(train["early_stop_patience"], 0)
        self.assertEqual(train["checkpoint_format"], "lightweight")
        self.assertFalse(train["use_text_guided"])
        self.assertEqual(train["start_epoch"], 0)
        self.assertEqual(train["class_to_id"],
                         {"clsA": 0, "clsB": 1})
        self.assertEqual(Path(train["save_path"]),
                         output.with_name(output.name + ".p6b_staging"))
        self.assertIsNone(train["manifest_path"])
        # locked args namespace carried through configure_*
        self.assertEqual(completion["seed"], 222)
        self.assertEqual(
            completion["trainable_parameter_whitelist"],
            ["image_adapter.det_proj.weight", "image_adapter.det_proj.bias",
             "prototype_mem.prototypes", "prototype_mem.logit_scale"])
        self.assertFalse(train["model"].prototype_text.weight.requires_grad)
        self.assertTrue(train["model"].image_adapter["det_proj"]
                        .weight.requires_grad)
        step_audit = json.loads((output / "ours_step_audit.json")
                                .read_text(encoding="utf-8"))
        # K=1, 2 samples, batch 8 -> 1 batch/pass; 20*4 passes
        self.assertEqual(step_audit["batches_per_pass"], 1)
        self.assertEqual(step_audit["passes"], 80)
        self.assertEqual(step_audit["expected_optimizer_steps"], 80)
        self.assertTrue((output / "multiclass_epoch_19.pth").is_file())
        names = [record["relative_path"] for record in
                 completion["artifacts"]]
        self.assertEqual(len(names), len(set(names)),
                         "ours: duplicate artifact registration")
        self.assertIn("multiclass_epoch_19.pth", names)

    # 13. late failure never leaves a formal directory ---------------------------
    def test_13_late_failure_leaves_nothing(self):
        def explode(**kwargs):
            Path(kwargs["save_path"], "partial.txt").write_text("partial\n",
                                                                encoding="utf-8")
            raise RuntimeError("simulated late failure")

        def fake_init(model, loader, prototype_mem, device, class_to_id,
                      **kwargs):
            return prototype_mem

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fusion_dim = models.EXPECTED_FUSION_DIM
                self.image_adapter = nn.ModuleDict(
                    {"det_proj": nn.Linear(4, 4)})

        from model.faiss_memory import FaissPrototypeMemory
        overrides = {
            "clip_model": object(),
            "model": TinyModel(),
            "prototype_mem": FaissPrototypeMemory(
                num_classes=2, feat_dim=models.EXPECTED_FUSION_DIM,
                orth_margin=0.3, init_logit_scale=10.0,
                prototype_init_mode="zero", prototypes_per_class=1),
            "init_prototypes": fake_init,
            "train_multiclass": explode,
        }
        output = self.class_temp / "out_late_failure"
        with self.assertRaises(RuntimeError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="ours", output_dir=output, device="cpu",
                allow_dirty_diagnostic=True,
                ours_overrides=overrides)  # Use diagnostic mode for injected overrides
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name(output.name +
                                          ".p6b_staging").exists())

    # 14. byte determinism (same fake inputs twice; timestamps excluded) ---------
    def test_14_byte_determinism(self):
        for method, artifact_names in (
                ("clip_proto",
                 ["p6b_support_meta.jsonl", "support_run_config.json",
                  "support_artifact_manifest.json",
                  "clip_proto_prototypes.pt"]),
                ("zero_shot_clip",
                 ["p6b_support_meta.jsonl", "support_run_config.json",
                  "zero_shot_clip_text_weights.pt"])):
            first, _ = self.run_method(method, "detA", diagnostic=True)
            second, _ = self.run_method(method, "detB", diagnostic=True)
            for name in artifact_names:
                bytes_a = (first / name).read_bytes()
                bytes_b = (second / name).read_bytes()
                self.assertEqual(bytes_a, bytes_b,
                                 f"{method}/{name} is not byte-deterministic")

    # 15. standalone import in a fresh subprocess --------------------------------
    def test_15_standalone_import(self):
        env = os.environ.copy()
        env.pop("KMP_DUPLICATE_LIB_OK", None)
        code = (
            "import sys; "
            f"sys.path.insert(0, r'{PROJECT_ROOT}'); "
            "from experiments.p6 import run_p6b_stage2_support as r; "
            "from experiments.p6 import p6b_stage2_models as m; "
            "assert len(r.METHODS) == 9; "
            "assert m.EXPECTED_FUSION_DIM == 1152; "
            "print('import-ok')"
        )
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True,
                                cwd=str(PROJECT_ROOT), env=env)
        self.assertEqual(result.returncode, 0,
                         f"stdout={result.stdout!r} stderr={result.stderr!r}")
        self.assertIn("import-ok", result.stdout)

    def test_16_ours_step_gate_exact_boundary(self):
        for steps in (0, 79, 81):
            overrides, _ = self.make_fake_ours_overrides(steps)
            output = self.class_temp / f"out_ours_steps_{steps}"
            with self.assertRaises(common.P6BFailClosedError):
                runner.execute_run(
                    project_root=self.repo, bundle_dir=self.bundle,
                    image_root=self.image_root, product="bottle", k_shot=1,
                    seed=111, method="ours", output_dir=output, device="cpu",
                    allow_dirty_diagnostic=True, ours_overrides=overrides)
            self.assertFalse(output.exists())
        overrides, _ = self.make_fake_ours_overrides(80)
        output = self.class_temp / "out_ours_steps_80"
        completion = runner.execute_run(
            project_root=self.repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1,
            seed=111, method="ours", output_dir=output, device="cpu",
            allow_dirty_diagnostic=True, ours_overrides=overrides)
        self.assertFalse(completion["formal_eligible"])
        audit = json.loads((output / "ours_step_audit.json")
                           .read_text(encoding="utf-8"))
        self.assertEqual(audit["actual_optimizer_steps"], 80)
        self.assertEqual(audit["expected_optimizer_steps"], 80)

    def test_17_formal_dependency_injection_rejected(self):
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="clip_proto",
                output_dir=self.class_temp / "out_formal_factory_injection",
                device="cpu", factory_bundle=self.factory)
        overrides, _ = self.make_fake_ours_overrides(80)
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="ours",
                output_dir=self.class_temp / "out_formal_ours_injection",
                device="cpu", ours_overrides=overrides)

    def test_18_canonical_lf_equivalence_and_bare_cr_rejection(self):
        roots = []
        for suffix, newline in (("lf", b"\n"), ("crlf", b"\r\n")):
            root = self.class_temp / f"canonical_{suffix}"
            roots.append(root)
            for _key, relative in runner.LOCKED_DOCUMENTS:
                source = (PROJECT_ROOT / relative).read_bytes()
                canonical = source.replace(b"\r\n", b"\n")
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(canonical.replace(b"\n", newline))
        self.assertEqual(runner.document_provenance_canonical_lf(roots[0]),
                         runner.document_provenance_canonical_lf(roots[1]))
        bad = roots[1] / runner.ADDENDUM_A_RELATIVE_PATH
        bad.write_bytes(bad.read_bytes() + b"bare\rreturn")
        with self.assertRaises(common.P6BFailClosedError):
            runner.document_provenance_canonical_lf(roots[1])

    def test_19_all_non_model_gates_precede_model_loading(self):
        def assert_clip_builder_not_called(**kwargs):
            with mock.patch.object(runner, "_build_lazy_clip_only") as build_clip:
                with self.assertRaises(common.P6BFailClosedError):
                    runner.execute_run(**kwargs)
                build_clip.assert_not_called()

        # Actual document content gate.
        doc_repo = make_temp_project_repo(self.class_temp / "doc_gate_case")
        protocol = doc_repo / runner.PROTOCOL_RELATIVE_PATH
        protocol.write_bytes(protocol.read_bytes() + b"\ncontent-tamper\n")
        assert_clip_builder_not_called(
            project_root=doc_repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1,
            seed=111, method="clip_proto",
            output_dir=self.class_temp / "out_doc_pre_model", device="cpu",
            allow_dirty_diagnostic=True)

        # Dirty-tree gate.
        dirty_marker = self.repo / "dirty_pre_model.tmp"
        dirty_marker.write_text("dirty\n", encoding="utf-8")
        try:
            assert_clip_builder_not_called(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="clip_proto",
                output_dir=self.class_temp / "out_dirty_pre_model",
                device="cpu")
        finally:
            dirty_marker.unlink(missing_ok=True)

        # Stage 1 pinned-bundle gate.
        stage1_variant = self.class_temp / "stage1_pre_model_variant"
        shutil.copytree(self.bundle, stage1_variant)
        completion_path = stage1_variant / "p6b_manifest_build_complete.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["status"] = "diagnostic"
        common.atomic_write_json(completion_path, completion)
        assert_clip_builder_not_called(
            project_root=self.repo, bundle_dir=stage1_variant,
            image_root=self.image_root, product="bottle", k_shot=1,
            seed=111, method="clip_proto",
            output_dir=self.class_temp / "out_stage1_pre_model", device="cpu",
            allow_dirty_diagnostic=True)

        # Support containment/existence/hash gate.
        def wrong_sha(item):
            item["rgb_sha256"] = "0" * 64

        variant = self._rebuilt_variant_bundle(
            wrong_sha, "pre_model_wrong_sha")
        output = self.class_temp / "out_pre_model_wrong_sha"
        assert_clip_builder_not_called(
            project_root=self.repo, bundle_dir=variant,
            image_root=self.image_root, product="bottle", k_shot=1,
            seed=111, method="clip_proto", output_dir=output,
            device="cpu", allow_dirty_diagnostic=True)
        self.assertFalse(output.exists())

    def test_20_method_specific_lazy_loading_matrix(self):
        cases = {
            "clip_proto": "clip", "zero_shot_clip": "clip",
            "tip_adapter": "clip", "dino_proto": "dino",
            "fusion_proto": "both", "linear_probe": "both",
        }
        for method, expected in cases.items():
            output = self.class_temp / f"out_lazy_{method}"
            with mock.patch.object(runner, "_build_lazy_clip_only",
                                   return_value=FakeFactory()) as clip_builder, \
                    mock.patch.object(runner, "_build_lazy_dino_only",
                                      return_value=FakeFactory()) as dino_builder, \
                    mock.patch.object(runner, "_build_lazy_clip_and_dino",
                                      return_value=FakeFactory()) as both_builder:
                runner.execute_run(
                    project_root=self.repo, bundle_dir=self.bundle,
                    image_root=self.image_root, product="bottle", k_shot=1,
                    seed=111, method=method, output_dir=output, device="cpu",
                    allow_dirty_diagnostic=True)
                calls = {"clip": clip_builder.call_count,
                         "dino": dino_builder.call_count,
                         "both": both_builder.call_count}
                self.assertEqual(calls[expected], 1, (method, calls))
                self.assertEqual(sum(calls.values()), 1, (method, calls))

    def test_21_prototype_parameter_contract_fail_closed(self):
        from model.faiss_memory import FaissPrototypeMemory

        class ExtraPrototypeMemory(FaissPrototypeMemory):
            def __init__(self):
                super().__init__(
                    num_classes=2, feat_dim=models.EXPECTED_FUSION_DIM,
                    orth_margin=0.3, init_logit_scale=10.0,
                    prototype_init_mode="zero", prototypes_per_class=1)
                self.unexpected = nn.Parameter(torch.ones(1))

        overrides, _ = self.make_fake_ours_overrides(
            80, prototype_mem=ExtraPrototypeMemory())
        output = self.class_temp / "out_bad_prototype_contract"
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="ours", output_dir=output, device="cpu",
                allow_dirty_diagnostic=True, ours_overrides=overrides)
            self.assertFalse(output.exists())

    def test_22_ours_import_neutralizes_legacy_hf_endpoint_side_effect(self):
        sentinel_module = object()
        previous = os.environ.pop("HF_ENDPOINT", None)
        try:
            def fake_import(name):
                self.assertEqual(name, "train_multiclass")
                self.assertIn("HF_ENDPOINT", os.environ)
                self.assertEqual(os.environ["HF_ENDPOINT"], "")
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                return sentinel_module

            loaded = runner.import_train_multiclass_offline_safe(fake_import)
            self.assertIs(loaded, sentinel_module)
            self.assertNotIn("HF_ENDPOINT", os.environ)

            blocked_import = mock.Mock()
            with mock.patch.dict(
                    os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}):
                with self.assertRaises(common.P6BFailClosedError):
                    runner.import_train_multiclass_offline_safe(blocked_import)
            blocked_import.assert_not_called()
        finally:
            os.environ.pop("HF_ENDPOINT", None)
            if previous is not None:
                os.environ["HF_ENDPOINT"] = previous


class _FakeDinoPatches:
    def forward_patches(self, x):
        B = x.shape[0]
        payload = bytes(x.detach().cpu().numpy().tobytes())
        rows = []
        for idx in range(B):
            tokens = torch.stack([
                _sha_vector(b"r1dino", payload + (idx * 4 + t).to_bytes(3, "big"),
                            models.EXPECTED_DINO_DIM)
                for t in range(4)])
            rows.append(tokens)
        return torch.stack(rows)


class FakeFrozenProtoModel(nn.Module):
    """AdaptedCLIP-shaped stub exposing exactly what R1 reuses: det_proj,
    forward_fusion (feeds pre-terminal tokens into det_proj), topk_avg_pool and
    dino_extractor.forward_patches. Deterministic in x for identity testing."""

    _T = 5

    def __init__(self):
        super().__init__()
        self.fusion_dim = models.EXPECTED_FUSION_DIM
        self.image_adapter = nn.ModuleDict(
            {"det_proj": nn.Linear(1024, 768)})
        self.dino_extractor = _FakeDinoPatches()
        self._last_pre_terminal = None

    def forward_fusion(self, x):
        B = x.shape[0]
        base = x.reshape(B, -1).mean(dim=1).view(B, 1, 1)
        grid = torch.arange(self._T * 1024, dtype=torch.float32).view(
            1, self._T, 1024)
        tokens = base + grid
        tokens = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-6)
        self._last_pre_terminal = tokens
        # Feed the pre-terminal tokens into the terminal adapter so R1's
        # forward-pre-hook captures the exact terminal-adapter input.
        _ = self.image_adapter["det_proj"](tokens)
        return torch.zeros(B, self.fusion_dim)

    def topk_avg_pool(self, tensor, k_ratio=0.1):
        B, N, D = tensor.shape
        K = max(1, int(N * k_ratio))
        vals, _ = torch.topk(tensor, K, dim=1)
        return vals.mean(dim=1)


def _frozen_proto_model(trainable: bool = False) -> FakeFrozenProtoModel:
    model = FakeFrozenProtoModel()
    for param in model.parameters():
        param.requires_grad_(bool(trainable))
    return model


class TestP6BR1R2BundleV2(unittest.TestCase):
    """Addendum B: R1 (p30_frozen_proto), R2 (tip_adapter_f), Bundle V2."""

    @classmethod
    def setUpClass(cls):
        cls.class_temp = Path(tempfile.mkdtemp())
        cls.bundle, cls.image_root = make_formal_bundle(cls.class_temp)
        cls.repo = make_temp_project_repo(cls.class_temp)
        cls._original_pin = stage2.STAGE1_COMPLETION_SHA256
        cls.synthetic_pin = stage2.file_sha256(
            cls.bundle / "p6b_manifest_build_complete.json")
        stage2.STAGE1_COMPLETION_SHA256 = cls.synthetic_pin

    @classmethod
    def tearDownClass(cls):
        stage2.STAGE1_COMPLETION_SHA256 = cls._original_pin
        shutil.rmtree(cls.class_temp, ignore_errors=True)

    # --- Protocol Bundle V2 provenance ----------------------------------------
    def test_bundle_v2_pins_three_documents(self):
        self.assertEqual(len(runner.LOCKED_DOCUMENTS), 3)
        keys = {k for k, _ in runner.LOCKED_DOCUMENTS}
        self.assertEqual(keys, {"protocol", "addendum_a", "addendum_b"})
        self.assertEqual(runner.EXPECTED_ADDENDUM_B_SHA256_CANONICAL_LF,
                         "15554bea2d61bbd92f349bbf3b6e106f5e4c1db97ef38e9663175e3a6ec54983")
        prov = runner.document_provenance_canonical_lf(self.repo)
        self.assertEqual(prov["addendum_b"]["sha256"],
                         runner.EXPECTED_ADDENDUM_B_SHA256_CANONICAL_LF)

    def test_bundle_v2_completion_records_all_shas(self):
        output = self.class_temp / "out_bundle_v2_completion"
        completion = runner.execute_run(
            project_root=self.repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1, seed=111,
            method="clip_proto", output_dir=output, device="cpu",
            allow_dirty_diagnostic=True, factory_bundle=FakeFactory())
        bundle = completion["protocol_bundle"]
        self.assertEqual(bundle["version"], "v2")
        self.assertEqual(bundle["main_protocol_sha256"],
                         runner.EXPECTED_PROTOCOL_SHA256_CANONICAL_LF)
        self.assertEqual(bundle["addendum_a_sha256"],
                         runner.EXPECTED_ADDENDUM_A_SHA256_CANONICAL_LF)
        self.assertEqual(bundle["addendum_b_sha256"],
                         runner.EXPECTED_ADDENDUM_B_SHA256_CANONICAL_LF)
        self.assertEqual(bundle["stage1_completion_sha256"], self.synthetic_pin)
        self.assertEqual(completion["addendum_b"]["hash_mode"], "canonical-lf")

    def test_addendum_b_missing_fails_closed(self):
        bad_repo = self.class_temp / "repo_no_addb"
        bad_repo.mkdir()
        (bad_repo / "docs").mkdir()
        for _key, name in runner.LOCKED_DOCUMENTS:
            if name == runner.ADDENDUM_B_RELATIVE_PATH:
                continue
            shutil.copy(PROJECT_ROOT / name,
                        bad_repo / "docs" / Path(name).name)
        with self.assertRaises(common.P6BFailClosedError):
            runner.document_provenance_canonical_lf(bad_repo)

    def test_addendum_b_mismatch_fails_before_model(self):
        tamper_repo = make_temp_project_repo(self.class_temp / "repo_addb_bad")
        addb = tamper_repo / runner.ADDENDUM_B_RELATIVE_PATH
        addb.write_bytes(addb.read_bytes() + b"\ncontent-tamper\n")
        with mock.patch.object(runner, "_build_lazy_clip_only") as build_clip:
            with self.assertRaises(common.P6BFailClosedError):
                runner.execute_run(
                    project_root=tamper_repo, bundle_dir=self.bundle,
                    image_root=self.image_root, product="bottle", k_shot=1,
                    seed=111, method="clip_proto",
                    output_dir=self.class_temp / "out_addb_bad", device="cpu",
                    allow_dirty_diagnostic=True)
            build_clip.assert_not_called()

    # --- R1 -------------------------------------------------------------------
    def test_r1_feature_identity_with_terminal_input(self):
        model = _frozen_proto_model(trainable=False)
        x = torch.randn(1, 3, 336, 336)
        fused, tokens = runner.extract_r1_pre_terminal_feature(model, x)
        # captured tokens are the EXACT det_proj input Ours would consume
        self.assertTrue(torch.equal(tokens, model._last_pre_terminal))
        self.assertEqual(tuple(fused.shape),
                         (1, runner.R1_FROZEN_PROTO_FUSION_DIM))
        self.assertAlmostEqual(float(fused.norm()), 1.0, places=5)

    def test_r1_zero_trainable_gate(self):
        # Frozen model: R1 runs end-to-end.
        output = self.class_temp / "out_r1_ok"
        completion = runner.execute_run(
            project_root=self.repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1, seed=111,
            method="p30_frozen_proto", output_dir=output, device="cpu",
            allow_dirty_diagnostic=True,
            r1_overrides={"model": _frozen_proto_model(trainable=False)})
        self.assertFalse(completion["formal_eligible"])
        self.assertFalse(completion["query_pixels_read"])
        self.assertFalse(completion["query_labels_read"])
        self.assertEqual(completion["trainable_parameter_whitelist"], [])
        self.assertFalse(completion["model_provenance"]["terminal_adaptation"])
        art = torch.load(output / "p30_frozen_proto_prototypes.pt",
                         weights_only=False)
        self.assertEqual(art["trainable_param_count"], 0)
        self.assertFalse(art["terminal_adaptation"])
        self.assertEqual(tuple(art["prototypes"].shape),
                         (2, runner.R1_FROZEN_PROTO_FUSION_DIM))
        # Trainable model: R1 gate fails closed.
        bad = self.class_temp / "out_r1_trainable"
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="p30_frozen_proto", output_dir=bad,
                device="cpu", allow_dirty_diagnostic=True,
                r1_overrides={"model": _frozen_proto_model(trainable=True)})
        self.assertFalse(bad.exists())

    def test_r1_formal_injection_rejected(self):
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="p30_frozen_proto",
                output_dir=self.class_temp / "out_r1_formal_inject",
                device="cpu",
                r1_overrides={"model": _frozen_proto_model(trainable=False)})

    # --- R2 -------------------------------------------------------------------
    def test_r2_module_locked_constants(self):
        keys = torch.eye(2, models.EXPECTED_CLIP_DIM)
        text = torch.eye(2, models.EXPECTED_CLIP_DIM)
        for kw in ({"alpha": 0.5}, {"beta": 6.0}, {"logit_scale": 50.0}):
            with self.assertRaises(common.P6BFailClosedError):
                stage2.TipAdapterFModule(keys, [0, 1], 2, text, **kw)
        module = stage2.TipAdapterFModule(keys, [0, 1], 2, text)
        trainable = [n for n, p in module.named_parameters() if p.requires_grad]
        self.assertEqual(trainable, ["cache_keys"])
        self.assertFalse(module.cache_values.requires_grad)
        self.assertFalse(module.text_weights.requires_grad)

    def test_r2_spec_fail_closed(self):
        import dataclasses
        stage2.validate_tip_adapter_f_spec(stage2.TipAdapterFSpec())
        for field, value in (("epochs", 10), ("lr", 1e-2), ("eps", 1e-3),
                             ("batch_size", 16), ("weight_decay", 0.0)):
            bad = dataclasses.replace(stage2.TipAdapterFSpec(), **{field: value})
            with self.assertRaises(common.P6BFailClosedError):
                stage2.validate_tip_adapter_f_spec(bad)

    def test_r2_end_to_end_support_only(self):
        output = self.class_temp / "out_r2_e2e"
        completion = runner.execute_run(
            project_root=self.repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1, seed=111,
            method="tip_adapter_f", output_dir=output, device="cpu",
            allow_dirty_diagnostic=True, factory_bundle=FakeFactory())
        self.assertFalse(completion["formal_eligible"])
        self.assertFalse(completion["query_pixels_read"])
        self.assertFalse(completion["query_labels_read"])
        self.assertEqual(completion["trainable_parameter_whitelist"],
                         ["cache_keys"])
        audit = json.loads((output / "tip_adapter_f_step_audit.json")
                           .read_text(encoding="utf-8"))
        # K=1, 2 samples, batch 8 -> 1 batch/pass; 20 epochs x 1 repeat.
        self.assertEqual(audit["expected_optimizer_steps"], 20)
        self.assertEqual(audit["actual_optimizer_steps"], 20)
        self.assertEqual(audit["checkpoint_selection"], "final_epoch_only")
        self.assertFalse(audit["query_used_for_selection"])
        self.assertFalse(audit["validation_used_for_selection"])
        cache = torch.load(output / "tip_adapter_f_cache.pt",
                           weights_only=False)
        self.assertEqual(cache["alpha"], 1.0)
        self.assertEqual(cache["beta"], 5.5)
        self.assertEqual(cache["checkpoint_selection"], "final_epoch_only")
        # no best-epoch / query field anywhere in the completion
        def all_keys(obj):
            keys = set()
            if isinstance(obj, dict):
                for key, value in obj.items():
                    keys.add(str(key).lower())
                    keys |= all_keys(value)
            elif isinstance(obj, list):
                for value in obj:
                    keys |= all_keys(value)
            return keys
        forbidden = {"best_acc", "best_epoch", "best_checkpoint", "accuracy",
                     "balanced_accuracy", "macro_f1", "query_predictions"}
        self.assertEqual(all_keys(completion) & forbidden, set())

    def test_r2_clip_encoder_unfrozen_fails_closed(self):
        class TrainableClipFactory(FakeFactory):
            def __init__(self):
                super().__init__()
                self.clip_model = nn.Linear(4, 4)  # a trainable param present

        output = self.class_temp / "out_r2_unfrozen_clip"
        with self.assertRaises(common.P6BFailClosedError):
            runner.execute_run(
                project_root=self.repo, bundle_dir=self.bundle,
                image_root=self.image_root, product="bottle", k_shot=1,
                seed=111, method="tip_adapter_f", output_dir=output,
                device="cpu", allow_dirty_diagnostic=True,
                factory_bundle=TrainableClipFactory())
        self.assertFalse(output.exists())

    def test_r2_support_only_no_query_leak(self):
        # Only support pixels are read; a pixel-consuming method fails under a
        # ban, proving no hidden query load path exists.
        output, completion = None, None
        out = self.class_temp / "out_r2_pixels"
        completion = runner.execute_run(
            project_root=self.repo, bundle_dir=self.bundle,
            image_root=self.image_root, product="bottle", k_shot=1, seed=111,
            method="tip_adapter_f", output_dir=out, device="cpu",
            allow_dirty_diagnostic=True, factory_bundle=FakeFactory())
        self.assertTrue(completion["support_pixels_read"])
        self.assertFalse(completion["query_pixels_read"])


if __name__ == "__main__":
    unittest.main()
