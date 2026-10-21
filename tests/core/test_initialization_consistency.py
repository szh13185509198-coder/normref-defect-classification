"""P1.4 integration tests for offline DINO backend and initialization parity."""

import os

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gc
import json
import socket
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.adapter import AdaptedCLIP
from provenance import (
    collect_backbone_provenance,
    initialize_training_manifest,
    state_dicts_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]
HUB_REPO = ROOT / "third_party" / "dinov2"
HUB_CHECKPOINT = ROOT / "pretrained" / "dinov2_vits14_pretrain.pth"


class DummyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Identity()


def project_state_fingerprint(model):
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.startswith(("clipmodel.", "dino_extractor."))
    }
    return state_dicts_fingerprint([("project", state)])


def build_model(backend):
    return AdaptedCLIP(
        clip_model=DummyClip(),
        dino_use_hub_first=backend == "hub",
        dino_strict_backend=backend == "hub",
        dino_hub_repo=str(HUB_REPO) if backend == "hub" else None,
        dino_hub_checkpoint=(
            str(HUB_CHECKPOINT) if backend == "hub" else None
        ),
        rng_protocol="isolated_rng",
        adapter_train_mode="terminal_only",
        img_size=336,
    )


class InitializationConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HUB_REPO.joinpath("hubconf.py").is_file():
            raise RuntimeError(f"offline DINO Hub source is missing: {HUB_REPO}")
        if not HUB_CHECKPOINT.is_file():
            raise RuntimeError(
                f"offline DINO Hub checkpoint is missing: {HUB_CHECKPOINT}"
            )
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator().manual_seed(20260808)
        cls.input = torch.randn(1, 3, 336, 336, generator=generator)

        torch.manual_seed(111)
        # Local source + explicit weights must never touch the network.
        with mock.patch.object(
            socket, "create_connection", side_effect=AssertionError("network used")
        ), mock.patch.object(
            urllib.request, "urlopen", side_effect=AssertionError("network used")
        ):
            hub_model = build_model("hub")
        cls.hub_project = project_state_fingerprint(hub_model)
        cls.hub_provenance = collect_backbone_provenance(
            model=hub_model,
            model_name="dummy-clip",
            actual_dino_backend=hub_model.dino_extractor.backend,
            dino_hub_repo=str(HUB_REPO),
            dino_hub_checkpoint=str(HUB_CHECKPOINT),
            clip_weight_path=None,
        )
        extractor = hub_model.dino_extractor.to(cls.device).eval()
        with torch.no_grad():
            cls.hub_cls = extractor(cls.input.to(cls.device)).cpu()
            cls.hub_patches = extractor.forward_patches(
                cls.input.to(cls.device)
            ).cpu()
        extractor.patch_policy = "legacy_drop_first"
        cls.hub_legacy_patches = extractor._apply_patch_policy(cls.hub_patches)
        extractor.patch_policy = "full"
        cls.hub_backend = extractor.backend
        del hub_model, extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.manual_seed(111)
        try:
            hf_model = build_model("transformers")
        except RuntimeError as error:
            raise RuntimeError(
                "facebook/dinov2-small is not present in the local HF cache: "
                f"{error}"
            ) from error
        cls.hf_project = project_state_fingerprint(hf_model)
        extractor = hf_model.dino_extractor.to(cls.device).eval()
        with torch.no_grad():
            cls.hf_cls = extractor(cls.input.to(cls.device)).cpu()
            cls.hf_patches = extractor.forward_patches(
                cls.input.to(cls.device)
            ).cpu()
        cls.hf_backend = extractor.backend
        del hf_model, extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_offline_hub_backend_and_manifest_identity(self):
        self.assertEqual(self.hub_backend, "hub_local")
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path = Path(temporary_dir) / "run_manifest.json"
            initialize_training_manifest(
                manifest_path, {"backbones": self.hub_provenance}
            )
            with manifest_path.open("r", encoding="utf-8") as source:
                manifest = json.load(source)
        dino = manifest["training"]["backbones"]["dino"]
        self.assertEqual(dino["actual_backend"], "hub_local")
        self.assertEqual(dino["weight_file"]["sha256"], (
            "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
        ))
        self.assertTrue(dino["local_source"]["sha256"])
        self.assertTrue(dino["loaded_state"]["sha256"])

    def test_frozen_backend_outputs_are_equivalent(self):
        self.assertEqual(self.hf_backend, "transformers")
        cls_cosine = F.cosine_similarity(self.hub_cls, self.hf_cls).item()
        patch_cosine = F.cosine_similarity(
            self.hub_patches.flatten(1), self.hf_patches.flatten(1)
        ).item()
        print(
            f"P1.4 backend cosine: cls={cls_cosine:.9f}, "
            f"patch_flat={patch_cosine:.9f}"
        )
        self.assertGreater(cls_cosine, 0.9999)
        # The official implementations use slightly different attention
        # kernels/layer ordering details; 0.999 is a strict representation-level
        # equivalence threshold while tolerating those numeric differences.
        self.assertGreater(patch_cosine, 0.999)

    def test_patch_counts_are_explicit(self):
        self.assertEqual(self.hub_patches.shape[1], 576)
        self.assertEqual(self.hub_legacy_patches.shape[1], 575)

    def test_post_backbone_project_initialization_matches(self):
        self.assertEqual(self.hub_project["sha256"], self.hf_project["sha256"])
        self.assertEqual(
            self.hub_project["parameter_names_sha256"],
            self.hf_project["parameter_names_sha256"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
