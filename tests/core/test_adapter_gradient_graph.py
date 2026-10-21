"""Fast P1.5 unit tests for the explicit adapter autograd protocols."""

import os

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import types
import unittest
import hashlib
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.adapter import AdaptedCLIP
from model.faiss_memory import FaissPrototypeMemory
from model.checkpoint_io import extract_model_state_dict
from train_multiclass import (
    _gradient_graph_audit,
    _optimizer_named_parameters,
    balanced_normal_aware_bce,
    configure_model_trainable_parameters,
)


class FrozenBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width), requires_grad=False)

    def forward(self, tokens, attn_mask=None):
        return tokens @ self.weight, None


class ToyVisual(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(
            [FrozenBlock(width) for _ in range(24)]
        )


class ToyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = ToyVisual()


class DummyDino(nn.Module):
    backend = "dummy"

    def __init__(self, **kwargs):
        super().__init__()


def non_head_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("normal_aware_head."):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_toy_model(mode, normal_aware_head=False):
    model = AdaptedCLIP.__new__(AdaptedCLIP)
    nn.Module.__init__(model)
    model.clipmodel = ToyClip()
    model.image_encoder = model.clipmodel.visual
    model.dino_extractor = nn.Identity()
    model.image_adapt_until = 6
    model.i_w = 0.2
    model.adapter_train_mode = mode
    model.fusion_dim = 3
    model.image_adapter = nn.ModuleDict(
        {
            "layer_adapters": nn.ModuleList(
                [nn.Linear(4, 4, bias=False) for _ in range(6)]
            ),
            "seg_proj": nn.ModuleList(
                [nn.Linear(4, 4, bias=False) for _ in range(4)]
            ),
            "det_proj": nn.Linear(4, 3, bias=False),
        }
    )
    model.normal_aware_head_enabled = bool(normal_aware_head)
    model.normal_aware_head = (
        nn.Linear(3, 1) if normal_aware_head else None
    )
    if model.normal_aware_head is not None:
        nn.init.zeros_(model.normal_aware_head.weight)
        nn.init.zeros_(model.normal_aware_head.bias)
    return model


def run_backward(mode):
    torch.manual_seed(17)
    model = make_toy_model(mode)
    tokens = torch.randn(5, 2, 4)
    hidden = model._run_clip_blocks(tokens)
    loss = model.image_adapter["det_proj"](hidden).square().mean()
    loss.backward()
    return model


def args_for(mode, protocol="clean_v1", normal_aware_head=False):
    return types.SimpleNamespace(
        training_protocol=protocol,
        adapter_train_mode=mode,
        freeze_cross_modal=False,
        use_text_guided=False,
        proto_text_refine="none",
        normal_aware_head=normal_aware_head,
    )


class AdapterGradientGraphTests(unittest.TestCase):
    def test_enabling_head_preserves_historical_initialization_and_rng(self):
        with mock.patch("model.adapter.Dinov2Extractor", DummyDino):
            torch.manual_seed(111)
            without_head = AdaptedCLIP(
                ToyClip(), rng_protocol="isolated_rng", normal_aware_head=False
            )
            historical_sha = non_head_state_sha256(without_head)
            rng_without_head = torch.get_rng_state().clone()
            del without_head

            torch.manual_seed(111)
            with_head = AdaptedCLIP(
                ToyClip(), rng_protocol="isolated_rng", normal_aware_head=True
            )
            self.assertEqual(historical_sha, non_head_state_sha256(with_head))
            self.assertTrue(torch.equal(rng_without_head, torch.get_rng_state()))

    def test_balanced_binary_loss_equalizes_one_good_and_eight_defect_classes(self):
        logits = torch.zeros(9)
        labels = torch.arange(9)
        loss = balanced_normal_aware_bce(logits, labels, good_id=0, num_classes=9)
        self.assertAlmostEqual(loss.item(), float(torch.log(torch.tensor(2.0))), places=6)

    def test_normal_aware_head_zero_start_and_terminal_optimizer_membership(self):
        model = make_toy_model("terminal_only", normal_aware_head=True)
        names = configure_model_trainable_parameters(
            model, args_for("terminal_only", normal_aware_head=True)
        )
        self.assertTrue(torch.equal(model.normal_aware_head.weight, torch.zeros_like(model.normal_aware_head.weight)))
        self.assertTrue(torch.equal(model.normal_aware_head.bias, torch.zeros_like(model.normal_aware_head.bias)))
        self.assertIn("normal_aware_head.weight", names)
        self.assertIn("normal_aware_head.bias", names)
        logits = model.forward_normal_aware(torch.randn(4, 3))
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
        lightweight_state = extract_model_state_dict(model, "lightweight")
        self.assertIn("normal_aware_head.weight", lightweight_state)
        self.assertIn("normal_aware_head.bias", lightweight_state)

    def test_legacy_graph_cut_keeps_early_adapters_fixed(self):
        model = run_backward("legacy_random_fixed")
        self.assertTrue(
            all(p.grad is None for p in model.image_adapter["layer_adapters"].parameters())
        )
        self.assertGreater(
            torch.count_nonzero(model.image_adapter["det_proj"].weight.grad).item(), 0
        )

    def test_split_seed_diagnostic_retains_exact_legacy_optimizer_membership(self):
        model = make_toy_model("legacy_random_fixed")
        legacy_names = configure_model_trainable_parameters(
            model, args_for("legacy_random_fixed", protocol="legacy_9359")
        )
        split_model = make_toy_model("legacy_random_fixed")
        split_names = configure_model_trainable_parameters(
            split_model,
            args_for("legacy_random_fixed", protocol="legacy_seed_split_diag"),
        )
        self.assertEqual(legacy_names, split_names)

    def test_terminal_only_disables_early_adapters(self):
        model = run_backward("terminal_only")
        self.assertTrue(
            all(p.grad is None for p in model.image_adapter["layer_adapters"].parameters())
        )
        names = configure_model_trainable_parameters(
            model, args_for("terminal_only")
        )
        self.assertTrue(all(name.startswith("image_adapter.det_proj.") for name in names))

    def test_frozen_random_early_is_active_but_not_optimized(self):
        torch.manual_seed(31)
        frozen_model = make_toy_model("frozen_random_early")
        terminal_model = make_toy_model("terminal_only")
        terminal_model.load_state_dict(frozen_model.state_dict())
        frozen_model.train()
        self.assertFalse(frozen_model.clipmodel.training)
        self.assertFalse(frozen_model.dino_extractor.training)
        tokens = torch.randn(5, 2, 4)

        frozen_hidden = frozen_model._run_clip_blocks(tokens)
        terminal_hidden = terminal_model._run_clip_blocks(tokens)
        self.assertFalse(torch.equal(frozen_hidden, terminal_hidden))

        loss = frozen_model.image_adapter["det_proj"](frozen_hidden).square().mean()
        loss.backward()
        self.assertTrue(
            all(
                p.grad is None
                for p in frozen_model.image_adapter["layer_adapters"].parameters()
            )
        )
        self.assertGreater(
            torch.count_nonzero(
                frozen_model.image_adapter["det_proj"].weight.grad
            ).item(),
            0,
        )

        names = configure_model_trainable_parameters(
            frozen_model, args_for("frozen_random_early")
        )
        self.assertTrue(all(name.startswith("image_adapter.det_proj.") for name in names))
        self.assertFalse(any("layer_adapters" in name for name in names))

    def test_learned_early_has_nonzero_gradient_for_every_adapter(self):
        model = run_backward("learned_early")
        for parameter in model.image_adapter["layer_adapters"].parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)
        names = configure_model_trainable_parameters(
            model, args_for("learned_early")
        )
        self.assertTrue(any("layer_adapters.0" in name for name in names))
        self.assertFalse(any("seg_proj" in name for name in names))

    def test_first_backward_audit_matches_terminal_optimizer(self):
        torch.manual_seed(23)
        model = make_toy_model("terminal_only")
        configure_model_trainable_parameters(model, args_for("terminal_only"))
        memory = FaissPrototypeMemory(2, feat_dim=3)
        with torch.no_grad():
            memory.prototypes.copy_(F.normalize(torch.randn(2, 3), dim=-1))
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.parameters() if p.requires_grad]},
                {"params": memory.parameters()},
            ],
            lr=1e-3,
        )
        hidden = model._run_clip_blocks(torch.randn(5, 2, 4))
        features = model.image_adapter["det_proj"](hidden).mean(dim=0)
        loss = F.cross_entropy(memory(features), torch.tensor([0, 1]))
        loss.backward()
        audit = _gradient_graph_audit(
            _optimizer_named_parameters(model, memory, optimizer)
        )
        self.assertEqual(audit["grad_none_names"], [])
        self.assertEqual(audit["zero_grad_names"], [])
        self.assertEqual(
            set(audit["nonzero_grad_names"]),
            {
                "model.image_adapter.det_proj.weight",
                "prototype_mem.prototypes",
                "prototype_mem.logit_scale",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
