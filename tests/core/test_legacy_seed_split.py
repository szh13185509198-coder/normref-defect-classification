"""Lightweight tests for the P1.9 post-construction RNG split diagnostic."""

import os

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_multiclass import (
    is_legacy_training_protocol,
    load_early_adapter_override,
    reset_split_training_rng,
)


def args_for(protocol, *, data_seed=222, augment_seed=222):
    return types.SimpleNamespace(
        training_protocol=protocol,
        data_seed=data_seed,
        augment_seed=augment_seed,
    )


def random_probe():
    return torch.rand(5), np.random.rand(3), [random.random() for _ in range(3)]


class ToyAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(6)]
                ),
                "seg_proj": nn.ModuleList(
                    [nn.Linear(4, 3, bias=False) for _ in range(4)]
                ),
                "det_proj": nn.Linear(4, 3, bias=False),
            }
        )


class LegacySeedSplitTests(unittest.TestCase):
    def test_protocol_classifier_keeps_both_legacy_graphs(self):
        self.assertTrue(is_legacy_training_protocol("legacy_9359"))
        self.assertTrue(is_legacy_training_protocol("legacy_seed_split_diag"))
        self.assertFalse(is_legacy_training_protocol("clean_v1"))

    def test_exact_legacy_does_not_reset_continuous_rng(self):
        torch.manual_seed(17)
        np.random.seed(17)
        random.seed(17)
        expected = random_probe()

        torch.manual_seed(17)
        np.random.seed(17)
        random.seed(17)
        self.assertFalse(reset_split_training_rng(args_for("legacy_9359")))
        observed = random_probe()

        self.assertTrue(torch.equal(expected[0], observed[0]))
        self.assertTrue(np.array_equal(expected[1], observed[1]))
        self.assertEqual(expected[2], observed[2])

    def test_split_protocol_resets_all_global_rngs_to_trajectory_seed(self):
        torch.manual_seed(999)
        np.random.seed(999)
        random.seed(999)
        self.assertTrue(
            reset_split_training_rng(
                args_for("legacy_seed_split_diag", data_seed=222, augment_seed=222)
            )
        )
        observed = random_probe()

        torch.manual_seed(222)
        torch.cuda.manual_seed(222)
        torch.cuda.manual_seed_all(222)
        np.random.seed(222)
        random.seed(222)
        expected = random_probe()

        self.assertTrue(torch.equal(expected[0], observed[0]))
        self.assertTrue(np.array_equal(expected[1], observed[1]))
        self.assertEqual(expected[2], observed[2])

    def test_split_protocol_rejects_two_post_construction_streams(self):
        with self.assertRaisesRegex(ValueError, "data_seed == augment_seed"):
            reset_split_training_rng(
                args_for("legacy_seed_split_diag", data_seed=222, augment_seed=333)
            )

    def test_early_adapter_override_changes_only_six_selected_tensors(self):
        torch.manual_seed(111)
        target = ToyAdapterModel()
        target_before = {
            name: tensor.detach().clone()
            for name, tensor in target.state_dict().items()
        }
        torch.manual_seed(222)
        donor_model = ToyAdapterModel()
        donor_state = {
            name: tensor.detach().clone()
            for name, tensor in donor_model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "donor.pth"
            torch.save(
                {
                    "model_state_dict": donor_state,
                    "checkpoint_format": "lightweight_v1",
                    "epoch": 20,
                    "run_config": {
                        "training_protocol": "legacy_seed_split_diag",
                        "adapter_train_mode": "legacy_random_fixed",
                        "seed": 222,
                        "init_seed": 222,
                        "trajectory_seed": 111,
                    },
                },
                checkpoint_path,
            )
            info = load_early_adapter_override(target, str(checkpoint_path))

        self.assertEqual(info["tensor_count"], 6)
        self.assertEqual(info["source_epoch"], 20)
        self.assertEqual(info["source_init_seed"], 222)
        self.assertEqual(info["source_trajectory_seed"], 111)
        self.assertTrue(info["checkpoint_sha256"])
        self.assertTrue(info["selected_state"]["sha256"])
        for name, observed in target.state_dict().items():
            if name.startswith("image_adapter.layer_adapters."):
                self.assertTrue(torch.equal(observed, donor_state[name]), name)
            else:
                self.assertTrue(torch.equal(observed, target_before[name]), name)

    def test_early_adapter_override_rejects_incomplete_donor(self):
        target = ToyAdapterModel()
        incomplete = {
            name: tensor
            for name, tensor in target.state_dict().items()
            if not name.startswith("image_adapter.layer_adapters.5.")
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "incomplete.pth"
            torch.save(
                {
                    "model_state_dict": incomplete,
                    "run_config": {
                        "adapter_train_mode": "legacy_random_fixed"
                    },
                },
                checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "donor keys"):
                load_early_adapter_override(target, str(checkpoint_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
