"""Regression tests for provenance tensor fingerprinting (0-D scalar support).

Covers the cloud smoke failure where ``_hash_named_tensors`` crashed on 0-D
scalar tensors (e.g. CLIP ``logit_scale``) with
``RuntimeError: self.dim() cannot be 0 to view Float as Byte``.
"""

import os

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import unittest

import numpy  # noqa: F401  (numpy before torch, cole/OpenMP safe)

import torch
import torch.nn as nn

from provenance import model_state_fingerprint, state_dicts_fingerprint


class ScalarBufferModule(nn.Module):
    """Minimal module with a 0-D scalar buffer (like CLIP ``logit_scale``)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scalar", torch.tensor(1.23))


class MixedShapeModule(nn.Module):
    """Module covering 0-D / 1-D / 2-D float32 state entries."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scalar", torch.tensor(1.23))
        self.register_buffer("vector", torch.tensor([1.0, 2.0, 3.0]))
        self.register_buffer("matrix", torch.randn(2, 2))


class TestModelStateFingerprintScalar(unittest.TestCase):
    def test_scalar_buffer_does_not_crash(self):
        # Regression: 0-D tensors previously crashed inside
        # _hash_named_tensors on ``view(torch.uint8)``.
        result = model_state_fingerprint(ScalarBufferModule())
        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["parameter_count"], 1)

    def test_mixed_shapes_do_not_crash(self):
        result = model_state_fingerprint(MixedShapeModule())
        self.assertEqual(result["tensor_count"], 3)
        self.assertEqual(result["parameter_count"], 1 + 3 + 4)

    def test_same_tensors_same_fingerprint(self):
        first = model_state_fingerprint(ScalarBufferModule())
        second = model_state_fingerprint(ScalarBufferModule())
        self.assertEqual(first["sha256"], second["sha256"])

    def test_scalar_value_change_changes_fingerprint(self):
        base = model_state_fingerprint(ScalarBufferModule())
        module = ScalarBufferModule()
        module.scalar = torch.tensor(2.34)
        changed = model_state_fingerprint(module)
        self.assertNotEqual(base["sha256"], changed["sha256"])

    def test_zero_dim_and_one_dim_are_not_confused(self):
        # shape metadata is hashed separately, so flattening bytes must not
        # collapse 0-D and shape-[1] tensors into the same fingerprint.
        zero_d = nn.Module()
        zero_d.register_buffer("scalar", torch.tensor(1.23))
        one_d = nn.Module()
        one_d.register_buffer("scalar", torch.tensor([1.23]))
        self.assertNotEqual(
            model_state_fingerprint(zero_d)["sha256"],
            model_state_fingerprint(one_d)["sha256"],
        )

    def test_dtype_change_changes_fingerprint(self):
        base = model_state_fingerprint(ScalarBufferModule())
        other = nn.Module()
        other.register_buffer("scalar", torch.tensor(1.23, dtype=torch.float64))
        self.assertNotEqual(base["sha256"], model_state_fingerprint(other)["sha256"])

    def test_state_dicts_fingerprint_scalar(self):
        mapping = {"clip": {"logit_scale": torch.tensor(4.6052)}}
        result = state_dicts_fingerprint([("clip", mapping["clip"])])
        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["parameter_count"], 1)


if __name__ == "__main__":
    unittest.main()
