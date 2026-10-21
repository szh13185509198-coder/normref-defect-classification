"""Independent deterministic streams for ordering and data augmentation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


MAX_TORCH_SEED = 2**63 - 1


def derive_seed(base_seed: int, *components) -> int:
    """Derive a stable torch-compatible seed without consuming any RNG state."""
    digest = hashlib.sha256(str(int(base_seed)).encode("ascii"))
    for component in components:
        digest.update(b"\0")
        digest.update(str(component).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big") % MAX_TORCH_SEED


class SeededEpochSampler(Sampler):
    """Yield ``(index, augmentation_seed)`` with independent RNG streams."""

    def __init__(
        self,
        data_source,
        *,
        data_seed: int,
        augment_seed: int,
        shuffle: bool = True,
    ):
        self.data_source = data_source
        self.data_seed = int(data_seed)
        self.augment_seed = int(augment_seed)
        self.shuffle = bool(shuffle)
        self.stage = "unconfigured"
        self.epoch = 0
        self.repeat = 0

    def set_context(self, stage: str, *, epoch: int = 0, repeat: int = 0) -> dict:
        self.stage = str(stage)
        self.epoch = int(epoch)
        self.repeat = int(repeat)
        return self.context()

    def context(self) -> dict:
        return {
            "stage": self.stage,
            "epoch": self.epoch,
            "repeat": self.repeat,
            "sampler_seed": derive_seed(
                self.data_seed, "order", self.stage, self.epoch, self.repeat
            ),
            "augmentation_seed": derive_seed(
                self.augment_seed, "augment", self.stage, self.epoch, self.repeat
            ),
        }

    def __iter__(self) -> Iterator:
        size = len(self.data_source)
        context = self.context()
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(context["sampler_seed"])
            order = torch.randperm(size, generator=generator).tolist()
        else:
            order = list(range(size))
        for index in order:
            # The sample's augmentation is independent of its position in the
            # shuffled order, so changing data_seed changes order only.
            sample_augment_seed = derive_seed(
                context["augmentation_seed"], "sample", index
            )
            yield index, sample_augment_seed

    def __len__(self) -> int:
        return len(self.data_source)


@dataclass(frozen=True)
class DeterministicWorkerSeeder:
    """Pickle-safe worker initializer for Windows spawn and Linux fork."""

    augment_seed: int

    def __call__(self, worker_id: int) -> None:
        seed = derive_seed(self.augment_seed, "worker", int(worker_id))
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)


def make_loader_generator(data_seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(derive_seed(data_seed, "dataloader_worker_base"))
    return generator


def set_loader_seed_context(
    loader, stage: str, *, epoch: int = 0, repeat: int = 0
) -> dict:
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, SeededEpochSampler):
        raise TypeError("DataLoader must use SeededEpochSampler for deterministic training")
    return sampler.set_context(stage, epoch=epoch, repeat=repeat)
