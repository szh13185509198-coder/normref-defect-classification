import os

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
import hashlib
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model.faiss_memory import FaissPrototypeMemory
from seed_protocol import (
    DeterministicWorkerSeeder,
    SeededEpochSampler,
    make_loader_generator,
)


class SyntheticSeededDataset(Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, key):
        index, augmentation_seed = key
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(augmentation_seed))
            value = torch.rand(4)
        return int(index), int(augmentation_seed), value


def collect_loader(num_workers):
    dataset = SyntheticSeededDataset()
    sampler = SeededEpochSampler(
        dataset, data_seed=222, augment_seed=333, shuffle=True
    )
    sampler.set_context("train", epoch=2, repeat=1)
    loader = DataLoader(
        dataset,
        batch_size=5,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=DeterministicWorkerSeeder(333),
        generator=make_loader_generator(222),
    )
    rows = []
    for indices, seeds, values in loader:
        rows.extend(
            (int(index), int(seed), value.clone())
            for index, seed, value in zip(indices, seeds, values)
        )
    return rows


class SeedProtocolTests(unittest.TestCase):
    def test_prototype_dimension_does_not_advance_rng(self):
        outputs = []
        for feature_dim in (384, 1920):
            torch.manual_seed(1234)
            memory = FaissPrototypeMemory(9, feature_dim)
            self.assertEqual(torch.count_nonzero(memory.prototypes).item(), 0)
            outputs.append(torch.rand(8))
        self.assertTrue(torch.equal(outputs[0], outputs[1]))

    def test_data_and_augmentation_streams_are_independent(self):
        dataset = SyntheticSeededDataset()

        def pairs(data_seed, augment_seed):
            sampler = SeededEpochSampler(
                dataset, data_seed=data_seed, augment_seed=augment_seed
            )
            sampler.set_context("train", epoch=3, repeat=2)
            return list(sampler)

        baseline = pairs(11, 22)
        changed_data = pairs(12, 22)
        changed_augment = pairs(11, 23)
        self.assertNotEqual([index for index, _ in baseline], [index for index, _ in changed_data])
        self.assertEqual(dict(baseline), dict(changed_data))
        self.assertEqual([index for index, _ in baseline], [index for index, _ in changed_augment])
        self.assertNotEqual(dict(baseline), dict(changed_augment))

    def test_worker_count_does_not_change_samples(self):
        single_process = collect_loader(0)
        worker_processes = collect_loader(2)
        self.assertEqual(
            [(index, seed) for index, seed, _ in single_process],
            [(index, seed) for index, seed, _ in worker_processes],
        )
        for left, right in zip(single_process, worker_processes):
            self.assertTrue(torch.equal(left[2], right[2]))

    def test_sampler_does_not_consume_global_rng(self):
        torch.manual_seed(99)
        random.seed(99)
        np.random.seed(99)
        expected = (torch.rand(4), random.random(), np.random.rand())

        torch.manual_seed(99)
        random.seed(99)
        np.random.seed(99)
        sampler = SeededEpochSampler(
            SyntheticSeededDataset(), data_seed=222, augment_seed=333
        )
        sampler.set_context("train", epoch=0, repeat=0)
        list(sampler)
        observed = (torch.rand(4), random.random(), np.random.rand())

        self.assertTrue(torch.equal(expected[0], observed[0]))
        self.assertEqual(expected[1:], observed[1:])

    def test_init_seed_and_data_seed_are_independent(self):
        def initial_fingerprint(init_seed):
            torch.manual_seed(init_seed)
            layer = torch.nn.Linear(8, 4)
            digest = hashlib.sha256()
            for value in layer.state_dict().values():
                digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        def order(data_seed):
            sampler = SeededEpochSampler(
                SyntheticSeededDataset(), data_seed=data_seed, augment_seed=333
            )
            sampler.set_context("train", epoch=0, repeat=0)
            return list(sampler)

        self.assertEqual(initial_fingerprint(111), initial_fingerprint(111))
        self.assertNotEqual(initial_fingerprint(111), initial_fingerprint(112))
        self.assertEqual(order(222), order(222))
        self.assertNotEqual(order(222), order(223))

        # Constructing/changing the sampler cannot alter parameter initialization.
        before = initial_fingerprint(111)
        order(999)
        after = initial_fingerprint(111)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
