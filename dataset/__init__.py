import os
import json
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .constants import CLASS_NAMES, DATA_PATH, DOMAINS


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Return a content fingerprint without depending on the machine path."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_metadata(meta_path: str):
    """Read non-empty JSONL records and return records plus provenance."""
    records = []
    with open(meta_path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {meta_path} at line {line_number}: {exc}"
                ) from exc
    return records, file_sha256(meta_path)


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        text: bool = False,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.text = text
        self.full_shot = "full-shot" in meta_path
        self.meta, self.metadata_sha256 = read_jsonl_metadata(meta_path)
        self.meta_path = os.path.abspath(meta_path)
        self.metadata_record_count = len(self.meta)

        # ================================================================
        # training augmentation, used together with --repeat_per_epoch;
        # all random augmentations run on PIL images, so every repeat sees a different variant;
        # no horizontal flip (HFlip = 0.0), which preserves left-right semantics (e.g. cable_swap);
        # cable-like datasets: rotation <= 3 deg, no translation, no cropping, preserving the spatial layout
        # ================================================================
        is_cable = "cable" in data_path.lower()
        transform_x = []
        if not text:
            if is_cable:
                # cable-like: fixed resize, no cropping/rotation/translation, preserving the spatial layout
                transform_x.append(
                    transforms.Resize((img_size, img_size), Image.BICUBIC)
                )
                # only a very small rotation jitter (<= 3 deg), preserving the absolute wire orientation
                transform_x.append(
                    transforms.RandomApply([
                        transforms.RandomRotation(degrees=3.0, interpolation=Image.BICUBIC)
                    ], p=0.3)
                )
            else:
                # other datasets: the original augmentation
                transform_x.append(
                    transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=Image.BICUBIC)
                )
                transform_x.append(
                    transforms.RandomApply([
                        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), interpolation=Image.BICUBIC)
                    ], p=0.3)
                )
            # photometric jitter: brightness/contrast 20%, saturation 15%, no hue change (wire colours matter)
            transform_x.append(
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.0)
                ], p=0.5)
            )
        else:
            # text mode: no augmentation, fixed resize
            transform_x.append(
                transforms.Resize((img_size, img_size), Image.BICUBIC)
            )
        self.transform_x = transforms.Compose(
            transform_x
            + [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ],
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        augmentation_seed = None
        if isinstance(idx, (tuple, list)):
            idx, augmentation_seed = idx
        idx = int(idx)
        meta = self.meta[idx]
        data_path = self.data_path
        # use forward slashes as path separators (mixed Windows/Linux paths)
        image_path = meta["image_path"].replace("\\", "/")
        img_path = os.path.join(data_path, image_path)
        # make sure the path uses the right separators
        img_path = img_path.replace("\\", "/")
        img = Image.open(img_path).convert("RGB")

        if augmentation_seed is None:
            img = self.transform_x(img)
        else:
            # Isolate per-sample augmentation from model/global RNG. This also
            # makes results independent of worker count and sample order.
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[]):
                    seed = int(augmentation_seed)
                    torch.manual_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    img = self.transform_x(img)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        # multi-class: load a mask only if mask_path exists and is not empty
        if "mask_path" in meta and meta["mask_path"]:
            mask_path_meta = meta["mask_path"].replace("\\", "/")
            mask_path = os.path.join(data_path, mask_path_meta)
            mask_path = mask_path.replace("\\", "/")
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])

        inputs = {
            "image": img,
            "mask": mask,
            "label": torch.tensor(meta["label"]).to(torch.int64),
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
            "sample_index": idx,
            "augmentation_seed": (
                int(augmentation_seed) if augmentation_seed is not None else -1
            ),
        }
        return inputs


class BaseSingleClassDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        meta_path: str,
        img_size: int,
        class_name: str,
        logger=None,
        strict_files: bool = False,
    ):

        assert class_name is not None, "class_name should be provided"
        self.data_path = data_path
        self.img_size = img_size
        all_meta, metadata_sha256 = read_jsonl_metadata(meta_path)
        self.meta_path = os.path.abspath(meta_path)
        self.metadata_sha256 = metadata_sha256
        self.metadata_record_count = len(all_meta)
        self.meta = [m for m in all_meta if m["class_name"] == class_name]
        self.class_metadata_record_count = len(self.meta)

        # Filter out entries whose image files don't exist on disk
        valid_meta = []
        missing_files = []
        for m in self.meta:
            img_path = os.path.join(self.data_path, m["image_path"].replace("\\", "/"))
            required_paths = [img_path]
            if m.get("mask_path"):
                required_paths.append(
                    os.path.join(
                        self.data_path, m["mask_path"].replace("\\", "/")
                    )
                )
            missing_for_record = [path for path in required_paths if not os.path.isfile(path)]
            if not missing_for_record:
                valid_meta.append(m)
            else:
                missing_files.extend(missing_for_record)
                if not strict_files:
                    for missing_path in missing_for_record:
                        if logger:
                            logger.warning(f"Skipping record with missing file: {missing_path}")
                        print(f"  WARNING: Skipping record with missing file: {missing_path}")
        self.missing_files = missing_files
        if missing_files and strict_files:
            preview = "\n".join(f"  - {path}" for path in missing_files[:20])
            more = "" if len(missing_files) <= 20 else f"\n  ... and {len(missing_files) - 20} more"
            raise FileNotFoundError(
                f"Formal evaluation requires every referenced file, but "
                f"{len(missing_files)} file(s) are missing for class '{class_name}':\n"
                f"{preview}{more}"
            )
        skipped = len(self.meta) - len(valid_meta)
        if skipped > 0:
            print(f"  Filtered out {skipped} missing image(s) for class '{class_name}'")
        self.meta = valid_meta

        # Define transforms
        self.transform_x = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(  # set image / mean metadata from pretrained_cfg if available, or use default
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

        # logging
        if logger:
            logger.info(f"Class name: {class_name}")
            logger.info(f"Sample number: {len(self.meta)}")
            logger.info("=====================================")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        meta = self.meta[idx]
        # use forward slashes as path separators (mixed Windows/Linux paths)
        image_path = meta["image_path"].replace("\\", "/")
        img_path = os.path.join(self.data_path, image_path)
        # make sure the path uses the right separators
        img_path = img_path.replace("\\", "/")
        img = Image.open(img_path).convert("RGB")
        img = self.transform_x(img)
        # multi-class: load a mask only if mask_path exists and is not empty
        if "mask_path" in meta and meta["mask_path"]:
            mask_path_meta = meta["mask_path"].replace("\\", "/")
            mask_path = os.path.join(self.data_path, mask_path_meta)
            mask_path = mask_path.replace("\\", "/")
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = (mask != 0).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size])
        inputs = {
            "image": img,
            "mask": mask,
            "label": meta["label"],
            "file_name": meta["image_path"],
            "class_name": meta["class_name"],
        }
        return inputs


def get_dataset(
    dataset_name: str,
    img_size: int,
    training_mode: str,
    shot: int = -1,
    stage: str = "train",
    logger=None,
    formal_eval: bool = False,
    meta_path_override: str | None = None,
):
    if "Med" not in dataset_name:
        assert dataset_name in DATA_PATH, (
            f"Dataset {dataset_name} not found; available datasets: {list(DATA_PATH.keys())}"
        )

    if stage == "train":
        if meta_path_override is not None:
            meta_path = os.path.abspath(meta_path_override)
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(
                    f"Explicit training metadata does not exist: {meta_path}"
                )
        elif training_mode == "few_shot":
            assert shot > 0, "shot should be positive"
            # the metadata folder of MT_Dataset is lower-case mt_dataset
            meta_folder = dataset_name.lower() if dataset_name == "MT_Dataset" else dataset_name
            meta_path = os.path.join(
                "./dataset/metadata", meta_folder, f"{shot}-shot.jsonl"
            )
        else:
            # the metadata folder of MT_Dataset is lower-case mt_dataset
            meta_folder = dataset_name.lower() if dataset_name == "MT_Dataset" else dataset_name
            meta_path = os.path.join(
                "./dataset/metadata", meta_folder, "full-shot.jsonl"
            )

        # PVEL-AD / MVTecAD-Cable use the full name as key (no split on "-")
        if dataset_name in ("PVEL-AD", "MVTecAD-Cable"):
            data_path = DATA_PATH[dataset_name]
        else:
            data_path = DATA_PATH[dataset_name.split("-")[0]]

        text_dataset = BaseDataset(data_path, meta_path, img_size, text=True)
        image_dataset = BaseDataset(data_path, meta_path, img_size, text=False)
        return text_dataset, image_dataset
    elif stage == "test":
        # the metadata folder of MT_Dataset is lower-case mt_dataset
        meta_folder = dataset_name.lower() if dataset_name == "MT_Dataset" else dataset_name
        
        # use test-shot.jsonl if it exists, otherwise full-shot.jsonl
        test_meta_path = os.path.join("./dataset/metadata", meta_folder, "test-shot.jsonl")
        full_meta_path = os.path.join("./dataset/metadata", meta_folder, "full-shot.jsonl")
        
        if meta_path_override is not None:
            meta_path = os.path.abspath(meta_path_override)
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(
                    f"Explicit evaluation metadata does not exist: {meta_path}"
                )
            if logger:
                logger.info(f"Using explicit evaluation metadata: {meta_path}")
        elif os.path.exists(test_meta_path):
            meta_path = test_meta_path
            if logger:
                logger.info(f"Using the test split: test-shot.jsonl")
        elif formal_eval:
            raise FileNotFoundError(
                "Formal evaluation requires the dedicated test metadata file: "
                f"{os.path.abspath(test_meta_path)}"
            )
        else:
            meta_path = full_meta_path
            if logger:
                logger.info(f"Warning: test-shot.jsonl not found, using full-shot.jsonl")
        
        class_names = CLASS_NAMES[dataset_name]
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                logger=logger,
                strict_files=formal_eval,
            )
            datasets[class_name] = image_dataset
        return datasets
    elif stage == "visualize":
        class_names = CLASS_NAMES[dataset_name]
        # the metadata folder of MT_Dataset is lower-case mt_dataset
        meta_folder = dataset_name.lower() if dataset_name == "MT_Dataset" else dataset_name
        meta_path = os.path.join("./dataset/metadata", meta_folder, "full-shot.jsonl")
        datasets = {}
        for class_name in class_names:
            image_dataset = BaseSingleClassDataset(
                data_path=DATA_PATH[dataset_name],
                meta_path=meta_path,
                img_size=img_size,
                class_name=class_name,
                logger=None,
            )
            datasets[class_name] = image_dataset
        return datasets
    else:
        raise ValueError(f"stage {stage} not found; available stages: train, test")
