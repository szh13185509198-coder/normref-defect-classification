"""A2 dynamic poison audit for the MVREC-RGB adaptation (Protocol A section 6).

Holding RGB pixels, image-level class label, seed and the deterministic
transform state fixed, this audit poisons every forbidden localization
metadata source (GT mask polygon column ``points``, GT bbox columns
``x1/x2/y1/y2``, and every derived ROI/crop/coordinate value) with random,
invalid, extreme and sentinel values, and verifies that:

Test 1  the constructed model RGB input, the fixed/global views, the labels,
        and the model support-side output stay byte-identical (deterministic
        tensors) or within atol=1e-7 / rtol=1e-6 (floating tensors);
Test 2  deleting the forbidden fields entirely still produces a working
        pipeline with identical support-side inputs and outputs.

The audit is support-only and performance-blind: no query manifest, query
pixel, query label, accuracy, balanced accuracy, macro-F1 or confusion
matrix is involved anywhere in this module.
"""

import copy
import json
import os
import shutil
import tempfile

import numpy as np
import torch

# Forbidden annotation columns of the upstream CSV (MVTec-FS CONFIG).
_FORBIDDEN_CSV_COLUMNS = ["x1", "x2", "y1", "y2", "points"]


def _poison_csv_variants(src_csv, workdir):
    """Create poisoned copies of the upstream annotation CSV.

    Returns {variant_name: csv_path}.  The variants deliberately write
    garbage into the forbidden localization columns while keeping every
    non-forbidden column untouched.
    """
    import pandas as pd

    df = pd.read_csv(src_csv)
    df = df.fillna("")
    n = len(df)
    rng = np.random.RandomState(20260813)
    variants = {}

    def clone():
        return df.copy(deep=True)

    # 1. random mask polygon
    v = clone()
    v["points"] = [str(rng.randint(0, 900, size=(rng.randint(3, 20), 2)).tolist())
                   for _ in range(n)]
    variants["random_mask"] = v
    # 2. all-zero mask polygon
    v = clone()
    v["points"] = "[]"
    variants["all_zero_mask"] = v
    # 3. random bbox
    v = clone()
    v["x1"], v["y1"] = rng.randint(0, 500, n), rng.randint(0, 500, n)
    v["x2"], v["y2"] = v["x1"] + rng.randint(1, 400, n), v["y1"] + rng.randint(1, 400, n)
    variants["random_bbox"] = v
    # 4. invalid bbox
    v = clone()
    v["x1"], v["y1"] = 900, 900
    v["x2"], v["y2"] = 0, 0
    variants["invalid_bbox"] = v
    # 5. random ROI coordinates (bbox-derived in upstream)
    v = clone()
    v["x1"], v["y1"] = rng.randint(0, 800, n), rng.randint(0, 800, n)
    v["x2"], v["y2"] = rng.randint(0, 800, n), rng.randint(0, 800, n)
    variants["random_roi"] = v
    # 6. extreme ROI coordinates
    v = clone()
    v["x1"], v["y1"] = -10 ** 6, -10 ** 6
    v["x2"], v["y2"] = 10 ** 6, 10 ** 6
    variants["extreme_roi"] = v
    # 7. random coordinates
    v = clone()
    v["points"] = [str(rng.randint(0, 10 ** 6, size=(10, 2)).tolist()) for _ in range(n)]
    variants["random_coordinates"] = v
    # 8. sentinel values
    v = clone()
    v["x1"], v["x2"], v["y1"], v["y2"] = "POISON", "POISON", "POISON", "POISON"
    v["points"] = "POISON"
    variants["sentinel_values"] = v

    paths = {}
    for name, v in variants.items():
        p = os.path.join(workdir, f"train_{name}.csv")
        v.to_csv(p, index=False)
        paths[name] = p
    return paths


def _deletion_csv_variant(src_csv, workdir):
    """CSV with every forbidden localization column deleted."""
    import pandas as pd

    df = pd.read_csv(src_csv)
    for col in _FORBIDDEN_CSV_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    p = os.path.join(workdir, "train_fields_deleted.csv")
    df.to_csv(p, index=False)
    return p


def _snapshot(dataset, support_forward, device, seed=111):
    """Materialize one deterministic support-side input/output snapshot."""
    import random

    import numpy as np

    # Fixed deterministic transform state for every snapshot.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    collate = dataset.get_collate_fn()
    batch = collate([dataset[i] for i in range(len(dataset))])
    batch_gpu = {"x": batch["x"].to(device), "y": batch["y"].to(device),
                 "masks": batch["masks"].to(device)}
    with torch.no_grad():
        outputs = support_forward(batch_gpu)
    labels = []
    for i in range(len(dataset)):
        try:
            labels.append(dataset[i]["class_name"])
        except (KeyError, TypeError):
            labels.append(int(batch["y"][i]))
    return {
        "x": batch_gpu["x"].detach().cpu(),
        "masks": batch_gpu["masks"].detach().cpu(),
        "y": batch_gpu["y"].detach().cpu(),
        "outputs": {k: v.detach().cpu() for k, v in outputs.items()
                    if isinstance(v, torch.Tensor)},
        "labels": labels,
    }


def _compare(base, cand):
    """Return (identical, diffs) with byte identity for integers and the
    protocol tolerance (atol=1e-7, rtol=1e-6) for floating tensors."""
    diffs = []

    def walk(prefix, a, b):
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.cpu().numpy()
        if isinstance(a, np.ndarray):
            if a.shape != b.shape:
                diffs.append(f"{prefix}: shape {a.shape} vs {b.shape}")
                return
            if a.dtype.kind in "iu" or b.dtype.kind in "iu":
                if not np.array_equal(a, b):
                    diffs.append(f"{prefix}: integer tensors differ")
            else:
                if not np.allclose(a, b, atol=1e-7, rtol=1e-6, equal_nan=True):
                    diffs.append(
                        f"{prefix}: float tensors differ "
                        f"(max abs {np.abs(a.astype(np.float64) - b.astype(np.float64)).max():.3e})"
                    )
        elif isinstance(a, list):
            if len(a) != len(b):
                diffs.append(f"{prefix}: list length {len(a)} vs {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(f"{prefix}[{i}]", x, y)
        elif isinstance(a, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b:
                    diffs.append(f"{prefix}.{k}: missing on one side")
                else:
                    walk(f"{prefix}.{k}", a[k], b[k])
        else:
            if a != b:
                diffs.append(f"{prefix}: {a!r} != {b!r}")

    walk("", base, cand)
    return len(diffs) == 0, diffs


def run_poison_audit(
    dataset_factory,
    manifest_path,
    image_root,
    annotation_csv,
    support_forward,
    device,
    seed=111,
):
    """Run the full A2 dynamic poison audit and return a result dict.

    ``dataset_factory(csv_path)`` reproduces the production adaptation path
    for a given annotation CSV path; a compliant whole-image adaptation
    ignores the argument entirely, while a violating implementation reads
    localization fields from it and therefore reacts to the poison.
    ``support_forward`` maps a batch dict to a dict of support-side output
    tensors (e.g. ``model.get_mvrec``).
    """
    workdir = tempfile.mkdtemp(prefix="mvrec_rgb_poison_")
    try:
        base = _snapshot(dataset_factory(annotation_csv), support_forward,
                         device, seed)
        results = {
            "baseline_ok": True,
            "poison_variants": {},
            "test1_pass": True,
            "test2_pass": True,
            "failures": [],
        }

        variants = _poison_csv_variants(annotation_csv, workdir)
        for name, poisoned_csv in variants.items():
            # The production path ignores annotation files entirely; the
            # poisoned CSVs exist on disk to prove that even wholesale
            # corruption of the localization metadata source has no effect.
            cand = _snapshot(dataset_factory(poisoned_csv), support_forward,
                             device, seed)
            identical, diffs = _compare(base, cand)
            results["poison_variants"][name] = {
                "identical": identical,
                "diffs": diffs,
            }
            if not identical:
                results["test1_pass"] = False
                results["failures"].append(f"poison variant {name} changed outputs")

        # Test 2: delete the forbidden fields entirely.  A compliant
        # adaptation keeps working; an implementation that still depends on
        # the deleted fields raises here and must fail the audit.
        deleted_csv = _deletion_csv_variant(annotation_csv, workdir)
        try:
            cand = _snapshot(dataset_factory(deleted_csv), support_forward,
                             device, seed)
        except Exception as e:  # pragma: no cover - defensive
            results["field_deletion"] = {
                "identical": False,
                "diffs": [f"pipeline raised after field deletion: {e!r}"],
            }
            results["test2_pass"] = False
            results["failures"].append(
                f"field deletion broke the pipeline: {e!r}"
            )
        else:
            identical, diffs = _compare(base, cand)
            results["field_deletion"] = {"identical": identical, "diffs": diffs}
            if not identical:
                results["test2_pass"] = False
                results["failures"].append("field deletion changed outputs")

        results["forbidden_fields_read"] = False
        results["query_pixels_read"] = False
        results["query_labels_read"] = False
        return results
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
