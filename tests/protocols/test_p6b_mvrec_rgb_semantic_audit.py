"""P6.B MVREC-RGB semantic audit focused regression tests.

Covers Protocol A section 14:

1. Protocol integrity (fail-closed)
   - locked protocol missing -> fail closed
   - locked protocol hash mismatch -> fail closed
   - wrong upstream MVREC commit -> fail closed
   - dirty formal semantic audit -> fail closed
2. RGB-only firewall
   - mask affects output -> fail
   - bbox affects output -> fail
   - ROI affects output -> fail
   - forbidden field deleted but code fails -> fail
3. Gradient ownership
   - unexpected trainable parameter -> fail
   - expected official trainable parameter missing -> fail
   - optimizer contains forbidden parameter -> fail
4. Query firewall
   - query dataset instantiated during semantic audit -> fail
   - query pixels touched -> fail
   - query labels touched -> fail
   - performance metric generated -> fail
5. Determinism
   - same manifest/seed/weights -> consistent diagnostic output

Synthetic manifest + tiny PNGs + fake support forward only; real AlphaCLIP
weights are never loaded in these tests.
"""

import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.mvrec_rgb import protocol_lock as pl
from experiments.p6.mvrec_rgb import poison_audit as pa
from experiments.p6.mvrec_rgb import semantic_smoke as smoke
from experiments.p6.mvrec_rgb.rgb_dataset import MVRecRgbSupportDataset

PKG_DIR = PROJECT_ROOT / "experiments" / "p6" / "mvrec_rgb"


# ---------------------------------------------------------------------------
# Synthetic assets
# ---------------------------------------------------------------------------

def write_image(path, color):
    """Spatially varying RGB image so that any crop offset changes content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    r, g, b = color
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[:, :, 0] = (r + np.arange(16)[None, :] * 11) % 256
    arr[:, :, 1] = (g + np.arange(16)[:, None] * 13) % 256
    arr[:, :, 2] = b
    Image.fromarray(arr, mode="RGB").save(path, format="PNG")
    return path


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def make_assets(base):
    """Manifest (3 classes x 1 support) + tiny RGB images + annotation CSV."""
    image_root = base / "mvtec-fs"
    manifest_path = base / "k1_s111.json"
    csv_path = base / "CONFIG" / "train.csv"
    classes = ["broken_large", "broken_small", "contamination"]
    manifest_classes = {}
    csv_rows = []
    csv_rows = [
        "imageName,imageData,imageHeight,imageWidth,img_id,part,id,"
        "img_rel_path,classification_rank,x1,x2,y1,y2,label,rank,points\n"
    ]
    for ci, cls in enumerate(classes):
        rel = f"image/bottle/training/{cls}/000.png"
        sha = file_sha256(write_image(image_root / rel, (ci * 60, 128, 200)))
        manifest_classes[cls] = {
            "candidate_count": 10,
            "items": [{"rel_path": rel, "rgb_sha256": sha}],
        }
        csv_rows.append(
            f"000.png,,900,900,{ci},bottle,{ci},{rel},,"
            f"{ci * 10},{ci * 10 + 5},{ci * 10},{ci * 10 + 5},{cls},-1,"
            f"\"[[{ci},{ci}],[{ci + 1},{ci + 1}]]\"\n"
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("".join(csv_rows), encoding="utf-8")

    manifest = {
        "K": 1,
        "candidate_ordering": "lexicographic_rgb_sha256",
        "class_order": classes,
        "class_universe": classes,
        "classes": manifest_classes,
        "draw_scheme": "p6b_draw_v1",
        "manifest_sha256": "synthetic",
        "product": "bottle",
        "rng_algorithm": "python_random_mersenne_twister",
        "schema_version": "p6b_support_manifest_v1",
        "seed": 111,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, image_root, csv_path


def fake_support_forward(batch):
    """Deterministic fake support-side forward: mean-pooled RGB + masks."""
    return {
        "rgb_mean": batch["x"].float().mean(dim=(-2, -1)),
        "mask_sum": batch["masks"].float().sum(),
    }


# ---------------------------------------------------------------------------
# Violating datasets (deliberately annotation-dependent)
# ---------------------------------------------------------------------------

class _AnnotationDataset(Dataset):
    def __init__(self, csv_path, image_root):
        import pandas as pd

        self.rows = pd.read_csv(csv_path).fillna("").to_dict("records")
        self.image_root = image_root

    def __len__(self):
        return len(self.rows)

    def _img(self, row):
        p = os.path.join(self.image_root, row["img_rel_path"])
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)

    def _base(self, i):
        row = self.rows[i]
        x = torch.from_numpy(self._img(row)).permute(2, 0, 1).unsqueeze(0)
        y = torch.tensor([i % 3])
        return row, x, y

    def get_collate_fn(self):
        def collate(batch):
            return {
                "x": torch.stack([b["x"] for b in batch], dim=0),
                "y": torch.stack([b["y"] for b in batch], dim=0).view(-1),
                "masks": torch.stack([b["masks"] for b in batch], dim=0),
            }

        return collate


class MaskViolatingDataset(_AnnotationDataset):
    """masks derived from the GT points column -> poison must change them."""

    def __getitem__(self, i):
        row, x, y = self._base(i)
        pts = json.loads(row["points"]) if row["points"] not in ("", "[]", "POISON") else [[0, 0]]
        mask = np.zeros((16, 16), dtype=np.float32)
        for px, py in pts[:4]:
            if 0 <= px < 16 and 0 <= py < 16:
                mask[py, px] = 1.0
        masks = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        return {"x": x, "y": y, "masks": masks}


class BboxViolatingDataset(_AnnotationDataset):
    """x crop offset depends on the GT bbox columns -> poison must change x."""

    def __getitem__(self, i):
        row, x, y = self._base(i)
        x1 = int(float(str(row["x1"]).replace("POISON", "0"))) % 8
        y1 = int(float(str(row["y1"]).replace("POISON", "0"))) % 8
        crop = x[:, :, y1:y1 + 8, x1:x1 + 8]
        masks = torch.ones(1, 1, 16, 16)
        return {"x": crop, "y": y, "masks": masks}


class RoiViolatingDataset(_AnnotationDataset):
    """x is a bbox-derived ROI crop (center/scale from annotation)."""

    def __getitem__(self, i):
        row, x, y = self._base(i)
        try:
            cx = (int(float(row["x1"])) + int(float(row["x2"]))) // 2 % 8
            cy = (int(float(row["y1"])) + int(float(row["y2"]))) // 2 % 8
        except ValueError:  # sentinel variant
            cx, cy = 4, 4
        cx, cy = int(cx), int(cy)
        masks = torch.ones(1, 1, 16, 16)
        return {"x": x[:, :, cy:cy + 8, cx:cx + 8], "y": y, "masks": masks}


class FieldDeletionBreakingDataset(_AnnotationDataset):
    """Reads x1 directly -> breaks when the forbidden columns are deleted."""

    def __getitem__(self, i):
        row, x, y = self._base(i)
        _ = row["x1"]  # raises KeyError after field deletion
        masks = torch.ones(1, 1, 16, 16)
        return {"x": x, "y": y, "masks": masks}


# ---------------------------------------------------------------------------
# S1 protocol integrity
# ---------------------------------------------------------------------------

class TestProtocolIntegrity(unittest.TestCase):
    def test_01_locked_protocol_missing_fails_closed(self):
        with mock.patch.object(pl, "PROTOCOL_A", str(Path(tempfile.gettempdir()) / "nope.md")):
            with self.assertRaises(OSError):
                pl.compute_protocol_hashes()

    def test_02_hash_mismatch_fails_closed(self):
        tmp = Path(tempfile.mkdtemp(prefix="mvrec_lock_"))
        try:
            lock_path = tmp / "lock.json"
            lock_path.write_text(json.dumps({
                "protocol_lock_commit": pl.PROTOCOL_LOCK_COMMIT,
                "protocol_A_canonical_lf_sha256": "deadbeef",
                "protocol_B_canonical_lf_sha256": "deadbeef",
                "decision_rule_canonical_lf_sha256": "deadbeef",
                "mvrec_upstream_commit": pl.MVREC_UPSTREAM_COMMIT,
            }), encoding="utf-8")
            with mock.patch.object(pl, "LOCK_ARTIFACT", str(lock_path)):
                artifact, ok, problems = pl.ensure_lock_artifact()
            self.assertFalse(ok)
            self.assertTrue(problems)
            self.assertIn("lock artifact mismatch", problems[0])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_03_wrong_upstream_commit_fails_closed(self):
        tmp = Path(tempfile.mkdtemp(prefix="mvrec_lock_"))
        try:
            # First create a valid lock artifact with the true upstream
            # commit, then verify that a different commit fails closed.
            with mock.patch.object(pl, "LOCK_ARTIFACT", str(tmp / "lock.json")):
                _, created_ok, _ = pl.ensure_lock_artifact()
                self.assertTrue(created_ok)
                artifact, ok, problems = pl.ensure_lock_artifact(
                    mvrec_upstream_commit="0" * 40
                )
            self.assertFalse(ok)
            self.assertTrue(any("mvrec_upstream_commit" in p for p in problems))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_04_dirty_audit_fails_closed(self):
        from experiments.p6.mvrec_rgb import audit_main

        tmp = Path(tempfile.mkdtemp(prefix="mvrec_audit_"))
        try:
            fake_provenance = {
                "execution_commit": pl.PROTOCOL_LOCK_COMMIT,
                "execution_dirty": True,
                "status_short": [" M dirty.py"],
                "mvrec_upstream_commit": pl.MVREC_UPSTREAM_COMMIT,
                "mvrec_patch_files": {},
                "upstream_tree_file_count": 0,
                "upstream_tree_sha256": "x" * 64,
                "environment": {},
                "pretrained_weight_hashes": {},
                "config_hashes": {},
            }
            with mock.patch.object(
                audit_main, "collect_provenance", return_value=fake_provenance
            ), mock.patch.object(
                audit_main, "ensure_lock_artifact",
                return_value=({
                    "protocol_lock_commit": pl.PROTOCOL_LOCK_COMMIT,
                    "protocol_A_canonical_lf_sha256": "a",
                    "protocol_B_canonical_lf_sha256": "b",
                    "decision_rule_canonical_lf_sha256": "c",
                    "mvrec_upstream_commit": pl.MVREC_UPSTREAM_COMMIT,
                }, True, []),
            ):
                final, report = audit_main.run_audit(
                    manifest_path=str(tmp / "m.json"),
                    image_root=str(tmp),
                    annotation_csv=str(tmp / "c.csv"),
                    alpha_ckpt_path=None,
                    out_dir=str(tmp / "out"),
                    run_model=False,
                    allow_dirty=False,
                )
            self.assertEqual(final["final_decision"], "INCONCLUSIVE_ENGINEERING")
            self.assertTrue(any("dirty" in p for p in report["problems"]))

            with mock.patch.object(
                audit_main, "collect_provenance", return_value=fake_provenance
            ), mock.patch.object(
                audit_main, "ensure_lock_artifact",
                return_value=({
                    "protocol_lock_commit": pl.PROTOCOL_LOCK_COMMIT,
                    "protocol_A_canonical_lf_sha256": "a",
                    "protocol_B_canonical_lf_sha256": "b",
                    "decision_rule_canonical_lf_sha256": "c",
                    "mvrec_upstream_commit": pl.MVREC_UPSTREAM_COMMIT,
                }, True, []),
            ):
                final2, report2 = audit_main.run_audit(
                    manifest_path=str(tmp / "m.json"),
                    image_root=str(tmp),
                    annotation_csv=str(tmp / "c.csv"),
                    alpha_ckpt_path=None,
                    out_dir=str(tmp / "out2"),
                    run_model=False,
                    allow_dirty=True,
                )
            # With allow_dirty the dirty-state problem is suppressed; the
            # remaining problems are the not-executed model gates only.
            self.assertEqual(final2["final_decision"], "INCONCLUSIVE_ENGINEERING")
            self.assertTrue(all("dirty" not in p for p in report2["problems"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# S2 RGB-only firewall
# ---------------------------------------------------------------------------

class TestRgbOnlyFirewall(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_fw_"))
        self.manifest, self.image_root, self.csv = make_assets(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, factory):
        return pa.run_poison_audit(
            dataset_factory=factory,
            manifest_path=str(self.manifest),
            image_root=str(self.image_root),
            annotation_csv=str(self.csv),
            support_forward=fake_support_forward,
            device="cpu",
            seed=111,
        )

    def test_05_compliant_dataset_passes(self):
        res = self._run(
            lambda csv_path: MVRecRgbSupportDataset(
                str(self.manifest), str(self.image_root), device="cpu"
            )
        )
        self.assertTrue(res["test1_pass"], res["failures"])
        self.assertTrue(res["test2_pass"], res["failures"])
        self.assertTrue(res["field_deletion"]["identical"])
        self.assertFalse(res["forbidden_fields_read"])
        self.assertFalse(res["query_pixels_read"])
        self.assertFalse(res["query_labels_read"])

    def test_06_mask_affects_output_fails(self):
        res = self._run(
            lambda csv_path: MaskViolatingDataset(
                csv_path, str(self.image_root)
            )
        )
        self.assertFalse(res["test1_pass"])
        self.assertTrue(any("random_mask" in f for f in res["failures"]) or
                        not res["poison_variants"]["random_mask"]["identical"])

    def test_07_bbox_affects_output_fails(self):
        res = self._run(
            lambda csv_path: BboxViolatingDataset(
                csv_path, str(self.image_root)
            )
        )
        self.assertFalse(res["test1_pass"])
        self.assertFalse(res["poison_variants"]["random_bbox"]["identical"])

    def test_08_roi_affects_output_fails(self):
        res = self._run(
            lambda csv_path: RoiViolatingDataset(
                csv_path, str(self.image_root)
            )
        )
        self.assertFalse(res["test1_pass"])
        self.assertFalse(res["poison_variants"]["random_roi"]["identical"])

    def test_09_field_deletion_break_fails(self):
        res = self._run(
            lambda csv_path: FieldDeletionBreakingDataset(
                csv_path, str(self.image_root)
            )
        )
        self.assertFalse(res["test2_pass"])
        self.assertTrue(any("field deletion" in f for f in res["failures"]))


# ---------------------------------------------------------------------------
# S3 gradient ownership
# ---------------------------------------------------------------------------

class _FakeHead(torch.nn.Module):
    def __init__(self, extra_trainable=False):
        super().__init__()
        self.support_key = torch.nn.Parameter(torch.zeros(2, 4))
        self.zifa = torch.nn.Module()
        adapter = torch.nn.Module()
        lin = torch.nn.Module()
        lin.weight = torch.nn.Parameter(torch.zeros(4, 4))
        lin.bias = torch.nn.Parameter(torch.zeros(4))
        adapter.add_module("0", lin)
        self.zifa.add_module("adapter", adapter)
        if extra_trainable:
            self.extra = torch.nn.Parameter(torch.zeros(3))
        self.frozen = torch.nn.Parameter(torch.zeros(5), requires_grad=False)


class TestGradientOwnership(unittest.TestCase):
    def test_10_whitelist_expected_set(self):
        names = ["head.support_key", "head.zifa.adapter.0.weight",
                 "head.zifa.adapter.0.bias"]
        self.assertTrue(smoke.check_trainable_whitelist(names))
        self.assertFalse(smoke.check_trainable_whitelist(names[:-1]))
        self.assertFalse(smoke.check_trainable_whitelist(names + ["head.extra"]))

    def _optimizer_for(self, steps_map, ft_epo=500):
        m = _FakeHead()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, eps=1e-6)
        for name, steps in steps_map.items():
            p = dict(m.named_parameters())[name]
            if steps is not None:
                opt.state[p] = {"step": steps}
        return opt

    def test_11_optimizer_full_budget_ok(self):
        opt = self._optimizer_for({
            "support_key": 500,
            "zifa.adapter.0.weight": 500,
            "zifa.adapter.0.bias": 500,
        })
        ok, reason, entries = smoke._optimizer_ownership(opt, 500)
        self.assertTrue(ok, reason)

    def test_12_optimizer_frozen_param_updated_fails(self):
        opt = self._optimizer_for({
            "support_key": 500,
            "zifa.adapter.0.weight": 500,
            "zifa.adapter.0.bias": 500,
            "frozen": 3,  # frozen parameter must never be updated
        })
        ok, reason, _ = smoke._optimizer_ownership(opt, 500)
        self.assertFalse(ok)
        self.assertIn("frozen", reason)

    def test_13_optimizer_partial_budget_fails(self):
        opt = self._optimizer_for({
            "support_key": 499,
            "zifa.adapter.0.weight": 500,
            "zifa.adapter.0.bias": 500,
        })
        ok, reason, _ = smoke._optimizer_ownership(opt, 500)
        self.assertFalse(ok)
        self.assertIn("full budget", reason)


# ---------------------------------------------------------------------------
# S4 query firewall (static AST scan of the audit package)
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORTS = {
    "fewshot_process", "create_valid_data", "create_valid_data_novel",
    "PartDatasetTool", "MultiviewRoiData", "gen_few_shot_support_query_dataset",
    "FewShotEvalFunc", "Accucary", "classificationMetric",
    "visualConfusionMatrix", "accuracy_score", "confusion_matrix", "f1_score",
}
FORBIDDEN_CALLS = {
    "accuracy_score", "balanced_accuracy_score", "f1_score", "confusion_matrix",
    "Accucary",
}


class TestQueryFirewall(unittest.TestCase):
    def _ast_scan(self):
        findings = {"imports": [], "calls": []}
        for path in sorted(PKG_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in FORBIDDEN_IMPORTS:
                            findings["imports"].append((path.name, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module in FORBIDDEN_IMPORTS:
                        findings["imports"].append((path.name, node.module))
                    for alias in node.names:
                        if alias.name in FORBIDDEN_IMPORTS:
                            findings["imports"].append((path.name, alias.name))
                elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
                    findings["calls"].append((path.name, node.id))
        return findings

    def test_14_no_query_dataset_import(self):
        findings = self._ast_scan()
        query_imports = [f for f in findings["imports"]
                         if f[1] in {"fewshot_process", "create_valid_data",
                                     "create_valid_data_novel", "PartDatasetTool",
                                     "MultiviewRoiData", "FewShotEvalFunc"}]
        self.assertEqual(query_imports, [])

    def test_15_no_performance_metric(self):
        findings = self._ast_scan()
        self.assertEqual(findings["calls"], [])
        self.assertEqual(findings["imports"], [])

    def test_16_poison_result_query_flags(self):
        tmp = Path(tempfile.mkdtemp(prefix="mvrec_qf_"))
        try:
            manifest, image_root, csv = make_assets(tmp)
            res = pa.run_poison_audit(
                dataset_factory=lambda csv_path: MVRecRgbSupportDataset(
                    str(manifest), str(image_root), device="cpu"),
                manifest_path=str(manifest),
                image_root=str(image_root),
                annotation_csv=str(csv),
                support_forward=fake_support_forward,
                device="cpu",
                seed=111,
            )
            for key in ("query_pixels_read", "query_labels_read"):
                self.assertIn(key, res)
                self.assertFalse(res[key])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# S5 determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mvrec_det_"))
        self.manifest, self.image_root, self.csv = make_assets(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_17_dataset_byte_identical_across_instances(self):
        d1 = MVRecRgbSupportDataset(str(self.manifest), str(self.image_root),
                                    device="cpu")
        d2 = MVRecRgbSupportDataset(str(self.manifest), str(self.image_root),
                                    device="cpu")
        self.assertEqual(len(d1), len(d2))
        for i in range(len(d1)):
            a, b = d1[i], d2[i]
            self.assertTrue(torch.equal(a["x"], b["x"]))
            self.assertTrue(torch.equal(a["masks"], b["masks"]))
            self.assertTrue(torch.equal(a["y"], b["y"]))

    def test_18_snapshot_repeatable(self):
        d = MVRecRgbSupportDataset(str(self.manifest), str(self.image_root),
                                   device="cpu")
        s1 = pa._snapshot(d, fake_support_forward, "cpu", seed=111)
        s2 = pa._snapshot(d, fake_support_forward, "cpu", seed=111)
        identical, diffs = pa._compare(s1, s2)
        self.assertTrue(identical, diffs)

    def test_19_support_identity_verify(self):
        d = MVRecRgbSupportDataset(str(self.manifest), str(self.image_root),
                                   device="cpu")
        ok, mismatches = d.verify_support_identity()
        self.assertTrue(ok, mismatches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
