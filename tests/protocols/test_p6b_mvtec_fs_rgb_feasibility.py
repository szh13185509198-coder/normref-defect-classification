import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class TestP6BMVTecFSRGBFeasibilityV2(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_root = Path(self.temp_dir) / "mvtec_fs"
        self.output_dir = Path(self.temp_dir) / "output"

        self.dataset_root.mkdir()
        self.output_dir.mkdir()

        self.image_dir = self.dataset_root / "image"
        self.config_dir = self.dataset_root / "CONFIG"
        self.image_dir.mkdir()
        self.config_dir.mkdir()

        self.cable_ref_root = self.image_dir / "cable"
        self.cable_ref_root.mkdir()

        self.script_path = Path(__file__).parent.parent.parent / "experiments" / "p6" / "probe_p6b_mvtec_fs_rgb_feasibility.py"

        (self.dataset_root / "readme.md").write_text("readme", encoding="utf-8")
        (self.dataset_root / "license.txt").write_text("license", encoding="utf-8")
        (self.cable_ref_root / "dummy.png").write_text("dummy", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def run_probe(self, dataset_root=None, cable_ref_root=None, output_dir=None):
        dr = dataset_root or self.dataset_root
        cr = cable_ref_root or self.cable_ref_root
        out = output_dir or self.output_dir
        cmd = [
            sys.executable, str(self.script_path),
            "--dataset-root", str(dr),
            "--cable-reference-root", str(cr),
            "--output-dir", str(out)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res

    def create_image(self, rel_path: str, content: bytes):
        path = self.image_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def create_json(self, rel_path: str, data: dict):
        path = self.image_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f)
        return path

    def create_csv(self, product: str, csv_name: str, rows: list):
        prod_config_dir = self.config_dir / f"{product}_config1"
        prod_config_dir.mkdir(parents=True, exist_ok=True)
        path = prod_config_dir / csv_name

        headers = ["imageName","imageData","imageHeight","imageWidth","img_id","part","id","img_rel_path","classification_rank","x1","x2","y1","y2","label","rank","points"]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                full_row = {h: "" for h in headers}
                full_row.update(r)
                writer.writerow(full_row)
        return path

    def test_01_multi_shape_group_id_null_valid(self):
        self.create_csv("prod1", "train.csv", [
            {"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"},
            {"img_id": "1", "part": "prod1", "id": "2", "img_rel_path": "training/defect/001.png", "label": "defectA"}
        ])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {
            "imagePath": "001.png",
            "shapes": [
                {"label": "defectA", "group_id": None},
                {"label": "defectA", "group_id": None}
            ]
        })

        res = self.run_probe()
        self.assertEqual(res.returncode, 0)

        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertTrue(feas["per_product"]["prod1"]["integrity_pass"])

    def test_02_two_img_id_same_rgb_folded(self):
        self.create_csv("prod1", "train.csv", [
            {"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"},
            {"img_id": "2", "part": "prod1", "id": "2", "img_rel_path": "training/defect/002.png", "label": "defectB"}
        ])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"same_content")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})
        self.create_image("prod1/training/defect/002.png", b"same_content")
        self.create_json("prod1/training/defect/002.json", {"imagePath": "002.png", "shapes": [{"label": "defectB"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_image_records.jsonl") as f:
            records = [json.loads(line) for line in f]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["instance_count"], 2)

    def test_03_same_rgb_cross_pool_collision_is_product_failure(self):
        self.create_csv("prod1", "train.csv", [
            {"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}
        ])
        self.create_csv("prod1", "valid.csv", [
            {"img_id": "2", "part": "prod1", "id": "2", "img_rel_path": "testing/defect/002.png", "label": "defectA"}
        ])
        self.create_image("prod1/training/defect/001.png", b"collision")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})
        self.create_image("prod1/testing/defect/002.png", b"collision")
        self.create_json("prod1/testing/defect/002.json", {"imagePath": "002.png", "shapes": [{"label": "defectA"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertEqual(feas["per_product"]["prod1"]["cross_pool_collision_count"], 1)

    def test_04_one_product_failure_doesnt_block_another(self):
        self.create_csv("fail_prod", "train.csv", [{"img_id": "1", "part": "fail_prod", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_csv("fail_prod", "valid.csv", [])
        self.create_csv("pass_prod", "train.csv", [{"img_id": "2", "part": "pass_prod", "id": "2", "img_rel_path": "training/defect/002.png", "label": "defectB"}])
        self.create_csv("pass_prod", "valid.csv", [])
        self.create_image("pass_prod/training/defect/002.png", b"img2")
        self.create_json("pass_prod/training/defect/002.json", {"imagePath": "002.png", "shapes": [{"label": "defectB"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["fail_prod"]["integrity_pass"])
            self.assertTrue(feas["per_product"]["pass_prod"]["integrity_pass"])

    def test_05_class_parse_failure_kept_as_0(self):
        self.create_csv("prod1", "train.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defect_fail"}])
        self.create_csv("prod1", "valid.csv", [])

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertEqual(feas["per_product"]["prod1"]["per_class_unique_rgb_count"]["before_filtering"]["defect_fail"], 0)

    def test_06_training_exact_mapping(self):
        self.create_csv("prod1", "train.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "testing/defect/001.png", "label": "defectA"}])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/testing/defect/001.png", b"img1")
        self.create_json("prod1/testing/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(any("does not start with 'training'" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

    def test_07_testing_exact_mapping(self):
        self.create_csv("prod1", "train.csv", [])
        self.create_csv("prod1", "valid.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])

    def test_08_csv_json_imagepath_mismatch(self):
        self.create_csv("prod1", "train.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "002.png", "shapes": [{"label": "defectA"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(any("JSON imagePath mismatch" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

    def test_09_path_escape(self):
        self.create_csv("prod1", "train.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "../../../../escape.png", "label": "defectA"}])
        self.create_csv("prod1", "valid.csv", [])

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])

    def test_10_output_schema_lf_determinism_atomic(self):
        self.create_csv("prod1", "train.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})

        # Capture mtime and hash of all inputs
        inputs = [
            self.image_dir / "prod1/training/defect/001.png",
            self.image_dir / "prod1/training/defect/001.json",
            self.config_dir / "prod1_config1/train.csv",
            self.config_dir / "prod1_config1/valid.csv",
            self.dataset_root / "readme.md",
            self.dataset_root / "license.txt",
            self.cable_ref_root / "dummy.png"
        ]
        import hashlib

        def snapshot(paths):
            return {p: (p.stat().st_mtime, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}

        pre_snap = snapshot(inputs)

        import sys
        sys.path.insert(0, str(self.script_path.parent.parent.parent))
        from experiments.p6.probe_p6b_mvtec_fs_rgb_feasibility import atomic_write_text

        import unittest.mock

        orig_named_temp = tempfile.NamedTemporaryFile
        with unittest.mock.patch("tempfile.NamedTemporaryFile") as mock_temp:
            def side_effect(*args, **kwargs):
                self.assertEqual(kwargs.get("dir"), self.output_dir)
                return orig_named_temp(*args, **kwargs)
            mock_temp.side_effect = side_effect

            import os
            orig_replace = os.replace
            def mock_replace(src, dst):
                raise OSError("Mock failure")

            with unittest.mock.patch("os.replace", side_effect=mock_replace):
                with self.assertRaises(OSError):
                    atomic_write_text(self.output_dir / "test_atomic.txt", "content")

        # Check no tmp left in output
        self.assertEqual(len(list(self.output_dir.glob("tmp*"))), 0)

        self.run_probe()

        json_file = self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json"
        jsonl_file = self.output_dir / "p6b_mvtec_fs_rgb_image_records.jsonl"
        md_file = self.output_dir / "p6b_mvtec_fs_rgb_feasibility_table.md"

        # Check LF
        with open(json_file, "rb") as bf:
            self.assertNotIn(b"\r\n", bf.read())

        with open(json_file) as f:
            feas = json.load(f)
            self.assertEqual(feas["schema_version"], "p6b_mvtec_fs_rgb_feasibility_v2")
            self.assertIn("project_provenance", feas)
            self.assertIn("dataset_provenance", feas)

            # Verify project/dataset provenance separation
            from experiments.p6.probe_p6b_mvtec_fs_rgb_feasibility import get_git_provenance
            actual_proj_prov = get_git_provenance(self.script_path.parent.parent.parent)
            self.assertEqual(feas["project_provenance"], actual_proj_prov)

            self.assertIsNone(feas["dataset_provenance"]["git_commit"])
            self.assertIsNone(feas["dataset_provenance"]["git_is_dirty"])

            readme_hash = hashlib.sha256((self.dataset_root / "readme.md").read_bytes()).hexdigest()
            license_hash = hashlib.sha256((self.dataset_root / "license.txt").read_bytes()).hexdigest()
            self.assertEqual(feas["dataset_provenance"]["readme_sha256"], readme_hash)
            self.assertEqual(feas["dataset_provenance"]["license_sha256"], license_hash)
            self.assertNotIn("readme_sha256", feas["project_provenance"])

            # Output hashes computable
            computed_md = hashlib.sha256(md_file.read_bytes()).hexdigest()
            computed_jsonl = hashlib.sha256(jsonl_file.read_bytes()).hexdigest()
            self.assertEqual(computed_md, feas["output_hashes"]["p6b_mvtec_fs_rgb_feasibility_table.md"])
            self.assertEqual(computed_jsonl, feas["output_hashes"]["p6b_mvtec_fs_rgb_image_records.jsonl"])

        # Capture content for determinism
        c_json = json_file.read_bytes()
        c_jsonl = jsonl_file.read_bytes()
        c_md = md_file.read_bytes()

        self.run_probe()

        # Check determinism
        self.assertEqual(c_json, json_file.read_bytes())
        self.assertEqual(c_jsonl, jsonl_file.read_bytes())
        self.assertEqual(c_md, md_file.read_bytes())

        # Check input integrity
        post_snap = snapshot(inputs)
        self.assertEqual(pre_snap, post_snap)

    def test_11_cable_hash_overlap(self):
        self.create_csv("prod1", "train.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"cable_content")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})

        (self.cable_ref_root / "test.png").write_bytes(b"cable_content")

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(any("Non-cable hash overlap" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

    def test_12_cable_path_overlap_cable_excluded(self):
        self.create_csv("cable", "train.csv", [{"img_id": "1", "part": "cable", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_csv("cable", "valid.csv", [])
        self.create_image("cable/training/defect/001.png", b"cable_content")
        self.create_json("cable/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertTrue(feas["excluded_product_audit"]["cable"]["is_excluded"])
            self.assertIn("image/cable/training/defect/001.png", feas["excluded_product_audit"]["cable"]["cable_path_overlaps"])

    def test_13_duplicate_instance_id_fails(self):
        self.create_csv("prod1", "train.csv", [
            {"img_id": "1", "part": "prod1", "id": "DUP", "img_rel_path": "training/defect/001.png", "label": "defectA"},
            {"img_id": "1", "part": "prod1", "id": "DUP", "img_rel_path": "training/defect/001.png", "label": "defectB"}
        ])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}, {"label": "defectB"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(any("Duplicate instance id" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

    def test_14_missing_images_triggers_failure(self):
        self.create_csv("prod1", "train.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_csv("prod1", "valid.csv", [])

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(any("Missing image" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

    def test_15_label_count_mismatch(self):
        self.create_csv("prod1", "train.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectB"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(any("Label count mismatch" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

    def test_16_non_cable_path_overlap(self):
        self.create_csv("prod1", "train.csv", [{"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})

        res = self.run_probe(cable_ref_root=(self.image_dir / "prod1"))

        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(any("Non-cable path overlap" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

    def test_17_valid_multi_type_removed(self):
        self.create_csv("prod1", "train.csv", [
            {"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"},
            {"img_id": "1", "part": "prod1", "id": "2", "img_rel_path": "training/defect/001.png", "label": "defectB"}
        ])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}, {"label": "defectB"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_image_records.jsonl") as f:
            records = [json.loads(line) for line in f]
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["multi_type_removed"])

    def test_18_same_img_id_two_paths_sha(self):
        self.create_csv("prod1", "train.csv", [
            {"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"},
            {"img_id": "1", "part": "prod1", "id": "2", "img_rel_path": "training/defect/002.png", "label": "defectB"}
        ])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})
        self.create_image("prod1/training/defect/002.png", b"img2")
        self.create_json("prod1/training/defect/002.json", {"imagePath": "002.png", "shapes": [{"label": "defectB"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(any("Inconsistent img_id 1" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

    def test_19_empty_required_fields(self):
        for field in ["img_id", "part", "id", "img_rel_path", "label"]:
            with self.subTest(field=field):
                base_row = {"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"}
                base_row[field] = "   "
                self.create_csv("prod1", "train.csv", [base_row])
                self.create_csv("prod1", "valid.csv", [])
                self.create_image("prod1/training/defect/001.png", b"img1")
                self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})

                res = self.run_probe()
                with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
                    feas = json.load(f)
                    self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
                    self.assertTrue(any(f"Empty required column '{field}'" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

    def test_20_global_fail_missing_readme_license(self):
        # Test readme missing
        (self.dataset_root / "readme.md").unlink()
        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertIn("Missing readme.md", feas["dataset_integrity_failures"])
            self.assertEqual(feas["probe_status"], "INCONCLUSIVE")

        # Test license missing as well
        (self.dataset_root / "license.txt").unlink()
        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertIn("Missing readme.md", feas["dataset_integrity_failures"])
            self.assertIn("Missing license.txt", feas["dataset_integrity_failures"])
            self.assertEqual(feas["probe_status"], "INCONCLUSIVE")

    def test_21_global_fail_empty_config_or_cable(self):
        shutil.rmtree(self.config_dir)
        self.config_dir.mkdir()
        (self.cable_ref_root / "dummy.png").unlink()
        (self.cable_ref_root / "dummy.txt").write_text("not an image")

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertIn("No valid product config directories found in CONFIG", feas["dataset_integrity_failures"])
            self.assertIn("cable-reference-root contains no supported image formats", feas["dataset_integrity_failures"])

    def test_22_shot_boundaries(self):
        # We need to test exact boundaries.
        # Prod1: support=1, query=1 -> feasible for 1, not 3, 5
        # Prod2: support=3, query=1 -> feasible for 1, 3, not 5
        # Prod3: support=5, query=1 -> feasible for 1, 3, 5
        # Prod4: support=3, query=0 -> fails all because query=0

        # Prod1 setup: K=1, query=1
        self.create_csv("prod1", "train.csv", [{"img_id": "t1", "part": "prod1", "id": "t1", "img_rel_path": "training/defect/001.png", "label": "defectA"}])
        self.create_csv("prod1", "valid.csv", [{"img_id": "v1", "part": "prod1", "id": "v1", "img_rel_path": "testing/defect/001.png", "label": "defectA"}])
        self.create_image("prod1/training/defect/001.png", b"img_t1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})
        self.create_image("prod1/testing/defect/001.png", b"img_v1")
        self.create_json("prod1/testing/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}]})

        # Prod2 setup: K=3, query=1
        self.create_csv("prod2", "train.csv", [{"img_id": f"t{i}", "part": "prod2", "id": f"t{i}", "img_rel_path": f"training/defect/00{i}.png", "label": "defectB"} for i in range(1, 4)])
        self.create_csv("prod2", "valid.csv", [{"img_id": "v1", "part": "prod2", "id": "v1", "img_rel_path": "testing/defect/001.png", "label": "defectB"}])
        for i in range(1, 4):
            self.create_image(f"prod2/training/defect/00{i}.png", f"img_t{i}".encode())
            self.create_json(f"prod2/training/defect/00{i}.json", {"imagePath": f"00{i}.png", "shapes": [{"label": "defectB"}]})
        self.create_image("prod2/testing/defect/001.png", b"img_v2")
        self.create_json("prod2/testing/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectB"}]})

        # Prod3 setup: K=5, query=1
        self.create_csv("prod3", "train.csv", [{"img_id": f"t{i}", "part": "prod3", "id": f"t{i}", "img_rel_path": f"training/defect/00{i}.png", "label": "defectC"} for i in range(1, 6)])
        self.create_csv("prod3", "valid.csv", [{"img_id": "v1", "part": "prod3", "id": "v1", "img_rel_path": "testing/defect/001.png", "label": "defectC"}])
        for i in range(1, 6):
            self.create_image(f"prod3/training/defect/00{i}.png", f"img_t{i}".encode())
            self.create_json(f"prod3/training/defect/00{i}.json", {"imagePath": f"00{i}.png", "shapes": [{"label": "defectC"}]})
        self.create_image("prod3/testing/defect/001.png", b"img_v3")
        self.create_json("prod3/testing/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectC"}]})

        # Prod4 setup: K=3, query=0
        self.create_csv("prod4", "train.csv", [{"img_id": f"t{i}", "part": "prod4", "id": f"t{i}", "img_rel_path": f"training/defect/00{i}.png", "label": "defectD"} for i in range(1, 4)])
        self.create_csv("prod4", "valid.csv", [])
        for i in range(1, 4):
            self.create_image(f"prod4/training/defect/00{i}.png", f"img_t{i}".encode())
            self.create_json(f"prod4/training/defect/00{i}.json", {"imagePath": f"00{i}.png", "shapes": [{"label": "defectD"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertTrue(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(feas["per_product"]["prod2"]["integrity_pass"])
            self.assertTrue(feas["per_product"]["prod3"]["integrity_pass"])
            self.assertTrue(feas["per_product"]["prod4"]["integrity_pass"])

            # Prod1: K=1, query=1
            self.assertIn("prod1", feas["feasible_products_by_shot"]["1"])
            self.assertNotIn("prod1", feas["feasible_products_by_shot"]["3"])
            self.assertNotIn("prod1", feas["feasible_products_by_shot"]["5"])

            # Prod2: K=3, query=1
            self.assertIn("prod2", feas["feasible_products_by_shot"]["1"])
            self.assertIn("prod2", feas["feasible_products_by_shot"]["3"])
            self.assertNotIn("prod2", feas["feasible_products_by_shot"]["5"])

            # Prod3: K=5, query=1
            self.assertIn("prod3", feas["feasible_products_by_shot"]["1"])
            self.assertIn("prod3", feas["feasible_products_by_shot"]["3"])
            self.assertIn("prod3", feas["feasible_products_by_shot"]["5"])

            # Prod4: K=3, query=0
            self.assertNotIn("prod4", feas["feasible_products_by_shot"]["1"])
            self.assertNotIn("prod4", feas["feasible_products_by_shot"]["3"])
            self.assertNotIn("prod4", feas["feasible_products_by_shot"]["5"])

    def test_23_two_img_id_same_path_diff_label_fails(self):
        # Two img_id sharing same path and SHA, but different labels in CSV. JSON has both labels.
        # This will fail closed because the label count per img_id won't match JSON.
        self.create_csv("prod1", "train.csv", [
            {"img_id": "1", "part": "prod1", "id": "1", "img_rel_path": "training/defect/001.png", "label": "defectA"},
            {"img_id": "2", "part": "prod1", "id": "2", "img_rel_path": "training/defect/001.png", "label": "defectB"}
        ])
        self.create_csv("prod1", "valid.csv", [])
        self.create_image("prod1/training/defect/001.png", b"img1")
        self.create_json("prod1/training/defect/001.json", {"imagePath": "001.png", "shapes": [{"label": "defectA"}, {"label": "defectB"}]})

        res = self.run_probe()
        with open(self.output_dir / "p6b_mvtec_fs_rgb_feasibility.json") as f:
            feas = json.load(f)
            self.assertFalse(feas["per_product"]["prod1"]["integrity_pass"])
            self.assertTrue(any("Label count mismatch" in msg for msg in feas["per_product"]["prod1"]["integrity_failures"]))

if __name__ == '__main__':
    unittest.main()
