from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.p6.p6b_execution_common import P6BFailClosedError, atomic_write_json
from experiments.p6 import p6b_stage2_execution as stage2
from experiments.p6 import run_p6b_stage2_support as support
from experiments.p6 import p6b_stage3_common as common
from experiments.p6 import run_p6b_formal_support_grid as grid
from experiments.p6 import evaluate_p6b_formal_grid as evaluator
from experiments.p6 import audit_p6b_formal_grid as auditor


class P6BStage3ContractsTest(unittest.TestCase):
    def test_01_exact_grid_is_1755_unique_units(self):
        units = list(common.expected_units())
        self.assertEqual(len(units), 1755)
        self.assertEqual(len({unit.key for unit in units}), 1755)
        self.assertEqual(units[0].key, "clip_proto/bottle/k1_s111")
        self.assertEqual(units[-1].key, "tip_adapter_f/zipper/k5_s555")

    def test_02_operational_shards_partition_without_scientific_change(self):
        args = []
        for index in range(7):
            args.extend(grid.selected_units(7, index))
        self.assertEqual(len(args), 1755)
        self.assertEqual({u.key for u in args}, {u.key for u in common.expected_units()})
        self.assertEqual(len({u.key for u in args}), 1755)
        with self.assertRaises(P6BFailClosedError):
            grid.selected_units(0, 0)
        with self.assertRaises(P6BFailClosedError):
            grid.selected_units(2, 2)

    def test_03_locked_metrics_known_example_and_zero_division(self):
        result = common.classification_metrics(
            ["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b", "c"])
        self.assertAlmostEqual(result["top1_accuracy"], 0.75)
        self.assertAlmostEqual(result["balanced_accuracy"], (0.5 + 1.0 + 0.0) / 3)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.8 + 0.0) / 3)

    def test_04_product_macro_then_seed_ddof1(self):
        rows = []
        for method_index, method in enumerate(common.METHODS):
            for product_index, product in enumerate(common.PRODUCTS):
                for k_shot in common.K_VALUES:
                    for seed_index, seed in enumerate(common.SEEDS):
                        value = method_index + product_index / 100 + k_shot / 1000 + seed_index / 10
                        rows.append({"method": method, "product": product, "K": k_shot,
                                     "seed": seed, "top1_accuracy": value,
                                     "balanced_accuracy": value, "macro_f1": value})
        result = common.aggregate_raw_metrics(rows)
        target = next(r for r in result["method_k_summary"]
                      if r["method"] == "clip_proto" and r["K"] == 1)
        self.assertAlmostEqual(target["top1_accuracy_mean"], 0.061 + 0.2)
        self.assertAlmostEqual(target["top1_accuracy_std_ddof1"], 0.15811388300841897)
        self.assertEqual(len(result["product_macro_per_seed"]), 9 * 3 * 5)
        self.assertEqual(len(result["per_product_five_seed"]), 9 * 13 * 3)

    def test_05_duplicate_raw_metric_is_fail_closed(self):
        row = {"method": "clip_proto", "product": "bottle", "K": 1, "seed": 111,
               "top1_accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0}
        with self.assertRaisesRegex(P6BFailClosedError, "duplicate raw metric"):
            common.aggregate_raw_metrics([row, dict(row)])

    def _make_unit(self, root: Path):
        unit = common.GridUnit("clip_proto", "bottle", 1, 111)
        unit_dir = unit.directory(root)
        unit_dir.mkdir(parents=True)
        artifacts = []
        meta_path = unit_dir / "p6b_support_meta.jsonl"
        meta_path.write_text(
            json.dumps({"class_name": "a", "label": 0}) + "\n" +
            json.dumps({"class_name": "b", "label": 1}) + "\n", encoding="utf-8")
        primary = unit_dir / "clip_proto_prototypes.pt"
        primary.write_bytes(b"prototype\n")
        for path in (meta_path, primary):
            artifacts.append({"relative_path": path.name, "size_bytes": path.stat().st_size,
                              "sha256": common.file_sha256(path)})
        config = {
            "schema_version": support.RUN_CONFIG_SCHEMA, "method": unit.method,
            "product": unit.product, "K": 1, "seed": 111,
            "class_order": ["a", "b"],
            "support_metadata": {"records": 2, "sha256": common.file_sha256(meta_path)},
        }
        config_path = unit_dir / "support_run_config.json"
        atomic_write_json(config_path, config)
        artifacts.append({"relative_path": config_path.name, "size_bytes": config_path.stat().st_size,
                          "sha256": common.file_sha256(config_path)})
        artifact_manifest = {
            "schema_version": support.ARTIFACT_MANIFEST_SCHEMA,
            "method": unit.method, "product": unit.product, "K": 1, "seed": 111,
            "artifacts": list(artifacts),
        }
        manifest_path = unit_dir / "support_artifact_manifest.json"
        atomic_write_json(manifest_path, artifact_manifest)
        artifacts.append({"relative_path": manifest_path.name, "size_bytes": manifest_path.stat().st_size,
                          "sha256": common.file_sha256(manifest_path)})
        completion = {
            "schema_version": support.COMPLETION_SCHEMA, "status": "completed",
            "formal_eligible": True, "formal_test_used": False,
            "query_pixels_read": False, "query_labels_read": False,
            "method": unit.method, "product": unit.product, "K": 1, "seed": 111,
            "class_order": ["a", "b"],
            "execution": {"commit": "a" * 40, "dirty": False,
                          "allow_dirty_diagnostic": False},
            "protocol_bundle": {
                "version": support.PROTOCOL_BUNDLE_VERSION,
                "main_protocol_sha256": support.EXPECTED_DOCUMENT_SHAS_CANONICAL_LF["protocol"],
                "addendum_a_sha256": support.EXPECTED_DOCUMENT_SHAS_CANONICAL_LF["addendum_a"],
                "addendum_b_sha256": support.EXPECTED_DOCUMENT_SHAS_CANONICAL_LF["addendum_b"],
                "stage1_completion_sha256": stage2.STAGE1_COMPLETION_SHA256,
                "hash_mode": "canonical-lf",
            },
            "model_provenance": {
                "clip": {
                    "network_download": False,
                    "weight_file": {"exists": True, "filename": "clip.pt",
                                    "sha256": "c" * 64, "size_bytes": 123},
                    "loaded_state": {"sha256": "d" * 64,
                                     "parameter_names_sha256": "e" * 64,
                                     "parameter_count": 10, "tensor_count": 2},
                },
                "dino": {"not_loaded": True}, "fusion": {"not_loaded": True},
            },
            "artifacts": artifacts,
        }
        atomic_write_json(unit_dir / common.SUPPORT_COMPLETION, completion)
        return unit, unit_dir

    def test_06_support_unit_full_hash_gate_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            unit, unit_dir = self._make_unit(Path(temp))
            common.validate_support_unit(unit, unit_dir, "a" * 40)
            (unit_dir / "clip_proto_prototypes.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(P6BFailClosedError, "artifact (size|SHA) mismatch"):
                common.validate_support_unit(unit, unit_dir, "a" * 40)

    def test_07_diagnostic_or_cross_commit_unit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            unit, unit_dir = self._make_unit(Path(temp))
            completion_path = unit_dir / common.SUPPORT_COMPLETION
            completion = common.load_json(completion_path)
            completion["status"] = "diagnostic"
            completion["formal_eligible"] = False
            atomic_write_json(completion_path, completion)
            with self.assertRaises(P6BFailClosedError):
                common.validate_support_unit(unit, unit_dir, "a" * 40)
            completion["status"] = "completed"
            completion["formal_eligible"] = True
            atomic_write_json(completion_path, completion)
            with self.assertRaisesRegex(P6BFailClosedError, "grid commit"):
                common.validate_support_unit(unit, unit_dir, "b" * 40)

    def test_08_query_contract_audits_raw_to_single_type_transition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            dataset.mkdir()
            records = []
            for index, (label, excluded) in enumerate((("a", False), ("b", False), ("a", True))):
                relative = f"q{index}.png"
                image = dataset / relative
                image.write_bytes(f"rgb-{index}".encode())
                records.append({
                    "schema_version": "p6b_mvtec_fs_rgb_image_record_v2",
                    "product": "bottle", "source_pools": ["query"],
                    "rgb_sha256": common.file_sha256(image),
                    "canonical_relative_path": relative,
                    "unique_defect_types": [label] if not excluded else ["a", "b"],
                    "multi_type_removed": excluded,
                })
            records_path = root / "records.jsonl"
            records_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            bundle = root / "bundle"
            bundle.mkdir()
            completion = {
                "schema_version": "p6b_support_manifest_build_v1", "status": "completed",
                "formal_eligible": True, "formal_test_used": False,
                "query_pixels_read": False, "query_labels_read": False,
                "records_jsonl_sha256": common.file_sha256(records_path),
                "per_product": {"bottle": {"class_universe": ["a", "b"],
                                             "query_sha_metadata_count": 3}},
            }
            completion_path = bundle / "p6b_manifest_build_complete.json"
            atomic_write_json(completion_path, completion)
            with (mock.patch.object(common, "EXPECTED_QUERY_RECORDS_SHA256", common.file_sha256(records_path)),
                  mock.patch.object(common, "EXPECTED_RAW_QUERY_COUNT", 3),
                  mock.patch.object(common, "EXPECTED_EXCLUDED_MULTI_TYPE", 1),
                  mock.patch.object(common, "EXPECTED_FORMAL_QUERY_COUNT", 2),
                  mock.patch.object(common, "PRODUCTS", ("bottle",)),
                  mock.patch.object(stage2, "STAGE1_COMPLETION_SHA256", common.file_sha256(completion_path))):
                selected, audit = common.load_query_contract(records_path, dataset, bundle)
            self.assertEqual(len(selected), 2)
            self.assertEqual(audit["raw_query_count"], 3)
            self.assertEqual(audit["multi_type_excluded_count"], 1)

    def test_09_evaluator_never_opens_query_when_grid_gate_fails(self):
        with mock.patch.object(evaluator, "validate_formal_grid",
                               side_effect=P6BFailClosedError("grid incomplete")), \
             mock.patch.object(evaluator, "load_query_contract") as query_loader:
            with self.assertRaisesRegex(P6BFailClosedError, "grid incomplete"):
                evaluator.evaluate(grid_root="x", records_path="secret", dataset_root="d",
                                   stage1_bundle="b", output_dir="o")
            query_loader.assert_not_called()

    def test_10_locked_query_arithmetic(self):
        self.assertEqual(common.EXPECTED_RAW_QUERY_COUNT, 549)
        self.assertEqual(common.EXPECTED_EXCLUDED_MULTI_TYPE, 19)
        self.assertEqual(common.EXPECTED_FORMAL_QUERY_COUNT, 530)
        self.assertEqual(549 - 19, 530)

    def _tiny_prediction_rows(self):
        queries = [
            {"product": "bottle", "rgb_sha256": "1" * 64,
             "true_label": "a", "class_order": ["a", "b"]},
            {"product": "bottle", "rgb_sha256": "2" * 64,
             "true_label": "b", "class_order": ["a", "b"]},
        ]
        rows = []
        for query, scores in zip(queries, ([2.0, 1.0], [3.0, 4.0])):
            rows.append({
                "schema_version": common.PREDICTION_SCHEMA,
                "method": "clip_proto", "product": "bottle", "K": 1, "seed": 111,
                "rgb_sha256": query["rgb_sha256"], "true_label": query["true_label"],
                "class_order": ["a", "b"], "scores": list(scores),
                "predicted_label": query["true_label"],
            })
        return queries, rows

    def test_11_independent_prediction_audit_recomputes_metrics(self):
        queries, rows = self._tiny_prediction_rows()
        unit = common.GridUnit("clip_proto", "bottle", 1, 111)
        with mock.patch.object(auditor, "EXPECTED_FORMAL_QUERY_COUNT", 2), \
             mock.patch.object(auditor, "EXPECTED_PREDICTION_ROWS", 2), \
             mock.patch.object(auditor, "expected_units", return_value=iter([unit])), \
             mock.patch.object(auditor, "aggregate_raw_metrics", return_value={"recomputed": True}):
            raw, aggregate = auditor.audit_prediction_rows(rows, queries)
        self.assertEqual(raw[0]["top1_accuracy"], 1.0)
        self.assertEqual(aggregate, {"recomputed": True})

    def test_12_prediction_audit_rejects_duplicate_and_nonfinite_scores(self):
        queries, rows = self._tiny_prediction_rows()
        with mock.patch.object(auditor, "EXPECTED_FORMAL_QUERY_COUNT", 2), \
             mock.patch.object(auditor, "EXPECTED_PREDICTION_ROWS", 2):
            duplicate = [rows[0], dict(rows[0])]
            with self.assertRaisesRegex(P6BFailClosedError, "duplicate prediction"):
                auditor.audit_prediction_rows(duplicate, queries)
            bad = [dict(rows[0]), dict(rows[1])]
            bad[1]["scores"] = [float("nan"), 0.0]
            with self.assertRaisesRegex(P6BFailClosedError, "invalid score"):
                auditor.audit_prediction_rows(bad, queries)

    def test_13_prediction_jsonl_requires_lf_canonical_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.jsonl"
            path.write_bytes(b'{"x":1}\r\n')
            with self.assertRaisesRegex(P6BFailClosedError, "non-canonical newline"):
                auditor.read_prediction_rows(path)

    def test_14_unit_prediction_manifest_and_global_concatenation_gate(self):
        queries, rows = self._tiny_prediction_rows()
        unit = common.GridUnit("clip_proto", "bottle", 1, 111)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            grid_root, output = root / "grid", root / "output"
            unit_grid = unit.directory(grid_root)
            unit_grid.mkdir(parents=True)
            (grid_root / common.GRID_COMPLETION).write_bytes(b"grid\n")
            support_completion = unit_grid / common.SUPPORT_COMPLETION
            support_completion.write_bytes(b"support\n")
            unit_prediction = output / "predictions/clip_proto/bottle/k1_s111.jsonl"
            unit_prediction.parent.mkdir(parents=True)
            unit_prediction.write_text("".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows), encoding="utf-8", newline="\n")
            unit_manifest_path = unit_prediction.with_suffix(".manifest.json")
            unit_manifest = {
                "schema_version": common.UNIT_PREDICTION_MANIFEST_SCHEMA,
                "method": unit.method, "product": unit.product, "K": 1, "seed": 111,
                "class_order": ["a", "b"], "query_count": 2,
                "query_rgb_sha256": [query["rgb_sha256"] for query in queries],
                "support_completion_sha256": common.file_sha256(support_completion),
                "predictions_relative_path": "predictions/clip_proto/bottle/k1_s111.jsonl",
                "predictions_sha256": common.file_sha256(unit_prediction),
                "predictions_size_bytes": unit_prediction.stat().st_size,
            }
            atomic_write_json(unit_manifest_path, unit_manifest)
            global_prediction = output / evaluator.PREDICTIONS_NAME
            global_prediction.write_bytes(unit_prediction.read_bytes())
            index = [{
                "key": unit.key,
                "predictions_relative_path": unit_manifest["predictions_relative_path"],
                "predictions_sha256": common.file_sha256(unit_prediction),
                "prediction_manifest_relative_path": "predictions/clip_proto/bottle/k1_s111.manifest.json",
                "prediction_manifest_sha256": common.file_sha256(unit_manifest_path),
                "query_count": 2,
            }]
            global_manifest_path = output / evaluator.PREDICTION_MANIFEST_NAME
            atomic_write_json(global_manifest_path, {
                "schema_version": common.PREDICTION_MANIFEST_SCHEMA, "status": "completed",
                "formal_test_used": False, "query_pixels_read": True,
                "query_labels_read": True, "prediction_row_count": 2,
                "formal_query_count": 2, "support_grid_execution_commit": "a" * 40,
                "evaluator_execution_commit": "b" * 40,
                "predictions_sha256": common.file_sha256(global_prediction),
                "unit_predictions": index,
            })
            patches = (
                mock.patch.object(auditor, "validate_formal_grid",
                                  return_value={"execution_commit": "a" * 40}),
                mock.patch.object(auditor.builder, "git_commit_and_dirty",
                                  return_value=("b" * 40, False)),
                mock.patch.object(auditor, "load_query_contract",
                                  return_value=(queries, {"formal_query_count": 2})),
                mock.patch.object(auditor, "EXPECTED_FORMAL_QUERY_COUNT", 2),
                mock.patch.object(auditor, "EXPECTED_PREDICTION_ROWS", 2),
                mock.patch.object(auditor, "expected_units", side_effect=lambda: iter([unit])),
                mock.patch.object(auditor, "audit_prediction_rows",
                                  return_value=([{"ok": True}], {"ok": True})),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                with mock.patch.object(auditor, "file_sha256", wraps=common.file_sha256) as sha:
                    # The records path is not opened by the mocked query loader;
                    # give the provenance field a concrete frozen fixture.
                    records_fixture = root / "records.jsonl"
                    records_fixture.write_bytes(b"records\n")
                    result = auditor.audit_files(
                        grid_root=grid_root, predictions_path=global_prediction,
                        prediction_manifest_path=global_manifest_path,
                        records_path=records_fixture, dataset_root="unused", stage1_bundle="unused",
                        output_path=output / "audit.json")
            self.assertEqual(result["prediction_row_count"], 2)
            unit_prediction.write_bytes(b"tampered\n")
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                with self.assertRaisesRegex(P6BFailClosedError, "index SHA mismatch"):
                    auditor.audit_files(
                        grid_root=grid_root, predictions_path=global_prediction,
                        prediction_manifest_path=global_manifest_path,
                        records_path=records_fixture, dataset_root="unused", stage1_bundle="unused",
                        output_path=output / "audit2.json")

    def test_15_cross_cloud_backbone_identity_is_path_independent_and_strict(self):
        completion = {
            "method": "clip_proto",
            "model_provenance": {
                "clip": {
                    "network_download": False,
                    "weight_file": {"exists": True, "filename": "clip.pt",
                                    "path": "/cloud-a/repo/clip.pt",
                                    "sha256": "a" * 64, "size_bytes": 100},
                    "loaded_state": {"sha256": "b" * 64,
                                     "parameter_names_sha256": "c" * 64,
                                     "parameter_count": 10, "tensor_count": 2},
                },
                "dino": {"not_loaded": True}, "fusion": {"not_loaded": True},
            },
        }
        cloud_a = common.extract_backbone_identities(completion, "cloud-a")
        variant = json.loads(json.dumps(completion))
        variant["model_provenance"]["clip"]["weight_file"]["path"] = "/cloud-b/repo/clip.pt"
        cloud_b = common.extract_backbone_identities(variant, "cloud-b")
        self.assertEqual(cloud_a, cloud_b)
        variant["model_provenance"]["clip"]["weight_file"]["sha256"] = "f" * 64
        self.assertNotEqual(
            cloud_a, common.extract_backbone_identities(variant, "cloud-b-tampered"))

    def test_16_stage3_clis_support_direct_file_launch_outside_repo(self):
        repo = Path(__file__).resolve().parents[2]
        scripts = (
            repo / "experiments/p6/run_p6b_formal_support_grid.py",
            repo / "experiments/p6/evaluate_p6b_formal_grid.py",
            repo / "experiments/p6/audit_p6b_formal_grid.py",
        )
        with tempfile.TemporaryDirectory() as outside_repo:
            for script in scripts:
                completed = subprocess.run(
                    [sys.executable, str(script), "--help"], cwd=outside_repo,
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(
                    completed.returncode, 0,
                    f"{script.name} direct launch failed: {completed.stderr}")
                self.assertIn("usage:", completed.stdout.lower())

    def test_17_bounded_parallel_dispatch_keeps_unit_process_isolation(self):
        parser = grid.build_parser()
        args = parser.parse_args([
            "--bundle-dir", "bundle", "--image-root", "images",
            "--output-root", "output", "--max-parallel", "3"])
        self.assertEqual(args.max_parallel, 3)
        self.assertEqual(parser.parse_args([
            "--bundle-dir", "b", "--image-root", "i",
            "--output-root", "o"]).max_parallel, 1)

        active = 0
        peak = 0
        lock = threading.Lock()
        def fake_run(command, cwd, check, env):
            nonlocal active, peak
            self.assertIn("-m", command)
            self.assertIn("experiments.p6.run_p6b_stage2_support", command)
            self.assertEqual(env["OMP_NUM_THREADS"], "2")
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return SimpleNamespace(returncode=0)

        units = list(common.expected_units())[:5]
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(grid.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(grid, "validate_support_unit"):
            with __import__("concurrent.futures").futures.ThreadPoolExecutor(
                    max_workers=3) as pool:
                futures = [pool.submit(
                    grid.run_one_unit, args, unit, unit.directory(Path(temp)),
                    "a" * 40, index, len(units))
                    for index, unit in enumerate(units, 1)]
                for future in futures:
                    future.result()
        self.assertEqual(peak, 3)

    def test_18_mixed_method_schedule_preserves_membership_and_child_env(self):
        original = list(common.expected_units())
        scheduled = grid.schedule_units(original)
        self.assertEqual({unit.key for unit in scheduled}, {unit.key for unit in original})
        self.assertEqual(len(scheduled), len(original))
        # First product/K/seed block must contain all methods instead of a long
        # clip_proto-only prefix.
        first = scheduled[:len(common.METHODS)]
        self.assertEqual({unit.method for unit in first}, set(common.METHODS))
        self.assertEqual({(unit.product, unit.k_shot, unit.seed) for unit in first},
                         {("bottle", 1, 111)})
        with mock.patch.dict(__import__("os").environ,
                             {"OMP_NUM_THREADS": "", "MKL_NUM_THREADS": "invalid",
                              "OPENBLAS_NUM_THREADS": " "}, clear=False):
            env = grid.child_environment(2)
        self.assertEqual(env["OMP_NUM_THREADS"], "2")
        self.assertEqual(env["MKL_NUM_THREADS"], "2")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "2")
        with self.assertRaises(P6BFailClosedError):
            grid.child_environment(0)

    def test_19_canonical_reassembly_is_independent_of_execution_order(self):
        canonical = [
            common.GridUnit("p30_frozen_proto", "bottle", 1, 111),
            common.GridUnit("p30_frozen_proto", "bottle", 1, 222),
        ]
        with tempfile.TemporaryDirectory() as temp:
            staging = Path(temp)
            records = {}
            # Deliberately create files in reverse execution order.
            for unit, payload in reversed(list(zip(canonical, (b"first\n", b"second\n")))):
                relative, path, _manifest = evaluator._unit_prediction_paths(staging, unit)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                records[unit.key] = {
                    "key": unit.key,
                    "predictions_relative_path": relative.as_posix(),
                    "predictions_sha256": common.file_sha256(path),
                    "prediction_manifest_relative_path": relative.with_suffix(
                        ".manifest.json").as_posix(),
                    "prediction_manifest_sha256": "a" * 64,
                    "query_count": 1,
                }
            with mock.patch.object(evaluator, "expected_units",
                                   side_effect=lambda: iter(canonical)), \
                    mock.patch.object(
                        evaluator, "_validate_unit_prediction",
                        side_effect=lambda _s, _g, unit, _q: records[unit.key]):
                path, index, rows = evaluator._write_canonical_global_predictions(
                    staging, Path("grid"), [])
            self.assertEqual(path.read_bytes(), b"first\nsecond\n")
            self.assertEqual([record["key"] for record in index],
                             [unit.key for unit in canonical])
            self.assertEqual(rows, 2)

    def test_20_resume_state_is_strict_and_late_files_are_regenerated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staging = root / "query.p6b_staging"
            output = root / "query"
            state = {
                "schema_version": evaluator.RESUME_STATE_SCHEMA,
                "status": "in_progress", "batch_size": 64,
            }
            evaluator._prepare_staging(
                staging=staging, output_dir=output, resume=False,
                expected_state=state)
            unit_file = staging / "predictions/keep.jsonl"
            unit_file.parent.mkdir(parents=True)
            unit_file.write_bytes(b"unit\n")
            for name in (evaluator.PREDICTIONS_NAME,
                         evaluator.PREDICTION_MANIFEST_NAME, evaluator.AUDIT_NAME):
                (staging / name).write_bytes(b"late\n")
            evaluator._prepare_staging(
                staging=staging, output_dir=output, resume=True,
                expected_state=state)
            self.assertTrue(unit_file.is_file())
            self.assertFalse((staging / evaluator.PREDICTIONS_NAME).exists())
            bad = dict(state, batch_size=32)
            with self.assertRaisesRegex(P6BFailClosedError, "batch_size"):
                evaluator._prepare_staging(
                    staging=staging, output_dir=output, resume=True,
                    expected_state=bad)

    def test_21_evaluator_preserves_staging_after_runtime_failure(self):
        unit = common.GridUnit("clip_proto", "bottle", 1, 111)
        query = {"product": "bottle", "rgb_sha256": "1" * 64,
                 "true_label": "a", "class_order": ["a"]}

        class FailingPredictor:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, *args, **kwargs):
                raise RuntimeError("injected accelerator interruption")

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            grid_root = root / "grid"
            unit_dir = unit.directory(grid_root)
            unit_dir.mkdir(parents=True)
            atomic_write_json(unit_dir / common.SUPPORT_COMPLETION,
                              {"class_order": ["a"]})
            records = root / "records.jsonl"
            records.write_bytes(b"records\n")
            stage1 = root / "stage1"
            stage1.mkdir()
            (stage1 / "p6b_manifest_build_complete.json").write_bytes(b"stage1\n")
            output = root / "query"
            with mock.patch.object(evaluator, "validate_formal_grid",
                                   return_value={"execution_commit": "a" * 40}), \
                    mock.patch.object(evaluator.builder, "git_commit_and_dirty",
                                      return_value=("b" * 40, False)), \
                    mock.patch.object(evaluator, "load_query_contract",
                                      return_value=([query], {"ok": True})), \
                    mock.patch.object(evaluator, "EXPECTED_FORMAL_QUERY_COUNT", 1), \
                    mock.patch.object(evaluator, "METHODS", ("clip_proto",)), \
                    mock.patch.object(evaluator, "expected_units",
                                      side_effect=lambda: iter([unit])), \
                    mock.patch.object(evaluator, "MethodPredictor", FailingPredictor):
                with self.assertRaisesRegex(RuntimeError, "accelerator interruption"):
                    evaluator.evaluate(
                        grid_root=grid_root, records_path=records,
                        dataset_root=root, stage1_bundle=stage1,
                        output_dir=output, device="cpu", batch_size=64)
            staging = output.with_name(output.name + ".p6b_staging")
            self.assertTrue(staging.is_dir())
            state = common.load_json(staging / evaluator.RESUME_STATE_NAME)
            self.assertEqual(state["support_grid_execution_commit"], "a" * 40)
            self.assertEqual(state["evaluator_execution_commit"], "b" * 40)
            self.assertFalse(output.exists())

    def test_22_parallel_method_dispatch_uses_isolated_subprocesses(self):
        active = 0
        peak = 0
        commands = []
        lock = threading.Lock()

        def fake_run(command, cwd, check):
            nonlocal active, peak
            self.assertEqual(cwd, str(evaluator.PROJECT_ROOT))
            self.assertFalse(check)
            with lock:
                commands.append(command)
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return SimpleNamespace(returncode=0)

        methods = ("ours", "p30_frozen_proto", "linear_probe", "fusion_proto")
        with mock.patch.object(evaluator.subprocess, "run", side_effect=fake_run):
            evaluator._run_parallel_methods(
                methods=methods, max_parallel_methods=4,
                grid_root="grid", records_path="records", dataset_root="dataset",
                stage1_bundle="stage1", output_dir="output", device="cuda:0",
                batch_size=64)
        self.assertEqual(peak, 4)
        self.assertEqual(len(commands), 4)
        selected = []
        for command in commands:
            self.assertIn("--worker-method", command)
            self.assertIn("--batch-size", command)
            self.assertEqual(command[command.index("--batch-size") + 1], "64")
            selected.append(command[command.index("--worker-method") + 1])
        self.assertEqual(set(selected), set(methods))
        with self.assertRaises(P6BFailClosedError):
            evaluator._run_parallel_methods(
                methods=methods, max_parallel_methods=0,
                grid_root="grid", records_path="records", dataset_root="dataset",
                stage1_bundle="stage1", output_dir="output", device="cuda:0",
                batch_size=64)


if __name__ == "__main__":
    unittest.main()
