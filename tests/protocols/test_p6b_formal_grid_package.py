from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

from experiments.p6.p6b_execution_common import P6BFailClosedError
from experiments.p6 import package_p6b_formal_grid as package
from experiments.p6.p6b_stage3_common import GridUnit, file_sha256


class P6BFormalShardPackageTest(unittest.TestCase):
    def _fake_completion(self, unit_dir: Path):
        artifact = unit_dir / "model.pt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"weights")
        value = {"execution": {"commit": "a" * 40}, "artifacts": [{
            "relative_path": "model.pt",
            "size_bytes": artifact.stat().st_size,
            "sha256": file_sha256(artifact),
        }]}
        completion = unit_dir / package.SUPPORT_COMPLETION
        completion.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return value

    def test_plan_contains_only_validated_registered_files(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "p6b_formal_support_grid_v3"
            unit = GridUnit("clip_proto", "bottle", 1, 111)
            completion = self._fake_completion(unit.directory(root))
            with mock.patch.object(package, "selected_units", return_value=[unit]), \
                    mock.patch.object(package.builder, "git_commit_and_dirty",
                                      return_value=("a" * 40, False)), \
                    mock.patch.object(package, "validate_support_unit",
                                      return_value=completion):
                plan = package.build_package_plan(root, 2, 0)
            self.assertEqual(plan["unit_count"], 1)
            self.assertEqual(plan["file_count"], 2)
            self.assertEqual(plan["execution_commit"], "a" * 40)
            self.assertEqual(plan["packaging_commit"], "a" * 40)
            self.assertFalse(plan["formal_test_used"])
            self.assertFalse(plan["query_pixels_read"])
            self.assertEqual(
                {row["relative_path"] for row in plan["files"]},
                {f"{unit.key}/{package.SUPPORT_COMPLETION}", f"{unit.key}/model.pt"},
            )

    def test_dirty_tree_and_global_completion_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "grid"
            root.mkdir()
            with mock.patch.object(package.builder, "git_commit_and_dirty",
                                   return_value=("a" * 40, True)):
                with self.assertRaisesRegex(P6BFailClosedError, "clean git tree"):
                    package.build_package_plan(root, 2, 0)
            (root / package.GRID_COMPLETION).write_text("{}\n", encoding="utf-8")
            with mock.patch.object(package.builder, "git_commit_and_dirty",
                                   return_value=("a" * 40, False)):
                with self.assertRaisesRegex(P6BFailClosedError, "package the full grid"):
                    package.build_package_plan(root, 2, 0)

    def test_unregistered_file_is_excluded_and_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "grid"
            unit = GridUnit("clip_proto", "bottle", 1, 111)
            completion = self._fake_completion(unit.directory(root))
            (unit.directory(root) / "debug.tmp").write_bytes(b"not packaged")
            patches = (
                mock.patch.object(package, "selected_units", return_value=[unit]),
                mock.patch.object(package.builder, "git_commit_and_dirty",
                                  return_value=("a" * 40, False)),
                mock.patch.object(package, "validate_support_unit", return_value=completion),
            )
            with patches[0], patches[1], patches[2]:
                plan = package.build_package_plan(root, 2, 0)
            self.assertNotIn("debug.tmp", {r["relative_path"] for r in plan["files"]})

            outside = unit.directory(root).parent / "outside.pt"
            outside.write_bytes(b"escape")
            completion["artifacts"] = [{
                "relative_path": "../outside.pt", "size_bytes": outside.stat().st_size,
                "sha256": file_sha256(outside),
            }]
            with mock.patch.object(package, "selected_units", return_value=[unit]), \
                    mock.patch.object(package.builder, "git_commit_and_dirty",
                                      return_value=("a" * 40, False)), \
                    mock.patch.object(package, "validate_support_unit",
                                      return_value=completion):
                with self.assertRaisesRegex(P6BFailClosedError, "escapes unit"):
                    package.build_package_plan(root, 2, 0)

    def test_archive_constraints_fail_before_tar(self):
        plan = {"files": [], "unit_count": 0, "file_count": 0}
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "grid"
            root.mkdir()
            with self.assertRaisesRegex(P6BFailClosedError, "\.tar\.zst"):
                package.create_archive(root, Path(temp_name) / "bad.zip", plan, 3)
            with self.assertRaisesRegex(P6BFailClosedError, "outside the grid"):
                package.create_archive(root, root / "bad.tar.zst", plan, 3)
            with self.assertRaisesRegex(P6BFailClosedError, "compression level"):
                package.create_archive(root, Path(temp_name) / "bad.tar.zst", plan, 20)

    def test_full_grid_requires_global_validation_and_includes_completion(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "grid"
            unit = GridUnit("clip_proto", "bottle", 1, 111)
            completion = self._fake_completion(unit.directory(root))
            global_path = root / package.GRID_COMPLETION
            global_path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(package.builder, "git_commit_and_dirty",
                                   return_value=("a" * 40, False)), \
                    mock.patch.object(package, "validate_formal_grid",
                                      return_value={"execution_commit": "b" * 40}), \
                    mock.patch.object(package, "expected_units", return_value=[unit]), \
                    mock.patch.object(package, "validate_support_unit",
                                      return_value=completion):
                plan = package.build_package_plan(root)
            self.assertEqual(plan["package_scope"], "full_grid")
            self.assertEqual(plan["execution_commit"], "b" * 40)
            self.assertEqual(plan["packaging_commit"], "a" * 40)
            self.assertIsNone(plan["shard_index"])
            self.assertIn(package.GRID_COMPLETION,
                          {row["relative_path"] for row in plan["files"]})

    def test_partial_shard_arguments_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name) / "grid"
            root.mkdir()
            with mock.patch.object(package.builder, "git_commit_and_dirty",
                                   return_value=("a" * 40, False)):
                with self.assertRaisesRegex(P6BFailClosedError, "supplied together"):
                    package.build_package_plan(root, shard_count=2)


if __name__ == "__main__":
    unittest.main()
