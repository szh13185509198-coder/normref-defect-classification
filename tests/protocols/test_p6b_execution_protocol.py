import ast
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6 import build_p6b_support_manifests as builder
from experiments.p6 import p6b_execution_common as common


def fake_sha(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def make_record(product, rel_path, pools, defect_types):
    sha = fake_sha(rel_path)
    return {
        "schema_version": "p6b_mvtec_fs_rgb_image_record_v2",
        "product": product,
        "original_image_id": sha,
        "rgb_sha256": sha,
        "canonical_relative_path": rel_path,
        "alias_paths": [rel_path],
        "source_pools": sorted(pools),
        "annotation_paths": [],
        "instance_count": 1,
        "unique_defect_types": sorted(defect_types),
        "multi_type_removed": len(defect_types) > 1,
        "same_type_multi_instance": False,
    }


def synthetic_fixture(root: Path, products=common.LOCKED_PRODUCTS,
                      support_per_class=6, query_per_class=2,
                      classes=("clsA", "clsB"), drop_product=None,
                      overlap_product=None, short_class=None):
    """Write a self-consistent frozen feasibility dir (JSONL + feasibility
    JSON, hash-pinned). No image files are ever created: Stage 1 must not
    need any pixels."""
    records = []
    included = []
    for product in products:
        if product == drop_product:
            continue
        included.append(product)
        for cls in classes:
            count = support_per_class
            if short_class == (product, cls):
                count = 2
            for i in range(count):
                records.append(make_record(
                    product, f"image/{product}/training/{cls}/{i:03d}.png",
                    ["support"], [cls]))
            for j in range(query_per_class):
                if overlap_product == product and j == 0:
                    # Same RGB content present in both pools (cross-pool record).
                    records.append(make_record(
                        product, f"image/{product}/training/{cls}/000.png",
                        ["support", "query"], [cls]))
                    continue
                records.append(make_record(
                    product, f"image/{product}/testing/{cls}/{j:03d}.png",
                    ["query"], [cls]))
    records.sort(key=lambda r: (r["product"], r["rgb_sha256"]))

    root.mkdir(parents=True, exist_ok=True)
    jsonl_bytes = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records
    ).encode("utf-8")
    (root / "p6b_mvtec_fs_rgb_image_records.jsonl").write_bytes(jsonl_bytes)
    feasibility = {
        "schema_version": "p6b_mvtec_fs_rgb_feasibility_v2",
        "probe_status": "INCONCLUSIVE",
        "formal_test_used": False,
        "cable_reference_used_for_overlap_only": True,
        "feasible_products_by_shot": {
            "1": sorted(included), "3": sorted(included), "5": sorted(included),
        },
        "common_products_feasible_for_all_shots": sorted(included),
        "output_hashes": {
            "p6b_mvtec_fs_rgb_image_records.jsonl":
                hashlib.sha256(jsonl_bytes).hexdigest(),
        },
    }
    (root / "p6b_mvtec_fs_rgb_feasibility.json").write_text(
        json.dumps(feasibility, sort_keys=True) + "\n", encoding="utf-8",
        newline="\n")
    return root


class TestP6BExecutionProtocol(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.feasibility_dir = self.temp_dir / "feasibility"
        self.output_dir = self.temp_dir / "out"
        self.script = (PROJECT_ROOT / "experiments" / "p6" /
                       "build_p6b_support_manifests.py")

    def tearDown(self):
        # git object files can be read-only on Windows; make everything
        # writable before removal (works across Python versions).
        for root, dirs, files in os.walk(self.temp_dir):
            for name in files + dirs:
                try:
                    os.chmod(os.path.join(root, name), stat.S_IWRITE)
                except OSError:
                    pass
        shutil.rmtree(self.temp_dir)

    def run_builder(self, feasibility_dir=None, output_dir=None,
                    allow_dirty=True):
        cmd = [sys.executable, str(self.script),
               "--feasibility-dir", str(feasibility_dir or self.feasibility_dir),
               "--output-dir", str(output_dir or self.output_dir)]
        if allow_dirty:
            cmd.append("--allow-dirty")
        return subprocess.run(cmd, capture_output=True, text=True)

    def read_completion(self, output_dir=None):
        return json.loads(((output_dir or self.output_dir) /
                           "p6b_manifest_build_complete.json")
                          .read_text(encoding="utf-8"))

    # 1. full manifest count: 13 x 3 x 5 = 195 --------------------------------
    def test_01_full_build_produces_195_manifests(self):
        self.assertEqual(common.EXPECTED_MANIFEST_COUNT, 195)
        synthetic_fixture(self.feasibility_dir)
        res = self.run_builder()
        self.assertEqual(res.returncode, 0, res.stderr)
        manifests = list((self.output_dir / "p6b_support_manifests").rglob("*.json"))
        self.assertEqual(len(manifests), 195)
        completion = self.read_completion()
        self.assertEqual(completion["manifest_count"], 195)
        # --allow-dirty is the explicit diagnostic switch: never disguised as
        # a formal completed build.
        self.assertEqual(completion["status"], "diagnostic")
        self.assertFalse(completion["formal_eligible"])
        self.assertEqual(sorted(completion["products"]),
                         sorted(common.LOCKED_PRODUCTS))

    # 2. deterministic second run byte-identical ------------------------------
    def test_02_second_run_byte_identical(self):
        synthetic_fixture(self.feasibility_dir)
        first = self.temp_dir / "out1"
        second = self.temp_dir / "out2"
        self.assertEqual(self.run_builder(output_dir=first).returncode, 0)
        self.assertEqual(self.run_builder(output_dir=second).returncode, 0)
        first_files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
        second_files = sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
        self.assertEqual(first_files, second_files)
        for rel in first_files:
            self.assertEqual((first / rel).read_bytes(), (second / rel).read_bytes(),
                             f"non-deterministic output: {rel}")

    # 3. nested K: K=1 subset K=3 subset K=5 ----------------------------------
    def test_03_nested_k_prefix_property(self):
        synthetic_fixture(self.feasibility_dir)
        self.assertEqual(self.run_builder().returncode, 0)
        product = "bottle"
        for seed in common.LOCKED_SEEDS:
            m1 = json.loads((self.output_dir / "p6b_support_manifests" / product /
                             f"k1_s{seed}.json").read_text(encoding="utf-8"))
            m3 = json.loads((self.output_dir / "p6b_support_manifests" / product /
                             f"k3_s{seed}.json").read_text(encoding="utf-8"))
            m5 = json.loads((self.output_dir / "p6b_support_manifests" / product /
                             f"k5_s{seed}.json").read_text(encoding="utf-8"))
            for cls in m1["class_universe"]:
                s1 = [item["rgb_sha256"] for item in m1["classes"][cls]["items"]]
                s3 = [item["rgb_sha256"] for item in m3["classes"][cls]["items"]]
                s5 = [item["rgb_sha256"] for item in m5["classes"][cls]["items"]]
                self.assertEqual(s3[:1], s1)
                self.assertEqual(s5[:3], s3)
                self.assertEqual(len(set(s5)), 5)  # without replacement

    # 4. class/product seed isolation ------------------------------------------
    def test_04_seed_isolation_across_class_and_product(self):
        base = common.sub_seed(111, "bottle", "clsA")
        self.assertNotEqual(base, common.sub_seed(222, "bottle", "clsA"))
        self.assertNotEqual(base, common.sub_seed(111, "capsule", "clsA"))
        self.assertNotEqual(base, common.sub_seed(111, "bottle", "clsB"))

        # Perturbing one class's candidate set must not change another class's
        # draw (per-class independent sub_seed streams).
        records_a = [make_record("bottle", f"image/bottle/training/clsA/{i:03d}.png",
                                 ["support"], ["clsA"]) for i in range(6)]
        records_a += [make_record("bottle", f"image/bottle/training/clsB/{i:03d}.png",
                                  ["support"], ["clsB"]) for i in range(6)]
        records_a = [
            {k: r[k] for k in common.SUPPORT_RECORD_FIELDS} for r in records_a]
        draw_before = common.build_support_manifest(records_a, "bottle", 3, 111)
        extra = make_record("bottle", "image/bottle/training/clsB/extra.png",
                            ["support"], ["clsB"])
        records_b = records_a + [
            {k: extra[k] for k in common.SUPPORT_RECORD_FIELDS}]
        draw_after = common.build_support_manifest(records_b, "bottle", 3, 111)
        self.assertEqual(draw_before["classes"]["clsA"]["items"],
                         draw_after["classes"]["clsA"]["items"])
        # same product/seed/class keeps the same sub_seed regardless of pool size
        self.assertEqual(common.sub_seed(111, "bottle", "clsB"),
                         common.sub_seed(111, "bottle", "clsB"))

    # 5. canonical self-hash ----------------------------------------------------
    def test_05_canonical_self_hash_excludes_sha_field(self):
        records = [make_record("bottle", f"image/bottle/training/clsA/{i:03d}.png",
                               ["support"], ["clsA"]) for i in range(4)]
        records = [{k: r[k] for k in common.SUPPORT_RECORD_FIELDS} for r in records]
        manifest = common.finalize_manifest(
            common.build_support_manifest(records, "bottle", 1, 111))
        self.assertEqual(manifest["manifest_sha256"],
                         common.manifest_self_sha256(manifest))
        tampered = dict(manifest)
        tampered["manifest_sha256"] = "0" * 64
        self.assertEqual(common.manifest_self_sha256(tampered),
                         manifest["manifest_sha256"])
        body_bytes = common.canonical_json_bytes(
            {k: v for k, v in manifest.items() if k != "manifest_sha256"})
        self.assertTrue(body_bytes.endswith(b"\n"))
        self.assertEqual(hashlib.sha256(body_bytes).hexdigest(),
                         manifest["manifest_sha256"])

    # 6. overlap fail-closed -----------------------------------------------------
    def test_06_support_query_overlap_fail_closed(self):
        records = [
            make_record("bottle", "image/bottle/training/clsA/000.png",
                        ["support"], ["clsA"]),
            make_record("bottle", "image/bottle/testing/clsA/000.png",
                        ["support", "query"], ["clsA"]),
        ]
        records = [{k: r[k] for k in (
            common.QUERY_RECORD_FIELDS if "query" in r["source_pools"]
            else common.SUPPORT_RECORD_FIELDS)} for r in records]
        with self.assertRaises(common.P6BFailClosedError):
            common.support_candidates_by_class(records, "bottle")

    # 7. missing product / class / K / seed fail-closed --------------------------
    def test_07_missing_product_class_k_seed_fail_closed(self):
        synthetic_fixture(self.feasibility_dir)
        records = common.load_frozen_records(
            self.feasibility_dir / "p6b_mvtec_fs_rgb_image_records.jsonl")
        # missing product: no records at all
        with self.assertRaises(common.P6BFailClosedError):
            common.build_support_manifest(records, "nonexistent", 1, 111)
        # excluded products are hard fail-closed
        with self.assertRaises(common.P6BFailClosedError):
            common.build_support_manifest(records, "cable", 1, 111)
        with self.assertRaises(common.P6BFailClosedError):
            common.build_support_manifest(records, "toothbrush", 1, 111)
        # class with fewer candidates than K
        short = [make_record("bottle", f"image/bottle/training/clsA/{i:03d}.png",
                             ["support"], ["clsA"]) for i in range(2)]
        short = [{k: r[k] for k in common.SUPPORT_RECORD_FIELDS} for r in short]
        with self.assertRaises(common.P6BFailClosedError):
            common.build_support_manifest(short, "bottle", 5, 111)
        # query-only classes never enter the support-side universe
        mixed = short + [make_record(
            "bottle", "image/bottle/testing/clsB/000.png", ["query"], ["clsB"])]
        mixed = [{k: r[k] for k in (
            common.QUERY_RECORD_FIELDS if "query" in r["source_pools"]
            else common.SUPPORT_RECORD_FIELDS)} for r in mixed]
        manifest = common.build_support_manifest(mixed, "bottle", 1, 111)
        self.assertEqual(manifest["class_universe"], ["clsA"])
        # invalid K / seed
        with self.assertRaises(common.P6BFailClosedError):
            common.build_support_manifest(records, "bottle", 2, 111)
        with self.assertRaises(common.P6BFailClosedError):
            common.build_support_manifest(records, "bottle", 1, 999)
        # missing product end-to-end: feasibility gate fails (set != locked 13),
        # builder exits non-zero and publishes nothing
        missing_dir = self.temp_dir / "feasibility_missing"
        synthetic_fixture(missing_dir, drop_product="zipper")
        out = self.temp_dir / "out_missing"
        res = self.run_builder(feasibility_dir=missing_dir, output_dir=out)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(out.exists())

    # 8. query firewall: field minimization + tamper invariance -------------------
    def test_08_query_tamper_invariance_real_dataflow(self):
        synthetic_fixture(self.feasibility_dir)
        jsonl_path = self.feasibility_dir / "p6b_mvtec_fs_rgb_image_records.jsonl"
        # (a) loaded query records expose ONLY the overlap-audit triple
        loaded = common.load_frozen_records(jsonl_path)
        query_records = [r for r in loaded if "query" in r["source_pools"]]
        self.assertTrue(query_records)
        for record in query_records:
            self.assertEqual(set(record.keys()),
                             {"product", "source_pools", "rgb_sha256"})
        # (b) clean build
        out_clean = self.temp_dir / "out_clean"
        self.assertEqual(self.run_builder(output_dir=out_clean).returncode, 0)
        # (c) tamper query-side labels/paths in a self-consistent copy
        tampered_dir = self.temp_dir / "feasibility_tampered"
        tampered_dir.mkdir()
        tampered_lines = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if "query" in raw["source_pools"]:
                raw["unique_defect_types"] = ["tampered_label"]
                raw["canonical_relative_path"] = "TAMPERED/" + raw["canonical_relative_path"]
                raw["alias_paths"] = ["TAMPERED/alias"]
                raw["annotation_paths"] = ["TAMPERED/ann"]
            tampered_lines.append(json.dumps(raw, sort_keys=True) + "\n")
        tampered_bytes = "".join(tampered_lines).encode("utf-8")
        (tampered_dir / "p6b_mvtec_fs_rgb_image_records.jsonl").write_bytes(tampered_bytes)
        feasibility = json.loads(
            (self.feasibility_dir / "p6b_mvtec_fs_rgb_feasibility.json")
            .read_text(encoding="utf-8"))
        feasibility["output_hashes"]["p6b_mvtec_fs_rgb_image_records.jsonl"] = (
            hashlib.sha256(tampered_bytes).hexdigest())
        (tampered_dir / "p6b_mvtec_fs_rgb_feasibility.json").write_text(
            json.dumps(feasibility, sort_keys=True) + "\n", encoding="utf-8",
            newline="\n")
        out_tampered = self.temp_dir / "out_tampered"
        self.assertEqual(
            self.run_builder(feasibility_dir=tampered_dir,
                             output_dir=out_tampered).returncode, 0)
        # (d) all 195 manifests must be byte-identical regardless of query
        # labels/paths — proves the real data flow never semantically consumes
        # them (JSONL lines are parsed, but those fields are not accessed)
        clean_manifests = sorted(
            (out_clean / "p6b_support_manifests").rglob("*.json"))
        self.assertEqual(len(clean_manifests), 195)
        for path in clean_manifests:
            twin = out_tampered / "p6b_support_manifests" / path.relative_to(
                out_clean / "p6b_support_manifests")
            self.assertTrue(twin.is_file(), f"missing {twin}")
            self.assertEqual(path.read_bytes(), twin.read_bytes(),
                             f"manifest changed under query tamper: {path.name}")

    # 9. LF newlines + atomic write ----------------------------------------------
    def test_09_lf_and_atomic_write(self):
        synthetic_fixture(self.feasibility_dir)
        self.assertEqual(self.run_builder().returncode, 0)
        outputs = [p for p in self.output_dir.rglob("*") if p.is_file()]
        self.assertTrue(outputs)
        for path in outputs:
            self.assertNotIn(b"\r", path.read_bytes(), f"CRLF found in {path}")
        self.assertEqual(list(self.temp_dir.rglob("*.tmp")), [])
        self.assertEqual(list(self.temp_dir.rglob("*" + builder.STAGING_SUFFIX)), [])

    # 10. Formal Test never used ---------------------------------------------------
    def test_10_formal_test_not_used_and_fail_closed(self):
        synthetic_fixture(self.feasibility_dir)
        self.assertEqual(self.run_builder().returncode, 0)
        self.assertFalse(self.read_completion()["formal_test_used"])
        # feasibility JSON claiming formal_test_used=true must fail-closed
        bad_dir = self.temp_dir / "feasibility_bad"
        synthetic_fixture(bad_dir)
        feasibility_path = bad_dir / "p6b_mvtec_fs_rgb_feasibility.json"
        feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
        feasibility["formal_test_used"] = True
        feasibility_path.write_text(json.dumps(feasibility, sort_keys=True) + "\n",
                                    encoding="utf-8", newline="\n")
        out = self.temp_dir / "out_bad"
        res = self.run_builder(feasibility_dir=bad_dir, output_dir=out)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(out.exists())

    # 11. no torch / CLIP / DINO / FAISS ------------------------------------------
    def test_11_no_model_imports(self):
        code = (
            "import ast, importlib, pathlib, sys; "
            "mods=[importlib.import_module('experiments.p6.p6b_execution_common'),"
            "importlib.import_module('experiments.p6.build_p6b_support_manifests')]; "
            "forbidden={'torch','clip','faiss','dinov2','transformers'}; "
            "assert 'torch' not in sys.modules; "
            "trees=[ast.parse(pathlib.Path(m.__file__).read_text(encoding='utf-8')) "
            "for m in mods]; "
            "names={a.name.split('.')[0] for t in trees for n in ast.walk(t) "
            "if isinstance(n,ast.Import) for a in n.names}; "
            "names|={n.module.split('.')[0] for t in trees for n in ast.walk(t) "
            "if isinstance(n,ast.ImportFrom) and n.module}; "
            "assert not (names & forbidden), names & forbidden; "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=PROJECT_ROOT,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    # 12. frozen-source gate fail-closed --------------------------------------------
    def test_12_frozen_source_gate_fail_closed(self):
        # (a) JSONL byte tamper -> sha mismatch with feasibility.output_hashes
        synthetic_fixture(self.feasibility_dir)
        jsonl_path = self.feasibility_dir / "p6b_mvtec_fs_rgb_image_records.jsonl"
        jsonl_path.write_bytes(jsonl_path.read_bytes() + b" ")
        out_a = self.temp_dir / "out_gate_a"
        res = self.run_builder(output_dir=out_a)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("FAIL-CLOSED", res.stderr)
        self.assertFalse(out_a.exists())
        # (b) feasible-product set tamper -> exact-match gate fails
        gate_dir = self.temp_dir / "feasibility_gate"
        synthetic_fixture(gate_dir)
        feasibility_path = gate_dir / "p6b_mvtec_fs_rgb_feasibility.json"
        feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
        feasibility["common_products_feasible_for_all_shots"].append("toothbrush")
        feasibility_path.write_text(json.dumps(feasibility, sort_keys=True) + "\n",
                                    encoding="utf-8", newline="\n")
        out_b = self.temp_dir / "out_gate_b"
        res = self.run_builder(feasibility_dir=gate_dir, output_dir=out_b)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(out_b.exists())
        # (c) single-shot feasible list tamper -> exact-match gate fails
        shot_dir = self.temp_dir / "feasibility_shot"
        synthetic_fixture(shot_dir)
        shot_feasibility_path = shot_dir / "p6b_mvtec_fs_rgb_feasibility.json"
        shot_feasibility = json.loads(shot_feasibility_path.read_text(encoding="utf-8"))
        shot_feasibility["feasible_products_by_shot"]["3"].remove("bottle")
        shot_feasibility_path.write_text(
            json.dumps(shot_feasibility, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        out_c = self.temp_dir / "out_gate_c"
        res = self.run_builder(feasibility_dir=shot_dir, output_dir=out_c)
        self.assertNotEqual(res.returncode, 0)
        self.assertFalse(out_c.exists())

    # 13. transactional bundle: late injection leaves nothing -------------------------
    def test_13_transactional_late_injection_leaves_nothing(self):
        synthetic_fixture(self.feasibility_dir)
        target = self.temp_dir / "out_tx"
        staging = target.parent / (target.name + builder.STAGING_SUFFIX)
        original = builder.atomic_write_json
        calls = {"n": 0}

        def explode(path, obj):
            calls["n"] += 1
            if calls["n"] == 150:
                raise RuntimeError("injected late failure")
            original(path, obj)

        builder.atomic_write_json = explode
        try:
            with self.assertRaises(RuntimeError):
                builder.build_bundle(self.feasibility_dir, target, PROJECT_ROOT,
                                     diagnostic=True)
        finally:
            builder.atomic_write_json = original
        self.assertGreaterEqual(calls["n"], 150)
        self.assertFalse(target.exists(), "official dir must never appear")
        self.assertFalse(staging.exists(), "staging must be cleaned up")

    # 14. existing output directory rejected ---------------------------------------------
    def test_14_existing_output_dir_rejected(self):
        synthetic_fixture(self.feasibility_dir)
        target = self.temp_dir / "out_existing"
        target.mkdir()
        sentinel = target / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(common.P6BFailClosedError):
            builder.build_bundle(self.feasibility_dir, target, PROJECT_ROOT,
                                 diagnostic=True)
        self.assertTrue(sentinel.exists(), "pre-existing dir must be untouched")

    # 15. formal git preflight on an isolated temp repo -----------------------------
    def test_15_formal_preflight_on_isolated_repo(self):
        # Independent of the main repository's dirty/clean state: build_bundle
        # is pointed at a dedicated temp git repo via its project_root argument.
        synthetic_fixture(self.feasibility_dir)
        repo = self.temp_dir / "git_repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True,
                       capture_output=True)
        (repo / "seed_file.txt").write_text("seed", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=repo, check=True,
                       capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=p6b@test.local", "-c", "user.name=p6b",
             "commit", "-q", "-m", "init"],
            cwd=repo, check=True, capture_output=True)
        # (a) clean repo + formal (diagnostic=False) succeeds as completed
        target_clean = self.temp_dir / "out_formal_clean"
        completion = builder.build_bundle(self.feasibility_dir, target_clean,
                                          repo, diagnostic=False)
        self.assertEqual(completion["status"], "completed")
        self.assertTrue(completion["formal_eligible"])
        self.assertFalse(completion["project_provenance"]["dirty"])
        self.assertEqual(completion["manifest_count"], 195)
        # (b) same repo made dirty -> formal build fail-closed
        (repo / "untracked_dirty.txt").write_text("dirty", encoding="utf-8")
        target_dirty = self.temp_dir / "out_formal_dirty"
        with self.assertRaises(common.P6BFailClosedError):
            builder.build_bundle(self.feasibility_dir, target_dirty, repo,
                                 diagnostic=False)
        self.assertFalse(target_dirty.exists())


if __name__ == "__main__":
    unittest.main()
