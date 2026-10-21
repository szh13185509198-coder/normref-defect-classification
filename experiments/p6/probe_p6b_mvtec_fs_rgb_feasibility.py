import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import tempfile

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def get_git_provenance(git_dir: Path):
    provenance = {"git_commit": None, "git_is_dirty": None}
    if (git_dir / ".git").exists() or git_dir.joinpath(".git").is_file():
        import subprocess
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_dir, text=True, stderr=subprocess.DEVNULL).strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=git_dir, text=True, stderr=subprocess.DEVNULL).strip() != ""
            provenance["git_commit"] = commit
            provenance["git_is_dirty"] = dirty
        except Exception:
            pass
    return provenance

def atomic_write_json(path: Path, data: dict):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False, newline="\n") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            temp_path = f.name
        os.replace(temp_path, path)
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def atomic_write_jsonl(path: Path, records: list):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False, newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            temp_path = f.name
        os.replace(temp_path, path)
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def atomic_write_text(path: Path, text: str):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False, newline="\n") as f:
            f.write(text)
            temp_path = f.name
        os.replace(temp_path, path)
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

class ProductContext:
    def __init__(self):
        self.integrity_pass = True
        self.integrity_failures = []
        self.csv_rows = []
        self.rgb_records = {} # sha256 -> RGBRecord
        self.classes = set()
        self.cross_pool_collision_count = 0
        self.img_id_map = {} # img_id -> (normalized_path, sha256)

class RGBRecord:
    def __init__(self, sha256, product):
        self.sha256 = sha256
        self.product = product
        self.canonical_relative_path = None
        self.alias_paths = set()
        self.source_pools = set()
        self.annotation_paths = set()
        self.instances = [] # list of lists of labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cable-reference-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    cable_ref_root = Path(args.cable_reference_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    project_root = Path(__file__).resolve().parent.parent.parent

    if dataset_root in output_dir.parents or dataset_root == output_dir:
        print("output-dir cannot be inside dataset-root")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_integrity_failures = []

    project_provenance = get_git_provenance(project_root)
    dataset_provenance = get_git_provenance(dataset_root)
    dataset_provenance["readme_sha256"] = None
    dataset_provenance["license_sha256"] = None
    dataset_provenance["csv_hashes"] = {}

    readme_path = dataset_root / "readme.md"
    license_path = dataset_root / "license.txt"
    if readme_path.exists():
        dataset_provenance["readme_sha256"] = file_sha256(readme_path)
    else:
        dataset_integrity_failures.append("Missing readme.md")

    if license_path.exists():
        dataset_provenance["license_sha256"] = file_sha256(license_path)
    else:
        dataset_integrity_failures.append("Missing license.txt")

    image_dir = dataset_root / "image"
    config_dir = dataset_root / "CONFIG"

    if not image_dir.exists():
        dataset_integrity_failures.append("Missing image/ directory in dataset-root")
    if not config_dir.exists():
        dataset_integrity_failures.append("Missing CONFIG/ directory in dataset-root")

    cable_hashes = set()
    cable_paths = set()
    has_cable_images = False
    if not cable_ref_root.exists() or not any(cable_ref_root.iterdir()):
        dataset_integrity_failures.append("cable-reference-root is missing or empty")
    else:
        for root, dirs, files in os.walk(cable_ref_root):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    has_cable_images = True
                    p = Path(root) / f
                    try:
                        cable_paths.add(p.relative_to(dataset_root).as_posix())
                    except ValueError:
                        cable_paths.add(p.relative_to(cable_ref_root).as_posix())
                    cable_hashes.add(file_sha256(p))
    if not has_cable_images:
        dataset_integrity_failures.append("cable-reference-root contains no supported image formats")

    products = {}
    if config_dir.exists():
        for prod_config_dir in sorted(config_dir.iterdir()):
            if not prod_config_dir.is_dir() or not prod_config_dir.name.endswith("_config1"):
                continue
            product_name = prod_config_dir.name.replace("_config1", "")
            pctx = ProductContext()
            products[product_name] = pctx

            for csv_name, expected_pool, expected_prefix in [("train.csv", "support", "training"), ("valid.csv", "query", "testing")]:
                csv_path = prod_config_dir / csv_name
                if not csv_path.exists():
                    pctx.integrity_failures.append(f"Missing {csv_name} for product {product_name}")
                    continue

                dataset_provenance["csv_hashes"][csv_path.relative_to(dataset_root).as_posix()] = file_sha256(csv_path)

                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        req_cols = {"img_id", "part", "id", "img_rel_path", "label"}
                        if not req_cols.issubset(set(reader.fieldnames or [])):
                            pctx.integrity_failures.append(f"Missing required columns in {csv_name}: expected {req_cols}")
                            continue

                        for row in reader:
                            # Non-empty validation
                            has_empty = False
                            for col in req_cols:
                                if not str(row[col]).strip():
                                    has_empty = True
                                    pctx.integrity_failures.append(f"Empty required column '{col}' in {csv_name}")
                                    break
                            if has_empty:
                                continue

                            row['__source_pool'] = expected_pool
                            row['__expected_prefix'] = expected_prefix
                            pctx.csv_rows.append(row)
                            label_val = str(row['label']).strip()
                            if label_val:
                                pctx.classes.add(label_val)
                except Exception as e:
                    pctx.integrity_failures.append(f"Failed to read {csv_name}: {e}")

    if not products:
        dataset_integrity_failures.append("No valid product config directories found in CONFIG")

    for product_name, pctx in products.items():
        if pctx.integrity_failures:
            pctx.integrity_pass = False

        product_image_dir = (image_dir / product_name).resolve()
        seen_instance_ids = set()

        rows_by_img_id = defaultdict(list)
        for row in pctx.csv_rows:
            rows_by_img_id[row['img_id']].append(row)

        for img_id, rows in rows_by_img_id.items():
            first_row = rows[0]
            expected_pool = first_row['__source_pool']
            expected_prefix = first_row['__expected_prefix']
            img_rel_path_str = first_row['img_rel_path']
            normalized_rel_path = img_rel_path_str.replace('\\', '/')

            for row in rows:
                if row['part'] != product_name:
                    pctx.integrity_failures.append(f"CSV part '{row['part']}' does not match product '{product_name}'")
                inst_id = row['id']
                if inst_id in seen_instance_ids:
                    pctx.integrity_failures.append(f"Duplicate instance id '{inst_id}' in product '{product_name}'")
                seen_instance_ids.add(inst_id)

                # Verify all rows for this img_id point to the same normalized_rel_path
                if row['img_rel_path'].replace('\\', '/') != normalized_rel_path:
                    pctx.integrity_failures.append(f"Inconsistent img_id {img_id}: maps to multiple paths")

            if not normalized_rel_path.startswith(expected_prefix + "/"):
                pctx.integrity_failures.append(f"Path '{img_rel_path_str}' does not start with '{expected_prefix}'")

            img_resolved = (image_dir / product_name / img_rel_path_str).resolve()

            try:
                relative_path = img_resolved.relative_to(dataset_root).as_posix()
                if not img_resolved.is_relative_to(product_image_dir):
                    pctx.integrity_failures.append(f"Path escape outside product dir: {img_resolved}")
            except ValueError:
                pctx.integrity_failures.append(f"Path escape outside dataset root: {img_resolved}")
                continue

            if not img_resolved.exists():
                pctx.integrity_failures.append(f"Missing image {img_resolved}")
                continue

            sha256 = file_sha256(img_resolved)

            # Record img_id mapping and check consistency across the product
            if img_id in pctx.img_id_map:
                prev_path, prev_sha = pctx.img_id_map[img_id]
                if prev_path != normalized_rel_path or prev_sha != sha256:
                    pctx.integrity_failures.append(f"Inconsistent img_id {img_id}: mapped to both ({prev_path}, {prev_sha}) and ({normalized_rel_path}, {sha256})")
            else:
                pctx.img_id_map[img_id] = (normalized_rel_path, sha256)

            json_path = img_resolved.with_suffix(".json")
            if not json_path.exists():
                pctx.integrity_failures.append(f"Missing JSON {json_path}")
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    jdata = json.load(f)
            except Exception as e:
                pctx.integrity_failures.append(f"Invalid JSON {json_path}")
                continue

            json_image_path = jdata.get("imagePath", "")
            json_img_resolved = (json_path.parent / json_image_path).resolve()
            if json_img_resolved != img_resolved:
                pctx.integrity_failures.append(f"JSON imagePath mismatch: {json_image_path} does not point to {img_resolved.name}")

            csv_labels = [str(r['label']).strip() for r in rows if str(r['label']).strip()]
            json_labels = [str(s.get('label', '')).strip() for s in jdata.get('shapes', []) if str(s.get('label', '')).strip()]

            from collections import Counter
            if Counter(csv_labels) != Counter(json_labels):
                pctx.integrity_failures.append(f"Label count mismatch between CSV and JSON {json_path}. CSV: {Counter(csv_labels)}, JSON: {Counter(json_labels)}")

            if sha256 not in pctx.rgb_records:
                rec = RGBRecord(sha256, product_name)
                rec.canonical_relative_path = relative_path
                pctx.rgb_records[sha256] = rec
            else:
                rec = pctx.rgb_records[sha256]

            rec.alias_paths.add(relative_path)
            for row in rows:
                rec.source_pools.add(row['__source_pool'])
            rec.annotation_paths.add(json_path.relative_to(dataset_root).as_posix())
            for lbl in csv_labels:
                rec.instances.append([lbl])

    excluded_product_audit = {
        "cable": {
            "is_excluded": False,
            "cable_hash_overlaps": [],
            "cable_path_overlaps": [],
            "excluded_instances": 0,
            "excluded_unique_rgb": 0,
            "excluded_classes": []
        }
    }

    for product_name, pctx in products.items():
        is_cable = (product_name.lower() == "cable")
        if is_cable:
            excluded_product_audit["cable"]["is_excluded"] = True

        for sha256, rec in pctx.rgb_records.items():
            if len(rec.source_pools) > 1:
                pctx.integrity_failures.append(f"Cross pool collision for RGB {sha256}")
                pctx.cross_pool_collision_count += 1

            hash_overlap = sha256 in cable_hashes
            path_overlap = any(p in cable_paths for p in rec.alias_paths)

            if is_cable:
                if hash_overlap:
                    excluded_product_audit["cable"]["cable_hash_overlaps"].append(sha256)
                if path_overlap:
                    excluded_product_audit["cable"]["cable_path_overlaps"].append(rec.canonical_relative_path)
            else:
                if hash_overlap:
                    pctx.integrity_failures.append(f"Non-cable hash overlap for RGB {sha256}")
                if path_overlap:
                    pctx.integrity_failures.append(f"Non-cable path overlap for RGB {rec.canonical_relative_path}")

        if pctx.integrity_failures:
            pctx.integrity_pass = False

    records_jsonl = []
    product_stats = {}

    for product_name, pctx in products.items():
        stats = {
            "integrity_pass": pctx.integrity_pass,
            "integrity_failures": sorted(list(set(pctx.integrity_failures))),
            "total_instances": 0,
            "total_unique_rgb": 0,
            "single_type_unique_rgb": 0,
            "same_type_multi_instance_rgb": 0,
            "multi_type_unique_rgb_removed": 0,
            "support_pool_unique_rgb": 0,
            "query_pool_unique_rgb": 0,
            "cross_pool_collision_count": pctx.cross_pool_collision_count,
            "per_class_unique_rgb_count": {
                "before_filtering": {c: 0 for c in pctx.classes},
                "support_after_filtering": {c: 0 for c in pctx.classes},
                "query_after_filtering": {c: 0 for c in pctx.classes},
                "total_after_filtering": {c: 0 for c in pctx.classes},
            }
        }

        is_cable = (product_name.lower() == "cable")

        for sha256, rec in pctx.rgb_records.items():
            instance_count = len(rec.instances)
            raw_labels = []
            for inst in rec.instances:
                raw_labels.extend(inst)

            unique_defect_types = sorted(list(set(raw_labels)))

            if is_cable:
                excluded_product_audit["cable"]["excluded_unique_rgb"] += 1
                excluded_product_audit["cable"]["excluded_instances"] += instance_count
                for cls in unique_defect_types:
                    if cls not in excluded_product_audit["cable"]["excluded_classes"]:
                        excluded_product_audit["cable"]["excluded_classes"].append(cls)
                continue

            stats["total_unique_rgb"] += 1
            stats["total_instances"] += instance_count

            if len(unique_defect_types) == 1:
                cls = unique_defect_types[0]
                stats["per_class_unique_rgb_count"]["before_filtering"][cls] += 1
                stats["single_type_unique_rgb"] += 1
                if instance_count > 1:
                    stats["same_type_multi_instance_rgb"] += 1

                pools = list(rec.source_pools)
                if "support" in pools:
                    stats["support_pool_unique_rgb"] += 1
                    stats["per_class_unique_rgb_count"]["support_after_filtering"][cls] += 1
                if "query" in pools:
                    stats["query_pool_unique_rgb"] += 1
                    stats["per_class_unique_rgb_count"]["query_after_filtering"][cls] += 1

                stats["per_class_unique_rgb_count"]["total_after_filtering"][cls] += 1
            else:
                if len(unique_defect_types) > 1:
                    stats["multi_type_unique_rgb_removed"] += 1
                for cls in unique_defect_types:
                    stats["per_class_unique_rgb_count"]["before_filtering"][cls] += 1

            out_record = {
                "schema_version": "p6b_mvtec_fs_rgb_image_record_v2",
                "product": product_name,
                "original_image_id": rec.sha256,
                "rgb_sha256": rec.sha256,
                "canonical_relative_path": rec.canonical_relative_path,
                "alias_paths": sorted(list(rec.alias_paths)),
                "source_pools": sorted(list(rec.source_pools)),
                "annotation_paths": sorted(list(rec.annotation_paths)),
                "instance_count": instance_count,
                "unique_defect_types": unique_defect_types,
                "multi_type_removed": len(unique_defect_types) > 1,
                "same_type_multi_instance": (len(unique_defect_types) == 1 and instance_count > 1)
            }
            records_jsonl.append(out_record)

        product_stats[product_name] = stats

    records_jsonl.sort(key=lambda x: (x["product"], x["original_image_id"]))
    jsonl_path = output_dir / "p6b_mvtec_fs_rgb_image_records.jsonl"
    atomic_write_jsonl(jsonl_path, records_jsonl)

    feasible_products_by_shot = {"1": [], "3": [], "5": []}

    md_lines = ["| Product | N classes | Unique RGB | Multi-type removed | RGB collision | 1-shot | 3-shot | 5-shot |"]
    md_lines.append("|---|---|---|---|---|---|---|---|")

    for product_name, stats in sorted(product_stats.items()):
        is_cable = (product_name.lower() == "cable")
        if is_cable:
            continue

        n_classes = len(stats["per_class_unique_rgb_count"]["before_filtering"])
        total_unique = stats["total_unique_rgb"]
        multi_removed = stats["multi_type_unique_rgb_removed"]
        collision = stats["cross_pool_collision_count"]

        shots_feasible = {}
        for shot in [1, 3, 5]:
            is_feasible = True
            if len(dataset_integrity_failures) > 0:
                is_feasible = False
            if not stats["integrity_pass"]:
                is_feasible = False

            for cls in stats["per_class_unique_rgb_count"]["before_filtering"].keys():
                if stats["per_class_unique_rgb_count"]["support_after_filtering"][cls] < shot:
                    is_feasible = False
                if stats["per_class_unique_rgb_count"]["query_after_filtering"][cls] < 1:
                    is_feasible = False
            shots_feasible[shot] = is_feasible
            if is_feasible:
                feasible_products_by_shot[str(shot)].append(product_name)

        md_lines.append(f"| {product_name} | {n_classes} | {total_unique} | {multi_removed} | {collision} | {'Yes' if shots_feasible[1] else 'No'} | {'Yes' if shots_feasible[3] else 'No'} | {'Yes' if shots_feasible[5] else 'No'} |")

    md_path = output_dir / "p6b_mvtec_fs_rgb_feasibility_table.md"
    atomic_write_text(md_path, "\n".join(md_lines) + "\n")

    jsonl_hash = file_sha256(jsonl_path)
    md_hash = file_sha256(md_path)

    any_failure = len(dataset_integrity_failures) > 0 or any(not s["integrity_pass"] for s in product_stats.values())
    probe_status = "INCONCLUSIVE" if any_failure else "COMPLETE"

    output_json = {
        "schema_version": "p6b_mvtec_fs_rgb_feasibility_v2",
        "probe_status": probe_status,
        "dataset_root": str(dataset_root),
        "cable_reference_root": str(cable_ref_root),
        "formal_test_used": False,
        "cable_reference_used_for_overlap_only": True,
        "project_provenance": project_provenance,
        "dataset_provenance": dataset_provenance,
        "dataset_integrity_failures": sorted(list(set(dataset_integrity_failures))),
        "per_product": product_stats,
        "aggregate_counts": {
            "total_records": len(records_jsonl),
            "total_rgb": sum(s["total_unique_rgb"] for k, s in product_stats.items() if k.lower() != "cable")
        },
        "excluded_product_audit": excluded_product_audit,
        "feasible_products_by_shot": feasible_products_by_shot,
        "common_products_feasible_for_all_shots": sorted(list(set(feasible_products_by_shot["1"]) & set(feasible_products_by_shot["3"]) & set(feasible_products_by_shot["5"]))),
        "output_hashes": {
            "p6b_mvtec_fs_rgb_image_records.jsonl": jsonl_hash,
            "p6b_mvtec_fs_rgb_feasibility_table.md": md_hash
        }
    }

    json_path = output_dir / "p6b_mvtec_fs_rgb_feasibility.json"
    atomic_write_json(json_path, output_json)

if __name__ == "__main__":
    main()
