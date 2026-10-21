#!/usr/bin/env python3
"""Independent fail-closed audit and metric recomputation for P6.B query output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Make direct-file launch equivalent to ``python -m`` before package imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.p6.p6b_execution_common import P6BFailClosedError, atomic_write_json
from experiments.p6 import build_p6b_support_manifests as builder
from experiments.p6.p6b_stage3_common import (
    AUDIT_SCHEMA, EXPECTED_FORMAL_QUERY_COUNT, K_VALUES, METHODS, PRODUCTS,
    PREDICTION_MANIFEST_SCHEMA, PREDICTION_SCHEMA, SEEDS, aggregate_raw_metrics,
    classification_metrics, expected_units, file_sha256, load_json,
    load_query_contract, validate_formal_grid, UNIT_PREDICTION_MANIFEST_SCHEMA,
)
EXPECTED_PREDICTION_ROWS = len(METHODS) * len(K_VALUES) * len(SEEDS) * EXPECTED_FORMAL_QUERY_COUNT


def queries_by_product_class_order(queries, product):
    orders = {tuple(query["class_order"]) for query in queries
              if query["product"] == product}
    if len(orders) != 1:
        raise P6BFailClosedError(f"query class_order is inconsistent for {product}")
    return list(next(iter(orders)))


def read_prediction_rows(path: Path):
    rows = []
    digest = __import__("hashlib").sha256()
    with Path(path).open("rb") as raw:
        for line_no, line in enumerate(raw, 1):
            digest.update(line)
            if not line.endswith(b"\n") or b"\r" in line:
                raise P6BFailClosedError(f"prediction JSONL line {line_no}: non-canonical newline")
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise P6BFailClosedError(f"prediction JSONL line {line_no}: {exc}") from exc
            rows.append(row)
    return rows, digest.hexdigest()


def audit_prediction_rows(rows: list[dict], query_records: list[dict]) -> tuple[list[dict], dict]:
    query_by_product = {}
    for record in query_records:
        query_by_product.setdefault(record["product"], {})[record["rgb_sha256"]] = record
    if sum(len(v) for v in query_by_product.values()) != EXPECTED_FORMAL_QUERY_COUNT:
        raise P6BFailClosedError("auditor query contract is not the locked 530-sample set")
    if len(rows) != EXPECTED_PREDICTION_ROWS:
        raise P6BFailClosedError(
            f"prediction row count {len(rows)} != {EXPECTED_PREDICTION_ROWS}")

    grouped = {}
    seen = set()
    for index, row in enumerate(rows, 1):
        if row.get("schema_version") != PREDICTION_SCHEMA:
            raise P6BFailClosedError(f"prediction row {index}: schema mismatch")
        method, product = row.get("method"), row.get("product")
        k_shot, seed, sha = row.get("K"), row.get("seed"), row.get("rgb_sha256")
        if method not in METHODS or product not in PRODUCTS or k_shot not in K_VALUES or seed not in SEEDS:
            raise P6BFailClosedError(f"prediction row {index}: key outside locked grid")
        key = (method, product, k_shot, seed, sha)
        if key in seen:
            raise P6BFailClosedError(f"duplicate prediction key: {key}")
        seen.add(key)
        query = query_by_product.get(product, {}).get(sha)
        if query is None:
            raise P6BFailClosedError(f"prediction for non-contract query: {product}/{sha}")
        class_order = row.get("class_order")
        if class_order != query["class_order"]:
            raise P6BFailClosedError(f"class_order mismatch: {key}")
        if row.get("true_label") != query["true_label"]:
            raise P6BFailClosedError(f"true label mismatch: {key}")
        scores = row.get("scores")
        if (not isinstance(scores, list) or len(scores) != len(class_order) or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                    not math.isfinite(float(x)) for x in scores)):
            raise P6BFailClosedError(f"invalid score vector: {key}")
        predicted = class_order[max(range(len(scores)), key=lambda i: scores[i])]
        if row.get("predicted_label") != predicted:
            raise P6BFailClosedError(f"argmax/tie-break mismatch: {key}")
        grouped.setdefault((method, product, k_shot, seed), []).append(row)

    raw_metrics = []
    for unit in expected_units():
        unit_rows = grouped.get((unit.method, unit.product, unit.k_shot, unit.seed), [])
        expected_queries = query_by_product.get(unit.product, {})
        if len(unit_rows) != len(expected_queries) or {r["rgb_sha256"] for r in unit_rows} != set(expected_queries):
            raise P6BFailClosedError(f"incomplete query coverage: {unit.key}")
        unit_rows.sort(key=lambda r: r["rgb_sha256"])
        metrics = classification_metrics(
            [r["true_label"] for r in unit_rows],
            [r["predicted_label"] for r in unit_rows],
            unit_rows[0]["class_order"],
        )
        raw_metrics.append({"method": unit.method, "product": unit.product,
                            "K": unit.k_shot, "seed": unit.seed, **metrics})
    return raw_metrics, aggregate_raw_metrics(raw_metrics)


def audit_files(*, grid_root, predictions_path, prediction_manifest_path,
                records_path, dataset_root, stage1_bundle, output_path) -> dict:
    if Path(output_path).exists():
        raise P6BFailClosedError(f"audit output already exists: {output_path}")
    grid = validate_formal_grid(grid_root, require_global_completion=True)
    audit_commit, audit_dirty = builder.git_commit_and_dirty(PROJECT_ROOT)
    if audit_dirty:
        raise P6BFailClosedError("formal auditor requires a clean git tree")
    queries, query_audit = load_query_contract(records_path, dataset_root, stage1_bundle)
    manifest = load_json(prediction_manifest_path)
    checks = {
        "schema_version": PREDICTION_MANIFEST_SCHEMA,
        "status": "completed",
        "formal_test_used": False,
        "query_pixels_read": True,
        "query_labels_read": True,
        "prediction_row_count": EXPECTED_PREDICTION_ROWS,
        "formal_query_count": EXPECTED_FORMAL_QUERY_COUNT,
    }
    for key, wanted in checks.items():
        if manifest.get(key) != wanted:
            raise P6BFailClosedError(f"prediction manifest {key} mismatch")
    if manifest.get("support_grid_execution_commit") != grid["execution_commit"]:
        raise P6BFailClosedError("prediction/support-grid commit mismatch")
    if manifest.get("evaluator_execution_commit") != audit_commit:
        raise P6BFailClosedError("prediction/evaluator-auditor commit mismatch")
    if file_sha256(predictions_path) != manifest.get("predictions_sha256"):
        raise P6BFailClosedError("prediction JSONL SHA mismatch")
    unit_records = manifest.get("unit_predictions")
    if not isinstance(unit_records, list) or len(unit_records) != len(list(expected_units())):
        raise P6BFailClosedError("unit prediction index count mismatch")
    prediction_root = Path(prediction_manifest_path).resolve().parent
    expected_index_keys = [unit.key for unit in expected_units()]
    if [record.get("key") for record in unit_records] != expected_index_keys:
        raise P6BFailClosedError("unit prediction index order/coverage mismatch")
    concatenated = __import__("hashlib").sha256()
    for unit, index_record in zip(expected_units(), unit_records):
        unit_prediction = (prediction_root / index_record["predictions_relative_path"]).resolve()
        unit_manifest_path = (prediction_root / index_record["prediction_manifest_relative_path"]).resolve()
        if (prediction_root not in unit_prediction.parents or
                prediction_root not in unit_manifest_path.parents):
            raise P6BFailClosedError(f"unit prediction path escapes output: {unit.key}")
        if not unit_prediction.is_file() or not unit_manifest_path.is_file():
            raise P6BFailClosedError(f"unit prediction artifact missing: {unit.key}")
        if (file_sha256(unit_prediction) != index_record.get("predictions_sha256") or
                file_sha256(unit_manifest_path) != index_record.get("prediction_manifest_sha256")):
            raise P6BFailClosedError(f"unit prediction index SHA mismatch: {unit.key}")
        unit_manifest = load_json(unit_manifest_path)
        expected_unit_fields = {
            "schema_version": UNIT_PREDICTION_MANIFEST_SCHEMA,
            "method": unit.method, "product": unit.product,
            "K": unit.k_shot, "seed": unit.seed,
            "predictions_relative_path": index_record["predictions_relative_path"],
            "predictions_sha256": index_record["predictions_sha256"],
            "query_count": index_record["query_count"],
        }
        for key, wanted in expected_unit_fields.items():
            if unit_manifest.get(key) != wanted:
                raise P6BFailClosedError(f"unit prediction manifest {key} mismatch: {unit.key}")
        expected_query_sha = [
            query["rgb_sha256"] for query in queries if query["product"] == unit.product]
        if (unit_manifest.get("class_order") != queries_by_product_class_order(
                queries, unit.product) or
                unit_manifest.get("query_rgb_sha256") != expected_query_sha or
                unit_manifest.get("query_count") != len(expected_query_sha)):
            raise P6BFailClosedError(f"unit prediction query contract mismatch: {unit.key}")
        support_completion = unit.directory(Path(grid_root)) / "p6b_stage2_support_complete.json"
        if unit_manifest.get("support_completion_sha256") != file_sha256(support_completion):
            raise P6BFailClosedError(f"unit prediction support provenance mismatch: {unit.key}")
        with unit_prediction.open("rb") as handle:
            for line in handle:
                concatenated.update(line)
    if concatenated.hexdigest() != manifest.get("predictions_sha256"):
        raise P6BFailClosedError("global predictions are not exact unit-file concatenation")
    rows, streamed_sha = read_prediction_rows(predictions_path)
    if streamed_sha != manifest["predictions_sha256"]:
        raise P6BFailClosedError("streamed prediction SHA mismatch")
    raw, aggregate = audit_prediction_rows(rows, queries)
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "status": "completed",
        "formal_eligible": True,
        "formal_test_used": False,
        "support_grid_execution_commit": grid["execution_commit"],
        "evaluator_execution_commit": audit_commit,
        "support_grid_completion_sha256": file_sha256(
            Path(grid_root) / "p6b_formal_support_grid_complete.json"),
        "query_records_sha256": file_sha256(Path(records_path)),
        "prediction_manifest_sha256": file_sha256(Path(prediction_manifest_path)),
        "predictions_sha256": manifest["predictions_sha256"],
        "prediction_row_count": len(rows),
        "query_contract": query_audit,
        "raw_metrics": raw,
        "aggregation": aggregate,
    }
    atomic_write_json(output_path, audit)
    return audit


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stage1-bundle", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    try:
        audit_files(**vars(build_parser().parse_args(argv)))
        print("[P6.B formal audit] completed")
        return 0
    except (P6BFailClosedError, OSError) as exc:
        print(f"[P6.B formal audit] INCONCLUSIVE / FAIL-CLOSED: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
