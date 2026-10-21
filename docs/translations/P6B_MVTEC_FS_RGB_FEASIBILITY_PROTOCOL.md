<!-- translation-header -->
> English translation of [`P6B_MVTEC_FS_RGB_FEASIBILITY_PROTOCOL.md`](../P6B_MVTEC_FS_RGB_FEASIBILITY_PROTOCOL.md), which is kept unchanged as the committed record (SHA-256 `a78c8d9e33854eafc158014d17793d9b935578e7e0fce63d5453901b4b459ba2`; commit dates in [`PROTOCOL_INDEX.md`](../PROTOCOL_INDEX.md)). The translation keeps the structure, numbers and decision rules of the original.

# P6.B MVTec-FS-RGB Feasibility Protocol

> Status: **PRE-SPECIFIED — FEASIBILITY PROBE ONLY**
> This file fixes only the P6.B MVTec-FS-RGB data construction and the probe contract; it does not pre-specify
> training, baseline hyperparameters or performance criteria. It ensures complete data parsing, leakage prevention
> and strict fail-closed protection, and involves no model conclusion.

## 1. Definitions

- **Formal name**: MVTec-FS RGB-Only Cross-Category Few-Shot Multiclass Method Generalization
- **Data adaptation name**: MVTec-FS-RGB
- P6.A stays unchanged; the method may not be named AA-CLIP; the name reserved for the paper is
  "P3.0 Method Transfer (Ours)".
- Products whose normalised name equals `cable` are excluded; the number of remaining products is not hard-coded but
  determined from the actual data.
- The split unit is the unique original RGB image, not the defect instance.
- The setting may not be called "the original official MVTec-FS protocol", and this round produces no model
  performance conclusion.

## 2. CLI contract

A formal run takes the following required arguments; no local path is hard-coded:
```bash
python -m experiments.p6.probe_p6b_mvtec_fs_rgb_feasibility \
  --dataset-root <MVTec-FS repository root> \
  --cable-reference-root <MVTecAD-Cable image root> \
  --output-dir <output dir>
```

## 3. Authoritative parsing rules (CSV-driven)

- `dataset-root` must point to the root of the MVTec-FS repository; the subdirectories `image/` and `CONFIG/` must
  exist.
- **Authoritative instance index**: the official `CONFIG/<product>_config1/train.csv` and `valid.csv` are the only
  source of truth for instances; each row is one defect instance.
- **Pool mapping**:
  - `train.csv` → `support` pool (the `training` directory);
  - `valid.csv` → `query` pool (the `testing` directory);
  - an unknown pool or an unmappable state causes a product-level fail-closed; nothing may silently enter an
    `unknown` pool.
- **CSV consistency constraints**:
  - `(product, id)` must be unique;
  - the `part` column must equal the product being parsed;
  - each `img_id` must map to exactly one `img_rel_path` and one RGB SHA-256; no ambiguity is allowed;
  - if the whole `CONFIG` directory or a required CSV is missing, a global or product-level integrity failure is
    raised, depending on the level of the missing item.
- **JSON cross-check**:
  - LabelMe JSON is used only for additional information and reconciliation; the `group_id` in the JSON is no longer
    verified (`group_id=null` is allowed);
  - reconciliation rule: for a given `(img_id, label)`, the number of CSV rows must equal the number of shapes with
    the same `label` in the corresponding JSON; any mismatch of count or label is an integrity failure of that
    product.

## 4. Aggregate records

The final aggregation folds all `img_id` by **(product, RGB_SHA256)** and collects the instance rows of all alias
paths.
- `img_id` is used only for consistency checks: each img_id must map to exactly one (normalized_rel_path, SHA256).
- When a CSV entry fails to parse or is aggregated, its original `label` (even if only a parse failure occurred)
  still counts as a base class in the statistics.

## 5. Output verification

The feasibility result is delivered as `p6b_mvtec_fs_rgb_feasibility_table.md`, with
`p6b_mvtec_fs_rgb_feasibility.json` and `p6b_mvtec_fs_rgb_image_records.jsonl` next to it, carrying detailed failure
reasons and SHA-256 fingerprint lists:
1. **Atomic write**: every output is written atomically (to a sibling `.tmp` file, then `os.replace`), so a crash
   leaves no intermediate state, with LF line endings.
2. **Deterministic output**: two consecutive runs on the same input produce byte-identical files (same key order and
   layout).
3. **Fail-closed gate**: on any of the following global integrity failures, no per-product feasibility data is
   written and the probe status is set to `INCONCLUSIVE` immediately:
   - `readme.md`, `license.txt` or `image/` missing under the dataset root;
   - no valid `*_config1` product directory under `CONFIG/`;
   - no supported image format in the Cable exclusion reference.
4. **Provenance**:
   - `project_provenance`: git commit and dirty state of the repository running the probe;
   - `dataset_provenance`: git commit and dirty state of the MVTec-FS dataset (dataset-root), and the hashes of the
     extracted `readme.md`, `license.txt` and all input CSV files.
- Output files may never be written inside `dataset-root`.

## 6. Cable exclusion and overlap audit

- The `cable` product is **removed first**, before the hash-overlap check; the counts before exclusion (e.g. unique
  RGB, overlaps) are recorded in `excluded_product_audit`, so that overlaps of Cable itself do not affect the global
  integrity gate.
- `cable-reference-root` must exist and contain at least one image; it is used only for the path/hash overlap audit.
- For valid non-Cable products:
  - a content SHA-256 overlap > 0 is a hard product-level integrity failure; the product fails but **does not**
    affect the other, clean products;
  - path overlaps are also reported and counted as failures.
- The output states `formal_test_used=false` and `cable_reference_used_for_overlap_only=true` explicitly.

## 7. Local and global error propagation

- **Global (dataset-level) integrity failure**, e.g. missing root structure or missing Cable reference: blocks the
  whole dataset; recorded in `dataset_integrity_failures`.
- **Local (product-level) integrity failure**, e.g. a missing product CSV, a failed CSV/JSON reconciliation or a
  non-Cable hash collision: only sets `integrity_pass=false` for that product.
- The overall `probe_status` is `INCONCLUSIVE` if any global or local error occurs, but the true feasibility of the
  unaffected compliant products is still reported.

## 8. Per-product and per-class statistics

For each product the following are counted and reported:
- `total_instances`
- `total_unique_rgb`
- `single_type_unique_rgb`
- `same_type_multi_instance_rgb`
- `multi_type_unique_rgb_removed`
- `support_pool_unique_rgb` / `query_pool_unique_rgb`
- classes with zero images after filtering are kept, never silently removed.

## 9. Shot feasibility

For each **non-excluded** product and K ∈ {1, 3, 5}, feasibility is the strict conjunction of:
1. the global `dataset_integrity_failures` is empty;
2. the product's `integrity_pass` is `true` (no local integrity error);
3. every official defect class has support unique RGB ≥ K;
4. every official defect class has query unique RGB ≥ 1;
5. support and query RGB are fully disjoint (no cross-pool collision).

## 10. Outputs and status

The only allowed overall statuses are `COMPLETE` and `INCONCLUSIVE`; this round outputs no CONFIRMED/NOT CONFIRMED.
Outputs:
1. `p6b_mvtec_fs_rgb_image_records.jsonl`
2. `p6b_mvtec_fs_rgb_feasibility.json`
3. `p6b_mvtec_fs_rgb_feasibility_table.md`

Runs are deterministic and consistent across repetitions; the dataset and the mtimes of local files are not
modified.
