# P15 — Two-stage inspection: normal-calibrated screening + ARR type classification

Status: **LOCKED BEFORE ANY P15 OUTPUT** (2026-09-27). User-approved exception to the P14
"last formal round" statement, limited to this analysis and disclosed in the paper.

## 0. What P15 does not change

- No existing method, constant or code path changes. The classification results of P10–P14
  (including ARR and CONS) are final and are not re-reported differently.
- P15 adds a **decision layer** on top of the frozen classifiers. Every test stage first
  recomputes the frozen argmax decisions and must reproduce the published per-query
  predictions exactly (ARR, A, SimpleShot on VisA / MTD vs P14; ARR, A, SimpleShot on the Cable
  OOF pool vs P14), otherwise it aborts before writing anything.
- The only free choice (the screening statistic) is fixed on dev data before any test run.

## 1. Pipeline

1. **Screening score** s(x) from the same M = 20 normal bank and CLIP layer-18 patch anomalies
   a(t) used by ARR (no extra forward pass): `max` = max_t a(t), or `top1pct` = mean of the
   ⌈0.01 · 576⌉ = 6 largest a(t) (AnomalyDINO-style). One is selected on dev (§2).
2. **Threshold from normal images only (split-conformal):** a calibration set of N = 100
   normal images, disjoint from the bank, supports and queries; τ_α = the k-th smallest
   calibration score with k = ⌈(N + 1)(1 − α)⌉; reject iff s(x) > τ_α. For exchangeable normals
   this bounds the expected false-reject rate by α. Targets α ∈ {0.01, 0.02, 0.05}.
3. **Type classification** of rejected parts: ARR scores restricted to the defect classes
   (argmax over defect columns). SimpleShot restricted in the same way is the reference.
4. **Three-way decision (review band):** auto-reject if s > τ_0.01, auto-pass if s ≤ τ_lo,
   manual review otherwise, with τ_lo = τ_α for α_lo ∈ {0.10, 0.30} (about 90 % / 70 % of
   normals auto-pass). Review is assumed correct; escapes come only from auto-pass, false
   rejects only from auto-reject.

Reported per dataset (VisA object macro; MTD; Cable test): image AUROC of s; realized
false-reject rate at each α (calibration check); defect recall / escape rate; type accuracy
among rejected defects; per 1,000 parts at 95 % good: false rejects, escapes, wrong-type
rejects, review items. **Descriptive oracle (uses test labels, not deployable):** the minimum
review rate that achieves escape ≤ 1 % with auto-reject FRR ≤ 1 %. Argmax rows from P14 are
shown alongside for comparison.

## 2. Dev stage (no test data)

MVTec AD `train/good` of the 13 products, excluding every image of the P10-A dev bank draws
(formal-bank images are reference normals, not test data, and may appear; excluding them too
would leave only 65 held-out normals for bottle): per product, 100 calibration + up to 100
evaluation normals (29–100 per product)
(`random.Random(f"p15dev|{product}")`). Defects: the MVTec-FS support pool. Banks: the P10-A
dev draws (seeds 9001–9005, M = 20). **Selection rule:** the statistic with the higher mean
image AUROC over products and seeds; ties → `top1pct`. Also reported: realized FRR at each α
on the evaluation normals.

## 3. Test stages (each once, lock + staging)

- **VisA / MTD:** tasks, orders, support draws, banks exactly as P10–P14. Calibration normals:
  per task, 100 images sampled with `random.Random(f"p15cal|{task}")` from the normal support
  candidates excluding every seed's bank and K5 normal supports; extracted in a separate call.
- **Cable sealed test (first and only read):** `dataset/metadata/MVTecAD-Cable/test-shot.jsonl`
  (79 images: 50 good, 29 defects). Supports: the whole 72-image dev pool (K = 8 per class,
  9 classes incl. good). Banks: the P10 R1 draws (5 seeds). Calibration: 100 MVTec AD cable
  `train/good` images outside all bank draws (`random.Random("p15cal|cable")`). Byte-level
  hash checks: test ∩ (pool ∪ bank ∪ calibration) = ∅. Also reported (descriptive, first
  read of this held-out split by the classifiers): 9-class BA and argmax normal-aware metrics of
  SimpleShot, normref, A, ARR, CONS with K = 8.

## 4. Outputs

`ckpt/p15/dev/`, `ckpt/p15/external/`, `ckpt/p15/cable_test/`.

## Amendment P15-A1 (2026-09-27, before any Cable-test output)

The first M3 launch stopped at the byte-level hash check, before feature extraction or any
prediction (no output, empty staging). Cause (hash audit only, no inference): the manifest
lists one image twice — `poke_insulation/train/images/004.png` and
`poke_insulation/test/images/004.png` are byte-identical. Overlap with the pool, the bank draws
and the calibration normals is zero. Fix: entries whose SHA256 was already seen are dropped
(first occurrence kept, manifest order), giving **78 unique images (50 good, 28 defects)**; the
dropped path is recorded in the output. The overlap check itself is unchanged. The empty
`cable_test.staging` and the `cable_test.lock` of the aborted launch are removed so that M3 can
run once; this is the only re-launch of M3.

## Clarification (wording only, 2026-09-27)

The Cable "sealed test" is **not an independent dataset**. Hash audit: the 72-image dev pool
and the 79-entry test manifest are a re-split of the MVTec AD `cable/test` set (58 good + 92
defects = 150 = 72 + 78 unique); banks and calibration normals come from MVTec AD
`cable/train/good`; the methods were developed on other MVTec AD products (MVTec-FS). It is an
image-disjoint, previously unread **same-source held-out split** (in-distribution check), not
external validation. External validation remains VisA (and MTD). No rule or result changes.
