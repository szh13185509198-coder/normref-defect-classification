# P12 — Threshold-source controls, industrial reliability of A, figures v3, evidence

Status: **LOCKED BEFORE ANY P12 OUTPUT** (2026-09-27). A = `fused_adareg_soft_q99`
(P11, confirmed on MVTec-FS formal and VisA). Nothing in P12 changes A or re-selects anything;
all P12 rows are controls or descriptive analyses. The formal sets are read again only to
evaluate the pre-declared controls below, each stage once.

## T. Threshold-source controls (does the normal calibration matter?)

All controls keep everything of A except where the threshold τ comes from: same normal bank
(M = 20), same patch anomaly a(t), same soft weights w(t) = max(a(t) − τ, 0) with the top-6
fallback, same SimpleShot and z-score fusion with the global SimpleShot score.

| Row | τ | Uses normal images for τ? | Per-image adaptive region? |
|---|---|---|---|
| `fused_adareg_soft_q99` (A) | 0.99-quantile of leave-one-image-out bank anomalies (one τ per bank) | yes | yes |
| `fused_ctrl_soft_top10` | per image: the 59th largest a(t), so exactly the top ⌈0.1·576⌉ = 58 patches exceed τ | no | no (fixed fraction) |
| `fused_ctrl_soft_otsu` | per image: Otsu on its own 576 anomaly values (exact, on sorted values: the split maximizing between-class variance; τ = largest value of the lower class; first maximum on ties) | no | yes |
| `fused_ctrl_soft_mebin` | per image, MEBin-inspired (AnomalyNCD, CVPR 2025), simplified: 64 thresholds uniform on [min_i, max_i] of the image's own map; count 8-connected components of {a > ε} on the 24×24 grid; δ̄ = most frequent non-zero count (ties → smaller count); τ = lowest threshold of the longest consecutive run with count = δ̄ (ties → lowest run) | no | yes |

MEBin deviations (declared): thresholds use the image's own range (independent query
inference; the original uses dataset-wide min/max), no run-length acceptance test, no cropping
or training. It is a thresholding control, not a re-implementation of AnomalyNCD.

**Stages (each once):**
- **T-dev** (P10 dev episodes, seeds 9001–9005, K = 1/3): all rows + references
  `simpleshot_cl2n`, `fused_normref_clip18`. Integrity: A must reproduce the P11 dev pilot rows
  exactly. Region diagnostics (patch-level precision / recall / F1 vs `mask_grid > 0`, mean size,
  fallback fraction) for every τ source. No gate: every control goes to T-formal regardless.
- **T-mvtec** (formal, 530 queries): integrity — SimpleShot, `fused_normref_clip18` and A equal
  P10 / P11 formal (max |Δ| < 1e-9). Contrasts A − each control, product-cluster bootstrap
  (seed 20260813), BA pooled over K. Per-query predictions of SimpleShot, normref, A and the
  controls are saved (needed for the qualitative figures).
- **T-external** (VisA confirmatory-style CI, MTD descriptive): integrity — SimpleShot, normref
  and A equal P10 / P11 external. Contrasts A − each control, object-cluster bootstrap
  (seed 20260814).

**Reading rule (fixed now):** "normal calibration adds beyond image-only adaptive thresholds"
is supported if A − `ctrl_soft_otsu` **and** A − `ctrl_soft_mebin` both have 95% CI lower
bound > 0 on MVTec-FS (pooled BA); "adaptive region adds beyond soft weighting of a fixed
fraction" is supported if A − `ctrl_soft_top10` has CI lower bound > 0. VisA is reported with
the same rule as a second dataset; disagreement between datasets is reported as such.

## R1'. Cable OOF with A and B

Same pool, folds, seeds, normal-bank draws, extraction path and V1 archive as P10 R1
(`experiments/p10/cable_oof.py`). Adds `fused_adareg_soft_q99` and
`fused_guided_xclass_normref`. Integrity: SimpleShot and `fused_normref_clip18` per-image
predictions equal the published P10 R1 predictions for every seed. Descriptive only.

## R2'. Normal-aware metrics of A and B on VisA / MTD (no GPU)

From `ckpt/p11_formal/external/p11_formal_external_predictions.jsonl`. Integrity: every
cell's classification metrics recomputed from the predictions equal the P11 raw metrics, and
SimpleShot / normref normal-aware cells equal P10 R2. Same metrics and per-1,000 conversion
(95 % good) as P10 R2; frozen methods (V1, MVREC-RGB) copied from P10 R2.

## F. Figures v3 (`docs/paper/figures_v3/`, each with a CSV of plotted numbers)

1. MVTec-FS line chart: mask oracle, A, P10 normal reference, residual, cross-class,
   SimpleShot, V1, MVREC-RGB.
2. Per-product Δ vs SimpleShot: A, P10 normal reference, cross-class (sorted by A).
3. VisA forest: E4 (A − normref), A − SimpleShot, E1 (normref − SimpleShot), E3 (cross-class − SimpleShot).
4. Industrial false rejects per 1,000 (VisA, MTD): SimpleShot, cross-class, normref, A, V1, MVREC-RGB.
5. Qualitative (deterministic): products = 3 largest and 3 smallest A − SimpleShot formal
   gains; K = 5, seed 111, queries sorted by `rgb_sha256`; success = first query with A
   correct and SimpleShot wrong; failure = first query with A wrong. Panels: image with GT
   outline (visualization only) and anomaly map with the A region (w > 0) outlined.
6. Region comparison (deterministic): the 3 products with the largest A − normref formal gain;
   first query (K = 5, seed 111, by `rgb_sha256`) with A correct and normref wrong; panels:
   image + GT outline, fixed top-58 region, Otsu region, MEBin region, A soft weights.
7. Cable: first two `combined` images among A's persistent errors (R1'), maps from the seed-111 bank.

## E. Evidence

SHA256 manifest `docs/paper/P12_EVIDENCE_MANIFEST.json` over all P10/P11/P12 result
directories (features hashed, not archived) and a new tar.gz in `D:\intergration\evidence_archive`.
