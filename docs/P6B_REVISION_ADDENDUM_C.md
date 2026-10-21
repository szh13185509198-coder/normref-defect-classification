# P6.B Revision Addendum C — Post-hoc Revision Analyses (Local Tier)

Status: **LOCKED BEFORE EXECUTION** (written 2026-09-26, before any C-analysis output exists)

Purpose: respond to the MVA pre-submission advice (`SCI论文/MVA投稿前详细修改建议.docx`,
sections C/D/E) with analyses that run on the local RTX 4050 (6 GB). Everything here is
a **post-hoc revision analysis** added after P6.B formal results were visible. It does
not alter P6.B, its method, its thresholds, its product/seed set, or any decision already
recorded. All outputs must be reported in full, whatever their direction, and labelled
"post-hoc (revision)" in the manuscript.

Common rules

- Grid: 13 locked products x K={1,3,5} x seeds {111,222,333,444,555}; 530 formal queries.
- No new tuning: every constant below is fixed now. No variant may be added, dropped,
  or re-run with other constants after outputs exist. A failed integrity gate may only
  be re-run with the identical definition.
- Interpreter: `D:\Anaconda\envs\cole\python.exe`; numpy is imported before torch.
- Outputs go to `ckpt/revision_c/` (gitignored `ckpt/`).

## C1 Paired product-cluster bootstrap for Group A contrasts

- Source: `ckpt/p6b_formal_query_v1/p6b_formal_query_audit.json` `raw_metrics`
  (per method/product/K/seed), plus C2 per-unit metrics when present.
- Contrasts (all reported): Ours minus each of `fusion_proto` (primary),
  `p30_frozen_proto` (R1), `clip_proto`, `dino_proto`, `linear_probe`; and, once C2
  exists, Ours minus `simpleshot_cl2n` and Ours minus `logreg_rfs`.
- Metrics: balanced accuracy (primary), macro-F1, top-1 accuracy.
- d[p,s,K] = M_ours - M_ref; D_K = mean_p mean_s d; Dbar = mean_K D_K.
- Uncertainty: identical estimator to the MVREC-RGB decision evaluator
  (`experiments/p6/mvrec_rgb_formal/decision_evaluator.percentile_bootstrap`):
  10,000 resamples of the 13 products, all seeds (and all K for Dbar) kept inside the
  product cluster, RNG seed 20260813, two-sided 95% percentile CI.
- Descriptives: per-product five-seed mean difference, count of products with positive
  difference, median product difference, count of seeds with positive product-macro
  difference.
- Interpretation: descriptive uncertainty only; no decision gate, no multiplicity
  claim. Nine shot x metric cells are not independent tests.

## C2 Frozen-feature baselines on the formal Frozen-Fusion features

Features

- Support: the exact `support_embeddings` / `support_labels` saved in each formal
  `ckpt/p6b_formal_support_grid_v3/fusion_proto/<product>/k<K>_s<seed>/fusion_proto_prototypes.pt`
  (1152-d, frozen CLIP native global embedding + DINOv2 full-patch mean, fused).
- Query: re-extracted locally with the same frozen factory
  (`run_p6b_stage2_support._build_lazy_clip_and_dino`, deterministic 336 view,
  `embed_images`).
- **Fidelity gate (must pass before any C2 metric is computed):** recomputing
  `fusion_proto` cosine scores from the re-extracted query features and the formal
  prototypes must reproduce the formal `fusion_proto` predicted label for 7,950/7,950
  rows, and max |score difference| <= 1e-3 (cross-hardware allowance; P8 observed
  7.57e-6). Otherwise STOP; do not run C2 methods.

Methods (both have zero hand-tuned constants beyond those fixed here)

1. `simpleshot_cl2n` (SimpleShot, Wang et al. 2019, CL2N variant). Centering vector
   mu = mean of all support embeddings of the current unit (all classes). Both support
   and query: x' = L2(x - mu). Prototype = L2(class mean of x'). Prediction = cosine
   argmax; ties resolve to manifest `class_order`. Deviation from the original: the
   base-class mean is unavailable under the image-level protocol (no base-class
   training), and query-set means are forbidden (transductive), so the support mean
   is used. This deviation is disclosed.
2. `logreg_rfs` (logistic-regression probe as in Tian et al., ECCV 2020 "RFS").
   Input = the formal L2-normalized fused features, no augmentation.
   `sklearn.linear_model.LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)`
   (multinomial), no class weighting, no feature standardization, no tuning.
   Prediction = argmax of decision function; ties resolve to `class_order`.

Metrics use `experiments/p6/p6b_stage3_common.classification_metrics` and the locked
equal-product macro aggregation. Results enter C1 as additional references.

Rationale for C2.2: the formal Frozen-Fusion Linear Probe (AdamW, lr 1e-4, 80 optimizer
steps, unit-norm inputs, no logit scale) is suspected to be under-optimized. `logreg_rfs`
is a standard closed-form-quality probe on the same features; the formal Linear Probe
row is retained unchanged and both rows are reported.

## C3 Resource profile (local device)

- Device: local NVIDIA GeForce RTX 4050 Laptop GPU (6 GB); report driver, torch, CUDA
  versions and CPU model in the output.
- Parameter counts per component of the Ours inference model (frozen CLIP visual,
  frozen CLIP text, frozen DINOv2, dormant AA-CLIP scaffold modules, trainable terminal
  projection, prototypes, logit scale) and trainable totals for Ours and each Group A
  baseline.
- Latency, batch 1, FP32, `torch.no_grad`, CUDA-synchronized: 50 warm-ups, 200 timed
  runs; report median / p90 / mean. Two boundaries: (a) model forward from a
  pre-loaded 336x336 GPU tensor to class scores; (b) end-to-end from a decoded PIL
  image including preprocessing. Pipelines: Ours query path, Frozen-Fusion Proto,
  CLIP-Proto, DINO-Proto. Peak allocated GPU memory per pipeline (fresh process per
  pipeline). Nothing else may run on the GPU during timing.
- Support-adaptation time: parsed from the 195 formal `ours_train_multiclass.log`
  files (first log line to "Checkpoint saved"), cloud hardware. Overlapping unit
  intervals must be detected and reported, because concurrent execution inflates
  wall time. Optionally one local diagnostic re-run of a single Ours unit
  (`bottle`, K=5, seed 111) with `--allow-dirty-diagnostic` for a local wall-time
  figure; that run is diagnostic and never replaces formal artifacts.
- Claims allowed: measured cost on the named device only. No edge / NPU / lightweight
  claim follows.

## Amendment C2-A1 — fidelity gate re-specification (2026-09-26)

**Recorded failure (original gate).** Local run on the RTX 4050: 7,950/7,950 rows
checked, max |score difference| `6.75e-4` (within `1e-3`), but label agreement
`7,938/7,950`; the original strict gate therefore returned `passed=false`, and T2 was
correctly skipped. The strict result is preserved in the report as
`passed_strict_original`.

**Diagnosis (integrity only; no C2 method output existed or was computed).** All 12
disagreements are cosine near-ties: formal top-1/top-2 margins lie in
`[9.5e-7, 1.7e-4]`, below the observed cross-hardware score noise (`6.75e-4`).
Locally recomputed `fusion_proto` product-macro BA differs from formal by
`+0.026 / +0.032 / +0.018` pp at K=1/3/5. Exact label identity across GPUs is therefore
not achievable for near-tie queries; the original gate was mis-specified, not the
features.

**Amended gate (decisive from now on).** Pass iff all of:
1. all 7,950 formal `fusion_proto` rows are matched, and max |score difference| <= 1e-3;
2. every label disagreement has formal top-1/top-2 margin <= 2 x the observed max
   |score difference| (i.e. is a numerical near-tie);
3. |local - formal| `fusion_proto` product-macro BA <= 0.1 pp at each K.

**Added within-hardware reference.** C2 additionally reports `fusion_proto_local`
(formal prototypes x local query features). SimpleShot and logreg are contrasted
against `fusion_proto_local` with the C1 estimator, so their difference to the Frozen
-Fusion baseline carries no cross-hardware term. Their contrasts against formal Ours
(C1) carry at most the ~0.03 pp cross-hardware term measured above, which is disclosed.

**Timing guarantee.** This amendment was written after the integrity failure and before
any `simpleshot_cl2n` / `logreg_rfs` metric was computed. No other constant changes.
