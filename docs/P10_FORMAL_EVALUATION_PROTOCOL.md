# P10 Formal Evaluation — normal-reference (C1) and cross-class (C0) cues on MVTec-FS-RGB

Status: **LOCKED BEFORE ANY FORMAL P10 OUTPUT** (2026-09-26). Code commit to be recorded
in the output. This is the single formal run of P10 on the 530 formal queries; it may be
repeated only after an integrity failure, with identical definitions.

Prior information disclosed: dev results in RESEARCH_LEDGER (P10, P10-A, P10-A2) and the
P6.B / Revision-C formal results.

## 1. Information conditions

- **C0 (strict image-level, identical to the original manuscript):** per class K support
  images with image-level labels; no masks, boxes, annotation-derived ROIs, or query-side
  localization input; each query classified independently.
- **C1 = C0 + M unlabeled normal images of the same product** (MVTec AD `train/good`,
  image-level "good"; no spatial annotation). M=20 primary, M=5 sensitivity.
- Neither condition uses localization annotations at any stage. Methods are compared
  within a condition; cross-condition differences are reported as the value of adding
  normal images.

## 2. Grid and data

- 13 locked products x K in {1,3,5} x formal seeds {111,222,333,444,555} = 195 units.
- Supports: the formal P6.B manifests
  (`data/p6b_stage1_manifest_bundle/p6b_support_manifests/<product>/k<K>_s<seed>.json`);
  their images are read from the P10 support-pool features by `rel_path` (every manifest
  item must be found, otherwise fail).
- Queries: the 530 formal queries via `load_query_contract` (frozen records SHA).
- Normal banks per product and formal seed:
  `random.Random(f"p10A|{seed}|{product}").sample(sorted(train/good), 20)`; M=5 = first 5.
- Features: same frozen backbones, deterministic 336 view, local RTX 4050: global fusion
  (formal `fusion_proto` definition), DINOv2 patches, CLIP ViT-L/14@336 layer-18 patches,
  L2-normalized. Query masks are rasterized **only** for the oracle rows.

## 3. Methods (definitions identical to the P10 dev protocol and addenda)

C0: `fusion_proto_local`, `simpleshot_cl2n`, `local_xclass_clip18`, `fused_xclass_clip18`.
C1: `normref_pool_clip18`, `fused_normref_clip18`, `normcenter_mean_clip18`,
`fused_normcenter_mean_clip18`, `fused_normref_clip18_m5`.
Oracle (uses GT masks of supports and queries; upper bound only, never ranked):
`oracle_maskpool_clip18`, `oracle_maskpool_dino`.
Existing formal comparators (read from frozen artifacts, not re-run): `ours`
(P6.B V1), `fusion_proto` (P6.B), `mvrec_rgb` (MVREC-RGB adaptation formal predictions,
`ckpt/mvrec_rgb_formal_query_227300b.tar.gz`, SHA256 `dc35eda5…f3e1`).

## 4. Hypotheses and decision rules (BA primary; Macro-F1 and Top-1 reported)

Estimator: C1 of Addendum C — paired differences per (product, K, seed), product-macro,
Dbar = mean over K; 95% product-cluster percentile bootstrap, 10,000 resamples, seed
20260813, all seeds and K of a product kept together.

- **H1 (primary, C1):** `fused_normref_clip18` minus `simpleshot_cl2n`.
  CONFIRMED iff the Dbar 95% CI lower bound > 0.
- **H2 (secondary, attribution within C1):** `fused_normref_clip18` minus
  `fused_normcenter_mean_clip18`. SUPPORTED iff the Dbar CI lower bound > 0; the number of
  positive products is reported alongside. If not supported, the manuscript states the
  C1 gain as "normal images help" and presents localization as product-dependent.
- **H3 (secondary, C0):** `fused_xclass_clip18` minus `simpleshot_cl2n`.
  CONFIRMED iff the Dbar CI lower bound > 0.

No multiplicity adjustment is claimed; H1 is the single primary hypothesis. Per-K CIs,
positive-product counts, and all descriptive contrasts (every method vs `simpleshot_cl2n`,
`ours`, `fusion_proto`, `mvrec_rgb`) are reported regardless of outcome. Contrasts against
the P6.B/MVREC artifacts include a cross-hardware term (Revision C: <= 0.04 pp at product-
macro level), disclosed.

## 5. Execution guards

- Evaluation refuses to run if `experiments/p10`, `experiments/revision`, or this
  protocol have uncommitted changes; it records `git HEAD`.
- Output `ckpt/p10_formal/results/` is written once (staging + atomic rename); an existing
  results directory aborts the run.
- Extraction writes only features; no metric is computed during extraction.
