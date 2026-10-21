# P27 protocol — missing VisA / MTD rows: Liu residual, logistic regression, mask oracle (report-only)

Locked 2026-09-29, before any P27 output exists.

## Why
A table audit of the paper (v4) found three methods that exist on MVTec-FS but were never run on VisA / MTD:
- the zero-training Liu et al. adaptation, "mean normal residual" (`residual_mean_clip18`, Table 8 row 5 on
  MVTec-FS; Table 6);
- logistic regression on the global features (RFS style, `logreg_rfs`, Table 6);
- the mask oracle (`oracle_maskpool_clip18`, Table 3; basis of the "oracle headroom recovered" figure).

P27 runs exactly these three (plus the fused residual variant, level only) on the frozen P10 external tasks. No
method, constant, bank, support draw, or query changes. Nothing on MVTec-FS or Real-IAD is touched. Real-IAD is
excluded on purpose: its blind protocol reads it once.

## Data and methods (all frozen from P10 external)
- Tasks: the 11 VisA objects and MTD 6-way (`experiments.p10.external_evaluate.TASKS`), their frozen class orders,
  support draws, and queries. Seeds 111–555, K = 1 / 3 / 5.
- Features: `external_evaluate.extract` (global fusion; CLIP ViT-L/14@336 layer 18 patch tokens). Normal banks:
  `external_evaluate.bank_draw`, M = 20.
- `residual_mean_clip18`: for every patch, subtract its nearest bank patch (cosine), then average over the image;
  SimpleShot on that descriptor. Same formula as `experiments/p10/supplement_s.normal_descriptors`, computed
  chunk by chunk. `fused_residual_mean_clip18` = z(SimpleShot) + z(residual), reported as a level only.
- `logreg_rfs`: `experiments/revision/c2_frozen_feature_baselines.logreg_rfs` (sklearn lbfgs, C = 1, max_iter 1000)
  on the global fused features.
- `oracle_maskpool_clip18` (**evaluation only; masks never touch any other method**): CLIP layer-18 patches pooled
  with the official defect-mask area per 24 × 24 cell (squash resize, as the evaluation view), for supports and
  queries; SimpleShot on the pooled descriptors. Same rule as `supplement_s.mask_pool` /
  `run_pilot.oracle_maskpool`: an image with an empty mask is pooled uniformly.
  - VisA masks: `<object>/Data/Masks/Anomaly/<id>.png`, values {0, 1}; a pixel counts when > 0.
  - MTD masks: the `.png` next to each image, 0 / 255 with compression noise; a pixel counts when ≥ 128.
    Four MT_Uneven masks are empty after this rule and are pooled uniformly.
  - Normal-class images (VisA `normal`, MTD `MT_Free`) have no mask and are pooled uniformly.

## Integrity gate (fail-closed, before anything is written)
`simpleshot_cl2n` and `fused_normref_clip18`, recomputed in the same run, must equal the P10 external raw metrics
exactly (max |Δ| < 1e-9 on BA, Macro-F1, top-1 for all 180 cells each). Otherwise the run stops and writes nothing.

## Report (every number, whatever it is; no method is selected, nothing is changed afterwards)
- Levels, VisA object macro and MTD, K = 1 / 3 / 5: the four new methods.
- Descriptive contrasts, VisA pooled over K with the object-cluster bootstrap (`external_evaluate.visa_contrast`)
  and the exact sign-flip test over the 11 objects; MTD seed mean ± SD per K:
  - R1 adaptive region (`fused_adareg_soft_q99`, P11) − residual_mean (Table 8 row 5);
  - R2 two-branch ARR (`fused_adareg_regres_soft_q99`, P14) − residual_mean;
  - R3 logreg_rfs − SimpleShot; R4 trainable adaptation (frozen `ours` cells) − SimpleShot;
    R5 oracle − SimpleShot (the Table 3 diagnosis on VisA / MTD);
  - R6 ARR − logreg_rfs.
- Oracle headroom recovered per K (macro means): (ARR − SimpleShot) / (oracle − SimpleShot), and the same for the
  adaptive region. With a normal class present the oracle can only help the defect classes; this is stated with
  the number.
- No Holm family: P27 adds no hypothesis. Nothing here can change the method or the frozen results.

## Execution
- Script `experiments/p27/external_fill.py`; single-shot output `ckpt/p27/external/` with a lock directory and
  per-task checkpoints in a staging directory (`--resume` continues after an interruption; checkpoints are bound
  to the git HEAD).
- Local GPU; features are never written to disk.

## Amendment A1 (2026-09-29, after attempt 1 failed the integrity gate; no P27 level was looked at)
- **Attempt 1:** extracted features in batches of 16, while P10 / P12 external used batches of 4.
  - Batched convolutions change the features at the 1e-6 level. That was enough to flip one query prediction of `fused_normref_clip18` in 8 of 180 cells (max |Δ| 0.0091, visa__pcb3). SimpleShot was exact.
  - The gate stopped the run before anything was written.
  - Attempt 1's lock and staging (`ckpt/p27/external.lock`, `external.staging`) are kept unchanged as a record.
- **Fix:** extraction batch fixed at 4 (`BATCH`), as in P10 / P12. With it, visa__pcb3 and visa__fryum reproduce P10 exactly for both integrity methods.
- **Attempt 2:** runs from scratch in its own staging and lock (`external.attempt2.*`). Nothing else changes: methods, contrasts, gate, and report are as above.
- Only the integrity rows of attempt 1 were read to find the cause.
