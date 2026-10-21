# P11 Pilot — Normal-calibrated adaptive region (A) and anomaly-guided cross-class matching (B)

Status: **DEVELOPMENT PROTOCOL, LOCKED BEFORE ANY P11 OUTPUT** (2026-09-27)

## 1. Motivation (from P10 results)

- The P10 normal-reference method (`fused_normref_clip18`) pools a **fixed** top 10 % (58)
  of the most anomalous CLIP layer-18 patches. The dev sensitivity grid showed it prefers a
  smaller fraction and that the best fraction differs by product; defect size varies a lot.
  The mask oracle, which pools exactly the defect area, is still ~11 pp higher at K = 5.
  **A** replaces the fixed fraction with a region whose size is decided per image by a
  threshold calibrated **on the normal images alone**.
- P10 qualitative maps: most failures are type confusions with correct localization, and
  the cross-class cue fails on grid / bottle where background patches do not align across
  classes. **B** chains the two cues: the normal reference says *where* (anomalous
  region), the cross-class margin says *which class*, computed only between anomalous
  regions of the query and of the supports.

Both are training-free and use the same information as P10 C1 (image-level labels +
M unlabeled normal images). No masks are used by any non-oracle method.

## 2. Firewall

- Identical to P10 / P10-A: only the MVTec-FS **support pool** dev features
  (`ckpt/p10_pilot/dev_features`) and the P10-A normal banks (`ckpt/p10_pilot/normal_features`,
  MVTec AD `train/good`, dev-seed draws). The 530 formal queries are never read.
- Dev episodes, seeds (9001–9005), K ∈ {1, 3}: exactly `run_pilot.episodes`.
- Masks are used only by the oracle row and the region-agreement diagnostic (§5), never by
  a method.
- Formal evaluation of any P11 method needs a separate formal addendum written after this
  pilot. It will be declared in the paper as a **post-hoc hypothesis added after the P10
  formal run**, evaluated once.

## 3. Common definitions

- Patches: CLIP ViT-L/14@336 layer-18 tokens, 576 per image, L2-normalized (P10 features).
- Bank: for dev seed s, the first M names of `normal["draws"][s]` (M = 20 primary, M = 5
  sensitivity), all their patches. Patch anomaly a(t) = 1 − max cosine to the bank
  (`run_pilot_a.patch_anomaly`).
- SimpleShot S, z-score fusion: as in P10 (`run_pilot.simpleshot`, `run_pilot._zscore`).
- Fixed region T(x): the ⌈0.1·576⌉ = 58 most anomalous patches (as `anomaly_pool`).

## 4. Candidate methods (constants fixed here)

### Family A — normal-calibrated adaptive region

- **Calibration (leave-one-image-out on the bank).** For each bank image j, compute the
  anomaly of its 576 patches against the other M − 1 bank images. Pool the M × 576 values;
  τ_q = their q-quantile (`torch.quantile`, linear interpolation). Uses normal images only.
- **Region.** R(x) = {t : a(t) > τ_q}. If |R(x)| < 6 (≈1 % of patches), R(x) = the 6 most
  anomalous patches. No upper cap.
- `fused_adareg_hard_q95`, `fused_adareg_hard_q99`, `fused_adareg_hard_q999`
  (q = 0.95, 0.99, 0.999): descriptor h(x) = mean of patches in R(x); SimpleShot on h;
  fused z(S) + z(·).
- `fused_adareg_soft_q99`: weights w(t) = max(a(t) − τ_0.99, 0); if fewer than 6 patches
  have w > 0, fall back to uniform weights on the 6 most anomalous patches;
  h(x) = Σ w t / Σ w; SimpleShot; fused.

### Family B — anomaly-guided cross-class matching

- Support class set U_c = union of T(x_i) patches over supports of class c.
- For each query patch t ∈ T(x_q): s_c(t) = max over u ∈ U_c of cos(t, u);
  margin m_c(t) = s_c(t) − max over c' ≠ c of s_c'(t).
- `fused_guided_plain`: G_c = mean over t ∈ T(x_q) of s_c(t); fused z(S) + z(G).
- `fused_guided_xclass`: G_c = mean over t ∈ T(x_q) of m_c(t); fused z(S) + z(G).
- `fused_guided_xclass_normref`: z(S) + z(R) + z(G_xclass), R = P10 normal-reference
  SimpleShot score on the fixed-region descriptor.

### References (recomputed in the same run)

`simpleshot_cl2n`, `normref_pool_clip18`, `fused_normref_clip18` (**comparison target**),
`fused_xclass_clip18`, `fused_residual_mean_clip18` (P10 Supplement S definition),
`oracle_maskpool_clip18`.

**Sanity:** `fused_normref_clip18` and `fused_xclass_clip18` must reproduce the P10 dev
values (+7.70 pp, +4.26 pp vs SimpleShot) to 1e-9; a mismatch is reported in the output
and the pilot is not interpreted until explained.

M = 5 rows (suffix `_m5`, calibration from the 5-image bank) are **sensitivity only**.

## 5. Gate, selection and diagnostics

- For each candidate, per product: mean over K and seeds of BA(candidate) − BA(`fused_normref_clip18`).
- **Gate:** mean over K of product-macro Δ ≥ **+1.0 pp** **and** Δ > 0 on **≥ 8 / 13** products.
- **Selection:** separately within family A and family B, among gate-passing M = 20
  candidates: most positive products, ties by larger mean Δ. At most one A and one B
  method can go to a formal addendum. If no candidate of a family passes, that family is
  reported as a negative dev result and is not run on formal data.
- Also reported (not gated): Δ vs SimpleShot, positive products vs SimpleShot, fraction of
  oracle headroom.
- **Region diagnostics (M = 20, over all seeds):** mean |R(x)| per image and fraction of
  images using the 6-patch fallback for each A variant; patch-level precision / recall / F1
  of the selected set vs `mask_grid > 0` for the fixed region T(x) and each hard A region
  (images with an empty mask skipped). Diagnostic only.

## 6. Outputs

`ckpt/p11_pilot/pilot_ab_results.json`, `ckpt/p11_pilot/pilot_ab_table.md` (write-once;
the script refuses to run if the directory exists).
