# P17 — Robustness analyses (report-only) and patch-level conformal calibration check

Status: **LOCKED BEFORE ANY P17 OUTPUT** (2026-09-27). User-approved reopening of the formal
sets **for report-only robustness analyses**. Rules:
- The main methods and every constant stay unchanged: ARR `fused_adareg_regres_soft_q99`
  (C1) and CONS `fused_cons_soft_supq90` (C0). **No P17 result may change the method, a
  constant or a claim of superiority of a variant.** Every grid cell is reported, none is selected.
- New method ideas are screened on dev first; at most one version per idea may later go to
  the formal sets, under its own protocol.

## D. Dev stage (no test data)

**D1 Patch-level conformal calibration check.** Claim to check: τ_q (q-quantile of
leave-one-image-out bank anomalies) bounds the fraction of *normal* patches selected as
evidence by ≈ 1 − q. Data: MVTec AD `train/good` held-out normals, the same selection as P15
dev (`random.Random(f"p15dev|{product}")`, excluding P10-A dev bank draws, up to 200 per
product); banks = P10-A dev draws (seeds 9001–9005, M = 20). For q ∈ {0.95, 0.97, 0.98, 0.99,
0.995, 0.999}: realized patch exceedance rate on held-out normals, fraction of normal images
with ≥ 1 selected patch, mean selected patches per normal image; on dev defect images (masks,
evaluation only): recall of mask patches and exceedance rate of non-mask patches.

**D2 Dev grids** (P10 dev episodes, K = 1/3): same variant list as §V; integrity — ARR equals
P13 dev, A equals P11 dev, CONS equals P13 dev (max |Δ| < 1e-12).

**D3 Dev-tuned fusion weights (the only fitted quantity, used only as a reported row):**
(w_S, w_A, w_R) ∈ {0, 0.5, 1, 1.5, 2}³ \ {0}; objective = mean over K = 1/3 of dev product-macro
BA; ties → smallest L1 distance to (1, 1, 1). The selected weights define the row
`arr_devweights`.

## V. Variants (all training-free; everything else identical to ARR / CONS)

| Group | Rows |
|---|---|
| Threshold quantile (ARR) | q ∈ {0.95, 0.97, 0.98, **0.99**, 0.995} |
| Fallback size (ARR, q = 0.99) | min region ∈ {1, 3, **6**, 12} |
| C0 threshold (CONS) | τ0 support quantile ∈ {0.80, 0.85, **0.90**, 0.95} |
| Fusion ablation | global only (SimpleShot); appearance only; deviation only; global + appearance (= A); global + deviation; appearance + deviation; **global + appearance + deviation (= ARR)**; dev-tuned weights |
| Patch backbone | ARR and A with DINOv2 ViT-S/14 patches (bank, τ, descriptors) instead of CLIP layer 18; global feature unchanged |

## F. Formal stages (report-only, each once; lock + staging)

- **mvtec:** MVTec-FS-RGB formal (cached features incl. DINOv2 patches). Integrity: SimpleShot,
  A, ARR, CONS equal P14 formal exactly.
- **external:** VisA + MTD, re-extracted exactly as P10–P15, additionally keeping DINOv2 patches
  from the same forward pass. Integrity: SimpleShot, A, ARR, CONS equal P14 external exactly.
- Reported per row: BA levels (K = 1/3/5), Δ vs ARR (C1 rows) or vs CONS (C0 rows) pooled
  over K with 95 % cluster-bootstrap CI (the P10 estimators) and exact product / object-level
  sign-flip p-value. **No verdicts.**

## Outputs

`ckpt/p17/{dev,mvtec,external}/`.
