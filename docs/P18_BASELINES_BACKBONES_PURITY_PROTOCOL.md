# P18 — Adapted published baselines, backbone swap, evidence purity (report-only)

Status: **LOCKED BEFORE ANY P18 OUTPUT** (2026-09-27). Same rules as P17: the main methods and
every constant stay unchanged — ARR `fused_adareg_regres_soft_q99` (C1) and CONS
`fused_cons_soft_supq90` (C0). **No P18 result may change a main method, a constant or a claim
about the main methods.** Every cell is reported. The only selections are made on dev, for the
*baselines* and the *alternative backbones*, by the rules below, before the formal stages.

## B. Adapted published baselines (main features: CLIP ViT-L/14@336 layer-18 patches + fusion global)

- **DN4, zero-training adaptation (C0).** Image-to-class local-descriptor measure of DN4
  (Li et al., CVPR 2019) without its episodic training: for every query patch, the sum of its
  k = 3 largest cosine similarities to the support patches of class c; class score = sum over all
  query patches. k = 3 is the DN4 default; nothing is tuned. Rows `dn4` (alone) and `dn4_fused`
  (= z(SimpleShot) + z(dn4), the same fusion as every other local cue).
- **Crop and re-encode (C1; AnomalyNCD / MVREC style "localise, crop, classify").** Region =
  patches with non-zero normal-calibrated weight (τ_0.99 from the same leave-one-out bank, soft
  weights, top-6 fallback — i.e. the region ARR uses). Box = bounding box of either the largest
  8-connected component (by summed weight) or of all region patches, enlarged about its centre by
  a context scale, at least 3 × 3 patch cells, shifted inside the image. The box is cropped from
  the original image and re-encoded with the frozen main extractor (CLIP ViT-L/14@336 projected
  global + DINOv2-S patch mean, concatenated, L2), then SimpleShot. Rows `crop` and `crop_fused`.
  **Dev selection** among component ∈ {largest, all} × scale ∈ {1.0, 1.5, 2.0}: highest dev mean
  BA of `crop_fused` (mean over K = 1/3); ties → smaller scale, then `largest`. This favours the
  baseline.

## K. Backbone swap (both global and patch features from one alternative backbone)

| Name | Weights | Input | Global | Patches (layer grid) |
|---|---|---|---|---|
| `dinov2_b` | DINOv2 ViT-B/14, official LVD-142M `pretrained/dinov2_vitb14_pretrain.pth` (SHA256 recorded by the dev stage; later stages require the same) | the main 336 × 336 view (576 patches) | final-layer normalised CLS token, L2 | `get_intermediate_layers(norm=True)` after block L ∈ {6, 8, 9, 10, 11, 12}, L2 |
| `clip_b16` | OpenAI CLIP ViT-B/16 `model/ViT-B-16.pt` (SHA256 `5806e77c…416f`) | r × r, r ∈ {224, 336} (bicubic resize, CLIP normalisation; position embeddings interpolated by the existing loader) → 196 / 441 patches | projected image embedding, L2 | residual stream after block L ∈ {6, 8, 9, 10, 11, 12}, CLS removed, L2 |

Everything else is identical to the main pipeline: same images, splits, supports, banks
(M = 20, same draws), LOO τ_0.99, soft weights, top-6 fallback (not rescaled to the patch
count), fixed fraction ⌈0.1 P⌉, cross-class top 10 %, consensus τ0 = 0.90, equal-weight z fusion.
Rows per backbone b: `ss_b`, `nr_b` (fixed fraction), `xc_b` (cross-class), `a_b`, `arr_b`,
`cons_b`, and `oracle_b` (mask pooling, MVTec-FS only, masks evaluation-only).
**Dev selection per backbone:** the configuration (layer, and resolution for `clip_b16`) with the
highest dev mean BA of `arr_b` (mean over K = 1/3); ties → lower layer, then lower resolution.
All configurations are reported.

## P. Evidence purity and background exceedance (dev only; masks for evaluation only)

On the dev pool with the P10 dev banks (CLIP layer 18), per image with a non-empty mask grid:
- pooling weights of seven schemes: global (uniform), fixed hard top 10 % (P10), fixed soft top 10 %
  (P12), Otsu soft, MEBin-style soft (P12), ours hard (w > 0), ours soft (ARR's weights);
- evidence purity λ_any = Σ_{m>0} w / Σ w and λ_area = Σ w·m / Σ w (m = mask area fraction per
  patch); defect fraction ρ = share of patches with m > 0;
- reported: mean λ per scheme; mean λ per scheme in fixed ρ bins [0, 2 %), [2, 5 %), [5, 15 %),
  [15 %, 100 %]; per-product λ; Spearman correlation over the 13 products between
  λ_any(ours soft) − λ_any(global) and the dev BA gain ARR − SimpleShot (descriptive, n = 13);
- background exceedance (a > τ_0.99) of non-mask patches by Chebyshev distance to the nearest
  mask patch: 1, 2, 3, 4–5, ≥ 6 cells (macro over product × seed units); share of the soft weight
  on mask / near background (d ≤ 2) / far background (d ≥ 3).

## F. Formal stages (each once; lock + staging)

- **mvtec:** MVTec-FS-RGB formal. Integrity: SimpleShot, fixed fraction, cross-class, A, ARR,
  CONS equal P14 formal exactly (else abort).
- **external:** VisA + MTD, re-extracted exactly as P14. Same integrity vs P14 external.
- Reported contrasts (pooled over K, 95 % cluster-bootstrap CI, exact sign-flip p; MTD
  descriptive): `crop_fused − SS`, `dn4_fused − SS`, `ARR − crop_fused`, `CONS − dn4_fused`,
  `ARR − dn4_fused`, `crop − SS`, `dn4 − SS`; per backbone b: `arr_b − ss_b`, `nr_b − ss_b`,
  `a_b − nr_b`, `arr_b − a_b`, `cons_b − xc_b`, `cons_b − ss_b`, `arr_b − ARR`; MVTec-FS
  oracle-headroom fraction (arr − ss) / (oracle − ss) per K for the main features and each b.
- **Declared descriptive readings (no method change):** "the principle transfers to backbone b"
  iff the pooled CI of `arr_b − ss_b` excludes 0 on both MVTec-FS and VisA; "normal calibration
  beats the fixed fraction under b" iff that holds for `a_b − nr_b`; for baselines, whether the CIs
  of `ARR − crop_fused` and `CONS − dn4_fused` exclude 0.

## Outputs

`ckpt/p18/{dev,mvtec,external}/`.
