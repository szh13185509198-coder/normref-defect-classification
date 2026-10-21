# P10 Figures and dev-only sensitivity — locked rules

Status: **LOCKED BEFORE ANY OUTPUT** (2026-09-26).

## 1. Dev-only hyperparameter sensitivity (supplementary)

- Data: the P10 dev episodes only (MVTec-FS support pool, dev seeds 9001–9005, K = 1/3,
  M = 20 normal banks from MVTec AD `train/good`). Formal queries are never read.
- Grid: CLIP layer L in {12, 18, 24} x top fraction rho in {0.05, 0.10, 0.20, 0.30}.
- Methods: `simpleshot_cl2n` (reference, grid-independent), `fused_xclass` (C0, uses L and
  rho), `fused_normref` (C1, uses L and rho), `residual_mean` (C1, uses L only).
- Report: mean over K of dev product-macro BA minus SimpleShot, and positive products.
- **No selection:** the formal variants (L = 18, rho = 0.10) stay fixed whatever the grid shows;
  the grid is reported in the supplement as robustness evidence only.
- Sanity: the (18, 0.10) cells must reproduce the P10 dev results
  (`fused_normref_clip18` +7.70 pp, `fused_xclass_clip18` +4.26 pp); a mismatch is reported.

## 2. Figures (all from frozen result files; each writes its plotted numbers to CSV)

- **Line chart** (MVTec-FS formal): product-macro BA vs K for mask oracle (CLIP-18),
  normal-reference, residual, cross-class, SimpleShot, V1 Ours, MVREC-RGB adaptation.
- **Per-product gains** (MVTec-FS formal): mean over K and seeds of
  `fused_normref_clip18` − SimpleShot and `fused_xclass_clip18` − SimpleShot per product.
- **VisA forest plot**: E1, E2, E3 per K and mean over K with 95% object-cluster CIs.
- **Industrial plot**: VisA and MTD false rejects and escapes per 1,000 parts at 95% good
  (R2), for SimpleShot, cross-class, normal-reference, V1 Ours, MVREC-RGB.
- **Qualitative anomaly maps — deterministic selection (no hand-picking):**
  - Products: the 3 largest and 3 smallest mean-over-K H1 per-product gains on MVTec-FS formal.
  - For each product, at K = 5, seed 111, among formal queries sorted by `rgb_sha256`:
    *success* = first query where `fused_normref_clip18` is correct and `simpleshot_cl2n` is
    wrong; *failure* = first query where `fused_normref_clip18` is wrong. If none exists, the
    panel says so.
  - Map: patch anomaly 1 − max cosine to the seed-111 M = 20 bank (CLIP layer 18), bilinearly
    upsampled; the GT polygon is drawn as an outline **for visualization only**.
  - Cable panel: the first two `combined` images in R1's persistent-error list of
    `fused_normref_clip18`, with maps from the seed-111 Cable bank.
