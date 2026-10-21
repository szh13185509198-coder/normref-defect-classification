# P11 formal F1 — MVTec-FS-RGB (single run, post-hoc hypotheses)

Commit `873c5dacb15aa6b27256afc81573bf654109e05c`. Integrity vs P10 formal: simpleshot_cl2n max|Δ|=0.0e+00, fused_normref_clip18 max|Δ|=0.0e+00, fused_normref_clip18_m5 max|Δ|=0.0e+00, oracle_maskpool_clip18 max|Δ|=0.0e+00

| Hypothesis | Contrast | K=1 | K=3 | K=5 | Pooled over K | Verdict |
|---|---|---|---|---|---|---|
| H4_primary_A | fused_adareg_soft_q99 − fused_normref_clip18 | +6.21 [+3.82, +8.40] | +6.25 [+3.33, +9.53] | +5.65 [+2.47, +9.31] | +6.04 [+3.68, +8.64] (12/13) | CONFIRMED |
| H5_secondary_B | fused_guided_xclass_normref − fused_normref_clip18 | +2.88 [+1.59, +4.32] | +1.69 [+0.74, +2.71] | +2.87 [+1.52, +4.22] | +2.48 [+1.45, +3.55] (12/13) | CONFIRMED |

Macro-F1 pooled: H4_primary_A +6.05 [+3.74, +8.57] (12/13); H5_secondary_B +2.74 [+1.66, +3.83] (12/13)

## Descriptive contrasts (BA, pooled over K)

| Contrast | Δ |
|---|---|
| fused_adareg_soft_q99_minus_simpleshot_cl2n | +12.25 [+8.64, +15.81] (13/13) |
| fused_guided_xclass_normref_minus_simpleshot_cl2n | +8.70 [+6.12, +11.14] (13/13) |
| fused_adareg_soft_q99_m5_minus_fused_normref_clip18_m5 | +6.05 [+3.86, +8.46] (12/13) |
| fused_guided_xclass_normref_m5_minus_fused_normref_clip18_m5 | +2.34 [+1.21, +3.53] (12/13) |
| fused_adareg_soft_q99_minus_residual_mean_clip18 | +5.18 [+3.08, +7.09] (12/13) |
| fused_adareg_soft_q99_minus_fused_residual_mean_clip18 | +6.78 [+4.24, +9.39] (12/13) |
| fused_adareg_soft_q99_minus_ours | +12.59 [+8.53, +16.36] (12/13) |
| fused_adareg_soft_q99_minus_mvrec_rgb | +19.39 [+14.04, +24.49] (13/13) |

## Levels (product-macro BA / Macro-F1 %)

| Method | K=1 | K=3 | K=5 |
|---|---|---|---|
| simpleshot_cl2n | 56.92 / 54.34 | 68.88 / 67.92 | 74.09 / 73.33 |
| fused_normref_clip18 | 62.99 / 60.52 | 75.89 / 75.04 | 79.67 / 79.24 |
| fused_adareg_soft_q99 | 69.19 / 66.96 | 82.14 / 81.34 | 85.32 / 84.65 |
| fused_guided_xclass_normref | 65.87 / 63.70 | 77.58 / 76.98 | 82.54 / 82.35 |
| fused_normref_clip18_m5 | 62.58 / 60.09 | 75.35 / 74.59 | 79.47 / 78.97 |
| fused_adareg_soft_q99_m5 | 68.74 / 66.69 | 81.64 / 80.82 | 85.19 / 84.45 |
| fused_guided_xclass_normref_m5 | 65.73 / 63.50 | 77.09 / 76.52 | 81.62 / 81.33 |
| oracle_maskpool_clip18 | 75.52 / 73.04 | 88.10 / 87.31 | 90.52 / 90.16 |
| residual_mean_clip18 | 62.87 / 60.59 | 77.18 / 76.13 | 81.06 / 80.75 |
| fused_residual_mean_clip18 | 62.35 / 59.93 | 74.75 / 74.00 | 79.21 / 78.77 |
| ours | 56.18 / 53.11 | 68.85 / 67.81 | 73.87 / 73.04 |
| mvrec_rgb | 47.44 / 44.70 | 61.10 / 59.69 | 69.94 / 69.27 |

Fraction of mask-oracle headroom over SimpleShot: K=1: fused_normref_clip18 0.33, fused_adareg_soft_q99 0.66, fused_guided_xclass_normref 0.48; K=3: fused_normref_clip18 0.36, fused_adareg_soft_q99 0.69, fused_guided_xclass_normref 0.45; K=5: fused_normref_clip18 0.34, fused_adareg_soft_q99 0.68, fused_guided_xclass_normref 0.51
